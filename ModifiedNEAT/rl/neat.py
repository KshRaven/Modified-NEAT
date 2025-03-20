
from ModifiedNEAT.nn.base import Model
from ModifiedNEAT.population import Population
from ModifiedNEAT.rl.base import Algorithm, TensorDict
from ModifiedNEAT.optim.scheduler import Scheduler
from ModifiedNEAT.util.datetime import unix_to_datetime_file, eta, clock
from ModifiedNEAT.util.qol import manage_params
from ModifiedNEAT.util.storage import STORAGE_DIR
from ModifiedNEAT.util.fancy_text import CM, Fore

from torch import Tensor
from typing import Union, Iterable

import torch
import numpy as np
import matplotlib.pyplot as plt


class NEAT(Algorithm):
    def __init__(self, population: Population, schedulers: Union[Scheduler, Iterable[Scheduler]],
                 device=torch.device('cpu'), dtype=torch.float32, **options):
        """
        Neuro-Evolution of Augmenting Topologies (NEAT) algorithm initialization.

        :param model: The model to be used, expected to be of type Reformer.
        :param device: The device to be used for computation (default is 'cpu').
        :param dtype: The data type for tensors (default is torch.float32).
        :param options: Additional keyword arguments for configuration.
        :keyword lr (learning_rate): The learning rate for the optimizer (default is 5e-4).
        :keyword wd (weight_decay): The weight decay (L2 penalty) for the optimizer (default is 0.0).
        :keyword norm (norm_rew, normalize_rewards): Whether to normalize rewards (default is True).
        :keyword gamma: The discount factor for rewards (default is 0.95).
        :keyword epsilon: A small value to ensure numerical stability (default is 1e-10).
        :keyword clip_range: The range for clipping the objective function (default is 0.2).
        :keyword ent_reg: The regularization coefficient for entropy (default is 0.01).
        :keyword val_reg: The regularization coefficient for value function loss (default is 1.0).
        :keyword clip_grad: Gradient clipping threshold (default is None).
        :keyword opt: Custom optimizer instance (default is AdamW with specified learning_rate and weight_decay).
        :keyword log_dir:
        :keyword log_name:
        """
        for var in ['schedulers', 'device', 'dtype']:
            if var in options:
                del options[var]
        super().__init__(population, schedulers, device, dtype, **options)

        # Buffers
        self.replay.add_buffers('state', 'action', 'reward', 'ep_map')

        # Options
        self.gamma: float       = manage_params(options, 'gamma', 0.95)
        self.alpha: float       = manage_params(options, 'alpha', 1.10)
        self.reverse: bool      = manage_params(options, 'reverse', False)
        self.epsilon: float     = manage_params(options, 'epsilon', 1e-10)
        self.rew_reg: float     = manage_params(options, 'rew_reg', 1.0)
        self.pol_reg: float     = manage_params(options, 'pol_reg', 0.0)
        self.validate: bool     = manage_params(options, 'validate', False)
        self.segr_size: Union[float, None] = manage_params(options, 'segr_size', None)
        self.target_kl: Union[float, None] = manage_params(options, 'target_kl', None)

        self.logging.add_buffers(
            'kl_divergence', 'explained_variance', 'std',
            'ep_len_mean', 'ep_len_std', 'ep_rew_mean', 'ep_rew_std', 'policy_acc', 'reward_acc'
        )

    def update(self, observations: Tensor, actions: Tensor, rewards: Tensor,
               terminated: Union[bool, list[bool]], force_stop=False, envs: list[object, list[object]] = None):
        if isinstance(terminated, bool):
            terminated = [terminated]
        if envs is not None and not isinstance(envs, (list, tuple)):
            envs = [envs]

        # Handle episode mapping
        if self.steps_done == self.prev_steps_done:
            # Give env new mapping if it is done
            if envs is not None:
                self.episode_mapping: dict[int, int] = {
                    env_idx: next(self.mapping_indexer)
                    if (self.episode_mapping.get(env_idx) is None or hasattr(env, 'done') and env.done)
                    else self.episode_mapping[env_idx]
                    for env_idx, env in enumerate(envs)
                }
            # When no env obj passed remap all env
            else:
                self.episode_mapping = {
                    env_idx: next(self.mapping_indexer) for env_idx in range(len(terminated))
                }
            # Delete invalid ep mapping
            for env_idx in list(self.episode_mapping.keys()):
                if 0 > env_idx > len(terminated) - 1:
                    del self.episode_mapping[env_idx]
            mapping_list = list(self.episode_mapping.values())
            # Initialize the episode lengths mapping
            if self.episode_lengths is None:
                self.episode_lengths: dict[int, int] = {
                    env_map: 0 for env_map in self.episode_mapping.values()
                }
            # Reset episode lengths after training regardless if env was not done and remove invalids
            else:
                for ep_map in list(self.episode_lengths.keys()):
                    if ep_map not in mapping_list:
                        del self.episode_lengths[ep_map]
                    else:
                        self.episode_lengths[ep_map] = 0
                for ep_map in mapping_list:
                    if ep_map not in self.episode_lengths:
                        self.episode_lengths[ep_map] = 0

        filled = False
        if self.steps_done < self.steps_limit:
            for idx, ended in enumerate(terminated):
                self.replay.update(
                    state       = observations,
                    action      = actions,
                    reward      = rewards,
                    ep_map      = self.episode_mapping[idx],
                    # key_index   = key_index
                )

                self.steps_done += 1
                self.episodes_done = max(self.episode_mapping.values()) + 1
                self.episode_lengths[self.episode_mapping[idx]] += 1

                if self.steps_done >= self.steps_limit or force_stop:
                    self.prev_steps_done = self.steps_done
                    filled = True
                else:
                    if ended:
                        self.episode_mapping[idx] = next(self.mapping_indexer)
                        self.episode_lengths[self.episode_mapping[idx]] = 0
        else:
            raise RuntimeError(f"Steps have already been filled.")

        return filled

    def set_scores(self, scores: dict[int, float], policy: dict[int, float]):
        assert all([key in policy for key in scores.keys()])
        if self.pol_reg != 0:
            scores = dict(sorted(self.normalize(scores).items(), key=lambda item: (item[1], -item[0]), reverse=True))
            policy = self.normalize(policy, None, self.segr_size, scores)
            true_scores = {key: self.rew_reg*scores[key] + self.pol_reg*policy[key] for key in scores.keys()}
            true_scores = dict(sorted(true_scores.items(), key=lambda item: (item[1], -item[0]), reverse=True))
        else:
            true_scores = dict(sorted(scores.items(), key=lambda item: (item[1], -item[0]), reverse=True))

        available_keys = list(true_scores.keys())
        available_scores = list(true_scores.values())

        min_fitness = min(list(true_scores.values()))
        # Assuming scores were set during evaluation function then norm the invalid keys below the valid ones
        pre_set = all([genome.fitness is not None for genome in self.population.genomes.values()])
        fitnesses = [genome.fitness for genome in self.population.genomes.values()]
        maximum = max(fitnesses) if pre_set else None
        minimum = min(fitnesses) if pre_set else None
        for genome in self.population.genomes.values():
            if genome.key in available_keys:
                genome.fitness = true_scores[genome.key]
            else:
                if pre_set:
                    genome.fitness = min_fitness - ((maximum - genome.fitness) / (maximum - minimum))
                else:
                    genome.fitness = min_fitness - self.epsilon

        criterion = self.population.config.general.fitness_criterion
        if criterion == 'max':
            best_genome_key = available_keys[np.argmax(available_scores)]
        elif criterion == 'min':
            best_genome_key = available_keys[np.argmin(available_scores)]
        elif criterion == 'mean':
            best_genome_key = available_keys[np.argmin((available_scores.mean() - available_scores) ** 2)]
        else:
            raise ValueError(f"unsupported fitness criteria")

        return best_genome_key

    def learn(self, evaluation_function: callable, steps: int, epochs: int = None, batch_size: int = None,
              accuracy_error=0.20, accuracy_type='continuous', verbose: int = None):
        print(f"Logging to {self.log_dir+self.log_name}")
        if epochs is None:
            epochs = np.inf
        epoch_done = 0
        while epoch_done < epochs:
            torch.cuda.empty_cache()

            # Running environment to collect rollout data
            ts = clock.perf_counter()
            self.init_limit(steps)
            self.population.run(evaluation_function, 1, verbose=verbose, skip=True, trainer=self)
            run_time = clock.perf_counter() - ts
            if verbose and verbose >= 2:
                print(f"collected data in {CM(f'{round(run_time, 2)}s', Fore.LIGHTCYAN_EX)}")

            # Rolling out genomes that are not to be deleted
            pts = clock.perf_counter()
            invalid_population = len(self.population.to_delete) == len(self.population.genomes)
            valid_keys = [
                key for key in self.replay.mapping.keys() if key not in self.population.to_delete or invalid_population
            ] if self.validate else list(self.population.genomes.keys())
            # TODO: Check whether only rolling out valid keys is necessary
            with torch.no_grad():
                # Roll out data from buffers
                ts = clock.perf_counter()
                states, actions, rewards, episode_mapping = self.replay.rollout(
                    ['state', 'action', 'reward', 'ep_map'], as_list=True, stack=True, keys=valid_keys
                )
                rollout_time = clock.perf_counter() - ts
                if verbose and verbose >= 2:
                    print(f"rolled out data in {CM(f'{round(rollout_time, 2)}s', Fore.LIGHTCYAN_EX)}")

                # Sort episodes wrt. episode mapping
                ts = clock.perf_counter()
                episode_lengths = self.sort_episodes(episode_mapping, states, actions, rewards)
                sort_time = clock.perf_counter() - ts
                if verbose and verbose >= 2:
                    print(f"sorted episodic data in {CM(f'{round(sort_time, 2)}s', Fore.LIGHTCYAN_EX)}")

                # Compute returns
                ts = clock.perf_counter()
                cum_rewards = self.compute_returns(rewards, self.gamma, self.alpha, self.reverse, episode_mapping)
                ret_comp_time = clock.perf_counter() - ts
                if verbose and verbose >= 2:
                    print(f"computed returns in {CM(f'{round(ret_comp_time, 2)}s', Fore.LIGHTCYAN_EX)}")

                # Compute scores from returns
                scores: dict[int, float] = {
                    key: np.mean([
                        torch.mean(cum_rewards[key][indices]).item()
                        for indices in [
                            [idx for idx, ep_idx in enumerate(episode_mapping[key]) if ep_idx == u_idx]
                            for u_idx in np.unique(episode_mapping[key])
                        ]
                    ])
                    for key in valid_keys
                }

                # Get batches wrt. steps done per key
                batch_indices = self.get_batches(valid_keys, batch_size, True)

            # Calculate and set scores
            with torch.no_grad():
                policy_accuracy = self.get_accuracy(batch_indices, states, actions, None,
                                                    accuracy_error, accuracy_type, verbose, keys=valid_keys)[0]
                ts = clock.perf_counter()
                best_genome_key = self.set_scores(scores, policy_accuracy)
                set_time = clock.perf_counter() - ts
                if verbose and verbose >= 2:
                    print(f"set scores in {CM(f'{round(set_time, 2)}s', Fore.LIGHTCYAN_EX)}")

            # Post Calculations
            with torch.no_grad():
                ts = clock.perf_counter()
                ep_len_mean = np.mean(episode_lengths[best_genome_key]).item()
                ep_len_std  = np.std(episode_lengths[best_genome_key]).item()
                episode_indices: list[int] = np.unique(episode_mapping[best_genome_key])
                episode_rewards = [[reward.mean().cpu().item()
                                    for idx, reward in enumerate(rewards[best_genome_key])
                                    if episode_mapping[best_genome_key][idx] == ep_idx]
                                   for ep_idx in episode_indices]
                cum_episode_rewards = [[reward.mean().cpu().item()
                                        for idx, reward in enumerate(cum_rewards[best_genome_key])
                                        if episode_mapping[best_genome_key][idx] == ep_idx]
                                       for ep_idx in episode_indices]
                ep_rew_mean = np.mean([np.mean(episode) for episode in episode_rewards]).item()
                ep_rew_std = np.mean([np.std(episode) for episode in episode_rewards]).item()
                ep_cum_rew = np.mean([np.mean(episode) for episode in cum_episode_rewards]).item()
                try:
                    policy_acc = policy_accuracy[best_genome_key] * 100
                except RuntimeError:
                    policy_acc = np.nan, np.nan
                explained_variance = self._explained_variance(batch_indices, states, rewards, best_genome_key)[best_genome_key]
                policy_reduction = 1 if len(policy_accuracy) <= 1 else sorted(
                    list(policy_accuracy.keys()), key=lambda k: policy_accuracy[k]
                ).index(best_genome_key) / (len(policy_accuracy)-1)
                buffer_sizes = self.replay.buffer_sizes()

                # noinspection PyBroadException
                def get_range(key: Union[int, None]):
                    try:
                        params = []
                        if key is not None:
                            for param in self.get_module(key).neat_parameters():
                                params.append(param[key].flatten())
                        else:
                            for module in self.models:
                                for param in module.neat_parameters():
                                    params.append(param[key].flatten())
                        params = torch.cat(params)
                        return torch.mean(params).cpu().item(), torch.std(params).cpu().item(), \
                               torch.max(params).cpu().item(), torch.min(params).cpu().item()
                    except Exception:
                        return torch.nan, torch.nan, torch.nan, torch.nan

                mean, std, maximum, minimum = get_range(best_genome_key)
                processing_time = np.floor(clock.perf_counter() - pts)

                if verbose and verbose >= 2:
                    print(f"calculated stats in {CM(f'{round(clock.perf_counter() - ts, 2)}s', Fore.LIGHTCYAN_EX)}")

            # Logging
            with torch.no_grad():
                self.logging.update(
                    ep_len_mean=ep_len_mean, ep_len_std=ep_len_std, ep_rew_mean=ep_rew_mean, ep_rew_std=ep_rew_std,
                    policy_acc=policy_acc, explained_variance=explained_variance, std=std,
                )

                # Rollout
                extra = 'rollout/'
                # rollout = self.writer.file_writer()
                self.writer.add_scalar(extra+'ep_len_mean', ep_len_mean, self.updates_done)
                self.writer.add_scalar(extra+'ep_len_std', ep_len_std, self.updates_done)
                self.writer.add_scalar(extra+'ep_rew_mean', ep_rew_mean, self.updates_done)
                self.writer.add_scalar(extra+'ep_rew_std', ep_rew_std, self.updates_done)
                self.writer.add_scalar(extra+'ep_cum_rew', ep_cum_rew, self.updates_done)
                self.writer.add_scalar(extra+'buffer_size', buffer_sizes[best_genome_key], self.updates_done)

                # Time
                extra = 'time/'
                self.writer.add_scalar(extra+'run_time', run_time, self.updates_done)
                # self.writer.add_scalar(extra+'train_time', train_time, self.updates_done)
                # self.writer.add_scalar(extra+'train_time', train_time, self.updates_done)
                # self.writer.add_scalar(extra+'train_time', train_time, self.updates_done)
                # self.writer.add_scalar(extra+'train_time', train_time, self.updates_done)
                self.writer.add_scalar(extra+'processing_time', processing_time, self.updates_done)

                # Training
                extra = 'policy/'
                self.writer.add_scalar(extra+'explained_variance', explained_variance, self.updates_done)
                self.writer.add_scalar(extra+'policy_accuracy', policy_acc, self.updates_done)
                self.writer.add_scalar(extra+'policy_reduction', policy_reduction, self.updates_done)
                # self.writer.add_scalar(extra+'kl_divergence', kl_divergence, self.updates_done)

                # Module
                extra = 'module/'
                self.writer.add_scalar(extra+'param_mean', mean, self.updates_done)
                self.writer.add_scalar(extra+'param_std', std, self.updates_done)
                self.writer.add_scalar(extra+'param_min', minimum, self.updates_done)
                self.writer.add_scalar(extra+'param_max', maximum, self.updates_done)
                for param, label in zip(get_range(None), ['mean', 'std', 'max', 'min']):
                    self.writer.add_scalar(extra+f'global_{label}', param, self.updates_done)

                # Population
                survival_rate = len(valid_keys) / len(self.population.genomes)
                best_genome = self.population.genomes[best_genome_key]
                extra = 'population/'
                self.writer.add_scalar(extra+'best_genome', best_genome.key, self.updates_done)
                self.writer.add_scalar(extra+'best_fitness', best_genome.fitness, self.updates_done)
                self.writer.add_scalar(extra+'survival_rate', survival_rate, self.updates_done)

                # Schedule
                extra = 'schedule/'
                for param, param_value in self.log_scheduler_params().items():
                    self.writer.add_scalar(extra+param, param_value, self.updates_done)

            self.writer.flush()
            self.updates_done += 1

            # Displaying
            if verbose:

                bar = "-"*54
                print(
                    f"\n{bar}"
                    f"\n{'|ROLLOUT:': <29}|{'': <22} |"
                    f"\n|\t{'ep_len_mean': <25}| {ep_len_mean: <21} |"
                    f"\n|\t{'ep_len_std': <25}| {ep_len_std: <21} |"
                    f"\n|\t{'ep_rew_mean': <25}| {ep_rew_mean: <21} |"
                    f"\n|\t{'ep_rew_std': <25}| {ep_rew_std: <21} |"
                    f"\n|\t{'ep_cum_rew': <25}| {ep_cum_rew: <21} |"
                    f"\n{'|TIME:': <29}|{'': <22} |"
                    f"\n|\t{'epochs_done': <25}| {epoch_done+1: <21} |"
                    f"\n|\t{'run_time': <25}| {run_time: <21} |"
                    # f"\n|\t{'train_time': <25}| {train_time: <21} |"
                    f"\n|\t{'steps_done': <25}| {self.steps_done: <21} |"
                    f"\n|\t{'episodes_done': <25}| {self.episodes_done+1: <21} |"
                    f"\n|\t{'current_episodes_done': <25}| {len(self.episode_lengths) if self.episode_lengths else 1: <21} |"
                    f"\n{'|TRAINING:': <29}|{'': <22} |"
                    # f"\n|\t{'kl_divergence': <25}| {kl_divergence: <21} |"
                    f"\n|\t{'updates_done': <25}| {self.updates_done: <21} |"
                    # f"\n|\t{'weight_mutate_power': <25}| {self.population.config.genome.weight_mutate_power: <21} |"
                    # f"\n|\t{'bias_mutate_power': <25}| {self.population.config.genome.bias_mutate_power: <21} |"
                    f"\n|\t{'explained_variance': <25}| {explained_variance: <21} |"
                    f"\n|\t{'std': <25}| {std: <21} |"
                    f"\n|\t{'policy_accuracy': <25}| {policy_acc: <21} |"
                    # f"\n|\t{'reward_accuracy': <25}| {reward_acc: <21} |"
                    f"\n{bar}"
                )

            if self.schedulers is not None:
                for s in self.schedulers:
                    s.step()

            if epoch_done == epochs - 1:
                self.population.run(evaluation_function, 1, verbose=verbose, skip=True, trainer=self,
                                    terminate_skip=True)

            epoch_done += 1

    def _explained_variance(self, batches: dict[int, list[list[int]]], states: TensorDict, rewards: TensorDict,
                            keys: Union[int, list[int]] = None):
        if isinstance(keys, (int, float)):
            keys = [keys]
        keys = list(self.replay.mapping.keys()) if keys is None else keys
        with torch.no_grad():
            ex_var = {}

            # Calculate accuracy for each key
            for key in keys:
                ev = []
                for batch in batches[key]:
                    # Calculate
                    state = states[key][batch].to(self.device)
                    value: Tensor = self.get_module(key).get_value(state.unsqueeze(0), keys=key).squeeze(0)
                    reward = rewards[key][batch].to(self.device)
                    value = ((torch.std(reward - value) ** 2) / (torch.std(reward) ** 2)) - 1
                    ev.append(value)

                ex_var[key] = torch.clamp(torch.mean(torch.stack(ev)), None, 1).cpu().item()

            return ex_var


if __name__ == '__main__':
    import ModifiedNEAT as neat
    from ModifiedNEAT.nn.modules.main import Linear
    # from ModifiedNEAT.util.datetime import eta, clock

    SEED = 100
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    class RModel(Model):
        def __init__(self, inputs, outputs, dim_size, device, dtype):
            super().__init__()
            self.act_proj   = Linear(inputs, dim_size, True, device, dtype)
            self.mean       = Linear(dim_size, outputs, True, device, dtype)
            self.log_std    = Linear(dim_size, outputs, True, device, dtype)
            self.rew_proj   = Linear(inputs, dim_size, True, device, dtype)
            self.decode     = Linear(dim_size, 1, True, device, dtype)

        def forward(self, state: Tensor):
            return self.get_policy(state)

        def get_mean(self, latent: Tensor, key: int = None) -> Tensor:
            return self.mean(latent, key=key) * 100

        def get_std(self, latent: Tensor, key: int = None) -> Tensor:
            return 10 ** (-2 + self.log_std(latent, key=key) * 3)

        def get_action(self, state: Tensor, key: int = None) -> tuple[Tensor, Tensor]:
            latent = self.act_proj(state, key=key)
            mean, std = self.get_mean(latent, key=key), self.get_std(latent, key=key)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob

        def evaluate_action(self, state: Tensor, action: Tensor, key: int = None) -> [Tensor, Union[Tensor, None]]:
            latent = self.act_proj(state, key=key)
            mean, std = self.get_mean(latent, key=key), self.get_std(latent, key=key)
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            return log_prob, entropy

        def get_policy(self, state: Tensor, key: int = None, **options) -> Tensor:
            latent = self.act_proj(state, key=key)
            mean, std = self.get_mean(latent, key=key), self.get_std(latent, key=key)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            return action

        def get_value(self, state: Tensor, key: int = None) -> Tensor:
            latent = self.rew_proj(state, key=key)
            value = self.decode(latent, key=key)
            return value

    DEVICE      = 'cuda'
    DTYPE       = torch.float32
    INPUTS      = 2
    OUTPUTS     = 2
    DIM_SIZE    = 64
    MODEL       = RModel(INPUTS, OUTPUTS, DIM_SIZE, DEVICE, DTYPE)
    CONFIG      = neat.Config("ppo_test")
    CONFIG.reproduction.elitism = 50
    CONFIG.reproduction.min_species_size = 100
    GENOMES     = 100
    POPULATION  = neat.Population(GENOMES, MODEL, CONFIG, init_reporter=True)
    TRAINER     = NEAT(POPULATION, DEVICE, DTYPE, loss_reg=0.1, gamma=0.0,
                       scheduler=neat.optim.scheduler.CosineAnnealing(CONFIG, 100, 50, 0.001, True, True))
    STEPS       = 100

    def evaluate(population: Population, **options):
        trainer: NEAT = options['trainer']
        trainer.update_mapping(population.get_mapping())
        genomes = len(population.genomes)

        terminate = False
        step = 0
        ts, ud, ut = clock.perf_counter(), 0, STEPS
        while not terminate:
            observations = torch.randint(0, 100, (genomes, INPUTS), device=DEVICE, dtype=DTYPE)
            actions, probs = MODEL.get_action(observations)
            # probs = torch.rand(GENOMES, OUTPUTS, device=DEVICE, dtype=DTYPE)
            rewards = 1 - ((observations - actions) ** 2)

            terminate = trainer.update(observations, actions, probs, rewards, step == STEPS-1)

            ud += 1
            eta(ts, ud, ut, 'fetching data')
            step += 1
        trainer.deque_episodes(20)
        print(f"")

    # POPULATION.load_dict(name='ppo_test')

    TRAINER.learn(evaluate, STEPS, 500, 64, 0.50, 'continuous', True)

    POPULATION.save_dict('ppo_test')
