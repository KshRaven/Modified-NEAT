
from build.nn.base import Model
from build.population import Population
from build.rl.base import Algorithm
from build.optim.scheduler import Scheduler
from build.util.datetime import eta, clock, unix_to_datetime_file
from build.util.qol import manage_params
from build.util.storage import STORAGE_DIR
from build.util.fancy_text import CM, Fore

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from typing import Union

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

TensorDict = dict[int, Tensor]


class PPO(Algorithm):
    def __init__(self, model: Model, population: Population, device=torch.device('cpu'), dtype=torch.float32, **options):
        """
        Proximal Policy Optimization (PPO) algorithm initialization.

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
        super().__init__(model, population)

        # Buffers
        self.replay.add_buffers('state', 'action', 'prob', 'reward', 'ep_map')

        # Options
        self.norm_rew: bool               = manage_params(options, ['norm_rew'], False)
        self.norm_adv: bool               = manage_params(options, ['norm_adv'], False)
        self.gamma: float                 = manage_params(options, 'gamma', 0.95)
        self.alpha: float                 = manage_params(options, 'alpha', 1.10)
        self.alpha_step: int              = manage_params(options, 'alpha_step', 5)
        self.epsilon: float               = manage_params(options, 'epsilon', 1e-10)
        self.clip_range: float            = manage_params(options, 'clip_range', 0.3)
        self.pol_reg: float               = manage_params(options, 'pol_reg', 1.0)
        self.val_reg: float               = manage_params(options, 'val_reg', 0.9)
        self.ent_reg: float               = manage_params(options, 'ent_reg', 0.3)
        self.loss_reg: float              = manage_params(options, 'loss_reg', 0.9)
        self.target_kl: [float, None]     = manage_params(options, 'target_kl', None)
        self.scheduler: [Scheduler, None] = manage_params(options, 'scheduler', None)

        # Tensorboard logging
        self.log_dir: str = manage_params(
            options, 'log_directory', STORAGE_DIR+f"neat_rl_logs\\{self.__class__.__name__}\\")
        self.log_sub_dir: str = manage_params(options, 'log_sub_dir', "")
        self.log_name: str = manage_params(
            options, 'log_name', f"log~{unix_to_datetime_file(clock.time())}")
        self.writer = SummaryWriter(self.log_dir+self.log_sub_dir+self.log_name)
        self.logging.add_buffers(
            'kl_divergence', 'clip_fraction', 'clip_range', 'explained_variance',
            'weight_mutate_power', 'bias_mutate_power',
            'loss', 'policy_loss', 'value_loss', 'entropy_loss', 'std',
            'ep_len_mean', 'ep_len_std', 'ep_rew_mean', 'ep_rew_std', 'policy_acc', 'reward_acc'
        )

        # States
        self.device         = device
        self.dtype          = dtype
        self.steps_done     = 0
        self._steps_limit: int = None
        self.episodes_done  = 0
        self._ep_mapping: list[int] = None
        self._ep_started: list[bool] = None
        self._ep_offset: int = 0
        self.updates_done   = 0

        # Timing
        self.__ts: int      = None
        self.__ud: int      = None
        self.__ut: int      = None

    def init(self, steps: int):
        self._steps_limit = self.steps_done + steps
        self._ep_mapping = None
        self._ep_started = None
        self._ep_offset = 0

    def update_mapping(self, mapping: dict[int, int]):
        self.replay.update_mapping(mapping)
        self.logging.update_mapping(mapping)

    def update(self, observations: Tensor, actions: Tensor, probs: Tensor, rewards: Tensor,
               terminated: Union[bool, list[bool]], key_index=-2, force_stop=False):
        if isinstance(terminated, bool):
            terminated = [terminated]

        envs = len(terminated)
        if self._ep_mapping is None:
            self._ep_mapping = [self.episodes_done + ex for ex in range(envs)]
            self._ep_started = [True for _ in range(envs)]
            self._ep_offset = envs - 1

        filled = False
        if self.steps_done < self._steps_limit:
            for idx, (ended, started) in enumerate(zip(terminated, self._ep_started)):
                if not started and not ended:
                    started = self._ep_started[idx] = True
                    self._ep_offset += 1
                    self._ep_mapping[idx] = self.episodes_done + self._ep_offset

                if started:
                    episode_index = self._ep_mapping[idx]
                    self.replay.update(
                        state       = observations,
                        action      = actions,
                        prob        = probs,
                        reward      = rewards,
                        ep_map      = episode_index,
                        # key_index   = key_index
                    )
                    self.steps_done += 1

                if ended:
                    self._ep_started[idx] = False

            if self.steps_done >= self._steps_limit or force_stop:
                self.episodes_done += self._ep_offset + 1
                filled = True
        else:
            raise ValueError(f"Steps have already been filled.")

        return filled

    def deque_episodes(self, episodes: int, keys: list[int] = None):
        """
        Deletes the episodes before the last n episodes.
        Used to compensate for the long data collection times of the NEAT evaluation functions.
        :param episodes: (int) Number of recent episodes to keep.
        :param keys: (list[int])
        :return: (none)
        """
        assert episodes >= 0
        filters = {}
        episode_mapping: dict[int, list[int]] = self.replay.rollout(buffers='ep_map', as_list=True)[0]
        for key in self.replay.mapping.keys():
            episodes_to_del = torch.tensor([ep for ep in range(self.episodes_done) if ep < (self.episodes_done-episodes)])
            mapping = torch.tensor(episode_mapping[key])
            episode_filter  = torch.isin(mapping, episodes_to_del)
            record_filter   = torch.nonzero(episode_filter, as_tuple=True)[0].tolist()
            # record_filter   = [elem.cpu().item() if elem.numel() == 1 else None for elem in record_filter]
            filters[key] = record_filter
        self.replay.deque(filters, keys)

    def deque_steps(self, steps: int, keys: list[int] = None):
        """
        Deletes the last n steps.
        Used to compensate for the large data sizes of the NEAT evaluation functions.
        :param steps: (int) Number of steps to keep.
        :param keys: (list[int])
        :return: (none)
        """
        assert steps >= 0
        filters = {}
        episode_mapping: dict[int, list[int]] = self.replay.rollout(buffers='ep_map', as_list=True)[0]
        for key in self.replay.mapping.keys():
            records = len(episode_mapping[key])
            limit = max(0, records - steps)
            filters[key] = [i for i in range(records) if i < limit]
        self.replay.deque(filters, keys)

    def reset(self):
        self.replay.reset()

    def _get_advantages(self, batches: dict[int, list[list[int]]], observations: TensorDict, rewards: TensorDict,
                        verbose: int = None):
        ts, ud, ut = clock.perf_counter(), 0, len(self.replay.mapping)
        with torch.no_grad():
            self.model.eval()
            advantages = {}
            for key in batches.keys():
                adv = []
                for batch in batches[key]:
                    # print(f"batch={self.model.pol_proj.embedder.embedding.weights.data.shape, observations[key].shape}")
                    value: Tensor = self.model.get_value(observations[key][batch].unsqueeze(0).to(self.device), keys=key).squeeze(0)
                    # print(value.shape, rewards[key][batch].shape)
                    adv.append(rewards[key][batch].to(self.device) - value)
                # print(f"end={torch.cat(adv, 0).cpu().shape}")
                advantages[key] = torch.cat(adv, 0).cpu()

                if verbose and verbose >= 2:
                    eta(ts, ud, ut, f"getting advantages")
            self.model.train()

            if verbose and verbose >= 2:
                print(f"\rcalculated advantages in {CM(f'{round(clock.perf_counter() - ts, 2)}s', Fore.LIGHTCYAN_EX)}")
            return advantages

    def _train(self, batches: dict[int, list[list[int]]], states: TensorDict, actions: TensorDict,
               old_log_probs: TensorDict, rewards: TensorDict, verbose: int = None):
        self.model.train()

        # Compute advantages
        advantages = self._get_advantages(batches, states, rewards)
        # Create dictionaries
        pl, vl, el, tl, cf, kd = {}, {}, {}, {}, {}, {}

        def cons(consolidation: list[Tensor]):
            return torch.mean(torch.stack(consolidation, dim=0)).cpu().item()

        ts, ud, ut = clock.perf_counter(), 0, len(self.replay.mapping)
        # Calculate policy loss for all keys
        for key in self.replay.mapping.keys():
            if key in batches:
                # handle batch for each key
                pl_, vl_, el_, tl_, cf_, kd_ = [], [], [], [], [], []
                for batch in batches[key]:
                    state           = states[key][batch].to(self.device)
                    action          = actions[key][batch].to(self.device)
                    old_log_prob    = old_log_probs[key][batch].to(self.device)
                    reward          = rewards[key][batch].to(self.device)
                    advantage       = advantages[key][batch].to(self.device)

                    # Run evaluations
                    log_prob, entropy = self.model.evaluate_action(state.unsqueeze(0), action, keys=key)
                    log_prob, entropy = log_prob.unsqueeze(0), entropy.unsqueeze(0)
                    value = self.model.get_value(state.unsqueeze(0), keys=key).squeeze(0)

                    # Normalize advantage when necessary
                    if self.norm_adv:
                        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                    # Policy Loss
                    ratio = torch.exp(log_prob - old_log_prob)
                    surr_loss_1 = advantage * ratio
                    surr_loss_2 = advantage * torch.clamp(ratio, 1-self.clip_range, 1+self.clip_range)
                    policy_loss = - torch.mean(torch.min(torch.stack([surr_loss_1, surr_loss_2]), 0)[0])
                    if torch.any(torch.isnan(policy_loss)):
                        def debug_(var: Tensor):
                            return f"shape={var.shape}, mean={torch.mean(var)}, std={torch.std(var)}, " \
                                   f"max={np.max(var.cpu().numpy())}, min={np.min(var.cpu().numpy())}"
                        print(f"")
                        with torch.no_grad():
                            print(f"rewards = {debug_(reward)}")
                            print(f"advantages = {debug_(advantage)}")
                            print(f"log_probs = {debug_(log_prob)}")
                            print(f"old_log_probs = {debug_(old_log_prob)}")
                            print(f"ratio = {debug_(ratio)}")
                            print(f"surr_loss_1 = {debug_(surr_loss_1)}")
                            print(f"surr_loss_2 = {debug_(surr_loss_2)}")
                            print(f"policy_loss = {policy_loss}")
                            largest_weight_val = None
                            smallest_weight_val = None
                            for param in self.model.parameters():
                                max_val = torch.max(param).cpu().item()
                                min_val = torch.min(param).cpu().item()
                                if largest_weight_val is None or max_val > largest_weight_val:
                                    largest_weight_val = max_val
                                if smallest_weight_val is None or min_val < smallest_weight_val:
                                    smallest_weight_val = min_val
                            print(f"largest weight value = {largest_weight_val}")
                            print(f"smallest weight value = {smallest_weight_val}")
                        raise ValueError(f"Infinity or NaN value found in policy loss")

                    # Value Loss
                    value_loss = torch.mean((reward - value) ** 2)
                    assert not torch.any(torch.isnan(value_loss))

                    # Entropy Loss
                    if entropy is not None:
                        entropy_loss = - torch.mean(entropy)
                    else:
                        entropy_loss = - torch.mean(-log_prob)
                    assert not torch.any(torch.isnan(entropy_loss))

                    try:
                        loss = (policy_loss * self.pol_reg) + (value_loss * self.val_reg) + (entropy_loss * self.ent_reg)
                    except RuntimeError as e:
                        def debug(var: Tensor, name: str):
                            print(f"\n{name} =>\n{var}\n\tshape = {var.shape}")
                        debug(policy_loss, 'policy_loss')
                        debug(value_loss, 'value_loss')
                        debug(entropy_loss, 'entropy_loss')
                        debug(entropy, 'entropy')
                        raise e

                    # Batch Logging
                    with torch.no_grad():
                        # TODO: Add the std-dev to the logging
                        clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float())
                        log_ratio = log_prob - old_log_prob
                        approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)

                        pl_.append(policy_loss)
                        vl_.append(value_loss)
                        el_.append(entropy_loss)
                        tl_.append(loss)
                        cf_.append(clip_fraction)
                        kd_.append(approx_kl_div)

                pl[key] = cons(pl_)
                vl[key] = cons(vl_)
                el[key] = cons(el_)
                tl[key] = cons(tl_)
                cf[key] = cons(cf_)
                kd[key] = cons(kd_)
            else:
                pl[key] = vl[key] = el[key] = tl[key] = cf[key] = kd[key] = np.nan

            ud += 1
            if verbose:
                eta(ts, ud, ut, f"ppo impl")

        if self.scheduler is not None:
            self.scheduler.step()

        def input(update: dict[int, float]):
            return list(update.values())

        # Logging
        self.logging.update(policy_loss=input(pl), value_loss=input(vl), entropy_loss=input(el),
                            loss=input(tl), clip_fraction=input(cf), kl_divergence=input(kd))
        self.updates_done += 1

    def learn(self, evaluation_function: callable, steps: int, epochs: int = None, batch_size: int = None,
              accuracy_error=0.20, accuracy_type='continuous', verbose: int = None):
        print(f"Logging to {self.log_dir+self.log_name}")
        if epochs is None:
            epochs = np.inf
        epoch_done = 0
        while epoch_done < epochs:
            # Running environment
            torch.cuda.empty_cache()
            ts = clock.perf_counter()
            self.init(steps)
            self.population.run(evaluation_function, 1, verbose=verbose, skip=True, trainer=self)
            run_time = np.floor(clock.perf_counter() - ts)
            if verbose and verbose >= 2:
                print(f"collected data in {CM(f'{round(clock.perf_counter() - ts, 2)}s', Fore.LIGHTCYAN_EX)}")

            # Rolling out data
            invalid_population = len(self.population.to_delete) == len(self.population.genomes)
            valid_keys = [key for key in self.replay.mapping.keys() if key not in self.population.to_delete or invalid_population]
            with torch.no_grad():
                ts = clock.perf_counter()
                states, actions, probabilities, raw_rewards, episode_mapping = self.replay.rollout(
                    as_list=True, stack=True, keys=valid_keys
                )
                if verbose and verbose >= 2:
                    print(f"rolled out data in {CM(f'{round(clock.perf_counter() - ts, 2)}s', Fore.LIGHTCYAN_EX)}")

                # Sort episodes wrt. episode index
                ts = clock.perf_counter()
                episode_lengths = self.sort_episodes(episode_mapping, states, actions, probabilities, raw_rewards)
                if verbose and verbose >= 2:
                    print(f"sorted episodes data in {CM(f'{round(clock.perf_counter() - ts, 2)}s', Fore.LIGHTCYAN_EX)}")
                ts = clock.perf_counter()
                cum_rewards     = self._get_rewards_to_go(raw_rewards, self.gamma, self.alpha, self.alpha_step,
                                                          episode_mapping, rollout=True)
                if verbose and verbose >= 2:
                    print(f"cumulated rewards in {CM(f'{round(clock.perf_counter() - ts, 2)}s', Fore.LIGHTCYAN_EX)}")
                scores: dict[int, float] = {
                    key: torch.mean(cum_rewards[key]).item()
                    for key in valid_keys
                }

                # Normalize rewards when necessary
                if self.norm_rew:
                    rewards = {}
                    for key in cum_rewards.keys():
                        rewards[key] = (cum_rewards[key] - cum_rewards[key].mean()) / (cum_rewards[key].std() + self.epsilon)
                else:
                    rewards = cum_rewards

                # Get batches wrt. steps done per key
                batch_indices = self._get_batches(valid_keys, batch_size, True)

                # Handling of non valid genomes
                fs = [genome.fitness if genome.fitness else 0 for genome in self.population.genomes.values()]
                score_range = max(fs) - min(fs)

            # Training model
            ts = clock.perf_counter()
            with torch.no_grad():
                self._train(batch_indices, states, actions, probabilities, rewards, verbose)
            train_time = np.floor(clock.perf_counter() - ts)
            if verbose and verbose >= 2:
                print(f"\rran training in {CM(f'{round(clock.perf_counter() - ts, 2)}s', Fore.LIGHTCYAN_EX)}")

            # Set values
            fitnesses: dict[int, float] = None
            ppo_score: dict[int, float] = None
            criterion = self.population.config.general.fitness_criterion
            buffer_sizes = self.replay.buffer_sizes()
            try:
                ts = clock.perf_counter()
                p_losses = self.normalize(self.logging.rollout('policy_loss', as_list=True, keys=valid_keys)[0], -1)
                v_losses = self.normalize(self.logging.rollout('value_loss', as_list=True, keys=valid_keys)[0], -1)
                e_losses = self.normalize(self.logging.rollout('entropy_loss', as_list=True, keys=valid_keys)[0], -1)
                losses: dict[int, int] = {key: (p_losses[key] * self.pol_reg +
                                                v_losses[key] * self.val_reg +
                                                e_losses[key] * self.ent_reg)
                                          for key in p_losses.keys()}

                ppo_score = {k: v * self.loss_reg for k, v in self.normalize(losses).items()}
                rew_score = self.normalize(scores)

                if criterion == 'max':
                    ppo_score = {k: -score for k, score in ppo_score.items()}
                elif criterion == 'min':
                    pass
                elif criterion == 'mean':
                    mean = np.mean(list(ppo_score.values())).item()
                    ppo_score = self.normalize({k: -((mean - score) ** 2) for k, score in ppo_score.items()})
                else:
                    raise ValueError(f"Unsupported fitness criterion '{criterion}'")

                fitnesses = {key: rew_score[key] + ppo_score[key] for key in rew_score.keys()}
                min_fitness = min(list(scores.values()))
                for genome in self.population.genomes.values():
                    if genome.key in valid_keys:
                        genome.fitness = fitnesses[genome.key]
                    else:
                        genome.fitness = min_fitness - score_range + genome.fitness
                if verbose and verbose >= 2:
                    print(f"set scores in {CM(f'{round(clock.perf_counter() - ts, 2)}s', Fore.LIGHTCYAN_EX)}")
            except Exception as e:
                print(f"\nFailed to set values due to error:\n\t{CM(e, Fore.LIGHTRED_EX)}")

            scores_ = np.array([fitnesses[key] for key in valid_keys])
            if criterion == 'max':
                best_genome_key = valid_keys[np.argmax(scores_)]
            elif self.population.config.general.fitness_criterion == 'min':
                best_genome_key = valid_keys[np.argmin(scores_)]
            elif self.population.config.general.fitness_criterion == 'mean':
                best_genome_key = valid_keys[np.argmin((scores_.mean() - scores_) ** 2)]
            else:
                raise ValueError(f"unsupported fitness criteria")

            # Logging
            with torch.no_grad():
                ts = clock.perf_counter()
                ep_len_mean = np.mean(episode_lengths[best_genome_key]).item()
                ep_len_std  = np.std(episode_lengths[best_genome_key]).item()
                episode_indices: list[int] = np.unique(episode_mapping[best_genome_key])
                episode_rewards = [[reward.mean().cpu().item()
                                    for idx, reward in enumerate(raw_rewards[best_genome_key])
                                    if episode_mapping[best_genome_key][idx] == ep_idx]
                                   for ep_idx in episode_indices]
                cum_episode_rewards = [[reward.mean().cpu().item()
                                        for idx, reward in enumerate(cum_rewards[best_genome_key])
                                        if episode_mapping[best_genome_key][idx] == ep_idx]
                                       for ep_idx in episode_indices]
                ep_rew_mean = np.mean([np.mean(episode) for episode in episode_rewards]).item()
                ep_rew_std = np.mean([np.std(episode) for episode in episode_rewards]).item()
                ep_cum_rew = np.mean([np.mean(episode) for episode in cum_episode_rewards]).item()
                acc = self._get_accuracy(batch_indices, states, actions, rewards, accuracy_error, accuracy_type, False, best_genome_key)
                try:
                    policy_acc, reward_acc = acc[0][best_genome_key] * 100, acc[1][best_genome_key] * 100
                except RuntimeError:
                    policy_acc, reward_acc = np.nan, np.nan
                explained_variance = self._explained_variance(batch_indices, states, rewards, best_genome_key)[best_genome_key]
                policy_reduction = 1 if len(ppo_score) <= 1 else sorted(
                    list(ppo_score.keys()), key=lambda k: ppo_score[k]).index(best_genome_key) / (len(ppo_score)-1)

                def get_range(key: int):
                    try:
                        params = []
                        for param in self.model.neat_parameters():
                            params.append(param[key])
                        params = torch.cat(params)
                        return torch.mean(params).cpu().item(), torch.std(params).cpu().item()
                    except Exception as e:
                        return torch.nan, torch.nan

                mean, std = get_range(best_genome_key)
                if verbose and verbose >= 2:
                    print(f"calculated stats in {CM(f'{round(clock.perf_counter() - ts, 2)}s', Fore.LIGHTCYAN_EX)}")

                self.logging.update(
                    ep_len_mean=ep_len_mean, ep_len_std=ep_len_std, ep_rew_mean=ep_rew_mean, ep_rew_std=ep_rew_std,
                    policy_acc=policy_acc, reward_acc=reward_acc, explained_variance=explained_variance, std=std,
                    weight_mutate_power=self.population.config.genome.weight_mutate_power,
                    bias_mutate_power=self.population.config.genome.bias_mutate_power, clip_range=self.clip_range,
                )
                temp_data = self.logging.rollout(
                    buffers=['kl_divergence', 'clip_fraction', 'policy_loss', 'value_loss', 'entropy_loss', 'loss'],
                    as_list=True
                )
                kl_divergence   = temp_data[0][best_genome_key][-1]
                clip_fraction   = temp_data[1][best_genome_key][-1]
                policy_loss     = temp_data[2][best_genome_key][-1]
                value_loss      = temp_data[3][best_genome_key][-1]
                entropy_loss    = temp_data[4][best_genome_key][-1]
                loss            = temp_data[5][best_genome_key][-1]

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
                self.writer.add_scalar(extra+'train_time', train_time, self.updates_done)

                # Training
                extra = 'training/'
                self.writer.add_scalar(extra+'clip_range', self.clip_range, self.updates_done)
                self.writer.add_scalar(extra+'explained_variance', explained_variance, self.updates_done)
                self.writer.add_scalar(extra+'param_mean', mean, self.updates_done)
                self.writer.add_scalar(extra+'param_std', std, self.updates_done)
                self.writer.add_scalar(extra+'policy_accuracy', policy_acc, self.updates_done)
                self.writer.add_scalar(extra+'reward_accuracy', reward_acc, self.updates_done)
                self.writer.add_scalar(extra+'policy_reduction', policy_reduction, self.updates_done)

                self.writer.add_scalar(extra+'kl_divergence', kl_divergence, self.updates_done)
                self.writer.add_scalar(extra+'clip_fraction', clip_fraction, self.updates_done)
                self.writer.add_scalar(extra+'policy_loss', policy_loss, self.updates_done)
                self.writer.add_scalar(extra+'value_loss', value_loss, self.updates_done)
                self.writer.add_scalar(extra+'entropy_loss', entropy_loss, self.updates_done)
                self.writer.add_scalar(extra+'loss', loss, self.updates_done)

                # Population
                survival_rate = len(valid_keys) / len(self.population.genomes)
                best_genome = self.population.genomes[best_genome_key]
                extra = 'population/'
                self.writer.add_scalar(extra+'weight_mutate_power', self.population.config.genome.weight_mutate_power, self.updates_done)
                self.writer.add_scalar(extra+'bias_mutate_power', self.population.config.genome.bias_mutate_power, self.updates_done)
                self.writer.add_scalar(extra+'best_genome', best_genome.key, self.updates_done)
                self.writer.add_scalar(extra+'best_fitness', best_genome.fitness, self.updates_done)
                self.writer.add_scalar(extra+'survival_rate', survival_rate, self.updates_done)

            self.writer.flush()
            self.model.eval()

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
                    f"\n|\t{'train_time': <25}| {train_time: <21} |"
                    f"\n|\t{'steps_done': <25}| {self.steps_done: <21} |"
                    f"\n|\t{'episodes_done': <25}| {self._ep_offset+1: <21} |"
                    f"\n|\t{'total_episodes_done': <25}| {self.episodes_done: <21} |"
                    f"\n{'|TRAINING:': <29}|{'': <22} |"
                    f"\n|\t{'kl_divergence': <25}| {kl_divergence: <21} |"
                    f"\n|\t{'clip_fraction': <25}| {clip_fraction: <21} |"
                    f"\n|\t{'clip_range': <25}| {self.clip_range: <21} |"
                    f"\n|\t{'policy_loss': <25}| {policy_loss: <21} |"
                    f"\n|\t{'value_loss': <25}| {value_loss: <21} |"
                    f"\n|\t{'entropy_loss': <25}| {entropy_loss: <21} |"
                    f"\n|\t{'loss': <25}| {loss: <21} |"
                    f"\n|\t{'updates_done': <25}| {self.updates_done: <21} |"
                    f"\n|\t{'weight_mutate_power': <25}| {self.population.config.genome.weight_mutate_power: <21} |"
                    f"\n|\t{'bias_mutate_power': <25}| {self.population.config.genome.bias_mutate_power: <21} |"
                    f"\n|\t{'explained_variance': <25}| {explained_variance: <21} |"
                    f"\n|\t{'std': <25}| {std: <21} |"
                    f"\n|\t{'policy_accuracy': <25}| {policy_acc: <21} |"
                    f"\n|\t{'reward_accuracy': <25}| {reward_acc: <21} |"
                    f"\n{bar}"
                )

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
                    value: Tensor = self.model.get_value(state.unsqueeze(0), keys=key).squeeze(0)
                    reward = rewards[key][batch].to(self.device)
                    value = ((torch.std(reward - value) ** 2) / (torch.std(reward) ** 2)) - 1
                    ev.append(value)

                ex_var[key] = torch.clamp(torch.mean(torch.stack(ev)), None, 1).cpu().item()

            return ex_var

    @staticmethod
    def plotter(name: str, title: str = None, **buffers: list[float]):
        for label, buffer in buffers.items():
            plt.plot(buffer, label=label)
        if len(buffers) > 1:
            plt.legend()
        if title is not None:
            plt.title(title)
        plt.savefig(STORAGE_DIR+f"plots\\{name}-{unix_to_datetime_file(clock.time())}")
        plt.close()


if __name__ == '__main__':
    import build as neat
    from build.models.main import Linear
    from build.util.datetime import eta, clock

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
    TRAINER     = PPO(MODEL, POPULATION, DEVICE, DTYPE, loss_reg=0.1, gamma=0.0,
                      scheduler=neat.scheduler.CosineAnnealing(CONFIG, 100, 50, 0.001, True, True))
    STEPS       = 100

    def evaluate(population: Population, **options):
        trainer: PPO = options['trainer']
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
