
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
from typing import Union, Any
from numpy import ndarray

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


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
        self.norm_rew: bool               = manage_params(options, ['norm_rew'], True)
        self.norm_adv: bool               = manage_params(options, ['norm_adv'], False)
        self.gamma: float                 = manage_params(options, 'gamma', 0.95)
        self.epsilon: float               = manage_params(options, 'epsilon', 1e-10)
        self.clip_range: float            = manage_params(options, 'clip_range', 0.3)
        self.pol_reg: float               = manage_params(options, 'pol_reg', 1.0)
        self.val_reg: float               = manage_params(options, 'val_reg', 0.5)
        self.ent_reg: float               = manage_params(options, 'ent_reg', 1e-6)
        self.loss_reg: float              = manage_params(options, 'loss_reg', 0.5)
        self.target_kl: [float, None]     = manage_params(options, 'target_kl', None)
        self.scheduler: [Scheduler, None] = manage_params(options, 'scheduler', None)

        # Tensorboard logging
        self.log_dir: str = manage_params(
            options, 'log_directory', STORAGE_DIR+f"neat_rl_logs\\{self.__class__.__name__}\\")
        self.log_name: str = manage_params(
            options, 'log_name', f"log~{unix_to_datetime_file(clock.time())}")
        self.writer         = SummaryWriter(self.log_dir+self.log_name)
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
        self.episodes_done  = 0
        self.updates_done   = 0

        # Timing
        self.__ts: int      = None
        self.__ud: int      = None
        self.__ut: int      = None

    def update(self, observation: Tensor, action: Tensor, probs: Tensor, reward: Tensor, terminated: bool):
        def convert(tensor: Union[Tensor, ndarray, list]):
            if isinstance(tensor, Tensor):
                return tensor.clone()
            elif isinstance(tensor, (ndarray, list, int, float)):
                return torch.tensor(tensor, device='cpu', dtype=torch.float32)
            else:
                raise ValueError(f"Unsupported tensor type '{type(tensor)}'")

        self.replay.update(
            add_dims=[0],
            state=convert(observation),
            action=convert(action),
            prob=convert(probs),
            reward=convert(reward),
            ep_map=convert(self.episodes_done),
        )
        if terminated:
            self.episodes_done += 1
            print(CM("TERMINATED!", Fore.LIGHTGREEN_EX))

    def deque(self, episodes: int):
        """
        Deletes the episodes before the last n episodes.
        Used to compensate for the long data collection times of the NEAT evaluation functions.
        :param episodes: (int) Number of recent episodes to keep.
        :return: (none)
        """
        assert episodes >= 1
        episodes_to_del = torch.tensor([ep for ep in range(self.episodes_done) if ep < (self.episodes_done-episodes)])
        episode_mapping = torch.cat(self.replay.rollout(buffers='ep_map', as_list=True)[0])
        episode_filter  = torch.isin(episode_mapping, episodes_to_del)
        record_filter   = torch.nonzero(episode_filter, as_tuple=True)[0].tolist()
        # record_filter   = [elem.cpu().item() if elem.numel() == 1 else None for elem in record_filter]
        self.replay.deque(record_filter)

    def reset(self):
        self.replay.reset()

    def _get_advantages(self, batches: list[list[int]], observations: Tensor, rewards: Tensor):
        with torch.no_grad():
            self.model.eval()
            value_est = []
            for batch in batches:
                val: Tensor = self.model.get_value(observations[batch]).to('cpu', torch.float32)
                value_est.append(val)
            self.model.train()
            return rewards - torch.cat(value_est, 0).to(self.device, self.dtype)

    def _train(self, epochs: int, batches: list[list[int]], states: Tensor, actions: Tensor, old_log_probs: Tensor,
               rewards: Tensor, adv_computations=1):
        self.model.train()
        assert epochs // adv_computations > 0
        div = epochs // adv_computations

        # NOTE: The second last dimension is the genome dimension, therefore it is not manipulated on
        state_dims  = self.get_dims(states)
        act_dims    = self.get_dims(actions)
        prob_dims   = self.get_dims(old_log_probs)
        rew_dims    = self.get_dims(rewards)

        def append(arrays: list[ndarray]):
            return np.mean(np.stack(arrays, 0), 0)

        advantages: Tensor = None
        for epoch_idx in range(epochs):
            if epoch_idx % div == 0:
                advantages = self._get_advantages(batches, states, rewards)
            adv_dims = self.get_dims(advantages)
            pl, vl, el, tl, cf, kd = [], [], [], [], [], []
            for batch in batches:
                log_probs, entropy = self.model.evaluate_action(states[batch], actions[batch])
                values = self.model.get_value(states[batch])

                if self.norm_adv:
                    adv_dims_std = adv_dims
                    if advantages[batch].shape[0] == 1:
                        adv_dims_std = adv_dims_std[1:]
                    adv = (advantages[batch] - advantages[batch].mean(adv_dims, True)) / (advantages[batch].std(adv_dims_std, True) + 1e-8)
                else:
                    adv = advantages[batch]
                if adv.shape[-1] > 1:
                    adv = torch.mean(adv, -1, True)

                # Policy Loss
                ratio = torch.exp(log_probs - old_log_probs[batch])
                if ratio.ndim != adv.ndim:
                    ratio = ratio.unsqueeze(-1)
                surr_loss_1 = ratio * adv
                surr_loss_2 = torch.clamp(ratio, 1-self.clip_range, 1+self.clip_range) * adv
                policy_loss = - torch.mean(torch.min(torch.stack([surr_loss_1, surr_loss_2]), 0)[0], prob_dims)
                if torch.any(torch.isnan(policy_loss)):
                    def debug_(tensor: Tensor):
                        dims = self.get_dims(tensor)
                        return f"shape={tensor.shape}, mean={torch.mean(tensor, dims)}, std={torch.std(tensor, dims)}, " \
                               f"max={np.max(tensor.cpu().numpy(), dims)}, min={np.min(tensor.cpu().numpy(), dims)}"
                    print(f"")
                    with torch.no_grad():
                        # mean, std = self.model.debug_mean_std(states[batch])
                        # print(f"mean = {debug_(mean)}")
                        # print(f"std = {debug_(std)}")
                        print(f"rewards = {debug_(rewards[batch])}")
                        print(f"advantages = {debug_(adv)}")
                        print(f"log_probs = {debug_(log_probs)}")
                        print(f"old_log_probs = {debug_(old_log_probs[batch])}")
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
                value_loss = torch.mean((rewards[batch] - values) ** 2, rew_dims)
                assert not torch.any(torch.isnan(value_loss))

                # Entropy Loss
                if entropy is not None:
                    entropy_loss = - torch.mean(entropy, prob_dims)
                else:
                    entropy_loss = - torch.mean(-log_probs, prob_dims)
                assert not torch.any(torch.isnan(entropy_loss))

                loss = (policy_loss * self.pol_reg) + (value_loss * self.val_reg) + (entropy_loss * self.ent_reg)

                # Batch Logging
                with torch.no_grad():
                    # TODO: Add the std-dev to the logging
                    clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float(), prob_dims)
                    # cf_mean, cf_std = clip_fraction.min().cpu().item(), clip_fraction.std().cpu().item()
                    log_ratio = log_probs - old_log_probs[batch]
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio, prob_dims)
                    # kd_mean, kd_std = (approx_kl_div[approx_kl_div != 0] + self.epsilon).min().cpu().item(), approx_kl_div.std().cpu().item()

                    pl.append(policy_loss.cpu().numpy())
                    vl.append(value_loss.cpu().numpy())
                    el.append(entropy_loss.cpu().numpy())
                    tl.append(loss.cpu().numpy())
                    cf.append(clip_fraction.cpu().numpy())
                    kd.append(approx_kl_div.cpu().numpy())

            if self.scheduler is not None:
                self.scheduler.step()

            # Logging
            self.logging.update(policy_loss=append(pl), value_loss=append(vl), entropy_loss=append(el),
                                loss=append(tl), clip_fraction=append(cf), kl_divergence=append(kd))
            self.updates_done += 1

    def learn(self, evaluation_function: callable, runs: int, epochs: int, batch_size: int = None, accuracy_error=0.20,
              adv_comp=1, verbose: int = None):
        print(f"Logging to {self.log_dir+self.log_name}")
        iterations = 0
        steps_done = 0
        runs_done = 0
        while runs_done < runs+1:
            # Running environment
            torch.cuda.empty_cache()
            ts = clock.perf_counter()
            self.population.run(evaluation_function, 1, verbose=verbose, skip=True, trainer=self)
            run_time = np.floor(clock.perf_counter() - ts)
            self.deque(1)

            # Rolling out data
            try:
                with torch.no_grad():
                    states, actions, probabilities, rewards, episode_mapping = self.replay.rollout(as_list=True)
                    episode_mapping = torch.cat(episode_mapping).cpu().numpy()
                    records = len(episode_mapping)

                    sorting_indices = torch.tensor(
                        sorted([i for i in range(records)], key=lambda idx: (episode_mapping[idx], idx)),
                        dtype=torch.int)
                    states          = torch.index_select(torch.cat(states), 0, sorting_indices).to(self.device, self.dtype)
                    actions         = torch.index_select(torch.cat(actions), 0, sorting_indices).to(self.device, self.dtype)
                    probabilities   = torch.index_select(torch.cat(probabilities), 0, sorting_indices).to(self.device, self.dtype)
                    raw_rewards     = torch.index_select(torch.cat(rewards), 0, sorting_indices).to(self.device, self.dtype)
                    cum_rewards     = self._get_rewards_to_go(raw_rewards, self.gamma, episode_mapping)
                    episode_mapping = np.array(episode_mapping)[sorting_indices.cpu().tolist()]
                    episodes_count = {}
                    for ep_idx in episode_mapping:
                        if ep_idx not in episodes_count:
                            episodes_count[ep_idx] = 1
                        else:
                            episodes_count[ep_idx] += 1
                    episode_lengths = [count for count in episodes_count.values()]
                    scores = torch.mean(cum_rewards, self.get_dims(cum_rewards))
                    if self.population.config.general.fitness_criterion == 'max':
                        best_genome_idx = scores.argmax().item()
                    elif self.population.config.general.fitness_criterion == 'min':
                        best_genome_idx = scores.argmin().item()
                    elif self.population.config.general.fitness_criterion == 'mean':
                        best_genome_idx = ((scores.mean() - scores) ** 2).argmin().item()
                    else:
                        raise ValueError(f"unsupported fitness criteria")

                    def fix(tensor: Tensor, idx: int):
                        return tensor[..., idx: idx+1, :].expand(*tensor.shape)

                    # states          = fix(states, best_genome_idx)
                    # actions         = fix(actions, best_genome_idx)
                    # probabilities   = fix(probabilities, best_genome_idx)
                    # cum_rewards     = fix(cum_rewards, best_genome_idx)
            except RuntimeError as e:
                pass

            if self.norm_rew:
                rewards = (cum_rewards - cum_rewards.mean()) / (cum_rewards.std() + 1e-8)
            else:
                rewards = cum_rewards
            batch_indices = self._get_batches(records, batch_size, True)

            # Training model
            ts = clock.perf_counter()
            self._train(epochs, batch_indices, states, actions, probabilities, rewards, adv_comp)
            train_time = np.floor(clock.perf_counter() - ts)

            # Set values
            try:
                # score_mapping = {idx: score.item() for idx, score in enumerate(scores)}
                # elite = []
                # cutoff = np.ceil(self.population.config.reproduction.survival_threshold * len(self.population.genomes))
                # for idx, score in sorted(score_mapping.items(), key=lambda item: item[1]):
                #     if len(elite) < cutoff:
                #         elite.append(idx)
                losses: ndarray = self.logging.rollout(None, 'loss', True)[0][-1]
                # loss_max = np.abs(np.max(losses))
                # loss_min = -np.abs(np.min(losses))
                s1 = self.normalize(scores).cpu().numpy()
                s2 = self.normalize(losses) * self.loss_reg
                if self.population.config.general.fitness_criterion == 'max':
                    s2 = -s2
                elif self.population.config.general.fitness_criterion == 'mean':
                    s2 = -((np.mean(s2) - s2) ** 2)
                fitness = s1 + s2
                for idx, (genome, fitness) in enumerate(zip(self.population.genomes.values(), fitness)):
                    # if idx == best_genome_idx:
                    #     genome.fitness = loss_min * 3
                    # elif idx in elite:
                    #     genome.fitness = loss_min - loss
                    # else:
                    #     genome.fitness = loss_max - loss
                    genome.fitness = fitness
                pass
            except Exception as e:
                print(f"\nFailed to set values due to error:\n\t{CM(e, Fore.LIGHTRED_EX)}")

            # Logging
            with torch.no_grad():
                ep_len_mean = np.mean(episode_lengths)
                ep_len_std  = np.std(episode_lengths)
                episode_indices = np.unique(episode_mapping)
                episode_rewards = [[reward.mean(self.get_dims(reward))[best_genome_idx].cpu().item() for idx, reward in enumerate(raw_rewards) if episode_mapping[idx] == ep_idx] for ep_idx in episode_indices]
                cum_episode_rewards = [[reward.mean(self.get_dims(reward))[best_genome_idx].cpu().item() for idx, reward in enumerate(cum_rewards) if episode_mapping[idx] == ep_idx] for ep_idx in episode_indices]
                ep_rew_mean = np.mean([np.mean(episode) for episode in episode_rewards])
                ep_rew_std = np.mean([np.std(episode) for episode in episode_rewards])
                ep_cum_rew = np.mean([np.mean(episode) for episode in cum_episode_rewards])
                # th_mean, th_std = self._get_threshold(batch_indices, states, actions)
                policy_acc, reward_acc = self._get_accuracy(batch_indices, states, actions, rewards, best_genome_idx, accuracy_error, 'binary', False)
                explained_variance = self._explained_variance(batch_indices, states, rewards).cpu().item()

                def get_std():
                    try:
                        params = []
                        for param in self.parameters:
                            for weight in param.weights:
                                params.append(weight.flatten())
                            if param.enable_bias:
                                for bias in param.biases:
                                    params.append(bias.flatten())
                        # for attr, val in vars(obj).items():
                        #     if 'log_std' in attr:
                        #         return val
                        #     if isinstance(val, Model):
                        #         res = get_std(obj)
                        #         if res is not None:
                        #             return res
                        # return None`
                        return torch.std(torch.cat(params))
                    except Exception as e:
                        return torch.nan

                std = get_std()

                self.logging.update(
                    add_dims=[0],
                    ep_len_mean=ep_len_mean, ep_len_std=ep_len_std, ep_rew_mean=ep_rew_mean, ep_rew_std=ep_rew_std,
                    policy_acc=policy_acc, reward_acc=reward_acc, explained_variance=explained_variance, std=std,
                    weight_mutate_power=self.population.config.genome.weight_mutate_power,
                    bias_mutate_power=self.population.config.genome.bias_mutate_power, clip_range=self.clip_range,
                )
                temp_data = self.logging.rollout(
                    buffers=['kl_divergence', 'clip_fraction', 'policy_loss', 'value_loss', 'entropy_loss', 'loss'])

                # Rollout
                extra = 'rollout/'
                # rollout = self.writer.file_writer()
                self.writer.add_scalar(extra+'ep_len_mean', ep_len_mean, self.updates_done)
                self.writer.add_scalar(extra+'ep_len_std', ep_len_std, self.updates_done)
                self.writer.add_scalar(extra+'ep_rew_mean', ep_rew_mean, self.updates_done)
                self.writer.add_scalar(extra+'ep_rew_std', ep_rew_std, self.updates_done)
                self.writer.add_scalar(extra+'ep_cum_rew', ep_cum_rew, self.updates_done)

                # Time
                extra = 'time/'
                self.writer.add_scalar(extra+'run_time', run_time, self.updates_done)
                self.writer.add_scalar(extra+'train_time', train_time, self.updates_done)

                # Training
                extra = 'training/'
                self.writer.add_scalar(extra+'clip_range', self.clip_range, self.updates_done)
                self.writer.add_scalar(extra+'explained_variance', explained_variance, self.updates_done)
                self.writer.add_scalar(extra+'std', std, self.updates_done)
                self.writer.add_scalar(extra+'policy_accuracy', policy_acc, self.updates_done)
                self.writer.add_scalar(extra+'reward_accuracy', reward_acc, self.updates_done)

                for x in range(self.updates_done-epochs, self.updates_done):
                    self.writer.add_scalar(extra+'kl_divergence', temp_data['kl_divergence'][x][best_genome_idx], x)
                    self.writer.add_scalar(extra+'clip_fraction', temp_data['clip_fraction'][x][best_genome_idx], x)
                    self.writer.add_scalar(extra+'policy_loss', temp_data['policy_loss'][x][best_genome_idx], x)
                    self.writer.add_scalar(extra+'value_loss', temp_data['value_loss'][x][best_genome_idx], x)
                    self.writer.add_scalar(extra+'entropy_loss', temp_data['entropy_loss'][x][best_genome_idx], x)
                    self.writer.add_scalar(extra+'loss', temp_data['loss'][x][best_genome_idx], x)

                # Population
                extra = 'population/'
                self.writer.add_scalar(extra+'weight_mutate_power', self.population.config.genome.weight_mutate_power, self.updates_done)
                self.writer.add_scalar(extra+'bias_mutate_power', self.population.config.genome.bias_mutate_power, self.updates_done)
                self.writer.add_scalar(extra+'best_genome', self.population.best_genome.key if self.population.best_genome is not None else np.nan, self.updates_done)

                self.writer.add_scalar(extra+'best_fitness', self.population.best_genome.fitness if self.population.best_genome is not None else np.nan, self.updates_done)

            self.writer.flush()
            self.model.eval()

            steps_done += records
            self.steps_done += np.sum(episode_mapping == np.max(episode_mapping))

            # Displaying
            if verbose:

                iterations += 1
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
                    f"\n|\t{'iterations': <25}| {iterations: <21} |"
                    f"\n|\t{'run_time': <25}| {run_time: <21} |"
                    f"\n|\t{'train_time': <25}| {train_time: <21} |"
                    f"\n|\t{'steps_done': <25}| {self.steps_done: <21} |"
                    f"\n|\t{'episodes_done': <25}| {len(episode_lengths): <21} |"
                    f"\n|\t{'total_episodes_done': <25}| {self.episodes_done: <21} |"
                    f"\n{'|TRAINING:': <29}|{'': <22} |"
                    f"\n|\t{'kl_divergence': <25}| {temp_data['kl_divergence'][-1][best_genome_idx]: <21} |"
                    f"\n|\t{'clip_fraction': <25}| {temp_data['clip_fraction'][-1][best_genome_idx]: <21} |"
                    f"\n|\t{'clip_range': <25}| {self.clip_range: <21} |"
                    f"\n|\t{'policy_loss': <25}| {temp_data['policy_loss'][-1][best_genome_idx]: <21} |"
                    f"\n|\t{'value_loss': <25}| {temp_data['value_loss'][-1][best_genome_idx]: <21} |"
                    f"\n|\t{'entropy_loss': <25}| {temp_data['entropy_loss'][-1][best_genome_idx]: <21} |"
                    f"\n|\t{'loss': <25}| {temp_data['loss'][-1][best_genome_idx]: <21} |"
                    f"\n|\t{'updates_done': <25}| {self.updates_done: <21} |"
                    f"\n|\t{'weight_mutate_power': <25}| {self.population.config.genome.weight_mutate_power: <21} |"
                    f"\n|\t{'bias_mutate_power': <25}| {self.population.config.genome.bias_mutate_power: <21} |"
                    f"\n|\t{'explained_variance': <25}| {explained_variance: <21} |"
                    f"\n|\t{'std': <25}| {std: <21} |"
                    f"\n|\t{'policy_accuracy': <25}| {policy_acc: <21} |"
                    f"\n|\t{'reward_accuracy': <25}| {reward_acc: <21} |"
                    f"\n{bar}"
                )

            runs_done += 1

    def _explained_variance(self, batches: list[list[int]], states: Tensor, rewards: Tensor):
        with torch.no_grad():
            estimations = []
            for batch in batches:
                value_est = self.model.get_value(states[batch])
                estimations.append(value_est)
            value_est = torch.cat(estimations, 0)
            dims = [i for i in range(rewards.ndim) if i != rewards.ndim-2]
            lim = 1 - ((torch.std(rewards-value_est, dims)**2) / (torch.std(rewards, dims)**2))
            return torch.min(torch.max(torch.stack([lim, torch.full_like(lim, -1)]), 0)[0])

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

    class RModel(Model):
        def __init__(self, inputs, outputs, dim_size, device, dtype):
            super().__init__()
            self.act_proj = Linear(inputs, dim_size, True, DEVICE, DTYPE, torch.nn.SiLU())
            self.mean = Linear(dim_size, outputs, True, DEVICE, DTYPE, torch.nn.SiLU())
            self.log_std = Linear(dim_size, outputs, True, DEVICE, DTYPE, torch.nn.SiLU())
            self.rew_proj = Linear(inputs, dim_size, True, DEVICE, DTYPE, torch.nn.SiLU())
            self.decode = Linear(dim_size, 1, True, DEVICE, DTYPE, torch.nn.SiLU())

        def forward(self, state: Tensor):
            return self.get_policy(state)

        def get_mean(self, latent: Tensor) -> Tensor:
            return self.mean(latent)

        def get_std(self, latent: Tensor) -> Tensor:
            return torch.exp(self.log_std(latent))

        def get_action(self, state: Tensor) -> tuple[Tensor, Tensor]:
            latent = self.act_proj(state)
            mean, std = self.get_mean(latent), self.get_std(latent)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob

        def evaluate_action(self, state: Tensor, action: Tensor) -> [Tensor, Union[Tensor, None]]:
            latent = self.act_proj(state)
            mean, std = self.get_mean(latent), self.get_std(latent)
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            return log_prob, entropy

        def get_policy(self, state: Tensor, **options) -> Tensor:
            latent = self.act_proj(state)
            mean, std = self.get_mean(latent), self.get_std(latent)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            return action

        def get_value(self, state: Tensor) -> Tensor:
            latent = self.rew_proj(state)
            value = self.decode(latent)
            return value

    DEVICE      = 'cuda'
    DTYPE       = torch.float32
    INPUTS      = 2
    OUTPUTS     = 2
    DIM_SIZE    = 256
    MODEL       = RModel(INPUTS, OUTPUTS, DIM_SIZE, DEVICE, DTYPE)
    CONFIG      = neat.Config("ppo_test")
    GENOMES     = 100
    POPULATION  = neat.Population(GENOMES, MODEL, CONFIG, init_reporter=True)
    TRAINER     = PPO(MODEL, POPULATION, DEVICE, DTYPE, )
    STEPS       = 100

    def evaluate(population: Population, **options):
        trainer: PPO = options['trainer']

        for step in range(STEPS):
            observations = torch.rand(GENOMES, INPUTS, device=DEVICE, dtype=DTYPE)
            actions, probs = MODEL.get_action(observations)
            # probs = torch.rand(GENOMES, OUTPUTS, device=DEVICE, dtype=DTYPE)
            rewards = torch.rand(GENOMES, 1, device=DEVICE, dtype=DTYPE)

            trainer.update(observations, actions, probs, rewards, step == STEPS-1)
        trainer.deque(1)

    POPULATION.load_dict(name='ppo_test')

    TRAINER.learn(evaluate, 30, 1, 64, 0.20, 1, True)

    POPULATION.save_dict('ppo_test')
