
from ModifiedNEAT.nn.base import NeatModule
from ModifiedNEAT.population import Population
from ModifiedNEAT.optim.scheduler import Scheduler
from ModifiedNEAT.util.replay import ReplayBuffer
from ModifiedNEAT.util.datetime import eta, clock
# from ModifiedNEAT.util.storage import save, load
from ModifiedNEAT.util.qol import manage_params
from ModifiedNEAT.util.datetime import unix_to_datetime_file
from ModifiedNEAT.util.fancy_text import CM, Fore

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from numba import njit
from numpy import ndarray
from typing import Any, Union, Callable
from itertools import count

import torch
import random
import numpy as np
import math


class NEATAlgoWarning(Warning):
    pass


TensorDict = dict[int, Tensor]


class Algorithm(object):
    mapping_indexer = count(0)

    def __init__(self, population: Population, schedulers: list[Scheduler] = None,
                 device=torch.device('cpu'), dtype: torch.dtype = torch.float32, **options):
        # ------------------------------ Handle input ------------------------------ #
        models = list(population.modules.values())
        if isinstance(models, NeatModule):
            models = [models]
        if isinstance(schedulers, Scheduler):
            schedulers = [schedulers]
        # ------------------------------ Build ------------------------------ #
        self.models                     = models
        self.population: Population     = population
        self.schedulers: list[Scheduler] = schedulers
        self.primary                    = ReplayBuffer()
        self.secondary                  = ReplayBuffer()
        self.logging                    = ReplayBuffer()
        # ------------------------------ Attributes ------------------------------ #
        self.steps_done                 = 0
        self.prev_steps_done            = self.steps_done
        self.steps_limit: int           = None
        self.episodes_done              = 0
        self.prev_episodes_done         = self.episodes_done
        self.updates_done               = 0
        self.batch_size                 = manage_params(options, 'batch_size', 512)
        self.terminated                 = True
        # ------------------------------ States ------------------------------ #
        self.device: torch.device       = device
        self.dtype: torch.dtype         = dtype
        self.episode_mapping: dict[int, int] = {}
        self.episode_lengths: dict[int, int] = {}
        self._global_mean = self._global_std = self._global_min = self._global_max = None
        # ------------------------------ Tensorboard logging ------------------------------ #
        from ModifiedNEAT.util.storage import STORAGE_DIR

        self.log_dir: str = manage_params(options, ['log_dir', 'log_directory'], STORAGE_DIR+f"neat_rl_logs\\{self.__class__.__name__}\\")
        self.log_sub_dir: str = manage_params(options, 'log_sub_dir', "")
        self.log_name: str = manage_params(options, 'log_name', f"log~{unix_to_datetime_file(clock.time())}")
        self.writer = SummaryWriter(self.log_dir+self.log_sub_dir+self.log_name)
        self._report_hook: Callable = None

    def set_report_hook(self, hook: Callable):
        """
        :param hook: Set a function that receives the Trainer as its parameter to report on a given generation's
            progress.
            You can also just add a custom Reporter object to the Population class for post-evaluation statistics.
        :return: None
        """
        self._report_hook = hook

    def get_module(self, key: int):
        for module in self.models:
            if key in module.mapping:
                return module
        raise ValueError(f"Cannot find module")

    def update_mapping(self, mapping: dict[int, int]):
        self.primary.update_mapping(mapping)
        self.secondary.update_mapping(mapping)
        self.logging.update_mapping(mapping)

    @staticmethod
    @njit
    def get_episode_filter(mapping: list[int], episodes_done: int, episode_wanted: int):
        episodes_to_del = [ep for ep in range(episodes_done) if ep < (episodes_done - episode_wanted)]
        record_filter = [i for i, ep in enumerate(mapping) if ep in episodes_to_del]
        return record_filter

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
        episode_mapping: dict[int, list[int]] = self.primary.rollout(buffers='ep_map', as_list=True)[0]
        for key in self.primary.mapping.keys():
            # episodes_to_del = torch.tensor([ep for ep in range(self.episodes_done) if ep < (self.episodes_done-episodes)])
            # mapping         = torch.tensor(episode_mapping[key])
            # episode_filter  = torch.isin(mapping, episodes_to_del)
            # record_filter   = torch.nonzero(episode_filter, as_tuple=True)[0].tolist()
            # record_filter   = [elem.cpu().item() if elem.numel() == 1 else None for elem in record_filter]
            filters[key] = self.get_episode_filter(episode_mapping[key], self.episodes_done, episodes) # record_filter
        self.primary.deque(filters, keys)

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
        episode_mapping: dict[int, list[int]] = self.primary.rollout(buffers='ep_map', as_list=True)[0]
        for key in self.primary.mapping.keys():
            records = len(episode_mapping[key])
            limit = max(0, records - steps)
            filters[key] = list(range(limit)) # [i for i in range(records) if i < limit]
        self.primary.deque(filters, keys)

    def deque_episodes_secondary(self, episodes: int, keys: list[int] = None):
        """
        Deletes the episodes before the last n episodes.
        Used to compensate for the long data collection times of the NEAT evaluation functions.
        :param episodes: (int) Number of recent episodes to keep.
        :param keys: (list[int])
        :return: (none)
        """
        assert episodes >= 0
        filters = {}
        episode_mapping: dict[int, list[int]] = self.secondary.rollout(buffers='ep_map', as_list=True)[0]
        for key in self.secondary.mapping.keys():
            max_ep = max(self.secondary.data[key]['ep_map']) + 1
            # episodes_to_del = torch.tensor([ep for ep in range(max_ep) if ep < (max_ep-episodes)])
            # mapping = torch.tensor(episode_mapping[key])
            # episode_filter  = torch.isin(mapping, episodes_to_del)
            # record_filter   = torch.nonzero(episode_filter, as_tuple=True)[0].tolist()
            # # record_filter   = [elem.cpu().item() if elem.numel() == 1 else None for elem in record_filter]
            filters[key] = self.get_episode_filter(episode_mapping[key], max_ep, episodes) # record_filter
        self.secondary.deque(filters, keys)

    def deque_steps_secondary(self, steps: int, keys: list[int] = None):
        """
        Deletes the last n steps.
        Used to compensate for the large data sizes of the NEAT evaluation functions.
        :param steps: (int) Number of steps to keep.
        :param keys: (list[int])
        :return: (none)
        """
        assert steps >= 0
        filters = {}
        episode_mapping: dict[int, list[int]] = self.secondary.rollout(buffers='ep_map', as_list=True)[0]
        for key in self.secondary.mapping.keys():
            records = len(episode_mapping[key])
            limit = max(0, records - steps)
            filters[key] = list(range(limit)) # [i for i in range(records) if i < limit]
        self.secondary.deque(filters, keys)

    def reset_buffers(self):
        self.primary.reset()
        self.secondary.reset()

    def handle_episode_mapping(self, terminated: bool | list[bool], force_reset: bool | list[bool] = False):
        # Type handling and error catching
        if isinstance(terminated, bool):
            terminated = [terminated]
        if isinstance(force_reset, bool):
            force_reset = [force_reset for _ in terminated]
        assert len(terminated) == len(force_reset)

        # Handle episode mapping
        if any(terminated) or any(force_reset) or len(self.episode_mapping) == 0 or self.terminated:
            # Give env new mapping if it is done
            for env_idx, (done, reset) in enumerate(zip(terminated, force_reset)):
                # Assign new mapping for each env instance on no mapping, instance termination or forced reset
                ep_map = self.episode_mapping.get(env_idx)
                # ep_len = self.episode_lengths.get(ep_map)
                if any([ep_map is None, done, reset]):
                    # print(cmod(f"Assigned mapping {self.episode_mapping.get(env_idx)} "
                    #            f"term: {len(terminated)} -> {[self.episode_mapping.get(env_idx) is None, done, force_reset[env_idx]]}", Fore.LIGHTRED_EX))
                    self.episode_mapping[env_idx] = next(self.mapping_indexer)
            # Delete invalid ep mapping; NOTE: For the case where num of envs change
            for env_idx in list(self.episode_mapping.keys()):
                if 0 > env_idx > len(terminated) - 1:
                    del self.episode_mapping[env_idx]
            # Delete invalid episode maps
            mapping_list = list(self.episode_mapping.values())
            # NOTE: Set self.terminated to False after running this method, because ep_lengths might be deleted on continuous episode runs
            if self.terminated:
                for ep_map in list(self.episode_lengths.keys()):
                    if ep_map not in mapping_list:
                        del self.episode_lengths[ep_map]
            # Initialize the episode lengths mapping for new episodes
            assert len(self.episode_mapping) == len(force_reset)
            for env_idx, env_map in enumerate(mapping_list):
                # Init for episodes that are not in dict, been completed or forcefully reset
                if self.episode_lengths.get(env_map) is None:
                    self.episode_lengths[env_map] = 0
        # For case where rollout buffer has been filled and new episodes have been made fore future env data collection
        if self.terminated: # self.steps_done == self.prev_steps_done:
            for ep_map in self.episode_lengths.keys():
                self.episode_lengths[ep_map] = 0

    def get_batches(self, keys: list[int], batch_size: int = None, shuffle=False):
        batches = {}
        buffer_sizes = self.primary.buffer_sizes()
        for key in keys:
            records = int(buffer_sizes[key])

            if batch_size is None:
                batch_size = records
            assert records > 0
            if batch_size is None:
                batch_size = records
            else:
                assert batch_size > 0
                batch_size = min(batch_size, records)

            indices: list[int] = list(range(records))
            if shuffle:
                random.shuffle(indices)
            batch_indices: list[list[int]] = list()
            batch: list[int] = list()
            for i, index in enumerate(indices):
                batch.append(index)
                batch_is_filled = len(batch) == batch_size
                no_more_records = i == len(indices) - 1 and not batch_is_filled
                if batch_is_filled or no_more_records:
                    batch_indices.append(batch)
                    batch = list()

            batches[key] = batch_indices
        return batches

    # TODO: Complete
    def compute_returns_and_advantage(
            self, rewards: TensorDict, values: Union[TensorDict, None],
            gamma=0.95, gae_lambda=0.91, alpha=1.10, reverse=False,
            episodes: dict[int, list[int]] = None, observations: TensorDict = None, device: torch.device = None
    ):
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.
        """

        keys = list(rewards.keys())
        max_records = int(np.max([tensor.shape for tensor in rewards.values()]))
        if episodes is None:
            episodes = {key: [0 for _ in range(max_records)] for key in keys}

        @njit
        def sorting_is_correct(eps_: list[int]):
            idx_ = eps_[0]
            for ep_idx_ in eps_:
                if ep_idx_ < idx_ or ep_idx_ > idx_ + 1:
                    return False
                idx_ = ep_idx_
            return True

        for key, eps in episodes.items():
            if not sorting_is_correct(eps):
                raise ValueError(f"{self.__class__.__name__} episodes have not been sorted well for key '{key}'. "
                                 f"Got: \n{eps}")

        # Get values in case they aren't available
        if values is None:
            if observations is None:
                raise RuntimeError(f"Cannot compute returns and advantages without values or observations")
            values = {}
            for key, observation in observations.items():
                stack = []
                for batch in self.get_batches([key], self.batch_size, False)[key]:
                    value = self.get_module(key).get_value(observation[batch].to(self.device), keys=key)
                    stack.append(value)
                values[key] = torch.cat(stack).cpu()

        advantages  = {}
        returns     = {}
        for (key, rewards_), (ck1, values_), (ck2, episodes_) in zip(rewards.items(), values.items(), episodes.items()):
            assert key == ck1 == ck2
            prev_ep_idx = episodes_[-1]
            adv_stack = []
            ret_stack = []
            future_value: Tensor = torch.zeros_like(values_[0])
            future_advantage: Tensor = torch.zeros_like(values_[0])
            factor = len(np.unique(episodes_))-1 if not reverse else 0
            for reward, value, ep_idx in reversed(list(zip(rewards_, values_, episodes_))):
                if prev_ep_idx != ep_idx:
                    future_value = torch.zeros_like(values_[0])
                    future_advantage = torch.zeros_like(values_[0])
                    if not reverse:
                        factor -= 1
                    else:
                        factor += 1
                reward = reward * (alpha ** factor)
                # reward = reward * ((alpha if (reward > 0 and alpha > 1) or (reward < 0 and alpha < 1) else 1) ** factor)
                delta = reward + gamma*future_value - value
                future_value = value
                future_advantage = delta + gamma*gae_lambda*future_advantage
                adv_stack.insert(0, future_advantage)
                ret_stack.insert(0, future_advantage + value)
                prev_ep_idx = ep_idx
            adv, ret = torch.stack(adv_stack), torch.stack(ret_stack)
            if device is not None:
                adv, ret = adv.to(device=device), ret.to(device=device)
            advantages[key] = adv
            returns[key] = ret

        return returns, advantages

    @staticmethod
    def compute_returns_static(
            rewards: TensorDict,
            gamma: float = 0.97, kappa: float = 0.87, alpha: float = 1.00, order=0, normalize: int = 1,
            episodes: dict[int, list[int]] = None, device: torch.device = None, self: 'Algorithm' = None
    ):
        if -1 >= order > 6:
            raise ValueError(f"Invalid alpha order: '{order}'")

        keys = list(rewards.keys())

        if episodes is None:
            try:
                max_records = int(np.max([tensor.shape if not isinstance(tensor, (float, int)) else tensor for tensor in rewards.values()]))
            except Exception as e:
                zero_rew_num = np.count_nonzero([len(tensor) for tensor in rewards.values()])
                print(CM(
                    f"\nThe number of genomes with 0 rewards are {zero_rew_num}/{len(rewards)}:\n" + (
                        f"\tgenomes = {len(self.primary.data)}"
                        f"\tmin_size = {self.primary.min_size()}"
                        f"\tmax_size = {self.primary.max_size()}"
                        f"\n"
                    ) if self is not None else '',
                    Fore.MAGENTA
                ))
                raise e
            episodes = {key: [0 for _ in range(max_records)] for key in keys}

        @njit
        def sorting_is_correct(episode_list: list[int]):
            # Start as first episodes
            current_ep = episode_list[0]
            # Loop until last episode index
            for ep_to_check in episode_list:
                if ep_to_check < current_ep:
                    return False, (current_ep, ep_to_check)
                current_ep = ep_to_check
            return True, None

        for key, eps in episodes.items():
            correct, error = sorting_is_correct(eps)
            if not correct:
                raise ValueError(f"Episodes have not been sorted well for '{self.__class__.__name__}'; "
                                 f"Error for key '{key}' at index {error[1]}, when testing for index {error[0]}.")

        def torch_std(tensor):
            if tensor.ndim < 1 or tensor.numel() <= 1:
                return torch.zeros_like(tensor).mean()
            return torch.std(tensor)

        def numpy_std(array):
            if len(array) <= 1:
                return 0.0
            return np.std(array)

        def fetch(attr: float, var: float, func: callable):
            if var is None:
                return None
            elif attr is None:
                return var
            else:
                return func(attr, var)

        global_mean, global_std = [
            g([f(r).cpu().item() for r in rewards.values()])
            for f, g in zip([torch.mean, torch_std], [np.mean, numpy_std])
        ] if normalize == 1 else (None, None)
        global_min, global_max = [
            g([f(r).cpu().item() for r in rewards.values()])
            for f, g in zip([torch.min, torch.max], [np.min, np.max])
        ] if normalize in [2, 3] else (None, None)
        # if normalize:
        #     try:
        #         self._global_mean = global_mean = fetch(self._global_mean, global_mean, max)
        #         self._global_std = global_std = fetch(self._global_std, global_std, min)
        #         self._global_min = global_min = fetch(self._global_min, global_min, min)
        #         self._global_max = global_max = fetch(self._global_max, global_max, max)
        #     except AttributeError:
        #         pass
        # max_ep_len = np.max([
        #     np.max([
        #         len([i for i, ep_idx in enumerate(listing) if ep_idx == u_idx])
        #         for u_idx in np.unique(listing)
        #     ]).item()
        #     for key, listing in episodes.items()
        # ]).item()
        returns: TensorDict = {}
        for (key, rewards_), (c_key, episodes_) in zip(rewards.items(), episodes.items()):
            if normalize == 1 and global_std != 0.0:
                rewards_ = (rewards_ - global_mean) / (global_std + 1e-9)
            elif normalize in [2, 3] and (global_max - global_min) != 0.0:
                rewards_ = (rewards_ - global_min) / (global_max - global_min)
                if normalize == 3:
                    rewards_ = -1 + 2 * rewards_
            if order in [2, 3, 4, 5]:
                if len(rewards_) != len(episodes_):
                    raise ValueError(f"Number of rewards (scores) must be equal to number of episodes for key '{key}' "
                                     f"when parameter 'best' is enabled.")
                scores = torch.mean(rewards_.view(rewards_.shape[0], -1), dim=-1)
                episodes_ = torch.tensor(episodes_, device=scores.device, dtype=scores.dtype)
                _, episode_ranking = torch.sort(scores, descending=False if order in [2, 4] else True) # Ensure the best is last
                rewards_ = rewards_[episode_ranking] # If episodes are re-ordered, ensures the same on the rewards
                episodes_ = episodes_[episode_ranking].tolist()
            assert key == c_key
            idx = episodes_[-1]
            prop_past_reward: Tensor = 0.
            prop_future_reward: Tensor = 0.
            ep_total = len(np.unique(episodes_))
            ep_factors = list(range(ep_total))[::(-1 if order not in [1,] else +1)]

            def get_ai(remaining_factors: list[int]) -> int:
                if len(remaining_factors) <= 0:
                    raise ValueError(f"No episodes rolled out or incorrect mapping")
                if order not in [4, 5, 6] or len(remaining_factors) == 1:
                    return 0
                else:
                    ef = np.arange(len(remaining_factors))
                    ef_ = ef[::-1] + 1
                    try:
                        return np.random.choice(ef, p=(ef_ / np.cumsum(ef_).max()) if order in [4, 5] else None)
                    except Exception as e:
                        print(ef)
                        print(ef / np.cumsum(ef).max())
                        print(remaining_factors)
                        raise e
            ep_factor = ep_factors.pop(get_ai(ep_factors))

            # Handle Alpha
            if alpha is not None and alpha > 1.0:
                # TODO: Reverse the listing of ep_factors
                for index, (reward, ep_idx) in reversed(list(enumerate(zip(rewards_, episodes_)))):
                    if idx != ep_idx:
                        try:
                            ep_factor = ep_factors.pop(get_ai(ep_factors))
                        except Exception as e:
                            print(ep_factors)
                            print(get_ai(ep_factors))
                            raise e
                    # TODO: Should length factor be re-enabled for Alpha?
                    # ep_len = len([e for e in episodes_ if e == ep_idx])
                    # len_factor = 1 # ep_len / max_ep_len
                    # diff = alpha - 1.0
                    rewards_[index] = reward * (alpha ** ep_factor) # ((1.0 + (diff * len_factor)) ** ep_factor)
                    idx = ep_idx

            # Handle Gamma
            rtg_gamma = []
            discounted_reward: Tensor | float = 0.
            if gamma > 0:
                for reward, ep_idx in reversed(list(zip(rewards_, episodes_))):
                    if idx != ep_idx:
                        discounted_reward = 0.
                    discounted_reward = reward + (discounted_reward * gamma)
                    rtg_gamma.insert(0, discounted_reward)
                    idx = ep_idx
            rtg_gamma = torch.stack(rtg_gamma).to(rewards_.device, rewards_.dtype) if len(rtg_gamma) > 0 else None

            # Handle Kappa
            rtg_kappa = []
            discounted_reward: Tensor | float = 0.
            if kappa > 0:
                for reward, ep_idx in zip(rewards_, episodes_):
                    if idx != ep_idx:
                        discounted_reward = 0.
                    discounted_reward = reward + (discounted_reward * gamma)
                    rtg_kappa.append(discounted_reward)
                    idx = ep_idx
            rtg_kappa = torch.stack(rtg_kappa).to(rewards_.device, rewards_.dtype) if len(rtg_kappa) > 0 else None

            if rtg_gamma is not None and rtg_kappa is not None:
                rewards_to_go = (rtg_gamma + rtg_kappa) / 2
            elif rtg_gamma is not None:
                rewards_to_go = rtg_gamma
            elif rtg_kappa is not None:
                rewards_to_go = rtg_kappa
            else:
                rewards_to_go = rewards_.clone()
            if device is not None:
                rewards_to_go = rewards_to_go.to(device=device)
            returns[key] = rewards_to_go
        return returns

    def compute_returns(self, rewards: TensorDict,
                        gamma: float = 0.97, kappa: float = 0.87, alpha: float = 1.00, order=0,
                        normalize=False, episodes: dict[int, list[int]] = None, device: torch.device = None):
        return self.compute_returns_static(rewards, gamma, kappa, alpha, order, normalize, episodes, device, self)

    def get_accuracy(self, batches: dict[int, list[list[int]]], observations: TensorDict, actions: TensorDict,
                     rewards: TensorDict = None, error=0.10, type='continuous', verbose: int = None,
                     keys: Union[int, list[int]] = None) -> tuple[dict[int, float], dict[int, float]]:
        if isinstance(keys, (int, float)):
            keys = [keys]
        all_keys = list(self.primary.mapping.keys())
        keys = all_keys if keys is None else keys
        with torch.no_grad():
            ts, ud, ut = clock.perf_counter(), 0, len(keys)
            actions_acc, rewards_acc = {}, {}

            # Calculate accuracy for each key
            for key in all_keys:
                if key in keys:
                    action_sum, reward_sum = [], []
                    for batch in batches[key]:
                        # Calculate
                        observation = observations[key][batch].to(self.device)

                        # Get action accuracy
                        action = actions[key][batch].to(self.device)
                        try:
                            action_pred: Tensor = self.get_module(key).get_policy(observation.unsqueeze(0), keys=key).squeeze(0)
                        except Exception as e:
                            self.get_module(key).get_policy(observation.unsqueeze(0), keys=key, verbose=2)
                            raise e
                        try:
                            if type == 'continuous':
                                action_res = ((action_pred <= action * (1+error)) & (action_pred >= action * (1-error))).float()
                            elif type == 'binary':
                                action_res = ((action_pred >= (1 - error)) == (action >= (1 - error))).float()
                            elif type == 'discrete':
                                action_res = (action_pred == action).float()
                            else:
                                raise ValueError(f"Unsupported accuracy type '{type}'")
                        except Exception as e:
                            print(CM(f'Action Prediction = {action_pred.shape}, Target = {action.shape}\n', Fore.LIGHTRED_EX))
                            raise e
                        action_sum.append(action_res)

                        # Get reward accuracy
                        if rewards is not None:
                            reward = rewards[key][batch].to(self.device)
                            reward_pred: Tensor = self.get_module(key).get_value(observation.unsqueeze(0), keys=key).squeeze(0)
                            reward_sum.append(
                                ((reward_pred <= reward * (1+error)) & (reward_pred >= reward * (1-error))).float()
                            )
                        else:
                            reward_sum.append(torch.zeros(1).float())
                    r = torch.concat(reward_sum)
                    a = torch.concat(action_sum)

                    actions_acc[key] = a.mean().cpu().item()
                    rewards_acc[key] = r.mean().cpu().item()
                else:
                    actions_acc[key] = 0
                    rewards_acc[key] = 0

                if verbose:
                    ud += 1
                    eta(ts, ud, ut, 'Getting accuracy')

            if verbose:
                print(f"\rGot accuracy in {round(clock.perf_counter() - ts, 2)}s")
            return actions_acc, rewards_acc

    def log_scheduler_params(self):
        values: dict[str, Union[int, float, bool]] = {}
        for scheduler in self.schedulers:
            for param in scheduler.params:
                values[param] = scheduler.get(param, scheduler.config)
        return values

    @staticmethod
    def segregate(ranking: dict[int, float], size: int):
        groups = math.ceil(len(ranking) / size)
        if groups < 1:
            raise ValueError(f"Invalid segregation size '{size}' with '{groups}' groups and '{len(ranking)}' keys")
        keys = list(ranking.keys())
        index = 0
        clusters: list[dict[int, float]] = [
            {key: ranking[key] for key in keys[index+size*x:index+size*(x+1)]} for x in range(groups)
        ]
        return clusters

    def normalize_array(self, array: dict[int, Any], index: int = None, segr_size: int = None, ranking: dict[int, float] = None):
        keys, source = list(array.keys()), list(array.values())

        # Handle errors
        if len(source) == 0:
            return array
        if isinstance(source[0], (list, tuple)):
            for i, (key, item) in enumerate(zip(keys, source)):
                if len(item) == 0:
                    raise ValueError(f"Empty Iterable found in key '{key}'")
                if isinstance(item[0], (int, float)):
                    if index is None:
                        source[i] = np.mean(item).item()
                    else:
                        source[i] = item[-1]
                else:
                    ValueError(f"Cannot convert variable of type '{type(item)}'")
        elif isinstance(source[0], ndarray):
            source = [item().mean().item() for item in source]
        elif isinstance(source[0], Tensor):
            source = [item().mean().item() for item in source]
        elif not isinstance(source[0], (float, int)):
            ValueError(f"Cannot consolidate variable of type '{type(source[0])}'")

        source = np.array(source)
        # print(source.shape)
        # print(np.max(source), np.min(source), source.mean(), source.std())
        maximum, minimum = np.max(source), np.min(source)
        if maximum > minimum:
            norm_source: ndarray = (source - minimum) / (maximum - minimum)
        else:
            norm_source = np.full_like(source, 1.0)

        norm_array = dict(zip(keys, norm_source))

        # Segregate normalization when enabled
        if segr_size is not None:
            # Sort if ranking is not given
            if ranking is None:
                norm_array = dict(sorted(norm_array.items(), key=lambda item: (item[1], -item[0]), reverse=True))
            else:
                norm_array = {key: norm_array[key] for key in ranking.keys()}

            segr_norm_array: dict[int, float] = {}
            clusters = self.segregate(norm_array, segr_size)
            segr_range = max(list(norm_array.values())) / len(clusters)
            for c_idx, cluster in enumerate(clusters):
                for key, norm_value in self.normalize_array(cluster).items():
                    segr_norm_array[key] = (norm_value*segr_range) + ((len(clusters)-1-c_idx)*segr_range)
            norm_array = segr_norm_array

        return norm_array

    @staticmethod
    def level_array(array: dict[int, Any]) -> dict[int, float]:
        keys, source = list(array.keys()), list(array.values())

        # Handle errors
        if len(source) == 0:
            raise ValueError("No keys in dict")
        if isinstance(source[0], list):
            for i, (key, item) in enumerate(zip(keys, source)):
                if len(item) == 0:
                    raise ValueError(f"Empty Iterable found in key '{key}'")
                if isinstance(item, (int, float)):
                    source[i] = np.mean(item).item()
                else:
                    ValueError(f"Cannot convert variable of type '{type(item)}'")
        elif isinstance(source[0], ndarray):
            source = [item().mean().item() for item in source]
        elif isinstance(source[0], Tensor):
            source = [item().mean().item() for item in source]
        elif not isinstance(source[0], (float, int)):
            ValueError(f"Cannot consolidate variable of type '{type(source[0])}'")

        source = np.array(source)
        level_source = source - np.min(np.clip(source, None, 0))
        return {k: v for k, v in zip(keys, level_source)}

    @staticmethod
    def sort_episodes(episode_mapping: dict[int, list[int]], *buffers: TensorDict):
        episode_lengths = {}
        for key in episode_mapping.keys():
            records = len(episode_mapping[key])
            sorting_indices = torch.tensor(
                sorted([i for i in range(records)], key=lambda idx: (episode_mapping[key][idx], idx)), dtype=torch.int
            )

            for buffer in buffers:
                buffer[key] = torch.index_select(buffer[key], 0, sorting_indices)

            episode_mapping[key] = np.array(episode_mapping[key])[sorting_indices.cpu().tolist()].tolist()

            length = {}
            for ep_idx in episode_mapping[key]:
                if ep_idx not in length:
                    length[ep_idx] = 1
                else:
                    length[ep_idx] += 1
            lengths = list(length.values())
            episode_lengths[key] = lengths

        return episode_lengths

    # # TODO: Implement saving and loading of Reinforcing
    # def save(self, name: str = None, directory: str = None, file_no: int = None, replace=False):
    #     # exclude = ['population', 'parameters', 'model', 'replay', 'logging', 'writer']
    #     # state = {var: getattr(self, var) for var in vars(self).keys() if var not in exclude}
    #     #
    #     # algo = self.__class__.__name__
    #     # if name is None:
    #     #     name = 'default'
    #     # if directory is None:
    #     #     directory = f'{algo.lower()}'
    #     #
    #     # # Save Trainer data
    #     # if save(state, name, directory, file_no, replace, items_name=f'NEAT {algo}')[0]:
    #     #     # Save population
    #     #     self.population.save_dict(name, directory, file_no, replace)
    #     #
    #     #     print(CM(f"Successfully saved trainer to '{directory}\\{name}'", Fore.LIGHTGREEN_EX))
    #     # else:
    #     #     print(CM(f"Failed to save trainer to '{directory}\\{name}'", Fore.LIGHTRED_EX))
    #     raise NotImplementedError()
    #
    # def load(self, name: str = None, directory: str = None, file_no: int = None):
    #     # algo = self.__class__.__name__
    #     # if name is None:
    #     #     name = 'default'
    #     # if directory is None:
    #     #     directory = f'{algo.lower()}'
    #     # state = load(name, directory, file_no, items_name=f'NEAT {algo}')
    #     # if state is not None:
    #     #     # Load Trainer data
    #     #     for var, val in state.items():
    #     #         setattr(self, var, val)
    #     #
    #     #     # Load Population
    #     #     self.population.load_dict(None, name, directory, file_no)
    #     #
    #     #     # Load Model and Params
    #     #     self._get_params(self.model)
    #     #     bind_modules(self.parameters, self.population.genomes, True)
    #     #
    #     #     print(CM(f"Successfully loaded trainer from '{directory}\\{name}'", Fore.LIGHTGREEN_EX))
    #     # else:
    #     #     print(CM(f"Failed to load trainer from '{directory}\\{name}'", Fore.LIGHTRED_EX))
    #     raise NotImplementedError()
