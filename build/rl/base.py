
from build.nn.base import Model
from build.population import Population
from build.optim.scheduler import Scheduler
from build.util.replay import ReplayBuffer
from build.util.datetime import eta, clock
from build.util.storage import save, load
from build.util.fancy_text import CM, Fore

from torch import Tensor
from numba import njit
from numpy import ndarray
from typing import Any, Union

import torch
import torch.nn as nn
import random
import numpy as np

TensorDict = dict[int, Tensor]


class Algorithm(object):
    def __init__(self, model: Model, population: Population):
        self.model                      = model
        self.population: Population     = population
        self.scheduler: Scheduler       = None
        self.replay                     = ReplayBuffer()
        self.logging                    = ReplayBuffer()
        self.alpha_steps_done           = 0

        self.device: torch.device       = 'cpu'
        self.dtype: torch.dtype         = torch.float32

    def _get_batches(self, keys: list[int], batch_size: int = None, shuffle=False):
        batches = {}
        for key in keys:
            records = len(list(self.replay.data[key].values())[0])

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

    @staticmethod
    def _get_rewards_to_go(rewards: TensorDict, gamma: float = 0.95, alpha: float = 1.10, step: int = 5,
                           episodes: dict[int, list[int]] = None, device: torch.device = None, rollout=False):
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
                raise ValueError(f"Episodes have not been sorted well for key '{key}'.")

        # print(episodes)
        cumulative_rewards: TensorDict = {}
        for (key, rewards_), (c_key, episodes_) in zip(rewards.items(), episodes.items()):
            assert key == c_key
            rewards_to_go = []
            idx = episodes_[-1]
            discounted_reward: Tensor = 0.
            factor = 0
            for reward, ep_idx in reversed(list(zip(rewards_, episodes_))):
                if idx != ep_idx:
                    discounted_reward = 0.
                    factor += 1
                discounted_reward = reward + (discounted_reward * gamma)
                # if rollout and step is not None and self.alpha_steps_done % step == 0:
                reward = discounted_reward * (alpha ** factor)
                # else:
                #     reward = discounted_reward
                rewards_to_go.insert(0, reward)
                idx = ep_idx
            cm = torch.stack(rewards_to_go).to(rewards_.device, rewards_.dtype)
            if device is not None:
                cm = cm.to(device=device)
            cumulative_rewards[key] = cm
        # if rollout:
        #     self.alpha_steps_done += 1
        return cumulative_rewards

    def _get_accuracy(self, batches: dict[int, list[list[int]]], observations: TensorDict, actions: TensorDict,
                      rewards: TensorDict, error=0.10, type='continuous', verbose: int = None,
                      keys: Union[int, list[int]] = None):
        if isinstance(keys, (int, float)):
            keys = [keys]
        keys = list(self.replay.mapping.keys()) if keys is None else keys
        with torch.no_grad():
            ts, ud, ut = clock.perf_counter(), 0, len(batches)
            actions_acc, rewards_acc = {}, {}

            # Calculate accuracy for each key
            for key in keys:
                action_sum, reward_sum = [], []
                for batch in batches[key]:
                    # Calculate
                    observation = observations[key][batch].to(self.device)
                    action      = actions[key][batch].to(self.device)
                    reward      = rewards[key][batch].to(self.device)
                    action_pred: Tensor = self.model.get_policy(observation.unsqueeze(0), keys=key).squeeze(0)
                    reward_pred: Tensor = self.model.get_value(observation.unsqueeze(0), keys=key).squeeze(0)

                    # Get action accuracy
                    if type == 'continuous':
                        action_res = ((action_pred <= action * (1+error)) & (action_pred >= action * (1-error))).float()
                    elif type == 'binary':
                        action_res = ((action_pred >= (1 - error)).float() == 1).float()
                    elif type == 'discrete':
                        action_res = (action_pred == action).float()
                    else:
                        raise ValueError(f"Unsupported accuracy type '{type}'")
                    action_sum.append(action_res)

                    # Get reward accuracy
                    reward_sum.append(
                        ((reward_pred <= reward * (1+error)) & (reward_pred >= reward * (1-error))).float()
                    )

                    if verbose:
                        ud += 1
                        eta(ts, ud, ut, 'getting accuracy')
                r = torch.concat(reward_sum)
                a = torch.concat(action_sum)

                actions_acc[key] = a.mean().cpu().item()
                rewards_acc[key] = r.mean().cpu().item()

                if verbose:
                    ud += 1
                    eta(ts, ud, ut, 'Getting accuracy')

            if verbose:
                print(f"\rGot accuracy in {round(clock.perf_counter() - ts, 2)}s")
            return actions_acc, rewards_acc

    @staticmethod
    def normalize(array: dict[int, Any], index: int = None) -> dict[int, float]:
        keys, source = list(array.keys()), list(array.values())

        # Handle errors
        if len(source) == 0:
            raise ValueError("No keys in dict")
        if isinstance(source[0], list):
            for i, (key, item) in enumerate(zip(keys, source)):
                if len(item) == 0:
                    raise ValueError(f"Empty array found in key '{key}'")
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
        return {k: v for k, v in zip(keys, norm_source)}

    @staticmethod
    def level(array: dict[int, Any]) -> dict[int, float]:
        keys, source = list(array.keys()), list(array.values())

        # Handle errors
        if len(source) == 0:
            raise ValueError("No keys in dict")
        if isinstance(source[0], list):
            for i, (key, item) in enumerate(zip(keys, source)):
                if len(item) == 0:
                    raise ValueError(f"Empty array found in key '{key}'")
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

            count = {}
            for ep_idx in episode_mapping[key]:
                if ep_idx not in count:
                    count[ep_idx] = 1
                else:
                    count[ep_idx] += 1
            lengths = list(count.values())
            episode_lengths[key] = lengths

        return episode_lengths

    def save(self, name: str = None, directory: str = None, file_no: int = None, replace=False):
        exclude = ['population', 'parameters', 'model', 'replay', 'logging', 'writer']
        state = {var: getattr(self, var) for var in vars(self).keys() if var not in exclude}

        algo = self.__class__.__name__
        if name is None:
            name = 'default'
        if directory is None:
            directory = f'{algo.lower()}'

        # Save Trainer data
        if save(state, name, directory, file_no, replace, items_name=f'NEAT {algo}')[0]:
            # Save population
            self.population.save_dict(name, directory, file_no, replace)

            print(CM(f"Successfully saved trainer to '{directory}\\{name}'", Fore.LIGHTGREEN_EX))
        else:
            print(CM(f"Failed to save trainer to '{directory}\\{name}'", Fore.LIGHTRED_EX))

    def load(self, name: str = None, directory: str = None, file_no: int = None):
        algo = self.__class__.__name__
        if name is None:
            name = 'default'
        if directory is None:
            directory = f'{algo.lower()}'
        state = load(name, directory, file_no, items_name=f'NEAT {algo}')
        if state is not None:
            # Load Trainer data
            for var, val in state.items():
                setattr(self, var, val)

            # Load Population
            self.population.load_dict(None, name, directory, file_no)

            # Load Model and Params
            self._get_params(self.model)
            bind_modules(self.parameters, self.population.genomes, True)

            print(CM(f"Successfully loaded trainer from '{directory}\\{name}'", Fore.LIGHTGREEN_EX))
        else:
            print(CM(f"Failed to load trainer from '{directory}\\{name}'", Fore.LIGHTRED_EX))
