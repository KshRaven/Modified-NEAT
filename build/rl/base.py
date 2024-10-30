
from build.nn.base import NeatModule, get_modules
from build.population import Population
from build.optim.scheduler import Scheduler
from build.util.replay import ReplayBuffer
from build.util.datetime import eta, clock

from torch import Tensor
from torch.optim import Optimizer
from numba import njit
from numpy import ndarray

import torch
import torch.nn as nn
import random
import numpy as np


class Algorithm(object):
    def __init__(self, model: nn.Module, population: Population):
        self.model: nn.Module             = model
        self.parameters: list[NeatModule] = self._get_params(model)
        self.population: Population       = population
        self.optimizer: Optimizer         = None
        self.scheduler: Scheduler         = None
        self.replay                       = ReplayBuffer()
        self.logging                      = ReplayBuffer()

    @staticmethod
    def _get_params(module: nn.Module):
        return get_modules(module)

    @staticmethod
    def _get_batches(records: int, batch_size: int = None, shuffle=False):
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

        return batch_indices

    @staticmethod
    def _handle_ndim(tensor: Tensor, req: int):
        extra = max(0, req - tensor.ndim)
        for _ in range(extra):
            tensor = tensor.unsqueeze(-1)
        return tensor

    @staticmethod
    def _get_rewards_to_go(rewards: Tensor, gamma: float = 0.95, episodes: list[int] = None):
        if episodes is None:
            episodes = [0 for _ in range(rewards.shape[0])]

        @njit
        def verify_sorting(episodes_: list[int]):
            idx_ = episodes_[0]
            for ep_idx_ in episodes_:
                if ep_idx_ < idx_ or ep_idx_ > idx_ + 1:
                    return False
                idx_ = ep_idx_
            return True

        if not verify_sorting(episodes):
            raise ValueError(f"Episodes have not been sorted")

        # print(episodes)
        rtg = []
        idx = episodes[-1]
        discounted_reward: Tensor = 0.
        for reward, ep_idx in reversed(list(zip(rewards, episodes))):
            if idx != ep_idx:
                discounted_reward = 0.
            discounted_reward = reward + (discounted_reward * gamma)
            rtg.insert(0, discounted_reward.unsqueeze(0))
            idx = ep_idx
        return torch.cat(rtg).to(rewards.device, rewards.dtype)

    def _get_accuracy(self, batches: list[list[int]], observations: Tensor, actions: Tensor, rewards: Tensor,
                      best_idx: int, error=0.10, type='continuous', verbose: int = None):
        with torch.no_grad():
            action_sum, reward_sum = [], []
            ts, ud, ut = clock.perf_counter(), 0, len(batches)
            for batch in batches:
                action_pred: Tensor = self.model.get_policy(observations[batch])
                reward_pred: Tensor = self.model.get_value(observations[batch])

                target = actions[batch]
                if type == 'continuous':
                    action_res = ((action_pred <= target * (1+error)) & (action_pred >= target * (1-error))).float().cpu()
                elif type == 'binary':
                    action_res = ((action_pred >= (1 - error)).float() == target).float().cpu()
                elif type == 'discrete':
                    action_res = (action_pred == target).float().cpu()
                else:
                    raise ValueError(f"Unsupported accuracy type '{type}'")
                action_sum.append(action_res)
                reward_sum.append(
                    ((reward_pred <= rewards[batch] * (1+error)) & (reward_pred >= rewards[batch] * (1-error))).float().cpu()
                )

                if verbose:
                    ud += 1
                    eta(ts, ud, ut, 'getting accuracy')
            r = torch.cat(reward_sum)
            a = torch.cat(action_sum)
            for _ in range(r.ndim-a.ndim):
                a = a.unsqueeze(-1)
            r = r[..., best_idx, :]
            a = a[..., best_idx, :]
            actions_acc: float = a.mean().item() * 100
            rewards_acc: float = r.mean().item() * 100
            if verbose:
                print(f"\r", end='')
            return actions_acc, rewards_acc

    @staticmethod
    def get_dims(tensor: Tensor):
        if not isinstance(tensor, Tensor):
            raise ValueError(f"variable provided is not a torch Tensor; type = {type(tensor)}")
        return tuple([i for i in range(tensor.ndim) if i != tensor.ndim-2])

    @staticmethod
    def normalize(tensor):
        if isinstance(tensor, Tensor):
            res = (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))
        elif isinstance(tensor, ndarray):
            res = (tensor - np.min(tensor)) / (np.max(tensor) - np.min(tensor))
        else:
            raise ValueError(f"unsupported dtype")
        return res

    @staticmethod
    def level(tensor):
        if isinstance(tensor, Tensor):
            excess = torch.min(torch.clamp(tensor, None, 0))
            res = tensor - excess
        elif isinstance(tensor, ndarray):
            excess = np.min(np.clip(tensor, None, 0))
            res = tensor - excess
        else:
            raise ValueError(f"unsupported dtype")
        return res
