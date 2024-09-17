
from build.util.qol import manage_params
from build.util.datetime import eta

from torch import Tensor
from torch.optim import Optimizer

import torch
import torch.nn as nn
import torch.nn.functional as F
import time as clock
import random


def _update_learning_rate(optimizer: Optimizer, learning_rate: float):
    for param_group in optimizer.param_groups:
        if learning_rate is not None:
            param_group['lr'] = learning_rate


def _update_weight_decay(optimizer: Optimizer, weight_decay: float):
    for param_group in optimizer.param_groups:
        if weight_decay is not None:
            param_group['weight_decay'] = weight_decay


def _update_gamma(optimizer: Optimizer, gamma: tuple[float, float]):
    for param_group in optimizer.param_groups:
        if gamma is not None:
            param_group['gamma'] = gamma


def _get_rewards_to_go(rewards: list[Tensor], gamma: float = 0.95):
    # Shape = (genomes, batch_size, rewards)
    rtg = list()
    discounted_reward = 0.
    for reward in reversed(rewards):
        discounted_reward = reward + (discounted_reward * gamma)
        rtg.insert(0, discounted_reward)
    return rtg


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


def mutate_ppo(eval_func: callable, optimizer: Optimizer,
               states: list[Tensor], policy_old: list[Tensor], rewards: list[Tensor], value_est: list[Tensor],
               epochs=1, batch_size: int = None, **ex):
    learning_rate   = manage_params(ex, ['learning_rate', 'lr'], 5e-4)
    weight_decay    = manage_params(ex, ['weight_decay', 'wd'], 0.0)
    norm_rew        = manage_params(ex, ['normalize_rewards', 'norm_rew'], True)
    gamma           = manage_params(ex, 'gamma', 0.95)
    epsilon         = manage_params(ex, 'epsilon', 1e-10)
    clip_range      = manage_params(ex, 'clip_range', 0.2)
    ent_reg         = manage_params(ex, 'ent_reg', 0.1)
    val_reg         = manage_params(ex, 'val_reg', 1.0)
    clip_grad       = manage_params(ex, 'clip_grad', None)
    model_params    = manage_params(ex, 'params', None)
    device          = manage_params(ex, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
    dtype           = manage_params(ex, 'dtype', torch.float64)
    best            = manage_params(ex, 'best', None)
    debug           = manage_params(ex, 'debug', True)

    _update_learning_rate(optimizer, learning_rate)
    _update_weight_decay(optimizer, weight_decay)

    assert len(states) == len(policy_old) and len(rewards) == len(value_est)
    records = len(states)
    batch_indices = _get_batches(records, batch_size)

    with torch.no_grad():
        rewards            = _get_rewards_to_go(rewards, gamma)
        states: Tensor     = torch.cat(states, dim=1).to(device, dtype)
        policy_old: Tensor = torch.cat(policy_old, dim=1).to(device, dtype)
        rewards: Tensor    = torch.cat(rewards, dim=1).to(device, dtype)
        value_est: Tensor  = torch.cat(value_est, dim=1).to(device, dtype)
        # print(states.shape, policy_old.shape, rewards.shape, value_est.shape)

        if best is not None:
            best_ = torch.zeros(*states.shape[: 2], 1)
            for i in range(records):
                best_[best[i], i] = 1
            best = best_.to(dtype=torch.bool)
            # print(best.shape)
            default    = (1, records, -1)
            states     = states[best.expand(*states.shape)].view(default).expand(*states.shape)
            policy_old = policy_old[best.expand(*policy_old.shape)].view(default).expand(*policy_old.shape)
            rewards    = rewards[best.expand(*rewards.shape)].view(default).expand(*rewards.shape)
            value_est  = value_est[best.expand(*value_est.shape)].view(default).expand(*value_est.shape)
        # print(states.shape, policy_old.shape, rewards.shape, value_est.shape)

        if norm_rew:
            # print(states.shape, policy_old.shape, rewards.shape, value_est.shape)
            # print((torch.std(rewards, 1, keepdim=True) + epsilon).shape)
            # print((rewards - torch.mean(rewards, 1, True)).shape)
            rewards = (rewards - torch.mean(rewards, 1, keepdim=True)) / (torch.std(rewards, 1, keepdim=True) + epsilon)
    advantages = rewards - value_est

    ts = clock.perf_counter()
    try:
        torch.cuda.empty_cache()
        units_done = 0
        units_total = epochs * len(batch_indices)
        for epoch in range(epochs):
            for batch in batch_indices:
                units_done += 1
                optimizer.zero_grad()
                actv: tuple[Tensor, Tensor, Tensor] = eval_func(states[:, batch])
                if len(actv) == 3:
                    policy, values, entropy = actv
                else:
                    policy, values = actv
                    entropy = None

                # Policy Loss
                ratios = policy / policy_old[:, batch]
                s1 = ratios * advantages[:, batch]
                s2 = torch.clamp(ratios, 1 - clip_range, 1 + clip_range)
                min = torch.min(s1, s2)
                policy_loss = -min.mean((-1, -2)).sum(-1)

                # Entropy Loss
                if entropy is not None:
                    entropy_loss = -entropy.mean((-1, -2)).sum(-1) * ent_reg
                else:
                    entropy_loss = None
                # print(entropy_loss)

                # Values Loss
                value_loss = F.mse_loss(values, rewards[:, batch]) * val_reg

                # Total Loss
                loss = policy_loss + value_loss
                if entropy is not None:
                    loss += entropy_loss

                # Backpropagate
                loss.backward()
                if clip_grad is not None and model_params is not None:
                    torch.nn.utils.clip_grad_norm(model_params, clip_grad)
                optimizer.step()

                # Debug
                if debug:
                    eta(ts, units_done, units_total,
                        f'ppo mutation tl={loss.item():.4f}, '
                        f'pl={policy_loss.item():.4f}, vl={value_loss.item():.4f}, el={entropy_loss.item():.4f}')

    except KeyboardInterrupt:
        torch.cuda.empty_cache()
        pass

    if debug:
        print(f"  ")
