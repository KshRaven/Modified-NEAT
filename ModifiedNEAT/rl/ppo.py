from ModifiedNEAT.nn.base import Model
from ModifiedNEAT.population import Population
from ModifiedNEAT.rl.base import Algorithm, TensorDict, NEATAlgoWarning
from ModifiedNEAT.optim.scheduler import Scheduler
from ModifiedNEAT.util.datetime import eta, clock
from ModifiedNEAT.util.qol import manage_params
from ModifiedNEAT.util.fancy_text import CM, Fore

from torch import Tensor
from typing import Union, Iterable

import torch
import numpy as np
import warnings
import math


class PPO(Algorithm):
    """
    Evolutionary Proximal Policy Optimization variant.

    Unlike standard PPO, genomes here are not updated through gradients: there is no clipped surrogate
    objective and no optimizer step. Instead, each genome's episodic return and its policy loss (under
    its own, NEAT-evolved policy) are used to rank genomes for the next generation, the same way NEAT
    ranks genomes by return/accuracy. ``Model.evaluate_action`` supplies the log-probabilities used to
    compute that policy loss.
    """

    def __init__(self, population: Population, schedulers: Union[Scheduler, Iterable[Scheduler]] = None,
                 device=torch.device('cpu'), dtype=torch.float32, use_critic: bool = False, **options):
        """
        :param population: Population being trained.
        :param schedulers: Scheduler or iterable of schedulers stepped once per epoch.
        :param device: Device used for computation (default is 'cpu').
        :param dtype: Data type used for tensors (default is torch.float32).
        :param use_critic: When True, ``Model.get_value`` is used to compute a value baseline that is
            subtracted from the returns to form advantages. When False (default), the raw returns are
            used directly as advantages.
        :param options: Additional keyword arguments for configuration.
        :keyword gamma: Discount factor applied to primary (per-step) rewards (default 0.99).
        :keyword kappa: Forward-discount factor mixed in with ``gamma`` for primary rewards (default 0.0).
        :keyword alpha: Per-episode reward multiplier used when reducing secondary returns (default 1.0).
        :keyword order: Episode ranking/ordering mode used when reducing secondary returns (default 0).
        :keyword normalize: Reward normalization mode used when reducing secondary returns (default 0).
        :keyword rew_reg: Weight of the (normalized) return component of the fitness score (default 1.0).
        :keyword loss_reg: Weight of the (normalized, inverted) policy-loss component of the fitness score (default 1.0).
        :keyword pol_reg: Weight of the (normalized, inverted) policy-consistency component of the fitness score
            (default 0.0). Measures the MSE between each action's old and new log-probability (from
            ``Model.evaluate_action``) as a cheap stand-in for MSE-between-action-means, avoiding a dedicated
            ``Model.get_mean`` call or exponentiating log-probabilities back into raw probabilities.
        :keyword ent_reg: Weight of the (normalized, inverted) policy std-dev component of the fitness score
            (default 0.0). Positive values reward genomes with a lower action std-dev (exploitation); negative
            values reward genomes with a higher action std-dev (exploration).
        :keyword use_entropy: When True, the policy std-dev used by ``ent_reg`` (and logged as ``policy_std``)
            is derived analytically from the action entropy returned by ``Model.evaluate_action``, instead of
            calling ``Model.get_std`` on the module (default False). Useful when a genome's module doesn't
            implement ``get_std`` but does return a valid entropy from ``evaluate_action``.
        :keyword div_reg: Weight applied to the std-dev of returns/loss across retained episodes (default 0.1).
        :keyword validate: Whether to restrict training to genomes not flagged for deletion (default False).
        :keyword max_steps: Maximum steps kept in the secondary buffer (default 1024).
        :keyword max_episodes: Maximum episodes kept in the secondary buffer (default None).
        :keyword log_dir: Directory tensorboard logs are written under.
        :keyword log_name: Name of this run's tensorboard log.
        """
        for var in ['schedulers', 'device', 'dtype']:
            if var in options:
                del options[var]
        super().__init__(population, schedulers, device, dtype, **options)

        # Base
        self.use_critic: bool = use_critic

        # Buffers
        self.primary.add_buffers('state', 'action', 'reward', 'log_prob', 'ep_map',)
        # Map each episode's mean return, mean policy loss and mean policy (log-prob) MSE to use as score
        self.secondary.add_buffers('ec_return', 'ep_loss', 'ep_map', 'ep_len', 'ep_mean', 'ep_std')
        # TODO: Fix deque and other buffer clearing methods to not raise Index or Value errors when clearing/dequeing shorter, unfilled or non-existent buffers
        self.score_idx = 0

        # Options
        self.gamma: float       = manage_params(options, 'gamma', 0.99)
        self.kappa: float       = manage_params(options, 'kappa', 0.00)
        self.alpha: float       = manage_params(options, 'alpha', 1.00)
        self.order: int         = manage_params(options, 'order', 0)
        self.beta: float        = manage_params(options, 'beta', 1.00)
        self.beta_order: int    = manage_params(options, 'beta_order', 0)
        self.normalize: int     = manage_params(options, 'normalize', 0)
        self.rew_reg: float     = manage_params(options, 'rew_reg', 1.0)
        self.loss_reg: float    = manage_params(options, 'loss_reg', 0.67)
        self.pol_reg: float     = manage_params(options, 'pol_reg', 0.50)
        self.ent_reg: float     = manage_params(options, 'ent_reg', 0.10)
        self.use_entropy: bool  = manage_params(options, 'use_entropy', True)
        self.div_reg: float     = manage_params(options, 'div_reg', 0.01)
        self.validate: bool     = manage_params(options, 'validate', False)
        self.epsilon: float     = manage_params(options, 'epsilon', 1e-24)
        self.max_steps: Union[float, None] = manage_params(options, 'max_steps', 1024)
        self.max_episodes: Union[float, None] = manage_params(options, 'max_episodes', None)
        if self.max_steps is not None and self.max_episodes is not None:
            warnings.warn(
                category=NEATAlgoWarning,
                message=f"Not recommended to apply both max_steps and max_episodes!"
            )

        self.logging.add_buffers(
            'ep_len_mean', 'ep_len_std', 'ep_rew_mean', 'ep_rew_std', 'policy_loss',
            'policy_accuracy', 'policy_std', 'policy_mean',
        )

        self.prev_valid_keys: list[int] = None

    def update(self, observations: Tensor, actions: Tensor, rewards: Tensor, log_probs: Tensor,
               terminated: Union[bool, list[bool]], force_stop=False, reset_mapping: bool | list[bool] = False,
               session_dim: int | None = None) -> Tensor:
        """
        Records one environment step for every environment instance into the primary buffer.

        :param observations: Observation tensor for this step.
        :param actions: Action tensor taken this step.
        :param rewards: Reward tensor received this step.
        :param log_probs: Log-probability of ``actions`` under the policy that produced them (eg. from
            ``Model.get_action``); re-evaluated later via ``Model.evaluate_action`` to compute the policy loss.
        :param terminated: Whether each environment instance terminated this step.
        :param force_stop: Force the rollout buffer to be treated as filled after this step.
        :param reset_mapping: Force a new episode mapping for each environment instance, regardless of termination.
        :param session_dim: Optional tensor dimension holding the per-environment session; inferred as 1 when omitted.
        :return: Whether the rollout buffer has been filled (steps limit reached or ``force_stop``).
        """
        if isinstance(terminated, bool):
            terminated = [terminated]
        sd_set = session_dim is not None
        if session_dim is None:
            session_dim = 1

        if self.steps_done == self.prev_steps_done:
            self.handle_episode_mapping(terminated, True)
            self.terminated = False

        def select(tensor: Tensor, session_idx: int):
            if len(terminated) > 1 or sd_set:
                assert tensor.shape[session_dim] == len(terminated)
                return torch.select(tensor, session_dim, session_idx)
            else:
                return tensor

        observations, actions, rewards, log_probs = observations.clone(), actions.clone(), rewards.clone(), log_probs.clone()

        filled = False
        if self.steps_done < self.steps_limit:
            # Loop through all environments
            for idx, ended in enumerate(terminated):
                self.primary.update(
                    state    = select(observations, idx),
                    action   = select(actions, idx),
                    reward   = select(rewards, idx),
                    log_prob = select(log_probs, idx),
                    ep_map   = self.episode_mapping[idx],
                )

                self.steps_done += 1
                self.episodes_done = max(self.episode_mapping.values()) + 1
                self.episode_lengths[self.episode_mapping[idx]] += 1

                if self.steps_done >= self.steps_limit or force_stop:
                    self.prev_steps_done = self.steps_done
                    self.prev_episodes_done = self.episodes_done
                    filled = True
        else:
            raise RuntimeError(f"Steps have already been filled.")

        if filled:
            self.terminated = True
        else:
            self.handle_episode_mapping(terminated, reset_mapping)

        return filled

    def _get_policy_loss(self, key: int, states: Tensor, actions: Tensor, old_log_probs: Tensor, advantages: Tensor, indices: list[int]):
        """
        Computes one genome's mean policy loss (``-new_log_prob * advantage``) over a set of records,
        using ``Model.evaluate_action`` to get that genome's current log-probability of ``actions``.

        :param key: Genome key being evaluated.
        :param states: Genome's full state tensor.
        :param actions: Genome's full action tensor, aligned with ``states``.
        :param advantages: Genome's full advantage tensor, aligned with ``states``.
        :param indices: Record indices (typically one episode) to compute the loss over.
        :return: Tuple of (mean policy loss, mean policy consistency loss, mean entropy-derived std-dev)
            over the given records.
        """
        state        = states[indices].to(self.device)
        action       = actions[indices].to(self.device)
        old_log_prob = old_log_probs[indices].to(self.device)
        advantage    = advantages[indices].to(self.device)

        new_log_prob, entropy = self.get_module(key).evaluate_action(state.unsqueeze(0), action.unsqueeze(0), keys=key)
        new_log_prob = new_log_prob.squeeze(0)
        assert new_log_prob.shape == old_log_prob.shape
        
        ratio = torch.exp(new_log_prob - old_log_prob)

        # Reduce any trailing action/reward dimensions to one value per record before combining them
        # new_log_prob = torch.mean(new_log_prob.reshape(new_log_prob.shape[0], -1), dim=-1)
        advantage = torch.mean(advantage.reshape(advantage.shape[0], -1), dim=-1, keepdim=True)

        try:
            ppo_loss = -(ratio * advantage).mean().cpu().item() # Improve rewards
        except RuntimeError as e:
            print(f"ratio = {ratio.shape}")
            print(f"advantage = {advantage.shape}")
            raise e
        if np.isinf(ppo_loss): raise ValueError(f"found NaN/Inf in PPO loss")

        # Policy consistency: MSE between old and new log-probs, used by 'pol_reg' as a cheap stand-in
        # for MSE-between-action-means (no separate get_mean() call or exp() needed).
        pol_loss = torch.mean((new_log_prob.exp() - old_log_prob.exp()) ** 2 + self.epsilon).log10().cpu().item() # Improve accuracy on policy
        if math.isinf(pol_loss) or math.isnan(pol_loss): raise ValueError("Encountered inf/nan loss")
        
        _ent_loss = -entropy.mean() # Reduce spread on policy
        ent_loss = self._get_true_entropy_loss(key, _ent_loss, [indices], states)

        return ppo_loss, pol_loss, ent_loss
    
    def _get_true_entropy_loss(self, key: int, entropy_loss: float, batches: list[list[int]], observations: Tensor, raw: bool = False):
        if self.use_entropy:
            loss = entropy_loss
        else:
            loss = -self._get_stdev(key, batches, observations, raw)
        return loss
    
    def set_scores(self, returns: dict[int, float], policy_loss: dict[int, float], std: dict[int, float] | None = None,
                   mean: dict[int, float] | None = None):
        """
        Normalizes and combines each genome's return and (inverted) policy loss into a final fitness.
        No accuracy-threshold term is used here (unlike NEAT): the mean and std-dev that would normally
        make up accuracy are already reflected in the returns and in the log-prob-based policy loss.
        Optionally also folds in a (normalized, inverted) policy std-dev term, the same way NEAT's
        ``std_reg`` does; unlike ``loss_reg``, ``std_reg`` may be negative to reward exploration instead.
        Optionally also folds in a (normalized, inverted) policy-consistency term (``pol_reg``): unlike
        NEAT's accuracy (a noisy threshold on raw action error), this is an MSE between each action's old
        and new log-probability, used as a cheap stand-in for MSE-between-action-means.

        :param returns: Per-genome mean return; higher is better.
        :param policy_loss: Per-genome mean policy loss; lower is better.
        :param std: Per-genome mean action std-dev; required when ``std_reg != 0``.
        :param pol: Per-genome mean log-prob MSE; lower is better; required when ``pol_reg != 0``.
        :return: Tuple of (best genome's key, global (min, max, mean, std) fitness stats).
        """
        def sort_key(item: tuple[int, float]):
            # Sort criteria: Higher score and older keys. TODO: Might wanna change to newers keys later
            key, score = item
            return score, -key

        use_std = self.ent_reg != 0.
        if use_std:
            assert std is not None and all(key in std for key in returns.keys())
        use_mean = self.pol_reg != 0.
        if use_mean:
            assert mean is not None and all(key in mean for key in returns.keys())

        norm_returns = self.normalize_array(returns, genus_separated=True)
        # Policy loss is minimized, so its normalized value is inverted before being combined additively
        norm_loss = {k: 1 - v for k, v in self.normalize_array(policy_loss, genus_separated=True).items()}
        # Log-prob MSE is minimized too, so it's inverted the same way as policy loss.
        norm_mean = {k: 1 - v for k, v in self.normalize_array(mean, genus_separated=True).items()} if use_mean else None
        # Std-dev is inverted the same way; std_reg's own sign (not this inversion) decides whether low
        # std-dev (exploitation) or high std-dev (exploration) is ultimately rewarded.
        norm_std  = {k: 1 - v for k, v in self.normalize_array(std, genus_separated=True).items()} if use_std else None
        scores = dict(sorted(
            {
                key: (self.rew_reg * norm_returns[key]) 
                     + (self.loss_reg * norm_loss[key])
                     + (self.ent_reg * norm_std[key] if use_std else 0.0)
                     + (self.pol_reg * norm_mean[key] if use_mean else 0.0)
                for key in returns.keys()
            }.items(),
            key=sort_key, reverse=True
        ))

        global_norm_returns = self.normalize_array(returns)
        global_norm_loss = {k: 1 - v for k, v in self.normalize_array(policy_loss).items()}
        global_norm_std  = {k: 1 - v for k, v in self.normalize_array(std).items()} if use_std else None
        global_norm_mean = {k: 1 - v for k, v in self.normalize_array(mean).items()} if use_mean else None
        global_scores = dict(sorted(
            {
                key: (self.rew_reg * global_norm_returns[key]) + (self.loss_reg * global_norm_loss[key])
                     + (self.ent_reg * global_norm_std[key] if use_std else 0.0)
                     + (self.pol_reg * global_norm_mean[key] if use_mean else 0.0)
                for key in returns.keys()
            }.items(),
            key=sort_key, reverse=True
        ))

        available_keys   = list(global_scores.keys())
        available_scores = list(global_scores.values())

        criterion = self.population.config.general.fitness_criterion
        score_min, score_max = np.min(available_scores).item(), np.max(available_scores).item()
        score_min -= abs(score_min) # Since normalize should ensure values between [0, 1]
        score_max += abs(score_max)
        if score_min == 0.: score_min = -score_max
        if score_max == 0.: score_max = -score_min
        for genome in self.population.genomes.values():
            if genome.key in available_keys:
                genome.fitness = scores[genome.key]
            else:
                genome.fitness = score_min if criterion == 'max' else score_max

        if criterion == 'max':    best_genome_key = available_keys[np.argmax(available_scores)]
        elif criterion == 'min':  best_genome_key = available_keys[np.argmin(available_scores)]
        elif criterion == 'mean': best_genome_key = available_keys[np.argmin((np.mean(available_scores) - available_scores) ** 2)]
        else: raise ValueError(f"unsupported fitness criteria")

        global_fitness_stats: tuple[float, ...] = tuple([
            func(available_scores).item() for func in [np.min, np.max, np.mean, np.std]
        ])

        return best_genome_key, global_fitness_stats

    def learn(self, evaluation_function: callable, steps: int, epochs: int = None, batch_size: int = None,
              accuracy_error=0.10, accuracy_type='continuous', verbose: int = None):
        """
        Runs the full evolution loop: collect rollout data, rank genomes by return and policy loss, log
        progress and step schedulers, until ``epochs`` generations have been trained (or forever, if
        ``epochs`` is ``None``).

        :param evaluation_function: Callable that runs the population against the environment for one epoch.
        :param steps: Number of primary-buffer steps to collect per epoch.
        :param epochs: Number of epochs to train for; trains indefinitely when ``None``.
        :param batch_size: Batch size used when iterating over collected records.
        :param accuracy_error: Relative error tolerance used when computing (logging-only) policy accuracy.
        :param accuracy_type: Accuracy comparison mode; one of ``'continuous'``, ``'binary'``, ``'discrete'``.
        :param verbose: Verbosity level; ``0``/``None`` is silent, ``1`` prints the summary each epoch, ``2`` also prints timings.
        """
        print(f"Logging to {CM(self.log_path, Fore.MAGENTA)}")
        if epochs is None:
            epochs = np.inf
        epoch_done = 0
        while epoch_done < epochs:
            torch.cuda.empty_cache()

            # Running environment to collect rollout data
            ts = clock.perf_counter()
            self.steps_limit = self.steps_done + steps
            self.population.run(evaluation_function, 1, verbose=verbose, skip=True, trainer=self)
            if len(self.population.to_delete) == len(self.population.genomes):
                self.population.to_delete.clear()
            invalid_population = len(self.population.to_delete) == len(self.population.genomes)
            valid_keys = [
                key for key in self.population.genomes.keys() if key not in self.population.to_delete or invalid_population
            ] if self.validate else list(self.population.genomes.keys())
            if len(valid_keys) == 0:
                valid_keys = list(self.population.genomes.keys())
            batch_indices = self.get_batches(valid_keys, batch_size, False)
            run_time = clock.perf_counter() - ts
            if verbose and verbose >= 2:
                print(f"collected data in {CM(f'{round(run_time, 2)}s', Fore.LIGHTCYAN_EX)}")

            # Rolling out genomes that are not to be deleted
            pts = clock.perf_counter()
            with torch.no_grad():
                # Roll out data from last episodes from primary buffers
                ts = clock.perf_counter()
                states, actions, rewards, log_probs, episode_mapping = self.primary.rollout(
                    ['state', 'action', 'reward', 'log_prob', 'ep_map'], as_list=True, stack=True, keys=valid_keys
                )
                if verbose and verbose >= 2:
                    print(f"rolled out data in {CM(f'{round(clock.perf_counter() - ts, 2)}s', Fore.LIGHTCYAN_EX)}")

                # Sort episodes wrt. episode mapping
                ts = clock.perf_counter()
                episode_lengths = self.sort_episodes(episode_mapping, states, actions, rewards, log_probs)
                if verbose and verbose >= 2:
                    print(f"sorted episodic data in {CM(f'{round(clock.perf_counter() - ts, 2)}s', Fore.LIGHTCYAN_EX)}")

                # Compute current returns
                ts = clock.perf_counter()
                returns_current = self.compute_returns(rewards, self.gamma, self.kappa, 1.0, 0, False, False, episode_mapping)
                if verbose and verbose >= 2:
                    print(f"computed primary returns in {CM(f'{round(clock.perf_counter() - ts, 2)}s', Fore.LIGHTCYAN_EX)}")

                # Values / advantages: with no critic, the raw returns double as the advantage
                ts = clock.perf_counter()
                if self.use_critic:
                    values = {
                        key: torch.cat([
                            self.get_module(key).get_value(states[key][batch].unsqueeze(0).to(self.device), keys=key).squeeze(0)
                            for batch in batch_indices[key]
                        ]).cpu()
                        for key in valid_keys
                    }
                    advantages = {key: returns_current[key] - values[key] for key in valid_keys}
                    # TODO: Add stats like value accuracy and explained varaince later on for this case
                else:
                    advantages = returns_current
                if verbose and verbose >= 2:
                    print(f"computed advantages in {CM(f'{round(clock.perf_counter() - ts, 2)}s', Fore.LIGHTCYAN_EX)}")

                # Set the new episodic returns, policy loss and policy (log-prob) MSE to the secondary buffer
                ts = clock.perf_counter()
                episodic_data: dict[int, dict[int, tuple[float, float, float, float, int]]] = {
                    key: {
                        uei: (
                            torch.mean(returns_current[key][episode_indices]).cpu().item(),
                            *self._get_policy_loss(key, states[key], actions[key], log_probs[key], advantages[key], episode_indices),
                            # TODO: For critic mode add collected stats like explained_variance and value_accuracy
                            #       here is order to view their aggregated values in the logs
                            len(episode_indices),
                        )
                        for episode_indices, uei in [
                            ([idx for idx, ep_idx in enumerate(episode_mapping[key]) if ep_idx == unique_ep_idx], unique_ep_idx)
                            for unique_ep_idx in np.unique(episode_mapping[key])
                        ] # List[ListOfIndicesForEachEpisode]
                    }
                    for key in valid_keys
                } # Dict[Key, Dict[EpisodeIndex, Tuple[MeanReturn, MeanPPOLoss, MeanPolicyMSE, MeanPolicyEntropy, EpisodeLength]]]
                episode_counts = [len(er) for er in episodic_data.values()]
                if not all([l == episode_counts[0] for l in episode_counts]):
                    raise RuntimeError(f"Ensure all genomes go through the same number of episodes in the environment;"
                                       f"Got:\n {episode_counts}")
                episodes = sorted(set(sum([list(d.keys()) for d in episodic_data.values()], [])))
                for ep_idx in episodes:
                    ec_return = torch.tensor([
                        episodic_data[key][ep_idx][0] if key in valid_keys else -np.inf
                        for key in self.secondary.mapping.keys()
                    ])
                    ep_loss = torch.tensor([
                        episodic_data[key][ep_idx][1] if key in valid_keys else +np.inf
                        for key in self.secondary.mapping.keys()
                    ])
                    ep_mean = torch.tensor([
                        episodic_data[key][ep_idx][2] if key in valid_keys else +np.inf
                        for key in self.secondary.mapping.keys()
                    ])
                    ep_std = torch.tensor([
                        episodic_data[key][ep_idx][3] if key in valid_keys else +np.inf
                        for key in self.secondary.mapping.keys()
                    ])
                    ep_len = [
                        episodic_data[key][ep_idx][4] if key in valid_keys else -np.inf
                        for key in self.secondary.mapping.keys()
                    ]
                    self.secondary.update(
                        ec_return=ec_return, ep_loss=ep_loss, ep_mean=ep_mean, ep_std=ep_std,
                        ep_map=int(ep_idx), ep_len=ep_len
                    )
                if verbose and verbose >= 2:
                    print(f"updated secondary buffers in {CM(f'{round(clock.perf_counter() - ts, 2)}s', Fore.LIGHTCYAN_EX)}")

                # Clean up memory for other calculations
                ts = clock.perf_counter()
                if self.max_steps is not None: self.deque_steps_secondary(self.max_steps, None)
                if self.max_episodes is not None: self.deque_episodes_secondary(self.max_episodes, None)
                buffer_sizes_primary = self.primary.buffer_sizes()
                buffer_sizes_secondary = self.secondary.buffer_sizes()
                self.deque_steps(0, None)
                if verbose and verbose >= 2:
                    print(f"deleted data in {CM(f'{round(clock.perf_counter() - ts, 2)}s', Fore.LIGHTCYAN_EX)}")

                # Get full returns/loss, combined across every retained episode the same way (alpha/order/normalize)
                ts = clock.perf_counter()
                returns_raw, ep_loss_raw, ep_pol_raw, ep_std_raw, full_mapping, episode_lengths = self.secondary.rollout(
                    ['ec_return', 'ep_loss', 'ep_mean', 'ep_std', 'ep_map', 'ep_len'], as_list=True, stack=True, keys=valid_keys
                )
                self.sort_episodes(full_mapping, returns_raw, ep_loss_raw, ep_pol_raw, ep_std_raw)
                if verbose and verbose >= 2:
                    print(f"fetched secondary data in {CM(f'{round(clock.perf_counter() - ts, 2)}s', Fore.LIGHTCYAN_EX)}")
                ts = clock.perf_counter()
                returns = self.compute_returns(returns_raw, 0, 0, self.alpha, self.order, False, self.normalize, full_mapping)
                # TODO: Might want to add norm parameter for these secondary scores
                beta_norm = False
                ppo_loss = self.compute_returns(ep_loss_raw, 0, 0, self.beta, self.beta_order, False, beta_norm, full_mapping) # NOTE: No normalization. Values should remain fully negative
                pol_loss = self.compute_returns(ep_pol_raw, 0, 0, self.beta, self.beta_order, True, beta_norm, full_mapping)
                ent_loss = self.compute_returns(ep_std_raw, 0, 0, self.beta, self.beta_order, False,beta_norm, full_mapping)
                if verbose and verbose >= 2:
                    print(f"computed secondary returns in {CM(f'{round(clock.perf_counter() - ts, 2)}s', Fore.LIGHTCYAN_EX)}")

                def fixed_std(array: list[float] | Tensor):
                    if len(array) == 0:
                        raise ValueError("Array or Tensor is empty!")
                    elif len(array) == 1:
                        return 0.0
                    else:
                        if isinstance(array, Tensor):
                            res = torch.std(array).cpu().item()
                        else:
                            res = np.std(array).item()
                        if np.isnan(res):
                            raise ValueError(f"std_dev raised NaN from array \n'{array}'")
                        return res

                # Reward consistency: mean minus spread. Penalize loss inconsistency: mean plus spread.
                _raw_returns: dict[int, list[float]] = {
                    key: [
                        torch.mean(returns[key][indices]).item()
                        for indices in [
                            [idx for idx, ep_idx in enumerate(full_mapping[key]) if ep_idx == u_idx]
                            for u_idx in np.unique(full_mapping[key])
                        ]
                    ]
                    for key in valid_keys
                }
                _raw_losses: dict[int, list[float]] = {
                    key: [
                        torch.mean(ppo_loss[key][indices]).item()
                        for indices in [
                            [idx for idx, ep_idx in enumerate(full_mapping[key]) if ep_idx == u_idx]
                            for u_idx in np.unique(full_mapping[key])
                        ]
                    ]
                    for key in valid_keys
                }
                _raw_losses_true: dict[int, list[float]] = {
                    key: [
                        torch.mean(ep_loss_raw[key][indices]).item()
                        for indices in [
                            [idx for idx, ep_idx in enumerate(full_mapping[key]) if ep_idx == u_idx]
                            for u_idx in np.unique(full_mapping[key])
                        ]
                    ]
                    for key in valid_keys
                }
                _raw_mean: dict[int, list[float]] = {
                    key: [
                        torch.mean(pol_loss[key][indices]).item()
                        for indices in [
                            [idx for idx, ep_idx in enumerate(full_mapping[key]) if ep_idx == u_idx]
                            for u_idx in np.unique(full_mapping[key])
                        ]
                    ]
                    for key in valid_keys
                }
                _raw_mean_true: dict[int, list[float]] = {
                    key: [
                        torch.mean(ep_pol_raw[key][indices]).item()
                        for indices in [
                            [idx for idx, ep_idx in enumerate(full_mapping[key]) if ep_idx == u_idx]
                            for u_idx in np.unique(full_mapping[key])
                        ]
                    ]
                    for key in valid_keys
                }
                # Entropy-derived std-dev, already computed once (per-episode) in _get_policy_loss and stored
                # in the secondary buffer, so no re-evaluation of actions is needed here.
                _raw_std: dict[int, list[float]] = {
                    key: [
                        torch.mean(ent_loss[key][indices]).item()
                        for indices in [
                            [idx for idx, ep_idx in enumerate(full_mapping[key]) if ep_idx == u_idx]
                            for u_idx in np.unique(full_mapping[key])
                        ]
                    ]
                    for key in valid_keys
                }
                _raw_std_true: dict[int, list[float]] = {
                    key: [
                        torch.mean(ep_std_raw[key][indices]).item()
                        for indices in [
                            [idx for idx, ep_idx in enumerate(full_mapping[key]) if ep_idx == u_idx]
                            for u_idx in np.unique(full_mapping[key])
                        ]
                    ]
                    for key in valid_keys
                }
                returns_score: dict[int, float] = {
                    key: np.mean(value).item() - (self.div_reg * fixed_std(value))
                    for key, value in _raw_returns.items()
                }

                for key, value in returns_score.items(): self.population.genomes[key]._actual = value
                
                loss_score: dict[int, float] = {
                    key: np.mean(value).item() + (self.div_reg * fixed_std(value))
                    for key, value in _raw_losses.items()
                }
                loss_true: dict[int, float] = {
                    key: np.mean(value).item() # + (self.div_reg * fixed_std(value))
                    for key, value in _raw_losses_true.items()
                    # TODO: Find a way to simplifiy/combine this calculation since it's only needed for logging,
                    #       just like the rewards
                }
                # Policy (log-prob MSE) consistency, only computed when actually used by set_scores (pol_reg != 0)
                pol_mean_score: dict[int, float] | None = {
                    key: np.mean(value).item() + (self.div_reg * fixed_std(value))
                    for key, value in _raw_mean.items()
                } if self.pol_reg != 0. else None
                mean_true: dict[int, float] = {
                    key: np.mean(value).item() # + (self.div_reg * fixed_std(value))
                    for key, value in _raw_mean_true.items()
                }
                
                # Policy std-dev, only computed when actually used by set_scores (std_reg != 0)
                pol_std_score: dict[int, float] | None = {
                    key: np.mean(value).item() + (self.div_reg * fixed_std(value))
                    for key, value in _raw_std.items()
                } if self.ent_reg != 0. else None
                std_true: dict[int, float] = {
                    key: np.mean(value).item() # + (self.div_reg * fixed_std(value))
                    for key, value in _raw_std_true.items()
                } if self.ent_reg != 0. else None

            # Calculate and set scores
            with torch.no_grad():
                ts = clock.perf_counter()
                best_genome_key, fitness_stats = self.set_scores(returns_score, loss_score, pol_std_score, pol_mean_score)
                if verbose and verbose >= 2:
                    print(f"set scores in {CM(f'{round(clock.perf_counter() - ts, 2)}s', Fore.LIGHTCYAN_EX)}")

            # Post Calculations
            with torch.no_grad():
                ts = clock.perf_counter()
                ep_len_mean = np.mean(episode_lengths[best_genome_key]).item()
                ep_len_std  = np.std(episode_lengths[best_genome_key]).item()

                def get(buffer: TensorDict, episodic_mapping: dict[int, list[int]], key: int) -> list[list[float]]:
                    return [
                        [
                            value.mean().cpu().item()
                            for idx, value in enumerate(buffer[key])
                            if episodic_mapping[key][idx] == ep_idx
                        ]
                        for ep_idx in np.unique(episodic_mapping[key])
                    ]
                episode_rewards = get(rewards, episode_mapping, best_genome_key)
                cum_episode_rewards = get(returns, full_mapping, best_genome_key)
                ep_rew_hold = [np.mean(episode) for episode in episode_rewards]
                ep_rew_mean = np.mean(ep_rew_hold).item()
                ep_cum_hold = [np.mean(episode) for episode in cum_episode_rewards]
                ep_cum_rew = np.mean(ep_cum_hold).item()
                ep_rew_std = np.std(ep_cum_hold).item() if len(ep_cum_hold) > 1 else 0.0
                global_rew = np.mean([
                    np.mean([
                        np.mean(episode) for episode in get(rewards, episode_mapping, key)
                    ]).item()
                    for key in rewards.keys()
                ])
                global_cum_rew = np.mean([
                    np.mean([
                        np.mean(episode) for episode in get(returns, full_mapping, key)
                    ]).item()
                    for key in returns.keys()
                ])
                policy_loss_best = loss_true[best_genome_key]

                # Policy accuracy: logging-only (not used in fitness), computed for the best genome only
                policy_accuracy_best, _ = self.get_accuracy(
                    batch_indices, states, actions, error=accuracy_error, type=accuracy_type,
                    keys=best_genome_key, strict=True,
                )
                policy_accuracy_best = policy_accuracy_best[best_genome_key]

                # Policy (log-prob MSE) consistency: reuse the score computation when pol_reg is active,
                # otherwise compute just for the best genome so it can still be logged
                policy_mean_best = (
                    mean_true[best_genome_key] if mean_true is not None
                    else np.mean(_raw_mean[best_genome_key]).item() + (self.div_reg * fixed_std(_raw_mean[best_genome_key]))
                )

                # Policy std-dev: reuse the score computation when std_reg is active, otherwise compute
                # just for the best genome so it can still be logged
                policy_std_best = (
                    std_true[best_genome_key] if std_true is not None
                    else np.mean(_raw_std[best_genome_key]).item() + (self.div_reg * fixed_std(_raw_std[best_genome_key]))
                ) # NOTE: Distributions like Categorical/MultivariateNormal have no direct std_dev hence use entropy from distributions as std_dev

                # noinspection PyBroadException
                def get_range(key: Union[int, None]):
                    try:
                        params = []
                        if key is not None:
                            for p in self.get_module(key).neat_parameters():
                                params.append(p[key].flatten())
                        else:
                            for module in self.models:
                                for p in module.neat_parameters():
                                    params.append(p[key].flatten())
                        params = torch.cat(params)
                        return torch.mean(params).cpu().item(), torch.std(params).cpu().item(), \
                               torch.max(params).cpu().item(), torch.min(params).cpu().item(), \
                               torch.sum(torch.abs(params) <= self.population.config.genome.param_epsilon).cpu().item() / params.numel()
                    except Exception:
                        return torch.nan, torch.nan, torch.nan, torch.nan, torch.nan

                mean, std, maximum, minimum, zero_count = get_range(best_genome_key)
                processing_time = np.floor(clock.perf_counter() - pts)

                if verbose and verbose >= 2:
                    print(f"calculated stats in {CM(f'{round(clock.perf_counter() - ts, 2)}s', Fore.LIGHTCYAN_EX)}")

            # Logging
            with torch.no_grad():
                self.logging.update(
                    ep_len_mean=ep_len_mean, ep_len_std=ep_len_std, ep_rew_mean=ep_rew_mean, ep_rew_std=ep_rew_std,
                    policy_loss=policy_loss_best, policy_accuracy=policy_accuracy_best, policy_std=policy_std_best,
                    policy_mean=policy_mean_best,
                )

                survival_rate = len([key for key in valid_keys if key not in self.population.to_delete]) / len(self.population.genomes)
                best_genome = self.population.genomes[best_genome_key]
                if self.prev_valid_keys is None:
                    creep = creep_max = creep_best = np.nan
                elif len(valid_keys) <= 0:
                    creep = creep_max = creep_best = 0.0
                else:
                    creep = len([key for key in self.prev_valid_keys if key in valid_keys]) / len(valid_keys)
                    buffer_size_best = buffer_sizes_secondary[best_genome_key]
                    buffer_size_max  = max(buffer_sizes_secondary.values())
                    creep_best = len([
                        key for key in self.prev_valid_keys
                        if key in valid_keys and buffer_sizes_secondary[key] >= buffer_size_best
                    ]) / len(valid_keys)
                    creep_max = len([
                        key for key in self.prev_valid_keys
                        if key in valid_keys and buffer_sizes_secondary[key] >= buffer_size_max
                    ]) / len(valid_keys)
                self.prev_valid_keys = valid_keys

                self.tb.log(
                    self.updates_done,
                    rollout={
                        'ep_len_mean': ep_len_mean, 'ep_len_std': ep_len_std, 'ep_rew_mean': ep_rew_mean,
                        'ep_rew_std': ep_rew_std, 'ep_cum_rew': ep_cum_rew, 'global_rew': global_rew,
                        'global_cum_rew': global_cum_rew, 'buffer_size_pri': buffer_sizes_primary[best_genome_key],
                        'buffer_size_sec': buffer_sizes_secondary[best_genome_key],
                    },
                    time={'run_time': run_time, 'processing_time': processing_time},
                    policy={
                        'policy_loss': policy_loss_best, 'return_score': returns_score[best_genome_key],
                        'policy_accuracy': policy_accuracy_best, 'policy_std': policy_std_best,
                        'policy_mean': policy_mean_best,
                    },
                    module={
                        'param_mean': mean, 'param_std': std, 'param_min': minimum, 'param_max': maximum,
                        'param_zeros': zero_count,
                        **{f'global_{label}': param for param, label in zip(get_range(None), ['mean', 'std', 'max', 'min', 'zeros'])},
                    },
                    population={
                        'best_genome': best_genome.key,
                        'best_genus': best_genome.genus if len(self.population.genera) > 0 else None,
                        'survival_rate': survival_rate, 'creep_score': creep, 'creep_score_best': creep_best,
                        'creep_score_max': creep_max,
                        **{f'fitness_{label}': param for param, label in zip(fitness_stats, ['min', 'max', 'mean', 'std'])},
                    },
                    schedule=self.log_scheduler_params() if self.schedulers else None,
                )
                self.tb.flush()
            self.updates_done += 1

            # Displaying
            if verbose:
                self.reporter.report({
                    'ROLLOUT': {
                        'ep_len_mean': ep_len_mean, 'ep_len_std': ep_len_std, 
                        'ep_rew_mean': ep_rew_mean, 'ep_rew_std': ep_rew_std, 
                        'ep_cum_rew': ep_cum_rew,
                    },
                    'TIME': {
                        'epochs_done': epoch_done + 1, 'run_time': run_time, 'steps_done': self.steps_done,
                        'episodes_done': self.episodes_done,
                        'current_episodes_done': len(self.episode_lengths) if self.episode_lengths else 1,
                    },
                    'TRAINING': {
                        'updates_done': self.updates_done, 'policy_loss': policy_loss_best,
                        'return_score': returns_score[best_genome_key], 
                        'policy_accuracy': policy_accuracy_best, 'policy_std': policy_std_best,
                        'policy_mean': policy_mean_best,
                    },
                    'POPULATION': {
                        'best_genome': best_genome.key, 'best_genus': best_genome.genus,
                        'survival_rate': survival_rate, 'creep': creep, 'creep_max': creep_max,
                    },
                })
            if self._report_hook is not None:
                self._report_hook(self)

            if self.schedulers is not None:
                for s in self.schedulers:
                    s.step()

            if epoch_done == epochs - 1:
                self.population.run(
                    evaluation_function, 1, verbose=verbose, skip=True, trainer=self, terminate_skip=True
                )

            epoch_done += 1
