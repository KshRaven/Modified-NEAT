from __future__ import annotations

import warnings
from contextlib import nullcontext
from typing import Iterable, Union

import numpy as np
import torch

from ModifiedNEAT.optim.scheduler import Scheduler
from ModifiedNEAT.population import Population
from ModifiedNEAT.rl.base import Algorithm, NEATAlgoWarning
from ModifiedNEAT.util.datetime import clock
from ModifiedNEAT.util.fancy_text import CM, Fore
from ModifiedNEAT.util.qol import manage_params

__all__ = ['PPO']


class PPO(Algorithm):
    """
    Evolutionary Proximal Policy Optimization variant.

    Unlike standard PPO, genomes here are not updated through gradients: there is no clipped surrogate
    objective and no optimizer step. Instead, each genome's episodic return and its policy loss (under
    its own, NEAT-evolved policy) are used to rank genomes for the next generation, the same way NEAT
    ranks genomes by return/accuracy. ``Model.evaluate_action`` supplies the log-probabilities used to
    compute that policy loss.

    All post-rollout calculations run on NumPy arrays through Numba kernels (see ``ModifiedNEAT.rl.functional``);
    only the model passes (``evaluate_action``, ``get_value``, ``get_policy``, ``get_std``) use Torch, through
    ``Algorithm.forward_pass`` which supports ``batching='genome'`` and ``batching='record'``.
    """

    def __init__(self, population: Population, schedulers: Union[Scheduler, Iterable[Scheduler]] = None,
                 device=None, dtype=torch.float32, use_critic: bool = False, **options):
        """
        :param population: Population being trained.
        :param schedulers: Scheduler or iterable of schedulers stepped once per epoch.
        :param device: Torch device used for the model passes (default: follows the ``ModifiedNEAT.base`` device).
        :param dtype: Torch data type used for the model passes and the rollout arrays (default is torch.float32).
        :param use_critic: When True, ``Model.get_value`` is used to compute a value baseline that is
            subtracted from the returns to form advantages. When False (default), the raw returns are
            used directly as advantages.
        :param options: Additional keyword arguments for configuration.
        :keyword gamma: Discount factor applied to primary (per-step) rewards (default 0.99).
        :keyword kappa: Forward-discount factor mixed in with ``gamma`` for primary rewards (default 0.0).
        :keyword alpha: Per-episode reward multiplier used when reducing secondary returns (default 1.0).
        :keyword order: Episode ranking/ordering mode used when reducing secondary returns (default 0).
        :keyword normalize: Reward normalization mode used when reducing secondary returns (default 0).
        :keyword norm_per_genus: Take the normalization statistics between genomes of the same genus only (default True).
        :keyword rew_reg: Weight of the (normalized) return component of the fitness score (default 1.0).
        :keyword loss_reg: Weight of the (normalized, inverted) policy-loss component of the fitness score (default 0.33).
        :keyword pol_reg: Weight of the (normalized, inverted) policy-consistency component of the fitness score
            (default 0.33). Measures the MSE between each action's old and new probability (from
            ``Model.evaluate_action``) as a cheap stand-in for MSE-between-action-means.
        :keyword ent_reg: Weight of the (normalized, inverted) policy std-dev component of the fitness score
            (default 1.0). Positive values reward genomes with a lower action std-dev (exploitation); negative
            values reward genomes with a higher action std-dev (exploration).
        :keyword use_entropy: When True, the policy std-dev used by ``ent_reg`` is the action entropy returned by
            ``Model.evaluate_action``, otherwise ``Model.get_std`` is used (default True).
        :keyword div_reg: Weight applied to the std-dev of returns/loss across retained episodes (default 0.01).
        :keyword validate: Whether to restrict training to genomes not flagged for deletion (default False).
        :keyword max_steps: Maximum steps kept in the secondary buffer (default 1024).
        :keyword max_episodes: Maximum episodes kept in the secondary buffer (default None).
        :keyword batching: ``'genome'`` (genomes one by one, ``batch_size`` records per batch) or ``'record'`` (all
            genomes of a genus at once, ``record_batch_size`` records per batch, default 1).
        :keyword log_dir: Directory logs are written under.
        :keyword log_name: Name of this run's log.
        """
        for var in ['schedulers', 'device', 'dtype']:
            if var in options:
                del options[var]
        base_options = dict(options)  # what the base class received before; parsing below may consume `options`

        # Options -- defined BEFORE super().__init__ so the base class can bind schedulers to these attributes
        self.gamma: float       = manage_params(options, 'gamma', 0.99)
        self.kappa: float       = manage_params(options, 'kappa', 0.00)
        self.alpha: float       = manage_params(options, 'alpha', 1.00)
        self.order: int         = manage_params(options, 'order', 0)
        self.beta: float        = manage_params(options, 'beta', 1.00)
        self.beta_order: int    = manage_params(options, 'beta_order', 0)
        self.normalize: int     = manage_params(options, 'normalize', 0)
        self.norm_per_genus: bool = manage_params(options, 'norm_per_genus', True)
        self.rew_reg: float     = manage_params(options, 'rew_reg', 1.0)
        self.loss_reg: float    = manage_params(options, 'loss_reg', 0.33)
        self.val_reg: float     = manage_params(options, 'val_reg', 0.0)
        self.pol_reg: float     = manage_params(options, 'pol_reg', 0.33)
        self.ent_reg: float     = manage_params(options, 'ent_reg', 1.0)
        # Attention-noise pressure (plain and differential attention alike; no backprop):
        #   noise_reg weight of (1 - normalized attention noise) in the fitness (0 = off), noise_sigma = probe size,
        #   noise_log = measure/log it even when noise_reg == 0 (baseline comparisons). See Attention.set_noise_probe.
        self.noise_reg: float   = manage_params(options, 'noise_reg', 0.0)
        self.noise_sigma: float = manage_params(options, 'noise_sigma', 0.1)
        self.noise_log: bool    = manage_params(options, 'noise_log', False)
        self.attention_noise: dict[int, float] = {}
        self.div_reg: float     = manage_params(options, 'div_reg', 0.01)
        self.use_entropy: bool  = manage_params(options, 'use_entropy', True)
        self.use_critic: bool   = manage_params(options, 'use_critic', use_critic)
        self.validate: bool     = manage_params(options, 'validate', False)
        self.epsilon: float     = manage_params(options, 'epsilon', 1e-36)
        self.max_steps: Union[float, None] = manage_params(options, 'max_steps', 1024)
        self.max_episodes: Union[float, None] = manage_params(options, 'max_episodes', None)
        if self.max_steps is not None and self.max_episodes is not None:
            warnings.warn(
                category=NEATAlgoWarning,
                message=f"Not recommended to apply both max_steps and max_episodes!"
            )

        super().__init__(population, schedulers, device, dtype, **base_options)

        # Buffers
        self.primary.add_buffers('state', 'action', 'reward', 'log_prob', 'ep_map',)
        # Map each episode's return and policy/value metrics to use as scores.
        self.secondary.add_buffers('ec_return', 'ep_loss', 'ep_val', 'ep_map', 'ep_len', 'ep_mean', 'ep_std')
        self.score_idx = 0

        self.logging.add_buffers(
            'ep_len_mean', 'ep_len_std', 'ep_rew_mean', 'ep_rew_std', 'policy_loss',
            'policy_accuracy', 'policy_std', 'policy_mean', 'value_loss', 'explained_variance',
        )

        self.prev_valid_keys: list[int] = None

    def update(self, observations, actions, rewards, log_probs,
               terminated: Union[bool, list[bool]], force_stop=False, reset_mapping: bool | list[bool] = False,
               session_dim: int | None = None) -> bool:
        """
        Records one environment step for every environment instance into the primary buffer.

        :param observations: Observation array (Tensor / ndarray) for this step, genomes on dim 0.
        :param actions: Action array taken this step.
        :param rewards: Reward array received this step.
        :param log_probs: Log-probability of ``actions`` under the policy that produced them (eg. from
            ``Model.get_action``); re-evaluated later via ``Model.evaluate_action`` to compute the policy loss.
        :param terminated: Whether each environment instance terminated this step.
        :param force_stop: Force the rollout buffer to be treated as filled after this step.
        :param reset_mapping: Force a new episode mapping for each environment instance, regardless of termination.
        :param session_dim: Optional dimension holding the per-environment session; inferred as 1 when omitted.
        :return: Whether the rollout buffer has been filled (steps limit reached or ``force_stop``).
        """
        return self._record_step(
            dict(state=observations, action=actions, reward=rewards, log_prob=log_probs),
            terminated, force_stop, reset_mapping, session_dim
        )

    @staticmethod
    def _clean(metrics: dict) -> dict:
        """Drops NaN metrics so they do not clutter the dashboard / the terminal."""
        return {k: v for k, v in metrics.items()
                if not (isinstance(v, (float, np.floating)) and np.isnan(v))}

    def set_scores(self, returns: dict[int, float], policy_loss: dict[int, float], std: dict[int, float] | None = None,
                   mean: dict[int, float] | None = None, value_loss: dict[int, float] | None = None,
                   noise: dict[int, float] | None = None):
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

        use_loss = self.loss_reg != 0.
        use_std = self.ent_reg != 0.
        if use_std:
            assert std is not None and all(key in std for key in returns.keys())
        use_mean = self.pol_reg != 0.
        if use_mean:
            assert mean is not None and all(key in mean for key in returns.keys())
        use_value = self.use_critic and self.val_reg != 0.
        if use_value:
            assert value_loss is not None and all(key in value_loss for key in returns.keys())

        use_noise = self.noise_reg > 0. and bool(noise)
        if use_noise:   # genomes the probe did not reach get the mean noise (neutral)
            fill = float(np.mean(list(noise.values())))
            noise = {k: noise.get(k, fill) for k in returns.keys()}
        norm_noise = {k: 1 - v for k, v in self.normalize_array(noise, genus_separated=True).items()} if use_noise else None
        norm_returns = self.normalize_array(returns, genus_separated=True)
        # Policy loss is minimized, so its normalized value is inverted before being combined additively
        norm_loss = {k: 1 - v for k, v in self.normalize_array(policy_loss, genus_separated=True).items()} if use_loss else None
        # Log-prob MSE is minimized too, so it's inverted the same way as policy loss.
        norm_mean = {k: 1 - v for k, v in self.normalize_array(mean, genus_separated=True).items()} if use_mean else None
        norm_value = {k: 1 - v for k, v in self.normalize_array(value_loss, genus_separated=True).items()} if use_value else None
        # Std-dev is inverted the same way; std_reg's own sign (not this inversion) decides whether low
        # std-dev (exploitation) or high std-dev (exploration) is ultimately rewarded.
        norm_std  = {k: 1 - v for k, v in self.normalize_array(std, genus_separated=True).items()} if use_std else None
        scores = dict(sorted(
            {
                key: (self.rew_reg * norm_returns[key]) 
                     + (self.loss_reg * norm_loss[key] if use_loss else 0.0)
                     + (self.ent_reg * norm_std[key] if use_std else 0.0)
                     + (self.pol_reg * norm_mean[key] if use_mean else 0.0)
                     + (self.val_reg * norm_value[key] if use_value else 0.0)
                     + (self.noise_reg * norm_noise[key] if use_noise else 0.0)
                for key in returns.keys()
            }.items(),
            key=sort_key, reverse=True
        ))

        global_norm_returns = self.normalize_array(returns)
        global_norm_noise = {k: 1 - v for k, v in self.normalize_array(noise).items()} if use_noise else None
        global_norm_loss  = {k: 1 - v for k, v in self.normalize_array(policy_loss).items()} if use_loss else None
        global_norm_std   = {k: 1 - v for k, v in self.normalize_array(std).items()} if use_std else None
        global_norm_mean  = {k: 1 - v for k, v in self.normalize_array(mean).items()} if use_mean else None
        global_norm_value = {k: 1 - v for k, v in self.normalize_array(value_loss).items()} if use_value else None
        global_scores = dict(sorted(
            {
                key: (self.rew_reg * global_norm_returns[key]) + (self.loss_reg * global_norm_loss[key] if use_loss else 0.0)
                     + (self.ent_reg * global_norm_std[key] if use_std else 0.0)
                     + (self.pol_reg * global_norm_mean[key] if use_mean else 0.0)
                     + (self.val_reg * global_norm_value[key] if use_value else 0.0)
                     + (self.noise_reg * global_norm_noise[key] if use_noise else 0.0)
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
              accuracy_error=0.10, accuracy_type='continuous', verbose: int = None,
              batching: str | None = None, record_batch_size: int | None = None):
        """
        Runs the full evolution loop: collect rollout data, rank genomes by return and policy loss, log
        progress and step schedulers, until ``epochs`` generations have been trained (or forever, if
        ``epochs`` is ``None``).

        :param evaluation_function: Callable that runs the population against the environment for one epoch.
        :param steps: Number of primary-buffer steps to collect per epoch.
        :param epochs: Number of epochs to train for; trains indefinitely when ``None``.
        :param batch_size: Records per batch when genomes are processed one by one (``batching='genome'``).
        :param accuracy_error: Relative error tolerance used when computing (logging-only) policy accuracy.
        :param accuracy_type: Accuracy comparison mode; one of ``'continuous'``, ``'binary'``, ``'discrete'``.
        :param verbose: Verbosity level; ``0``/``None`` is silent, ``1`` prints the summary each epoch, ``2`` also prints timings.
        :param batching: ``'genome'``: all genomes one by one with their own batches of ``batch_size`` records.
            ``'record'``: all genomes of a genus at once, ``record_batch_size`` records per batch (smaller values
            use less memory with many genomes). Defaults to the trainer's setting (``'genome'``).
        :param record_batch_size: Records per batch for ``batching='record'`` (default 1).
        """
        print(f"Logging to {CM(self.log_path, Fore.MAGENTA)}")
        if batch_size is not None: self.batch_size = batch_size
        if batching is not None: self.batching = batching
        if record_batch_size is not None: self.record_batch_size = record_batch_size
        if epochs is None:
            epochs = np.inf
        epoch_done = 0
        while epoch_done < epochs:
            torch.cuda.empty_cache()

            # Running environment to collect rollout data
            valid_keys, run_time = self._collect_rollout(evaluation_function, steps, verbose)
            prim, sec = self.primary, self.secondary
            pts = ts = clock.perf_counter()

            # Sort the rollout by episode and find the episode segments of every genome
            rows, runs = self.prepare_rollout(prim, valid_keys)
            lengths, ep = prim.lengths, prim.data['ep_map']
            kept_lengths = lengths.copy()  # the primary buffer is emptied (lengths -> 0) before the best genome is known
            n_eps = int(runs.counts[rows[0]])
            G = len(lengths)
            if verbose and verbose >= 2:
                print(f"Valid keys: {len(valid_keys)}, records: {prim.max_size()}, episodes: {n_eps}")
                print(f"states = {prim.data['state'].shape[2:]}, actions = {prim.data['action'].shape[2:]}")
            ts = self._tick(verbose, "sorted episodic data", ts)

            # Current returns; with no critic the raw returns double as the advantage
            returns_current = self.compute_returns(
                prim.data['reward'], ep, lengths, rows, self.gamma, self.kappa, 1.0, 0, False, False
            )
            ep_rewards = self.episode_means(prim.data['reward'], lengths, rows, runs)  # (genomes, episodes)
            ts = self._tick(verbose, "computed primary returns", ts)

            values = None
            if self.use_critic:
                values = self.forward_pass('get_value', ['state'], valid_keys, strict=True)[0]
                advantages = returns_current - values
            else:
                advantages = returns_current
            ts = self._tick(verbose, "computed advantages", ts)

            # Terms whose coefficient is 0 do not influence fitness: skip them for the population and only
            # evaluate them for the best genome afterwards (logging only).
            use_loss = self.loss_reg != 0.
            use_pol = self.pol_reg != 0.
            use_ent = self.ent_reg != 0.
            use_val = self.use_critic and self.val_reg != 0.
            need_eval = use_loss or use_pol or (use_ent and self.use_entropy)
            need_std = use_ent and not self.use_entropy
            zeros = np.zeros((G, prim.max_size()), np.float64)
            ppo_rec = sq_rec = ent_rec = zeros
            measure_noise = self.noise_reg > 0 or self.noise_log
            if need_eval:
                # the full-sequence pass also carries the attention-noise probe (no extra pass)
                with (self.noise_probe(valid_keys) if measure_noise else nullcontext()):
                    new_log_prob, entropy = self.forward_pass(
                        'evaluate_action', ['state', 'action'], valid_keys, strict=True
                    )[:2]
                ppo_rec, sq_rec, ent_rec = self.ops.ppo_terms(
                    new_log_prob, prim.data['log_prob'], advantages, entropy, lengths, rows
                )
                if not np.isfinite(ppo_rec[rows]).all():
                    raise ValueError(f"found NaN/Inf in PPO loss")
            elif measure_noise:
                with self.noise_probe(valid_keys):
                    self.forward_pass('get_policy', ['state'], valid_keys)
            if need_std:  # std-dev from the module instead of the entropy
                std_out = self.forward_pass('get_std', ['state'], valid_keys, strict=True)[0]
                ent_rec = -self.ops.record_mean(std_out, lengths, rows)
            val_rec = self.ops.value_terms(values, returns_current, lengths, rows, self.epsilon) if use_val else zeros
            ep_return = self.ops.segment_mean(self.ops.record_mean(returns_current, lengths, rows), runs, rows)
            if need_eval or need_std or use_val:
                ep_loss = -self.ops.segment_mean(ppo_rec, runs, rows)
                # Policy consistency: MSE between old and new probabilities (cheap stand-in for MSE-between-means)
                ep_pol = np.log10(self.ops.segment_mean(sq_rec, runs, rows) + self.epsilon)
                ep_ent = self.ops.segment_mean(ent_rec, runs, rows)
                ep_val = self.ops.segment_mean(val_rec, runs, rows)
            else:
                ep_loss = ep_pol = ep_ent = ep_val = np.zeros((G, n_eps), np.float64)
            if need_eval and not (np.isfinite(ep_pol[rows, :n_eps]).all()):
                raise ValueError("Encountered inf/nan loss")
            ts = self._tick(verbose, "computed policy terms", ts)

            # Set the new episodic returns, policy loss and policy (probability) MSE to the secondary buffer
            def full(fill, matrix):
                block = np.full((G, n_eps), fill, np.float64)
                block[rows] = matrix[rows, :n_eps]
                return block
            ep_ids = np.broadcast_to(runs.ids[rows[0], :n_eps], (G, n_eps)).astype(np.int64)
            sec.append_block(
                ec_return=full(-np.inf, ep_return), ep_loss=full(+np.inf, ep_loss), ep_val=full(+np.inf, ep_val),
                ep_mean=full(+np.inf, ep_pol), ep_std=full(+np.inf, ep_ent), ep_map=ep_ids,
                ep_len=full(-np.inf, runs.lengths),
            )
            ts = self._tick(verbose, "updated secondary buffers", ts)

            # Clean up memory for other calculations
            if self.max_steps is not None: self.deque_steps_secondary(self.max_steps, None)
            if self.max_episodes is not None: self.deque_episodes_secondary(self.max_episodes, None)
            buffer_sizes_primary = prim.buffer_sizes()
            buffer_sizes_secondary = sec.buffer_sizes()
            self.deque_steps(0, None)  # only resets lengths; the arrays stay valid through ``kept_lengths``
            ts = self._tick(verbose, "deleted data", ts)

            # Full returns/loss, combined across every retained episode the same way (alpha/order/normalize)
            srows, sruns = self.prepare_rollout(sec, valid_keys, check=False)
            sl, sep = sec.lengths, sec.data['ep_map']
            returns = self.compute_returns(sec.data['ec_return'], sep, sl, srows, 0, 0, self.alpha, self.order, False, self.normalize)
            # NOTE: No normalization of the secondary scores. Values should remain fully negative
            ppo_loss = None if not use_loss else self.compute_returns(sec.data['ep_loss'], sep, sl, srows, 0, 0, self.beta, self.beta_order, False, False)
            pol_loss = None if not use_pol else self.compute_returns(sec.data['ep_mean'], sep, sl, srows, 0, 0, self.beta, self.beta_order, True, False)  # Inverted: limits [0, +inf]
            value_loss = None if not use_val else self.compute_returns(sec.data['ep_val'], sep, sl, srows, 0, 0, self.beta, self.beta_order, True, False)
            ent_loss = None if not use_ent else self.compute_returns(sec.data['ep_std'], sep, sl, srows, 0, 0, self.beta, self.beta_order, False, False)
            ts = self._tick(verbose, "computed secondary returns", ts)

            def stat(array: np.ndarray):
                """(mean, std) over the retained episodes of every genome."""
                return self.row_stats(self.episode_means(array, sl, srows, sruns), sruns.counts, srows, 0)

            # Reward consistency: mean minus spread. Penalize loss inconsistency: mean plus spread.
            ret_m, ret_s = stat(returns)
            returns_score = self.as_dict(ret_m - self.div_reg * ret_s, valid_keys, srows)
            for key, value in returns_score.items(): self.population.genomes[key]._actual = value

            keys_zero = {key: 0.0 for key in returns_score}
            loss_score = (lambda ms: self.as_dict(ms[0] + self.div_reg * ms[1], valid_keys, srows))(stat(ppo_loss)) if use_loss else keys_zero
            loss_true = self.as_dict(stat(sec.data['ep_loss'])[0], valid_keys, srows) if use_loss else dict(keys_zero)

            if use_pol:
                pm, ps = stat(pol_loss)
                pol_mean_score = self.as_dict(pm + self.div_reg * ps, valid_keys, srows)
                mean_true = self.as_dict(stat(sec.data['ep_mean'])[0], valid_keys, srows)
            else:
                pol_mean_score, mean_true = None, dict(keys_zero)

            if use_ent:
                em, es = stat(ent_loss)
                pol_std_score = self.as_dict(em + self.div_reg * es, valid_keys, srows)
                std_true = self.as_dict(stat(sec.data['ep_std'])[0], valid_keys, srows)
            else:
                pol_std_score, std_true = None, dict(keys_zero)

            if use_val:
                vm, vs = stat(value_loss)
                value_loss_score = self.as_dict(vm + self.div_reg * vs, valid_keys, srows)
                value_loss_true = self.as_dict(stat(sec.data['ep_val'])[0], valid_keys, srows)
            else:
                value_loss_score, value_loss_true = None, dict(keys_zero)

            # Calculate and set scores
            best_genome_key, fitness_stats = self.set_scores(
                returns_score, loss_score, pol_std_score, pol_mean_score, value_loss_score,
                self.attention_noise if self.noise_reg > 0 else None
            )
            ts = self._tick(verbose, "set scores", ts)

            # Post Calculations
            best_row        = sec.mapping[best_genome_key]
            ep_lens         = sec.data['ep_len'][best_row, :sl[best_row]]
            ep_len_mean     = np.mean(ep_lens).item()
            ep_len_std      = np.std(ep_lens).item()
            ep_cum          = self.episode_means(returns, sl, srows, sruns)
            best_rew        = ep_rewards[best_row, :n_eps]
            ep_rew_mean     = np.mean(best_rew).item()
            best_cum        = ep_cum[best_row, :sruns.counts[best_row]]
            ep_cum_rew      = np.mean(best_cum).item()
            ep_rew_std      = np.std(best_cum).item() if len(best_cum) > 1 else 0.0
            global_rew      = np.mean(np.nanmean(ep_rewards[rows], axis=1)).item()
            global_cum_rew  = np.mean(np.nanmean(ep_cum[srows], axis=1)).item()
            policy_loss_best = loss_true[best_genome_key]
            policy_mean_best = mean_true[best_genome_key]
            policy_std_best = std_true[best_genome_key]  # NOTE: Categorical/MultivariateNormal have no direct std_dev, so entropy stands in
            value_loss_best = value_loss_true[best_genome_key]

            # Best-genome-only evaluation of the terms skipped for the population (logging only)
            if not need_eval:
                bk = best_genome_key
                br = np.asarray([prim.mapping[bk]], np.int64)
                r0 = int(br[0])
                nlp, ent_b = self.forward_pass('evaluate_action', ['state', 'action'], [bk], kept_lengths, strict=True)[:2]
                pr, sq, en = self.ops.ppo_terms(nlp, prim.data['log_prob'], advantages, ent_b, kept_lengths, br)
                policy_loss_best = float(-np.nanmean(self.ops.segment_mean(pr, runs, br)[r0, :n_eps]))
                policy_mean_best = float(np.log10(np.nanmean(self.ops.segment_mean(sq, runs, br)[r0, :n_eps]) + self.epsilon))
                policy_std_best = float(np.nanmean(self.ops.segment_mean(en, runs, br)[r0, :n_eps]))
                if not self.use_entropy:
                    std_b = self.forward_pass('get_std', ['state'], [bk], kept_lengths, strict=True)[0]
                    policy_std_best = float(-np.mean(self.ops.record_mean(std_b, kept_lengths, br)[r0, :int(kept_lengths[r0])]))

            # Policy / value accuracy: logging-only (not used in fitness), computed for the best genome only
            best_prim_row = prim.mapping[best_genome_key]
            best_rows = np.asarray([best_prim_row], np.int64)
            policy_accuracy_best = value_accuracy_best = explained_variance_best = np.nan
            policy = self.forward_pass('get_policy', ['state'], [best_genome_key], kept_lengths)
            if policy is not None:
                accuracy = self.ops.accuracy(policy[0], prim.data['action'], kept_lengths, best_rows, accuracy_error, accuracy_type)
                policy_accuracy_best = float(np.mean(accuracy[best_prim_row, :kept_lengths[best_prim_row]]))
            if self.use_critic:
                n = int(kept_lengths[best_prim_row])
                target = returns_current[best_prim_row, :n].reshape(n, -1).mean(axis=1)
                prediction = values[best_prim_row, :n].reshape(n, -1).mean(axis=1)
                target_variance = np.var(target)
                if target_variance > self.epsilon:
                    explained_variance_best = (1.0 - np.var(target - prediction) / target_variance).item()
                if self.val_reg != 0 and values.shape[2:] == returns_current.shape[2:]:
                    v_acc = self.ops.accuracy(values, returns_current, kept_lengths, best_rows, accuracy_error, 'continuous')
                    value_accuracy_best = float(np.mean(v_acc[best_prim_row, :n]))

            mean, std, maximum, minimum, zero_count = self._param_stats(best_genome_key)
            processing_time = np.floor(clock.perf_counter() - pts)
            self._tick(verbose, "calculated stats", ts)

            # Logging
            self.logging.update(
                ep_len_mean=ep_len_mean, ep_len_std=ep_len_std, ep_rew_mean=ep_rew_mean, ep_rew_std=ep_rew_std,
                policy_loss=policy_loss_best, policy_accuracy=policy_accuracy_best, policy_std=policy_std_best,
                policy_mean=policy_mean_best, value_loss=value_loss_best, explained_variance=explained_variance_best,
            )

            survival_rate = len([key for key in valid_keys if key not in self.population.to_delete]) / len(self.population.genomes)
            best_genome = self.population.genomes[best_genome_key]
            creep, creep_best, creep_max = self._creep(valid_keys, best_genome_key, buffer_sizes_secondary, verbose)

            external_report = self._write_logs(
                self.updates_done,
                rollout={
                    'ep_len_mean': ep_len_mean, 'ep_len_std': ep_len_std, 'ep_rew_mean': ep_rew_mean,
                    'ep_rew_std': ep_rew_std, 'ep_cum_rew': ep_cum_rew, 'global_rew': global_rew,
                    'global_cum_rew': global_cum_rew, 'buffer_size_pri': buffer_sizes_primary[best_genome_key],
                    'buffer_size_sec': buffer_sizes_secondary[best_genome_key],
                },
                time={'run_time': run_time, 'processing_time': processing_time},
                policy=self._clean({
                    'policy_loss': policy_loss_best, 'return_score': returns_score[best_genome_key],
                    'policy_accuracy': policy_accuracy_best, 'policy_std': policy_std_best,
                    'policy_mean': policy_mean_best,
                    **({'value_loss': value_loss_best, 'value_accuracy': value_accuracy_best,
                        'explained_variance': explained_variance_best} if self.use_critic else {}),
                }),
                module={
                    'param_mean': mean, 'param_std': std, 'param_min': minimum, 'param_max': maximum,
                    'param_zeros': zero_count,
                    **{f'global_{label}': param for param, label in zip(self._param_stats(None), ['mean', 'std', 'max', 'min', 'zeros'])},
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
            self.updates_done += 1

            # Displaying
            if verbose:
                self.reporter.report({
                    'ROLLOUT': {
                        'ep_len_mean': ep_len_mean, 'ep_len_std': ep_len_std,
                        'ep_rew_mean': ep_rew_mean, 'ep_rew_std': ep_rew_std, 'ep_cum_rew': ep_cum_rew,
                    },
                    'TIME': {
                        'epochs_done': epoch_done + 1, 'run_time': run_time, 'steps_done': self.steps_done,
                        'episodes_done': self.episodes_done,
                        'current_episodes_done': len(self.episode_lengths) if self.episode_lengths else 1,
                    },
                    'TRAINING': self._clean({
                        'updates_done': self.updates_done, 'policy_loss': policy_loss_best,
                        'return_score': returns_score[best_genome_key],
                        'policy_accuracy': policy_accuracy_best, 'policy_std': policy_std_best,
                        'policy_mean': policy_mean_best,
                        **({'value_loss': value_loss_best, 'explained_variance': explained_variance_best}
                           if self.use_critic else {}),
                    }),
                    'POPULATION': {
                        'best_genome': best_genome.key, 'best_genus': best_genome.genus,
                        'survival_rate': survival_rate, 'creep': creep, 'creep_max': creep_max,
                    },
                    **external_report,
                })

            self._finish_epoch(evaluation_function, epoch_done, epochs, verbose)
            epoch_done += 1
