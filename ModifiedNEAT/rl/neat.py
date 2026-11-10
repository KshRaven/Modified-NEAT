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

__all__ = ['NEAT', 'Ghost']


class NEAT(Algorithm):
    """
    Neuro-Evolution of Augmenting Topologies (NEAT) algorithm.

    Also doubles as the "Ghost Policy" variant: passing a ``ghosts`` mapping activates an extra
    regularization term that pulls a genus' policy towards another (source) genus' policy. When
    ``ghosts`` is left as ``None`` every ghost-only computation is skipped entirely.

    All post-rollout calculations run on NumPy arrays through Numba kernels (see ``ModifiedNEAT.rl.functional``);
    only the model passes use Torch.
    """

    def __init__(self, population: Population, schedulers: Union[Scheduler, Iterable[Scheduler]] = None,
                 device=None, dtype=torch.float32,
                 ghosts: dict[int, int] | None = None, **options):
        """
        :param population: Population being trained.
        :param schedulers: Scheduler or iterable of schedulers stepped once per epoch.
        :param device: Torch device the models run on; defaults to the device ``ModifiedNEAT.base`` uses.
        :param dtype: Data type used for the model inputs / stored records (default is torch.float32).
        :param ghosts: Optional map of {genus: source_genus}. When provided, enables the Ghost Policy
            regularization that penalizes a genus' policy mean for straying from its source genus'.
            Requires a Population with multiple genera and a Model exposing ``get_mean``/``get_mean_std``.
        :param options: Additional keyword arguments for configuration.
        :keyword batching: ``'genome'`` (default; genomes one by one, ``batch_size`` records per batch) or ``'record'``
            (every genome of a genus at once, ``record_batch_size`` records per batch).
        :keyword batch_size: Records per batch in ``'genome'`` batching (default 64).
        :keyword record_batch_size: Records per batch in ``'record'`` batching (default 1).
        :keyword gamma: Discount factor applied to primary (per-step) rewards (default 0.97).
        :keyword kappa: Forward-discount factor mixed in with ``gamma`` for primary rewards (default 0.0).
        :keyword alpha: Per-episode reward multiplier used when reducing secondary returns (default 1.0).
        :keyword order: Episode ranking/ordering mode used when reducing secondary returns (default 0).
        :keyword normalize: Reward normalization mode used when reducing secondary returns (default 0).
        :keyword norm_per_genus: Take the normalization statistics between genomes of the same genus only (default True).
        :keyword epsilon: Small value used for numerical stability (default 1e-12).
        :keyword rew_reg: Weight of the reward score component in the final fitness (default 1.0).
        :keyword cpy_reg: (Ghost only) Weight of the ghost-reduction penalty in the final fitness (default 1.0).
        :keyword pol_reg: Weight of the policy-accuracy component in the final fitness (default 0.0).
        :keyword std_reg: Weight of the policy-stddev component in the final fitness (default 0.2).
        :keyword div_reg: Weight subtracted for the std-dev of returns/accuracy/reduction across episodes (default 0.1).
        :keyword beta: (Ghost only) Reward multiplier applied to genera with a ghost source (default 1.0).
        :keyword beta_order: (Ghost only) Episode ranking/ordering mode used when reducing the ghost reduction (default 0).
        :keyword checks: (Ghost only) Number of top genomes per genus used as the ghost source (default 3).
        :keyword validate: Whether to restrict training to genomes not flagged for deletion (default False).
        :keyword segr_size: Optional cluster size used to segregate policy-accuracy normalization.
        :keyword target_kl: Optional target KL-divergence (currently unused, reserved).
        :keyword max_steps: Maximum steps kept in the secondary buffer (default 1024).
        :keyword max_episodes: Maximum episodes kept in the secondary buffer (default None).
        :keyword log_dir: Directory logs are written under.
        :keyword log_name: Name of this run's log.
        """
        if ghosts is not None:
            if population.genera is None or len(population.genera) <= 1:
                raise RuntimeError(f"Ghost Policy requires a Population with multiple genera; found {population.genera}")
            required_methods = ('get_mean', 'get_mean_std')
            for m in population.modules.values():
                if any(hasattr(m, meth) for meth in required_methods): continue
                raise NotImplementedError(f"Model must have any of the following fetch policy methods: {required_methods}")

        for var in ['schedulers', 'device', 'dtype']:
            if var in options:
                del options[var]
        super().__init__(population, schedulers, device, dtype, **options)

        # Base
        self.ghosts = ghosts

        # Buffers
        self.primary.add_buffers('state', 'action', 'reward', 'ep_map',)
        # Map each episodes' mean cumulative reward, mean policy accuracy and (when Ghost is active) mean
        # ghost reduction to use as score
        self.secondary.add_buffers('ec_reward', 'ep_accuracy', 'ep_reduction', 'ep_stdev', 'ep_map', 'ep_len',)
        self.score_idx = 0

        # Options
        self.gamma: float       = manage_params(options, 'gamma', 0.97)
        self.kappa: float       = manage_params(options, 'kappa', 0.00)
        self.alpha: float       = manage_params(options, 'alpha', 1.00)
        self.order: int         = manage_params(options, 'order', 0)
        self.normalize: int     = manage_params(options, 'normalize', 0)
        self.norm_per_genus: bool = manage_params(options, 'norm_per_genus', True)
        self.epsilon: float     = manage_params(options, 'epsilon', 1e-12)
        self.rew_reg: float     = manage_params(options, 'rew_reg', 1.0)
        self.pol_reg: float     = manage_params(options, 'pol_reg', 0.0)
        self.std_reg: float     = manage_params(options, 'std_reg', 0.2)
        # Attention-noise pressure (works for plain and differential attention alike, no backprop needed):
        #   noise_reg   weight of (1 - normalized attention noise) in the fitness; 0 = off (default)
        #   noise_sigma relative input perturbation of the probe (see Attention.set_noise_probe)
        #   noise_log   measure + keep ``attention_noise`` for logging even when noise_reg == 0 (baseline comparisons)
        self.noise_reg: float   = manage_params(options, 'noise_reg', 0.0)
        self.noise_sigma: float = manage_params(options, 'noise_sigma', 0.1)
        self.noise_log: bool    = manage_params(options, 'noise_log', False)
        self.attention_noise: dict[int, float] = {}
        self.div_reg: float     = manage_params(options, 'div_reg', 0.1)
        self.validate: bool     = manage_params(options, 'validate', False)
        self.segr_size: Union[float, None] = manage_params(options, 'segr_size', None)
        self.target_kl: Union[float, None] = manage_params(options, 'target_kl', None)
        self.max_steps: Union[float, None] = manage_params(options, 'max_steps', 1024)
        self.max_episodes: Union[float, None] = manage_params(options, 'max_episodes', None)
        if self.max_steps is not None and self.max_episodes is not None:
            warnings.warn(
                category=NEATAlgoWarning,
                message=f"Not recommended to apply both max_steps and max_episodes!"
            )

        # Ghost-only options; only parsed (and only ever used) when a ghost mapping is supplied
        if self.ghosts is not None:
            self.cpy_reg: float     = manage_params(options, 'cpy_reg', 1.0)
            self.beta: float        = manage_params(options, 'beta', 1.00)
            self.beta_order: int    = manage_params(options, 'beta_order', 0)
            self.checks: int        = manage_params(options, 'checks', 3)
        else:
            self.cpy_reg = self.beta = self.beta_order = self.checks = None

        self.logging.add_buffers(
            'kl_divergence', 'std', # 'explained_variance'
            'ep_len_mean', 'ep_len_std', 'ep_rew_mean', 'ep_rew_std', 'policy_acc', 'reward_acc'
        )

        self.prev_valid_keys: list[int] = None

    def update(self, observations, actions, rewards,
               terminated: Union[bool, list[bool]], force_stop=False, reset_mapping: bool | list[bool] = False,
               session_dim: int | None = None) -> bool:
        """
        Records one environment step for every environment instance into the primary buffer.

        :param observations: Observation array (Tensor / ndarray) for this step, genomes on dim 0.
        :param actions: Action array taken this step.
        :param rewards: Reward array received this step.
        :param terminated: Whether each environment instance terminated this step.
        :param force_stop: Force the rollout buffer to be treated as filled after this step.
        :param reset_mapping: Force a new episode mapping for each environment instance, regardless of termination.
        :param session_dim: Optional dimension holding the per-environment session; inferred as 1 when omitted.
        :return: Whether the rollout buffer has been filled (steps limit reached or ``force_stop``).
        """
        return self._record_step(
            dict(state=observations, action=actions, reward=rewards),
            terminated, force_stop, reset_mapping, session_dim
        )

    def _ghost_sources(self, valid_keys: list[int], rows: np.ndarray, ep_returns: np.ndarray, counts: np.ndarray,
                       means: np.ndarray):
        """
        (Ghost only) Picks the best ``checks`` genomes of every genus by mean episodic return and maps every genome
        to the rows of its genus' ghost source.

        :return: ``(src_rows, use_src)``: (genomes, checks) source rows (-1 = none) and whether a source exists AND
            has a non-zero mean policy (a zero source is treated as "no source", like before).
        """
        mapping = self.primary.mapping
        rows_total = len(self.primary.lengths)
        returns_mean = {
            key: float(np.mean(ep_returns[mapping[key], :counts[mapping[key]]])) for key in valid_keys
        }
        best_rows: dict[int, np.ndarray] = {}
        for genus in self.population.genera:
            genus_keys = [k for k in valid_keys if self._genus_of(k) == genus]
            if not genus_keys:
                continue
            scores = [returns_mean[k] for k in genus_keys]
            order = np.argsort(scores)[::-1]  # best first
            best = [mapping[genus_keys[i]] for i in order[:min(self.checks, len(genus_keys))]]
            best_rows[genus] = np.asarray(best, np.int64)

        src_rows = np.full((rows_total, self.checks), -1, np.int64)
        use_src = np.zeros(rows_total, np.bool_)
        non_zero: dict[int, bool] = {}
        lengths = self.primary.lengths
        for key in valid_keys:
            ghost_genus = self.ghosts.get(self._genus_of(key), None)
            if ghost_genus is None or ghost_genus not in best_rows:
                continue
            if ghost_genus not in non_zero:
                src = best_rows[ghost_genus]
                non_zero[ghost_genus] = bool(means[src, :int(lengths[src].max())].mean() != 0)
            if non_zero[ghost_genus]:
                row = mapping[key]
                src = best_rows[ghost_genus]
                src_rows[row, :len(src)] = src
                use_src[row] = True
        return src_rows, use_src

    @staticmethod
    def _aggr_scores(keys: Iterable[int], buffers: Iterable[dict[int, float]], coefficients: Iterable[float], aggr: str = 'sum'):
        """
        Combines several per-genome score dicts into one, weighted by ``coefficients``, and sorts the
        result by descending score (oldest genome key wins ties).

        :param keys: Genome keys to combine scores for.
        :param buffers: Per-genome score dicts to combine, one per coefficient.
        :param coefficients: Weight applied to each buffer, in the same order.
        :param aggr: Aggregation mode; ``'sum'`` for a weighted sum, ``'prod'`` for a weighted product.
        :return: Dict of combined scores, sorted by descending score.
        """
        assert len(buffers) == len(coefficients) and len(buffers) > 0
        zipped = list(zip(buffers, coefficients))
        scores = {k: (0.0 if aggr == 'sum' else 1.0) for k in keys}
        for k in keys:
            for buffer, coeff in zipped:
                if aggr == 'sum':    scores[k] += coeff * buffer[k]
                elif aggr == 'prod': scores[k] *= coeff * buffer[k]
                else: raise ValueError(f"Invalid aggregation type '{aggr}'")
        # Sort criteria: Descending, High score (+inf), Old genome key (-inf)
        return dict(sorted(scores.items(), key=lambda item: (item[1], -item[0]), reverse=True))

    def set_scores(self, scores: dict[int, float], policy: dict[int, float] | None, stdev: dict[int, float] | None,
                   reduction: dict[int, float] | None = None, noise: dict[int, float] | None = None):
        """
        Normalizes and combines the score components into a final fitness per genome, then assigns it.

        :param scores: Per-genome reward score.
        :param policy: Per-genome policy accuracy; required when ``pol_reg > 0``.
        :param stdev: Per-genome normalized policy stddev; required when ``pol_reg > 0``.
        :param reduction: (Ghost only) Per-genome normalized ghost reduction; required when Ghost is active.
        :param noise: Per-genome attention noise (lower is better); used when ``noise_reg > 0``.
        :return: Tuple of (best genome's key, global (min, max, mean, std) fitness stats).
        """
        def sort_key(item: tuple[int, float]):
            key, score = item
            return score, -key

        keys = list(scores.keys())

        scores = dict(sorted(self.normalize_array(scores, genus_separated=True).items(), key=sort_key, reverse=True))
        global_scores = dict(sorted(self.normalize_array(scores).items(), key=sort_key, reverse=True))

        # Only run the weighted aggregation when there is more than the plain reward score to combine; this
        # keeps the un-regularized default behaviour (rew_reg is otherwise applied as a no-op multiplier).
        use_noise = self.noise_reg > 0. and bool(noise)
        if use_noise:   # genomes the probe did not reach get the mean noise (neutral)
            fill = float(np.mean(list(noise.values())))
            noise = {k: noise.get(k, fill) for k in keys}
        if self.pol_reg > 0. or self.ghosts is not None or use_noise:
            local_components  = [(scores, self.rew_reg)]
            global_components = [(global_scores, self.rew_reg)]

            if self.ghosts is not None:
                assert reduction is not None
                # NOTE: The ghost reduction penalty only ever affects the per-genus 'scores' used to set
                # genome.fitness; it's intentionally excluded from 'global_scores', which is used purely to
                # pick and report the best genome/fitness stats without the ghost regularization mixed in.
                local_components.append((reduction, self.cpy_reg))

            if self.pol_reg > 0.:
                assert policy is not None and stdev is not None
                assert all([key in policy for key in scores.keys()])
                policy = self.normalize_array(policy, None, self.segr_size, scores, genus_separated=True)
                global_policy = self.normalize_array(policy, None, self.segr_size, scores)
                global_stdev  = self.normalize_array(stdev)
                local_components  += [(policy, self.pol_reg), (stdev, self.std_reg)]
                global_components += [(global_policy, self.pol_reg), (global_stdev, self.std_reg)]

            if use_noise:
                local_components.append(
                    ({k: 1 - v for k, v in self.normalize_array(noise, genus_separated=True).items()}, self.noise_reg))
                global_components.append(
                    ({k: 1 - v for k, v in self.normalize_array(noise).items()}, self.noise_reg))

            scores = self._aggr_scores(keys, *zip(*local_components))
            global_scores = self._aggr_scores(keys, *zip(*global_components))

        available_keys = list(global_scores.keys())
        available_scores = list(global_scores.values())

        criterion = self.population.config.general.fitness_criterion
        score_min, score_max = np.min(available_scores).item(), np.max(available_scores).item()
        score_min -= abs(score_min)
        score_max += abs(score_max)
        if score_min == 0.: score_min = -score_max
        if score_max == 0.: score_max = -score_min
        reg_sum = self.rew_reg + self.pol_reg + (self.std_reg if self.pol_reg != 0 else 0) + (self.cpy_reg if self.ghosts is not None else 0) + (self.noise_reg if use_noise else 0)
        for genome in self.population.genomes.values():
            if genome.key in available_keys:
                genome.fitness = scores[genome.key]
            else:
                genome.fitness = (
                    (score_min if criterion == 'max' else score_max)
                    if self.pol_reg <= 0.0 else
                    (-reg_sum if criterion == 'max' else reg_sum * 2)
                )

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
        Runs the full evolution loop: collect rollout data, score genomes, log progress and step schedulers,
        until ``epochs`` generations have been trained (or forever, if ``epochs`` is ``None``).

        :param evaluation_function: Callable that runs the population against the environment for one epoch.
        :param steps: Number of primary-buffer steps to collect per epoch.
        :param epochs: Number of epochs to train for; trains indefinitely when ``None``.
        :param batch_size: Records per batch when genomes are processed one by one (``batching='genome'``).
        :param accuracy_error: Relative error tolerance used when computing policy accuracy.
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
            n_eps = int(runs.counts[rows[0]])
            if verbose and verbose >= 2:
                print(f"Valid keys: {len(valid_keys)}, records: {prim.max_size()}, episodes: {n_eps}")
                print(f"states = {prim.data['state'].shape[2:]}, actions = {prim.data['action'].shape[2:]}")
            ts = self._tick(verbose, "sorted episodic data", ts)

            # Compute current returns
            returns_current = self.compute_returns(
                prim.data['reward'], ep, lengths, rows, self.gamma, self.kappa, 1.0, 0, False, False
            )
            ep_returns = self.episode_means(returns_current, lengths, rows, runs)  # (genomes, episodes)
            ep_rewards = self.episode_means(prim.data['reward'], lengths, rows, runs)
            ts = self._tick(verbose, "computed primary returns", ts)

            # Model passes (one pass per method over every record, reduced per episode afterwards)
            ep_accuracy = np.ones_like(ep_returns)
            measure_noise = self.noise_reg > 0 or self.noise_log
            if self.pol_reg != 0 or measure_noise:
                # one full-sequence pass serves both the policy accuracy and the attention-noise probe
                with (self.noise_probe(valid_keys) if measure_noise else nullcontext()):
                    policy = self.forward_pass('get_policy', ['state'], valid_keys, strict=True)[0]
                if self.pol_reg != 0:
                    accuracy = self.ops.accuracy(policy, prim.data['action'], lengths, rows, accuracy_error, accuracy_type)
                    ep_accuracy = self.ops.segment_mean(accuracy, runs, rows)
            std_out = self.forward_pass('get_std', ['state'], valid_keys)
            ep_stdev = (
                self.episode_means(std_out[0], lengths, rows, runs) if std_out is not None else np.zeros_like(ep_returns)
            )
            reduction = np.zeros(len(lengths), np.float64)
            if self.ghosts is not None:
                means = self.forward_pass('get_mean', ['state'], valid_keys, strict=True)[0]
                src_rows, use_src = self._ghost_sources(valid_keys, rows, ep_returns, runs.counts, means)
                reduction = self.ops.ghost_reduction(means, src_rows, use_src, lengths, rows, accuracy_type == 'discrete')
            ts = self._tick(verbose, "computed policy metrics", ts)

            # Set the new episodic returns, accuracy, (ghost) reduction and std to the secondary buffer
            G = len(lengths)
            def full(fill, values):
                block = np.full((G, n_eps), fill, np.float64)
                block[rows] = values[rows, :n_eps]
                return block
            ep_ids = np.broadcast_to(runs.ids[rows[0], :n_eps], (G, n_eps)).astype(np.int64)
            sec.append_block(
                ec_reward=full(-np.inf, ep_returns), ep_accuracy=full(0.0, ep_accuracy),
                ep_reduction=full(+np.inf, np.broadcast_to(reduction[:, None], (G, n_eps))),
                ep_stdev=full(+np.inf, ep_stdev), ep_map=ep_ids, ep_len=full(-np.inf, runs.lengths),
            )
            ts = self._tick(verbose, "updated secondary buffers", ts)

            # Clean up memory for other calculations
            if self.max_steps is not None: self.deque_steps_secondary(self.max_steps, None)
            if self.max_episodes is not None: self.deque_episodes_secondary(self.max_episodes, None)
            buffer_sizes_primary = prim.buffer_sizes()
            buffer_sizes_secondary = sec.buffer_sizes()
            self.deque_steps(0, None)
            ts = self._tick(verbose, "deleted data", ts)

            # Get full returns over every retained episode
            srows, sruns = self.prepare_rollout(sec, valid_keys, check=False)
            sl, sep = sec.lengths, sec.data['ep_map']
            returns = self.compute_returns(sec.data['ec_reward'], sep, sl, srows, 0, 0, self.alpha, self.order, False, self.normalize)
            if self.ghosts is not None:
                dependant_genera = list(self.ghosts.keys())
                reduction_reduced = self.compute_returns(
                    sec.data['ep_reduction'], sep, sl, srows, 0, 0, 1, self.beta_order, False, False,
                    dependant_genera, self.beta
                )
            ts = self._tick(verbose, "computed secondary returns", ts)

            # Compute scores from returns: mean over episodes minus the spread between episodes
            ep_cum = self.episode_means(returns, sl, srows, sruns)
            raw_mean, raw_std = self.row_stats(ep_cum, sruns.counts, srows, 0)
            scores = self.as_dict(raw_mean - self.div_reg * raw_std, valid_keys, srows)
            for key, value in scores.items(): self.population.genomes[key]._actual = value

            # Calculate and set scores
            if self.ghosts is not None:
                rm, rs = self.row_stats(self.ops.record_mean(reduction_reduced, sl, srows), sl, srows, 1)
                red_dev = self.as_dict(rm + self.div_reg * rs, valid_keys, srows)
                red_dev_norm = {k: 1 - v for k, v in self.normalize_array(red_dev, genus_separated=True, default=0.).items()}
                std_default = {'default': 0.}
            else:
                red_dev = red_dev_norm = None
                std_default = {}
            am, a_s = self.row_stats(sec.data['ep_accuracy'], sl, srows, 1)
            policy_accuracy = self.as_dict(am - self.div_reg * a_s, valid_keys, srows)
            sm, s_s = self.row_stats(sec.data['ep_stdev'], sl, srows, 1)
            std_dev = self.as_dict(sm + self.div_reg * s_s, valid_keys, srows)
            std_dev_norm = {k: 1 - v for k, v in self.normalize_array(std_dev, genus_separated=True, **std_default).items()}
            assert all(0 <= v <= 1 for v in std_dev_norm.values())
            best_genome_key, fitness_stats = self.set_scores(scores, policy_accuracy, std_dev_norm, red_dev_norm, self.attention_noise if self.noise_reg > 0 else None)
            ts = self._tick(verbose, "set scores", ts)

            # Post Calculations
            best_row = sec.mapping[best_genome_key]
            ep_lens = sec.data['ep_len'][best_row, :sl[best_row]]
            ep_len_mean = np.mean(ep_lens).item()
            ep_len_std  = np.std(ep_lens).item()
            best_rew = ep_rewards[best_row, :n_eps]
            ep_rew_mean = np.mean(best_rew).item()
            best_cum = ep_cum[best_row, :sruns.counts[best_row]]
            ep_cum_rew = np.mean(best_cum).item()
            ep_rew_std = np.std(best_cum).item() if len(best_cum) > 1 else 0.0
            global_rew = np.mean(np.nanmean(ep_rewards[rows], axis=1)).item()
            global_cum_rew = np.mean(np.nanmean(ep_cum[srows], axis=1)).item()
            policy_acc = policy_accuracy[best_genome_key]
            if self.pol_reg > 0:
                policy_reduction = 1 if len(policy_accuracy) <= 1 else sorted(
                    list(policy_accuracy.keys()), key=lambda k: policy_accuracy[k]
                ).index(best_genome_key) / (len(policy_accuracy)-1)
            else:
                policy_reduction = 0

            mean, std, maximum, minimum, zero_count = self._param_stats(best_genome_key)
            processing_time = np.floor(clock.perf_counter() - pts)
            self._tick(verbose, "calculated stats", ts)

            # Logging
            self.logging.update(
                ep_len_mean=ep_len_mean, ep_len_std=ep_len_std, ep_rew_mean=ep_rew_mean, ep_rew_std=ep_rew_std,
                std=std, policy_acc=policy_acc,
            )

            # Population stats shared by both the dashboard log and the terminal report
            survival_rate = len([key for key in valid_keys if key not in self.population.to_delete]) / len(self.population.genomes)
            best_genome = self.population.genomes[best_genome_key]
            creep, creep_best, creep_max = self._creep(valid_keys, best_genome_key, buffer_sizes_secondary, verbose)

            # Ghost-only reduction stats, computed once and shared by both outputs below
            if self.ghosts is not None:
                raw_red = sec.data['ep_reduction']
                valid_red = np.concatenate([raw_red[r, :sl[r]] for r in srows])
                reduction_global = (valid_red.mean(), valid_red.std(ddof=1) if len(valid_red) > 1 else 0.0,
                                    valid_red.max(), valid_red.min())
                ghost_tb_section = {
                    'raw_reduction': red_dev[best_genome_key],
                    'policy_reduction': red_dev_norm[best_genome_key],
                    **{f'reduction_{label}': float(param) for param, label in zip(reduction_global, ['mean', 'std', 'max', 'min'])},
                }
                ghost_report_section = {'reduction': red_dev[best_genome_key]}
            else:
                ghost_tb_section = ghost_report_section = None

            external_report = self._write_logs(
                self.updates_done,
                rollout={
                    'ep_len_mean': ep_len_mean, 'ep_len_std': ep_len_std, 'ep_rew_mean': ep_rew_mean,
                    'ep_rew_std': ep_rew_std, 'ep_cum_rew': ep_cum_rew, 'global_rew': global_rew,
                    'global_cum_rew': global_cum_rew, 'buffer_size_pri': buffer_sizes_primary[best_genome_key],
                    'buffer_size_sec': buffer_sizes_secondary[best_genome_key],
                },
                time={'run_time': run_time, 'processing_time': processing_time},
                policy={
                    'policy_accuracy': policy_acc, 'policy_reduction': policy_reduction,
                    'raw_stddev': std_dev[best_genome_key], 'policy_stddev': std_dev_norm[best_genome_key],
                    **(ghost_tb_section or {}),
                },
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
                        'ep_len_mean': ep_len_mean, 'ep_len_std': ep_len_std, 'ep_rew_mean': ep_rew_mean,
                        'ep_rew_std': ep_rew_std, 'ep_cum_rew': ep_cum_rew,
                    },
                    'TIME': {
                        'epochs_done': epoch_done + 1, 'run_time': run_time, 'steps_done': self.steps_done,
                        'episodes_done': self.episodes_done,
                        'current_episodes_done': len(self.episode_lengths) if self.episode_lengths else 1,
                    },
                    'TRAINING': {
                        'updates_done': self.updates_done, 'policy_accuracy': policy_acc,
                        **(ghost_report_section or {}), 'std': std_dev[best_genome_key],
                    },
                    'POPULATION': {
                        'best_genome': best_genome.key, 'best_genus': best_genome.genus,
                        'survival_rate': survival_rate, 'creep': creep, 'creep_max': creep_max,
                    },
                    **external_report,
                })

            self._finish_epoch(evaluation_function, epoch_done, epochs, verbose)
            epoch_done += 1


class Ghost(NEAT):
    """
    Ghost Policy NEAT: ``NEAT`` with a mandatory ``ghosts`` mapping ({genus: source genus}). Kept with the former
    ``Ghost(population, ghosts, schedulers, ...)`` signature; it is the same trainer as ``NEAT(..., ghosts=ghosts)``.
    """

    def __init__(self, population: Population, ghosts: dict[int, int],
                 schedulers: Union[Scheduler, Iterable[Scheduler]] = None,
                 device=None, dtype=torch.float32, **options):
        if not ghosts:
            raise ValueError("Ghost requires a non-empty 'ghosts' mapping {genus: source genus}")
        super().__init__(population, schedulers, device, dtype, ghosts=ghosts, **options)
