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


class NEAT(Algorithm):
    """
    Neuro-Evolution of Augmenting Topologies (NEAT) algorithm.

    Also doubles as the "Ghost Policy" variant: passing a ``ghosts`` mapping activates an extra
    regularization term that pulls a genus' policy towards another (source) genus' policy. When
    ``ghosts`` is left as ``None`` every ghost-only computation is skipped entirely.
    """

    def __init__(self, population: Population, schedulers: Union[Scheduler, Iterable[Scheduler]] = None,
                 device=torch.device('cpu'), dtype=torch.float32,
                 ghosts: dict[int, int] | None = None, **options):
        """
        :param population: Population being trained.
        :param schedulers: Scheduler or iterable of schedulers stepped once per epoch.
        :param device: Device used for computation (default is 'cpu').
        :param dtype: Data type used for tensors (default is torch.float32).
        :param ghosts: Optional map of {genus: source_genus}. When provided, enables the Ghost Policy
            regularization that penalizes a genus' policy mean for straying from its source genus'.
            Requires a Population with multiple genera and a Model exposing ``get_mean``/``get_mean_std``.
        :param options: Additional keyword arguments for configuration.
        :keyword gamma: Discount factor applied to primary (per-step) rewards (default 0.97).
        :keyword kappa: Forward-discount factor mixed in with ``gamma`` for primary rewards (default 0.0).
        :keyword alpha: Per-episode reward multiplier used when reducing secondary returns (default 1.0).
        :keyword order: Episode ranking/ordering mode used when reducing secondary returns (default 0).
        :keyword normalize: Reward normalization mode used when reducing secondary returns (default 0).
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
        :keyword log_dir: Directory tensorboard logs are written under.
        :keyword log_name: Name of this run's tensorboard log.
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
        self.epsilon: float     = manage_params(options, 'epsilon', 1e-12)
        self.rew_reg: float     = manage_params(options, 'rew_reg', 1.0)
        self.pol_reg: float     = manage_params(options, 'pol_reg', 0.0)
        self.std_reg: float     = manage_params(options, 'std_reg', 0.2)
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

    def update(self, observations: Tensor, actions: Tensor, rewards: Tensor,
               terminated: Union[bool, list[bool]], force_stop=False, reset_mapping: bool | list[bool] = False,
               session_dim: int | None = None) -> Tensor:
        """
        Records one environment step for every environment instance into the primary buffer.

        :param observations: Observation tensor for this step.
        :param actions: Action tensor taken this step.
        :param rewards: Reward tensor received this step.
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
                assert tensor.shape[session_dim] ==  len(terminated)
                return torch.select(tensor, session_dim, session_idx)
            else:
                return tensor

        observations, actions, rewards = observations.clone(), actions.clone(), rewards.clone()

        filled = False
        if self.steps_done < self.steps_limit:
            # Loop through all environments
            for idx, ended in enumerate(terminated):
                self.primary.update(
                    state   = select(observations, idx),
                    action  = select(actions, idx),
                    reward  = select(rewards, idx),
                    ep_map  = self.episode_mapping[idx],
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

    def _combine_returns(
        self, episode_mapping: dict[int, list[int]], batches: dict[int, list[list[int]]],
        states: dict[int, Tensor], returns: dict[int, Tensor],
        keys: int | Iterable[int] | None
    ):
        """
        (Ghost only) Computes per-episode mean returns and, for each genus with a ghost source, stacks the
        best-performing genomes' policy means to later compare other genomes' policies against.

        :param episode_mapping: Per-genome list mapping each record to its episode index.
        :param batches: Per-genome list of record-index batches.
        :param states: Per-genome state tensor.
        :param returns: Per-genome per-step return tensor.
        :param keys: Genome keys to include; defaults to every genome in the population.
        :return: Tuple of (per-genome per-episode mean return, per-genus stacked ghost-source policy means).
        """
        if keys is None: keys = list(self.population.genomes.keys())
        elif isinstance(keys, (int, float)): keys = [int(keys)]

        genera: list[int] = self.population.genera

        # Get mean returns for each episode and each genome
        ep_returns_mean: dict[int, dict[int, float]] = {
            key: {
                uei: torch.mean(returns[key][episode_indices]).cpu().item()
                for episode_indices, uei in [
                    ([idx for idx, ep_idx in enumerate(episode_mapping[key]) if ep_idx == unique_ep_idx], unique_ep_idx)
                    for unique_ep_idx in np.unique(episode_mapping[key])
                ] # List[ListOfIndicesForEachEpisode]
            }
            for key in keys
        } # Dict[Key, Dict[EpisodeIndex, MeanReturn]]

        # Get best genomes (up to self.checks) per genus
        returns_mean = {key: np.mean(list(episodes.values())).item() for key, episodes in ep_returns_mean.items()}
        best_keys: dict[int, list[int]] = {g: [] for g in genera}
        for genus in genera:
            _genus_keys = [k for k, g in self.population.genomes.items() if g.genus == genus]
            genus_keys  = [k for k in returns_mean.keys() if k in _genus_keys]
            if len(genus_keys) > 0:
                scores = [returns_mean[k] for k in genus_keys]
                order = np.argsort(scores)[::-1]  # best first
                top_n = min(self.checks, len(genus_keys))
                genus_best = [genus_keys[i] for i in order[:top_n]]
            else:
                # This is only encountered when validation is True
                # TODO: Ensure top genomes are pushed to the beginning during update of Population.genomes
                available_keys = [k for k in _genus_keys if k in keys]
                if len(available_keys) > 0:
                    top_n = min(self.checks, len(available_keys))
                    genus_best = available_keys[:top_n]
                else:
                    genus_best = _genus_keys[:self.checks]
            best_keys[genus] = genus_best

        # Get sources
        sources: dict[int, Tensor] = {
            genus: torch.stack([
                self._get_mean(key, batches[key], states[key], raw=True)
                if key in keys else
                torch.tensor(0.0)
                for key in key_list
            ], dim=0)
            for genus, key_list in best_keys.items() # TODO: Add support for multiple sources later
        }

        return ep_returns_mean, sources

    def _get_reduction(
        self, key: int, sources: dict[int, Tensor], batches: dict[int, list[list[int]]], states: dict[int, Tensor],
        type: str = "continuous",
    ):
        """
        (Ghost only) Computes how far a genome's policy mean has drifted from its genus' ghost source.

        :param key: Genome key to evaluate.
        :param sources: Per-genus stacked ghost-source policy means, as returned by ``_combine_returns``.
        :param batches: Per-genome list of record-index batches.
        :param states: Per-genome state tensor.
        :param type: Distribution type; ``'discrete'`` compares softmax-normalized policies.
        :return: Mean squared distance to the ghost source, or ``0.`` when the genome's genus has no source.
        """
        current_genus = self.population.genomes[key].genus
        ghost_genus = self.ghosts.get(current_genus, None)

        reduction = 0.
        if ghost_genus is not None:
            source = sources[ghost_genus]
            if source.mean() != 0:
                values = self._get_mean(key, batches[key], states[key], raw=True).unsqueeze(0)
                if type == "discrete":
                    source = torch.softmax(source, dim=-1)
                    values = torch.softmax(values, dim=-1)
                # batches = batches[key] # TODO: Are batches really necessary if memory can already store all genomes replay already
                reduction = ((values - source) ** 2).mean().cpu().item()

        return reduction

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
                   reduction: dict[int, float] | None = None):
        """
        Normalizes and combines the score components into a final fitness per genome, then assigns it.

        :param scores: Per-genome reward score.
        :param policy: Per-genome policy accuracy; required when ``pol_reg > 0``.
        :param stdev: Per-genome normalized policy stddev; required when ``pol_reg > 0``.
        :param reduction: (Ghost only) Per-genome normalized ghost reduction; required when Ghost is active.
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
        if self.pol_reg > 0. or self.ghosts is not None:
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
        reg_sum = self.rew_reg + self.pol_reg + (self.std_reg if self.pol_reg != 0 else 0) + (self.cpy_reg if self.ghosts is not None else 0)
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
              accuracy_error=0.10, accuracy_type='continuous', verbose: int = None):
        """
        Runs the full evolution loop: collect rollout data, score genomes, log progress and step schedulers,
        until ``epochs`` generations have been trained (or forever, if ``epochs`` is ``None``).

        :param evaluation_function: Callable that runs the population against the environment for one epoch.
        :param steps: Number of primary-buffer steps to collect per epoch.
        :param epochs: Number of epochs to train for; trains indefinitely when ``None``.
        :param batch_size: Batch size used when iterating over collected records.
        :param accuracy_error: Relative error tolerance used when computing policy accuracy.
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
            # TODO: The method for updating mapping of population keys should be here. Should support grouped keys maybe
            self.steps_limit = self.steps_done + steps
            self.population.run(evaluation_function, 1, verbose=verbose, skip=True, trainer=self)
            if len(self.population.to_delete) == len(self.population.genomes):
                self.population.to_delete.clear()
            # TODO: Might need to remove the check below since it might be redundant
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
                states, actions, rewards, episode_mapping = self.primary.rollout(
                    ['state', 'action', 'reward', 'ep_map'], as_list=True, stack=True, keys=valid_keys
                )
                rollout_time = clock.perf_counter() - ts
                if verbose and verbose >= 2:
                    print(f"rolled out data in {CM(f'{round(rollout_time, 2)}s', Fore.LIGHTCYAN_EX)}")
                    print(f"Valid keys: {len(valid_keys)}")
                    print(f"states = {list(states.values())[0].shape}")
                    print(f"actions = {list(actions.values())[0].shape}")
                    print(f"rewards = {list(rewards.values())[0].shape}")

                # Sort episodes wrt. episode mapping
                ts = clock.perf_counter()
                episode_lengths = self.sort_episodes(episode_mapping, states, actions, rewards)
                sort_time = clock.perf_counter() - ts
                if verbose and verbose >= 2:
                    print(f"sorted episodic data in {CM(f'{round(sort_time, 2)}s', Fore.LIGHTCYAN_EX)}")

                # Compute current returns
                ts = clock.perf_counter()
                returns_current = self.compute_returns(rewards, self.gamma, self.kappa, 1.0, 0, False, episode_mapping)
                ret_comp_time = clock.perf_counter() - ts
                if verbose and verbose >= 2:
                    print(f"computed primary returns in {CM(f'{round(ret_comp_time, 2)}s', Fore.LIGHTCYAN_EX)}")

                # (Ghost only) Combine returns to get each genus' ghost source policy mean; skipped otherwise
                if self.ghosts is not None:
                    ep_mean_returns, ghost_sources = self._combine_returns(
                        episode_mapping, batch_indices, states, returns_current, valid_keys
                    )
                else:
                    ep_mean_returns = ghost_sources = None

                # Set the new episodic returns, accuracy, (ghost) reduction and std to the secondary buffer
                ts = clock.perf_counter()
                episodic_data: dict[int, dict[int, tuple[float, float, float, float, int]]] = {
                    # Get episode collection for all valid keys
                    key: {
                        # Get the mean - (std * factor) of all returns per unique episode
                        uei: (
                            ep_mean_returns[key][uei] if self.ghosts is not None else
                            torch.mean(returns_current[key][episode_indices]).cpu().item(), # NOTE: Standard deviation
                            # regularization is no longer done between returns in an episode but between full scores
                            # from rollout loops
                            self._get_accuracy(
                                key, batch_indices[key], states[key][episode_indices], actions[key][episode_indices],
                                None, accuracy_error, accuracy_type,
                            )[0] if self.pol_reg != 0 else 1.0,
                            self._get_reduction(
                                key, ghost_sources, batch_indices, states, accuracy_type
                            ) if self.ghosts is not None else 0.0,
                            self._get_stdev(
                                key, batch_indices[key], states[key][episode_indices],
                            ),
                            len(episode_indices),
                        )
                        for episode_indices, uei in [
                            ([idx for idx, ep_idx in enumerate(episode_mapping[key]) if ep_idx == unique_ep_idx], unique_ep_idx)
                            for unique_ep_idx in np.unique(episode_mapping[key])
                        ] # List[ListOfIndicesForEachEpisode]
                    }
                    for key in valid_keys
                } # Dict[Key, Dict[EpisodeIndex, Tuple[MeanReturn, MeanAccuracy, MeanReduction, MeanStdev, EpisodeLength]]]
                # TODO: Refactor the check below to verify both episodes done and steps done are valid for all genomes
                episode_counts = [len(er) for er in episodic_data.values()]
                # step_counts = [...]
                # Verify that all genomes have done the same total episodes
                if not all([l == episode_counts[0] for l in episode_counts]):
                    raise RuntimeError(f"Ensure all genomes go through the same number of episodes in the environment;"
                                       f"Got:\n {episode_counts}")
                episodes = sorted(set(sum([list(d.keys()) for d in episodic_data.values()], []))) # list of all possible unique episode indices
                for ep_idx in episodes:
                    ec_reward = torch.tensor([
                        episodic_data[key][ep_idx][0] if key in valid_keys else -np.inf
                        for key in self.secondary.mapping.keys()
                    ])
                    ep_accuracy = torch.tensor([
                        episodic_data[key][ep_idx][1] if key in valid_keys else 0.0
                        for key in self.secondary.mapping.keys()
                    ])
                    ep_reduction = torch.tensor([
                        episodic_data[key][ep_idx][2] if key in valid_keys else +np.inf
                        for key in self.secondary.mapping.keys()
                    ])
                    ep_stdev = torch.tensor([
                        episodic_data[key][ep_idx][3] if key in valid_keys else +np.inf
                        for key in self.secondary.mapping.keys()
                    ])
                    ep_len = [
                        episodic_data[key][ep_idx][4] if key in valid_keys else -np.inf
                        for key in self.secondary.mapping.keys()
                    ]
                    self.secondary.update(
                        ec_reward=ec_reward, ep_accuracy=ep_accuracy, ep_reduction=ep_reduction, ep_stdev=ep_stdev,
                        ep_map=int(ep_idx), ep_len=ep_len
                    )
                sec_upd_time = clock.perf_counter() - ts
                if verbose and verbose >= 2:
                    print(f"updated secondary buffers in {CM(f'{round(sec_upd_time, 2)}s', Fore.LIGHTCYAN_EX)}")

                # Clean up memory for other calculations
                ts = clock.perf_counter()
                if self.max_steps is not None: self.deque_steps_secondary(self.max_steps, None) # valid_keys)
                if self.max_episodes is not None: self.deque_episodes_secondary(self.max_episodes, None) # valid_keys)
                buffer_sizes_primary = self.primary.buffer_sizes()
                buffer_sizes_secondary = self.secondary.buffer_sizes()
                self.deque_steps(0, None) # valid_keys)
                del_time  = clock.perf_counter() - ts
                if verbose and verbose >= 2:
                    print(f"deleted data in {CM(f'{round(del_time, 2)}s', Fore.LIGHTCYAN_EX)}")

                # Get full returns
                ts = clock.perf_counter()
                returns, accuracy, reduction_raw, stdev_raw, full_mapping, episode_lengths = self.secondary.rollout(
                    ['ec_reward', 'ep_accuracy', 'ep_reduction', 'ep_stdev', 'ep_map', 'ep_len'],
                    as_list=True, stack=True, keys=valid_keys
                ) # list[TensorDict]
                self.sort_episodes(full_mapping, returns)
                sec_fetch_time = clock.perf_counter() - ts
                if verbose and verbose >= 2:
                    print(f"fetched secondary data in {CM(f'{round(sec_fetch_time, 2)}s', Fore.LIGHTCYAN_EX)}")
                ts = clock.perf_counter()
                returns = self.compute_returns(returns, 0, 0, self.alpha, self.order, self.normalize, full_mapping)
                if self.ghosts is not None:
                    dependant_genera = list(self.ghosts.keys())
                    reduction_raw_reduced = self.compute_returns(
                        reduction_raw, 0, 0, 1, self.beta_order, False, full_mapping,
                        dependant_genera, self.beta
                    )
                else:
                    reduction_raw_reduced = None
                ret_comp_time2 = clock.perf_counter() - ts
                if verbose and verbose >= 2:
                    print(f"computed secondary returns in {CM(f'{round(ret_comp_time2, 2)}s', Fore.LIGHTCYAN_EX)}")

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

                # Compute scores from returns
                _raw_scores: dict[int, list[float]] = {
                    key: [
                        torch.mean(returns[key][indices]).item()
                        for indices in [
                            [idx for idx, ep_idx in enumerate(full_mapping[key]) if ep_idx == u_idx]
                            for u_idx in np.unique(full_mapping[key])
                        ]
                    ]
                    for key in valid_keys
                }
                scores: dict[int, float] = {
                    key: np.mean(value).item() - (self.div_reg * fixed_std(value))
                    for key, value in _raw_scores.items()
                }

                for key, value in scores.items(): self.population.genomes[key]._actual = value

            # Calculate and set scores
            with torch.no_grad():
                ts = clock.perf_counter()
                if self.ghosts is not None:
                    red_dev: dict[int, float] = {
                        key: torch.mean(reduction_raw_reduced[key]).cpu().item() + (self.div_reg * fixed_std(reduction_raw_reduced[key]))
                        for key in valid_keys
                    }
                    red_dev_norm = {k: 1 - v for k, v in self.normalize_array(red_dev, genus_separated=True, default=0.).items()}
                    std_default = {'default': 0.}
                else:
                    red_dev = red_dev_norm = None
                    std_default = {}
                policy_accuracy: dict[int, float] = {
                    key: torch.mean(accuracy[key]).cpu().item() - (self.div_reg * fixed_std(accuracy[key]))
                    # TODO: Check whether to implement the consolidation like scores in order to maintain episodic 
                    #       integrity at secondary buffer level (If it's even necessary)
                    for key in valid_keys
                }
                std_dev: dict[int, float] = {
                    key: torch.mean(stdev_raw[key]).cpu().item() + (self.div_reg * fixed_std(stdev_raw[key]))
                    # TODO: Test if standard deviation of stdev is necessary
                    for key in valid_keys
                }
                std_dev_norm = {k: 1 - v for k, v in self.normalize_array(std_dev, genus_separated=True, **std_default).items()}
                assert all(0 <= v <= 1 for v in std_dev_norm.values())
                best_genome_key, fitness_stats = self.set_scores(scores, policy_accuracy, std_dev_norm, red_dev_norm)
                set_time = clock.perf_counter() - ts
                if verbose and verbose >= 2:
                    print(f"set scores in {CM(f'{round(set_time, 2)}s', Fore.LIGHTCYAN_EX)}")

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
                # ep_rew_std = np.std(ep_rew_hold).item() if len(ep_rew_hold) > 1 else 0.0
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
                try:
                    policy_acc = policy_accuracy[best_genome_key]
                    if self.pol_reg > 0:
                        policy_reduction = 1 if len(policy_accuracy) <= 1 else sorted(
                            list(policy_accuracy.keys()), key=lambda k: policy_accuracy[k]
                        ).index(best_genome_key) / (len(policy_accuracy)-1)
                    else:
                        policy_reduction = 0
                    # explained_variance = self._explained_variance(batch_indices, states, rewards, best_genome_key)[best_genome_key]
                except RuntimeError:
                    policy_acc = np.nan
                    policy_reduction = np.nan

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

                # noinspection PyBroadException
                def get_range_reduction(key: Union[int, None]) -> tuple[float, float, float, float]:
                    """(Ghost only) Mean/std/max/min of the raw ghost reduction across one or all genomes."""
                    try:
                        aggregation = []
                        all_keys = valid_keys if key is None else [key]
                        for k in all_keys: aggregation.append(reduction_raw[k])
                        aggregation = torch.cat(aggregation)
                        return tuple(
                            func(aggregation).cpu().item()
                            for func in [
                                torch.mean, torch.std, torch.max, torch.min
                            ]
                        )
                    except Exception:
                        return torch.nan, torch.nan, torch.nan, torch.nan

                mean, std, maximum, minimum, zero_count = get_range(best_genome_key)
                processing_time = np.floor(clock.perf_counter() - pts)

                if verbose and verbose >= 2:
                    print(f"calculated stats in {CM(f'{round(clock.perf_counter() - ts, 2)}s', Fore.LIGHTCYAN_EX)}")

            # Logging
            with torch.no_grad():
                self.logging.update(
                    ep_len_mean=ep_len_mean, ep_len_std=ep_len_std, ep_rew_mean=ep_rew_mean, ep_rew_std=ep_rew_std,
                    std=std, policy_acc=policy_acc, # explained_variance=explained_variance,
                )

                # Population stats shared by both the tensorboard log and the terminal report
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
                    if verbose:
                        print(f"max_buffer_size = {buffer_size_max}")
                    creep_best = len([
                        key for key in self.prev_valid_keys
                        if key in valid_keys and buffer_sizes_secondary[key] >= buffer_size_best
                    ]) / len(valid_keys)
                    creep_max = len([
                        key for key in self.prev_valid_keys
                        if key in valid_keys and buffer_sizes_secondary[key] >= buffer_size_max
                    ]) / len(valid_keys)
                self.prev_valid_keys = valid_keys

                # Ghost-only reduction stats, computed once and shared by both outputs below
                if self.ghosts is not None:
                    reduction_global = get_range_reduction(None)
                    ghost_tb_section = {
                        'raw_reduction': red_dev[best_genome_key],
                        'policy_reduction': red_dev_norm[best_genome_key],
                        **{f'reduction_{label}': param for param, label in zip(reduction_global, ['mean', 'std', 'max', 'min'])},
                    }
                    ghost_report_section = {'reduction': red_dev[best_genome_key]}
                else:
                    ghost_tb_section = ghost_report_section = None

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
                        'policy_accuracy': policy_acc, 'policy_reduction': policy_reduction,
                        'raw_stddev': std_dev[best_genome_key], 'policy_stddev': std_dev_norm[best_genome_key],
                        **(ghost_tb_section or {}),
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

    def _explained_variance(self, batches: dict[int, list[list[int]]], states: TensorDict, rewards: TensorDict,
                            keys: Union[int, list[int]] = None):
        """
        Computes the fraction of return variance explained by the value function, per genome.

        :param batches: Per-genome list of record-index batches.
        :param states: Per-genome state tensor.
        :param rewards: Per-genome return tensor.
        :param keys: Genome keys to compute for; defaults to every key in ``states``.
        :return: Per-genome explained variance, clamped to at most 1.
        """
        if isinstance(keys, (int, float)):
            keys = [keys]
        keys = list(states.keys()) if keys is None else keys
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
                    value = (torch.var(reward - value) / torch.var(reward)) - 1
                    ev.append(value)

                ex_var[key] = torch.clamp(torch.mean(torch.stack(ev)), None, 1).cpu().item()

            return ex_var
