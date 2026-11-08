from __future__ import annotations

import math
import warnings
from itertools import count
from typing import Any, Callable, Iterable, Union

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from ModifiedNEAT.nn.base import Model, NeatModule
from ModifiedNEAT.optim.scheduler import Scheduler
from ModifiedNEAT.population import Population
from ModifiedNEAT.util.datetime import clock, eta, unix_to_datetime_file
from ModifiedNEAT.util.fancy_text import CM, Fore
from ModifiedNEAT.util.qol import manage_params

from .functional import EpisodeRuns, get_rl_backend
from .replay import LogBuffer, ReplayBuffer, RolloutBuffer

__all__ = ['Algorithm', 'NEATAlgoWarning', 'TensorDict', 'TensorboardLogger', 'TerminalReporter']


class NEATAlgoWarning(Warning):
    pass


# Kept for imports of the former Torch based API; the trainers now work on arrays shaped (genomes, records, *features)
TensorDict = dict[int, Tensor]


def _scalar(value):
    """Converts Tensor / numpy / python numbers to something SummaryWriter.add_scalar accepts."""
    if isinstance(value, Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.numpy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bool):
        return int(value)
    return value


class TensorboardLogger:
    """
    Groups scalars into named sections (eg. ``rollout``, ``policy``) and writes them all to a single
    Tensorboard ``SummaryWriter`` in one call, so algorithms build a plain dict per section instead of
    repeating ``writer.add_scalar(...)`` for every metric.

    Values that are arrays with more than one element are written as histograms instead of scalars, and ``call`` gives
    raw access to any other ``SummaryWriter`` method (text, images, figures, hparams ...).
    """

    def __init__(self, log_path: str):
        """
        :param log_path: Directory ``SummaryWriter`` will write its event files to.
        """
        self.log_path = log_path
        self.writer: SummaryWriter | None = None

    def init(self):
        """Creates the underlying ``SummaryWriter`` the first time it's needed, and returns it."""
        if self.writer is None:
            self.writer = SummaryWriter(self.log_path)
        return self.writer

    def log(self, step: int, **sections: dict[str, Any]):
        """
        Writes every scalar of every section under ``'<section>/<name>'``.

        :param step: Shared x-axis step used for every scalar logged this call.
        :param sections: One kwarg per section (eg. ``rollout={'ep_len_mean': 12.0}``). A ``None``
            value for a scalar or an empty/``None`` section is skipped, so optional metrics (eg. only
            present for some algorithms) can be passed unconditionally.
        """
        self.init()
        for section, scalars in sections.items():
            if not scalars:
                continue
            for name, value in scalars.items():
                if value is None:
                    continue
                value = _scalar(value)
                if isinstance(value, np.ndarray):
                    self.writer.add_histogram(f"{section}/{name}", value, step)
                else:
                    self.writer.add_scalar(f"{section}/{name}", value, step)

    def call(self, method: str, *args, **kwargs):
        """Calls any ``SummaryWriter`` method, eg. ``tb.call('add_text', 'env/notes', 'hello', step)``."""
        return getattr(self.init(), method)(*args, **kwargs)

    def flush(self):
        """Flushes any buffered events to disk."""
        if self.writer is not None:
            self.writer.flush()


class TerminalReporter:
    """
    Prints a bordered, column-aligned debug summary to the terminal. Column widths are derived from the
    labels/values actually passed in and from the terminal's own width, so algorithms no longer need to
    hand-pick (and keep in sync) padding widths for every field, which used to break across terminals
    with different tab sizes.
    """

    def __init__(self, min_width: int = 54, value_width: int = 21, label_padding: int = 2):
        """
        :param min_width: Minimum width of the printed border, in characters.
        :param value_width: Minimum column width reserved for values.
        :param label_padding: Extra spacing appended after the longest label found.
        """
        self.min_width = min_width
        self.value_width = value_width
        self.label_padding = label_padding

    def report(self, sections: dict[str, dict[str, Any]]):
        """
        Prints one bordered block containing every section and its rows.

        :param sections: Mapping of section title (eg. ``'ROLLOUT'``) to a ``{label: value}`` dict of
            rows printed beneath it, in insertion order. A falsy/empty section is skipped entirely, so
            algorithm-specific rows (eg. ghost reduction stats) can be included conditionally.
        """
        import shutil

        labels = [label for rows in sections.values() if rows for label in rows.keys()]
        label_width = (max(len(label) for label in labels) if labels else 0) + self.label_padding
        term_width = shutil.get_terminal_size(fallback=(self.min_width, 20)).columns
        width = max(self.min_width, min(term_width, label_width + self.value_width + 7))

        bar = "-" * width
        lines = [bar]
        for title, rows in sections.items():
            if not rows:
                continue
            lines.append(f"|{(title + ':'): <{width - 1}}|")
            for label, value in rows.items():
                lines.append(f"|\t{label: <{label_width}}| {str(value): <{self.value_width}} |")
        lines.append(bar)
        print("\n" + "\n".join(lines))


def _numpy_dtype(dtype: torch.dtype) -> np.dtype:
    try:
        return np.dtype(torch.empty(0, dtype=dtype).numpy().dtype)
    except Exception:  # eg. bfloat16
        return np.dtype(np.float32)


class Algorithm(object):
    """
    Shared rollout / scoring machinery of the NEAT-style trainers.

    Data lives in dense NumPy arrays (``RolloutBuffer``, shape ``(genomes, records, *features)``) and all the
    post-rollout bookkeeping (sorting by episode, returns, per-episode reductions, accuracies, losses) is done by
    Numba kernels (``ModifiedNEAT.rl.functional``). Torch is only used for the model passes, through
    :meth:`forward_pass`, which supports two batching systems:

    * ``batching='genome'``: genomes are processed one by one, each with its own batches of ``batch_size`` records.
    * ``batching='record'``: every genome of a genus (module) is processed at once, ``record_batch_size`` records at
      a time (default 1, which keeps the memory use of large populations low).
    """
    mapping_indexer = count(0)

    def __init__(self, population: Population, schedulers: list[Scheduler] = None,
                 device=None, dtype: torch.dtype = torch.float32, **options):
        # ------------------------------ Handle input ------------------------------ #
        models = list(population.modules.values())
        if isinstance(models, NeatModule):
            models = [models]
        if schedulers is None:
            schedulers = []
        elif isinstance(schedulers, Scheduler):
            schedulers = [schedulers]
        else:
            schedulers = list(schedulers)
        # ------------------------------ Device ------------------------------ #
        # The bookkeeping backend follows the device ModifiedNEAT.base uses; the torch device only runs the models.
        self.ops = get_rl_backend()
        if device is None:
            device = torch.device('cuda' if self.ops.name == 'cuda' else 'cpu')
        # ------------------------------ Build ------------------------------ #
        self.models                     = models
        self.population: Population     = population
        self.schedulers: list[Scheduler] = schedulers
        self.np_dtype                   = _numpy_dtype(dtype)
        self.primary                    = RolloutBuffer(dtype=self.np_dtype)  # rollout records
        self.secondary                  = RolloutBuffer(dtype=np.float64)     # per-episode scores
        self.logging                    = LogBuffer()                        # logging / reporting series
        # ------------------------------ Attributes ------------------------------ #
        self.steps_done                 = 0
        self.prev_steps_done            = self.steps_done
        self.steps_limit: int           = None
        self.episodes_done              = 0
        self.prev_episodes_done         = self.episodes_done
        self.updates_done               = 0
        self.batch_size                 = manage_params(options, 'batch_size', 256)
        self.record_batch_size          = manage_params(options, 'record_batch_size', 1)
        self.batching                   = manage_params(options, 'batching', 'genome')
        self.terminated                 = True
        # ------------------------------ States ------------------------------ #
        self.device: torch.device       = device
        self.dtype: torch.dtype         = dtype
        self.episode_mapping: dict[int, int] = {}
        self.episode_lengths: dict[int, int] = {}
        self.key_genus: dict[int, int]  = {}
        self.row_genus: np.ndarray      = np.zeros(0, np.int64)
        # ------------------------------ Tensorboard logging ------------------------------ #
        from ModifiedNEAT.util.storage import STORAGE_DIR

        self.log_dir: str = manage_params(options, ['log_dir', 'log_directory'], STORAGE_DIR+f"neat_rl_logs/{self.__class__.__name__}/")
        self.log_sub_dir: str = manage_params(options, 'log_sub_dir', "")
        self.log_name: str = manage_params(options, 'log_name', f"log~{unix_to_datetime_file(clock.time())}")
        self.log_path: str = self.log_dir+self.log_sub_dir+self.log_name
        self.writer: SummaryWriter | None = None
        self.tb: TensorboardLogger = TensorboardLogger(self.log_path)
        self.reporter: TerminalReporter = TerminalReporter()
        self._report_hook: Callable = None
        self._log_sources: dict[str, tuple[Callable | dict, bool]] = {}
        self._pending_logs: dict[str, dict[str, Any]] = {}
        self.prev_valid_keys: list[int] = None
        # ------------------------------ Scheduling ------------------------------ #
        # Last step on purpose: subclasses define their tunable attributes BEFORE calling super().__init__,
        # so every attribute a scheduler targets already exists when it is bound.
        self._bind_schedulers()

    def _bind_schedulers(self):
        """
        Binds every scheduler to this algorithm: variables missing from the config take their starting value
        from the attribute of the same name here, and each scheduler ``step`` / ``reset`` pushes the scheduled
        value back onto that attribute. Safe to call again (eg. by a subclass that defines attributes after
        ``super().__init__``).
        """
        config = getattr(self.population, 'config', None)
        for scheduler in self.schedulers:
            if config is not None and scheduler.config is not config:
                warnings.warn(
                    category=NEATAlgoWarning,
                    message=f"Scheduler for {scheduler.params} was built from a different Config than the "
                            f"population's; NEAT sections will not follow it. Build schedulers from population.config."
                )
            scheduler.bind(self)

    def _init_writer(self):
        """Lazily creates the Tensorboard writer. Kept for backwards compatibility; prefer ``self.tb``."""
        if self.writer is None:
            self.writer = self.tb.init()

    def set_report_hook(self, hook: Callable):
        """
        Add a custom post env calculation report hook for when you do not want to add/modify the Population's Reporter
        :param hook: Set a function that receives the Trainer as its parameter to report on a given generation's
            progress.
        :return: None
        """
        self._report_hook = hook

    # ============================================================================================== #
    #                                  External logging (Tensorboard)                                #
    # ============================================================================================== #
    def add_log_source(self, section: str, source: Callable[['Algorithm'], dict] | dict, report: bool = False):
        """
        Registers an external source that is written to Tensorboard under ``section`` every update (e.g. how your
        environment's goals are changing).

        :param section: Tensorboard section (``'<section>/<name>'``) the source's scalars are written under.
        :param source: Either a callable ``fn(trainer) -> {name: value}`` evaluated once per update, or a dict that is
            read (so it can be mutated by the environment) every update. A callable may also return
            ``{section: {name: value}}`` to write several sections. Arrays with more than one element become histograms.
        :param report: Also print the source's values in the terminal report.
        """
        self._log_sources[section] = (source, report)

    def remove_log_source(self, section: str):
        self._log_sources.pop(section, None)

    def log_external(self, section: str, **scalars):
        """
        Queues scalars from anywhere (eg. your evaluation function) for the next update's Tensorboard write, e.g.
        ``trainer.log_external('env', goal=ENV.goal, difficulty=ENV.difficulty)``. Later values of the same name
        within one epoch overwrite earlier ones.
        """
        self._pending_logs.setdefault(section, {}).update(scalars)

    def log_now(self, step: int | None = None, **sections: dict[str, Any]):
        """Writes sections to Tensorboard immediately (``step`` defaults to the number of updates done)."""
        self.tb.log(self.updates_done if step is None else step, **sections)
        self.tb.flush()

    def _collect_external(self) -> tuple[dict[str, dict], dict[str, dict]]:
        tb_sections: dict[str, dict] = {}
        rep_sections: dict[str, dict] = {}
        for section, (source, report) in self._log_sources.items():
            try:
                values = source(self) if callable(source) else dict(source)
            except Exception as e:  # a broken external source must not kill a long training run
                warnings.warn(category=NEATAlgoWarning, message=f"Log source '{section}' failed: {e!r}")
                continue
            if not values:
                continue
            nested = all(isinstance(v, dict) for v in values.values())
            for name, scalars in (values.items() if nested else [(section, values)]):
                tb_sections.setdefault(name, {}).update(scalars)
                if report:
                    rep_sections.setdefault(name.upper(), {}).update(scalars)
        for section, scalars in self._pending_logs.items():
            tb_sections.setdefault(section, {}).update(scalars)
        self._pending_logs = {}
        return tb_sections, rep_sections

    def _write_logs(self, step: int, **sections: dict[str, Any]) -> dict[str, dict]:
        """Writes the algorithm's own sections plus every external one; returns external sections to report."""
        external, report = self._collect_external()
        merged: dict[str, dict] = {section: dict(scalars) for section, scalars in external.items()}
        for section, scalars in sections.items():
            if scalars:
                merged.setdefault(section, {}).update(scalars)  # the algorithm's own metrics win on name clashes
        self.tb.log(step, **merged)
        self.tb.flush()
        return report

    # ============================================================================================== #
    #                                          Mapping / buffers                                     #
    # ============================================================================================== #
    def get_module(self, key: int) -> Model:
        for module in self.models:
            if key in module.mapping:
                return module
        raise ValueError(f"Cannot find module")

    def update_mapping(self, mapping: dict[int, int] | None = None):
        if mapping is None:
            mapping = self.population.get_mapping(consolidated=True, grouped=False)
        self.primary.update_mapping(mapping)
        self.secondary.update_mapping(mapping)
        genomes = self.population.genomes
        self.key_genus = {key: genomes[key].genus for key in mapping if key in genomes}
        self.row_genus = np.full(len(mapping), -1, np.int64)
        for key, index in mapping.items():
            self.row_genus[index] = self.key_genus.get(key, -1)

    def rows_of(self, keys: Iterable[int]) -> np.ndarray:
        return self.primary.rows_of(list(keys))

    def deque_episodes(self, episodes: int, keys: list[int] = None):
        """Keeps only the last ``episodes`` episodes of the primary (rollout) buffer."""
        self.primary.keep_last_episodes(episodes, keys, latest=self.episodes_done)

    def deque_steps(self, steps: int, keys: list[int] = None):
        """Keeps only the last ``steps`` records of the primary (rollout) buffer; ``0`` empties it."""
        self.primary.keep_last_steps(steps, keys)

    def deque_episodes_secondary(self, episodes: int, keys: list[int] = None):
        """Keeps only the last ``episodes`` episodes of each genome in the secondary (score) buffer."""
        self.secondary.keep_last_episodes(episodes, keys)

    def deque_steps_secondary(self, steps: int, keys: list[int] = None):
        """Keeps only the last ``steps`` episode records of each genome in the secondary (score) buffer."""
        self.secondary.keep_last_steps(steps, keys)

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
                if any([ep_map is None, done, reset]):
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
        if self.terminated:
            for ep_map in self.episode_lengths.keys():
                self.episode_lengths[ep_map] = 0

    def _record_step(self, data: dict[str, Any], terminated, force_stop=False,
                     reset_mapping: bool | list[bool] = False, session_dim: int | None = None) -> bool:
        """
        Appends one environment step for every environment instance to the primary buffer.

        :param data: ``{buffer name: array}``; arrays (Tensor / ndarray / cupy) have the genomes on dim 0 and, when
            several environment sessions are run, the sessions on ``session_dim`` (default 1).
        :return: Whether the rollout buffer has been filled (steps limit reached or ``force_stop``).
        """
        terminated = [bool(t) for t in np.atleast_1d(terminated)] if not isinstance(terminated, list) else terminated
        sd_set = session_dim is not None
        if session_dim is None:
            session_dim = 1

        if self.steps_done == self.prev_steps_done:
            self.handle_episode_mapping(terminated, True)
            self.terminated = False

        data = {name: self.ops.to_host(value) for name, value in data.items()}

        def select(array: np.ndarray, session_idx: int):
            if len(terminated) > 1 or sd_set:
                assert array.shape[session_dim] == len(terminated)
                return np.take(array, session_idx, axis=session_dim)
            return array

        filled = False
        if self.steps_done < self.steps_limit:
            # Loop through all environments
            for idx, ended in enumerate(terminated):
                self.primary.append(
                    **{name: select(value, idx) for name, value in data.items()},
                    ep_map=self.episode_mapping[idx],
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

    def get_batches(self, keys: Iterable[int], batch_size: int = None, shuffle=False):
        """``{key: [index arrays]}`` batches over each genome's primary records (kept for external use)."""
        batches = {}
        if batch_size is None:
            batch_size = self.batch_size
        assert batch_size > 0
        for key in keys:
            records = self.primary.size(key)
            indices = np.random.permutation(records) if shuffle else np.arange(records)
            batches[key] = [indices[i:i + batch_size] for i in range(0, records, batch_size)]
        return batches

    # ============================================================================================== #
    #                                            Model passes                                        #
    # ============================================================================================== #
    def _module_groups(self, keys: Iterable[int]) -> list[tuple[Model, list[int]]]:
        """Splits genome keys per module (genus), each ordered like the module's own mapping."""
        keys = list(keys)
        groups = []
        for module in self.models:
            module_keys = [k for k in keys if k in module.mapping]
            if module_keys:
                module_keys.sort(key=lambda k: module.mapping[k])
                groups.append((module, module_keys))
        return groups

    def _to_tensor(self, array: np.ndarray) -> Tensor:
        return torch.from_numpy(np.ascontiguousarray(array)).to(self.device)

    @staticmethod
    def _to_numpy(tensor) -> np.ndarray:
        if not isinstance(tensor, Tensor):
            return np.asarray(tensor)
        tensor = tensor.detach()
        if tensor.dtype in (torch.float16, torch.bfloat16):
            tensor = tensor.float()
        return tensor.cpu().numpy()

    def _call(self, func: Callable, inputs: list[np.ndarray], keys, strict: bool):
        try:
            result = func(*[self._to_tensor(x) for x in inputs], keys=keys)
        except NotImplementedError as e:
            if strict:
                raise NotImplementedError(f"Must implement '{getattr(func, '__name__', func)}' in the NEATModule; {e}")
            return None
        if not isinstance(result, (tuple, list)):
            result = (result,)
        return [self._to_numpy(r) for r in result]

    def forward_pass(
            self, method: str, inputs: list[str | np.ndarray], keys: Iterable[int], lengths: np.ndarray = None,
            batching: str | None = None, batch_size: int | None = None, record_batch_size: int | None = None,
            strict: bool = False,
    ) -> tuple[np.ndarray, ...] | None:
        """
        Runs a Model method (``get_policy``, ``get_std``, ``get_mean``, ``get_value``, ``evaluate_action`` ...) over
        every stored record of the given genomes and returns its outputs as arrays of shape
        ``(genomes, capacity, *output_features)`` (same layout as the primary buffer).

        :param method: Name of the Model method; called as ``method(*inputs, keys=keys)`` with inputs shaped
            ``(genomes, batch, *record_shape)``. In ``'record'`` batching ``keys`` is a list ordered like the
            module's mapping (or None for the whole module), in ``'genome'`` batching a single key.
        :param inputs: Primary buffer names or arrays of shape ``(genomes, capacity, ...)``.
        :param keys: Genome keys to evaluate.
        :param lengths: Valid records per row (defaults to the primary buffer's).
        :param batching: ``'genome'``: genomes one by one with ``batch_size`` records per batch. ``'record'``: all genomes
            of a module at once, ``record_batch_size`` records per batch. Defaults to the trainer's setting.
        :param strict: Raise instead of returning None when the module does not implement ``method``.
        :return: Tuple of output arrays, or None when ``method`` is missing / not implemented and ``strict`` is False.
        """
        batching = batching or self.batching
        if batching not in ('genome', 'record'):
            raise ValueError(f"Unknown batching '{batching}'; use 'genome' or 'record'")
        batch_size = int(batch_size or self.batch_size)
        record_batch_size = int(record_batch_size or self.record_batch_size)
        lengths = self.primary.lengths if lengths is None else lengths
        arrays = [self.primary.data[x] if isinstance(x, str) else x for x in inputs]
        rows_total, cap = arrays[0].shape[:2]
        mapping = self.primary.mapping
        outputs: list[np.ndarray] | None = None

        def store(result, sel, c0, c1):
            nonlocal outputs
            if outputs is None:
                outputs = [np.zeros((rows_total, cap, *o.shape[2:]), o.dtype) for o in result]
            for out, o in zip(outputs, result):
                out[sel, c0:c1] = o

        with torch.no_grad():
            for module, module_keys in self._module_groups(keys):
                if not hasattr(module, method):
                    if strict:
                        raise NotImplementedError(f"Module has no method '{method}'")
                    return None
                func = getattr(module, method)
                rows = np.asarray([mapping[k] for k in module_keys], np.int64)
                row_lengths = lengths[rows]
                uniform = bool((row_lengths == row_lengths[0]).all())
                if batching == 'record' and not uniform:
                    warnings.warn(category=NEATAlgoWarning,
                                  message="Genomes hold different record counts; falling back to genome batching")
                if batching == 'record' and uniform:
                    keys_arg = None if len(module_keys) == len(module.mapping) else module_keys
                    contiguous = bool((np.diff(rows) == 1).all()) if len(rows) > 1 else True
                    sel = slice(int(rows[0]), int(rows[-1]) + 1) if contiguous else rows
                    total = int(row_lengths[0])
                    for c0 in range(0, total, record_batch_size):
                        c1 = min(c0 + record_batch_size, total)
                        result = self._call(func, [a[sel, c0:c1] for a in arrays], keys_arg, strict)
                        if result is None:
                            return None
                        store(result, sel, c0, c1)
                else:
                    for key, row in zip(module_keys, rows):
                        sel = slice(int(row), int(row) + 1)
                        total = int(lengths[row])
                        for c0 in range(0, total, batch_size):
                            c1 = min(c0 + batch_size, total)
                            result = self._call(func, [a[sel, c0:c1] for a in arrays], key, strict)
                            if result is None:
                                return None
                            store(result, sel, c0, c1)
        return tuple(outputs) if outputs is not None else None

    # ============================================================================================== #
    #                                      Episodes / returns / GAE                                  #
    # ============================================================================================== #
    def prepare_rollout(self, buffer: RolloutBuffer, keys: Iterable[int], check: bool = True) -> tuple[np.ndarray, EpisodeRuns]:
        """
        Sorts a buffer's records by episode (stable, in place) for the given genomes and returns their rows and the
        episode segments. With ``check`` it raises when genomes did not all go through the same number of episodes.
        """
        keys = list(keys)
        rows = buffer.rows_of(keys)
        buffer.sort_by('ep_map', keys)
        runs = self.ops.episode_runs(buffer.data['ep_map'], buffer.lengths, rows)
        counts = runs.counts[rows]
        if check and len(counts) and not (counts == counts[0]).all():
            raise RuntimeError(f"Ensure all genomes go through the same number of episodes in the environment;"
                               f"Got:\n {counts.tolist()}")
        return rows, runs

    def compute_returns(
        self, rewards: np.ndarray, episodes: np.ndarray, lengths: np.ndarray, rows: np.ndarray,
        gamma: float = 0.97, kappa: float = 0.00, alpha: float = 1.00, order=0, inverse: bool = False,
        normalize=False, genus_filter: list[int] | None = None, beta: float = 1.00,
        per_genus: bool | None = None,
    ) -> np.ndarray:
        """
        Rewards -> returns for every genome in ``rows`` at once (Numba). ``rewards`` / ``episodes`` are buffer arrays
        shaped ``(genomes, capacity, ...)`` / ``(genomes, capacity)`` that must be sorted by episode. See
        ``CpuRLBackend.compute_returns``. The input is not modified.

        ``per_genus`` (default: the trainer's ``norm_per_genus`` option, True): the normalization statistics are taken
        between genomes of the same genus only, so a weak genus is not flattened by a stronger one.
        """
        if per_genus is None:
            per_genus = getattr(self, 'norm_per_genus', True)
        return self.ops.compute_returns(
            rewards, episodes, lengths, rows, self.row_genus, gamma, kappa, alpha, order, inverse, normalize,
            genus_filter, beta, per_genus
        )

    def compute_returns_and_advantage(
            self, rewards: np.ndarray, values: np.ndarray | None, episodes: np.ndarray, lengths: np.ndarray,
            rows: np.ndarray, gamma=0.95, gae_lambda=0.91, alpha=1.10, reverse=False,
            observations: np.ndarray | str | None = None, keys: Iterable[int] | None = None,
    ):
        """
        Lambda-return (TD(lambda)) and GAE(lambda) advantage (https://arxiv.org/abs/1506.02438) per episode, with the
        per-episode ``alpha`` scaling of the rewards. ``gae_lambda=1`` gives the Monte-Carlo advantage; TD(0) is
        ``gae_lambda=0``. When ``values`` is None they are computed with ``Model.get_value`` from ``observations``.

        :return: ``(returns, advantages)`` float64 arrays shaped like ``rewards``.
        """
        if values is None:
            if observations is None:
                raise RuntimeError(f"Cannot compute returns and advantages without values or observations")
            keys = [self.primary.reverse[r] for r in rows] if keys is None else keys
            result = self.forward_pass('get_value', [observations], keys, lengths, strict=True)
            values = result[0]
        return self.ops.compute_gae(rewards, values, episodes, lengths, rows, gamma, gae_lambda, alpha, reverse)

    def episode_means(self, array: np.ndarray, lengths: np.ndarray, rows: np.ndarray, runs: EpisodeRuns) -> np.ndarray:
        """``(genomes, capacity, *features)`` -> ``(genomes, episodes)`` mean of every episode (NaN-padded)."""
        return self.ops.segment_mean(self.ops.record_mean(array, lengths, rows), runs, rows)

    def row_stats(self, matrix: np.ndarray, counts: np.ndarray, rows: np.ndarray, ddof: int = 0):
        """Mean and std over the first ``counts[g]`` entries per row; raises on NaN (as the former fixed_std)."""
        mean, std = self.ops.row_mean_std(matrix, counts, rows, ddof)
        if np.isnan(std[rows]).any():
            raise ValueError("std_dev raised NaN")
        return mean, std

    @staticmethod
    def as_dict(array: np.ndarray, keys: list[int], rows: np.ndarray) -> dict[int, float]:
        return dict(zip(keys, array[rows].tolist()))

    # ============================================================================================== #
    #                                          Epoch helpers                                         #
    # ============================================================================================== #
    @staticmethod
    def _tick(verbose: int | None, label: str, start: float) -> float:
        """Prints the time since ``start`` when verbose >= 2 and returns a new start."""
        now = clock.perf_counter()
        if verbose and verbose >= 2:
            print(f"{label} in {CM(f'{round(now - start, 2)}s', Fore.LIGHTCYAN_EX)}")
        return now

    def _collect_rollout(self, evaluation_function: callable, steps: int, verbose: int | None) -> tuple[list[int], float]:
        """Runs the environment for ``steps`` records and returns the genome keys to train on and the run time."""
        ts = clock.perf_counter()
        self.steps_limit = self.steps_done + steps
        self.primary.reserve(steps)
        self.population.run(evaluation_function, 1, verbose=verbose, skip=True, trainer=self)
        if len(self.population.to_delete) == len(self.population.genomes):
            self.population.to_delete.clear()
        invalid_population = len(self.population.to_delete) == len(self.population.genomes)
        valid_keys = [
            key for key in self.population.genomes.keys() if key not in self.population.to_delete or invalid_population
        ] if self.validate else list(self.population.genomes.keys())
        if len(valid_keys) == 0:
            valid_keys = list(self.population.genomes.keys())
        run_time = clock.perf_counter() - ts
        if verbose and verbose >= 2:
            print(f"collected data in {CM(f'{round(run_time, 2)}s', Fore.LIGHTCYAN_EX)}")
        return valid_keys, run_time

    def _finish_epoch(self, evaluation_function: callable, epoch_done: int, epochs, verbose: int | None):
        if self._report_hook is not None:
            self._report_hook(self)

        if self.schedulers is not None:
            for s in self.schedulers:
                s.step()

        if epoch_done == epochs - 1:
            self.population.run(
                evaluation_function, 1, verbose=verbose, skip=True, trainer=self, terminate_skip=True
            )

    def _creep(self, valid_keys: list[int], best_key: int, sizes: dict[int, float], verbose: int | None = None):
        """Share of genomes that were also trained on last epoch (overall, as the best genome and as the largest)."""
        if self.prev_valid_keys is None:
            creep = creep_max = creep_best = np.nan
        elif len(valid_keys) <= 0:
            creep = creep_max = creep_best = 0.0
        else:
            creep = len([key for key in self.prev_valid_keys if key in valid_keys]) / len(valid_keys)
            size_best = sizes[best_key]
            size_max = max(sizes.values())
            if verbose:
                print(f"max_buffer_size = {size_max}")
            creep_best = len([
                key for key in self.prev_valid_keys if key in valid_keys and sizes[key] >= size_best
            ]) / len(valid_keys)
            creep_max = len([
                key for key in self.prev_valid_keys if key in valid_keys and sizes[key] >= size_max
            ]) / len(valid_keys)
        self.prev_valid_keys = valid_keys
        return creep, creep_best, creep_max

    # noinspection PyBroadException
    def _param_stats(self, key: Union[int, None]):
        """Mean / std / max / min / share of ~zero NEAT parameters of one genome (or of all when ``key`` is None)."""
        try:
            with torch.no_grad():
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
            return math.nan, math.nan, math.nan, math.nan, math.nan

    def log_scheduler_params(self):
        values: dict[str, Union[int, float, bool]] = {}
        for scheduler in self.schedulers:
            for param in scheduler.params:
                values[param] = scheduler.get(param, scheduler.config)
        return values

    # ============================================================================================== #
    #                                    Dict based score helpers                                    #
    # ============================================================================================== #
    def _genus_of(self, key: int) -> int:
        genus = self.key_genus.get(key)
        return self.population.genomes[key].genus if genus is None else genus

    @staticmethod
    def segregate(ranking: dict[int, float], size: int):
        groups = math.ceil(len(ranking) / size)
        if groups < 1:
            raise ValueError(f"Invalid segregation size '{size}' with '{groups}' groups and '{len(ranking)}' keys")
        keys = list(ranking.keys())
        clusters: list[dict[int, float]] = [
            {
                key: ranking[key] 
                for key in keys[size*x:size*(x+1)]
            } 
            for x in range(groups)
        ]
        return clusters

    def normalize_array(self, array: dict[int, Any], index: int = None, segr_size: int = None, 
                        ranking: dict[int, float] = None, genus_separated: bool = False, default: float = 1.):
        # TODO: Might want to convert genus separation to specie separation
        verbose = False
        full_norm_array = {key: default for key in array.keys()}
        # Filter by genera
        for genus in (self.population.genera if genus_separated else [-1]):
            keys    = [key for key in array.keys() if not genus_separated or self._genus_of(key) == genus]
            source  = [array[key] for key in keys]

            # Handle errors
            if len(source) == 0: continue
            sample = source[0]
            if isinstance(sample, (list, tuple)):
                # Ensure iterable can be reduced for all keys
                for i, (key, item) in enumerate(zip(keys, source)):
                    if len(item) == 0:
                        raise ValueError(f"Empty Iterable found in key '{key}'")
                    if isinstance(item[0], (int, float)):
                        if index is None:
                            source[i] = np.mean(item).item()
                        else:
                            source[i] = item[index]
                    else:
                        ValueError(f"Cannot convert variable of type '{type(item)}'")
            elif isinstance(sample, (ndarray, Tensor)):
                source = [a.mean().item() for a in source]
            elif not isinstance(sample, (float, int)):
                ValueError(f"Cannot consolidate variable of type '{type(sample)}'")

            source = np.array(source) # Should be a 1-D array of shape (genomes,)
            assert np.all(~np.isinf(source)), "Infinities not supported in this normalization"
            if verbose:
                print(source.shape)
                print(np.max(source), np.min(source), source.mean(), source.std())
            minimum, maximum = np.min(source), np.max(source)
            if maximum > minimum:
                norm_source: ndarray = (source - minimum) / (maximum - minimum)
            else:
                norm_source = np.full_like(source, default)

            norm_array: dict[int, float] = dict(zip(keys, norm_source))

            # Segregate normalization when enabled
            if segr_size is not None:
                # Sort if ranking is not given
                if ranking is None:
                    norm_array = dict(sorted(norm_array.items(), key=lambda item: (item[1], -item[0]), reverse=True))
                else:
                    norm_array = {key: norm_array[key] for key in ranking.keys() if key in norm_array}

                segr_norm_array: dict[int, float] = {}
                clusters = self.segregate(norm_array, segr_size)
                segr_range: float = np.max(norm_source).item() / len(clusters)
                for c_idx, cluster in enumerate(clusters):
                    for key, norm_value in self.normalize_array(cluster).items():
                        segr_norm_array[key] = (norm_value*segr_range) + ((len(clusters)-1-c_idx)*segr_range)
                norm_array = segr_norm_array

            for key, value in norm_array.items():
                full_norm_array[key] = value

        return full_norm_array

    def level_array(self, array: dict[int, Any], index: int = None, genus_separated: bool = False) -> dict[int, float]:
        full_level_array = {key: np.nan for key in array.keys()}
        # Filter by genera
        for genus in (self.population.genera if genus_separated else [-1]):
            keys    = [key for key in array.keys() if not genus_separated or self._genus_of(key) == genus]
            source  = [array[key] for key in keys]

            # Handle errors
            if len(source) == 0:
                raise ValueError("No keys in dict")
            sample = source[0]
            if isinstance(sample, list):
                for i, (key, item) in enumerate(zip(keys, source)):
                    if len(item) == 0:
                        raise ValueError(f"Empty Iterable found in key '{key}'")
                    if isinstance(item, (int, float)):
                        if index is None:
                            source[i] = np.mean(item).item()
                        else:
                            source[i] = item[index]
                    else:
                        ValueError(f"Cannot convert variable of type '{type(item)}'")
            elif isinstance(sample, (ndarray, Tensor)):
                source = [a.mean().item() for a in source]
            elif not isinstance(sample, (float, int)):
                ValueError(f"Cannot consolidate variable of type '{type(sample)}'")

            source = np.array(source)
            level_source = source - np.min(np.clip(source, None, 0))

            for key, value in zip(keys, level_source):
                full_level_array[key] = value

        return full_level_array
