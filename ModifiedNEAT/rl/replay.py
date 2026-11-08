from __future__ import annotations

import json
import os
from typing import Any, Union

import numpy as np
import torch
from numba import njit, prange
from numpy import ndarray
from torch import Tensor

__all__ = ['ReplayBuffer', 'replay_path', 'write_replay', 'read_replay', 'RolloutBuffer', 'LogBuffer']


class ReplayBuffer(object):
    def __init__(self, buffers: list[str] = None):
        if buffers is None:
            buffers = []
        self.buffer_names: list[str] = buffers
        self.data: dict[int, dict[str, list[Any]]] = {}
        self.mapping: dict[int, int] = {}
        self.reverse: dict[int, int] = {}

    def episodes(self):
        listing = []
        if 'ep_map' in self.buffer_names:
            for key_data in self.data.values():
                for ep_idx in key_data['ep_map']:
                    if ep_idx not in listing:
                        listing.append(ep_idx)
        return listing

    def buffer_sizes(self, normalize=False):
        lengths: dict[int, float] = {key: np.mean([len(buffer) for buffer in buffers.values()]) for key, buffers in self.data.items()}
        if normalize:
            maximum, minimum = np.max(list(lengths.values())), np.min(list(lengths.values()))
            if maximum > minimum:
                lengths = {key: (value - minimum) / (maximum - minimum) for key, value in lengths.items()}
            else:
                lengths = {key: 1.0 for key in lengths.keys()}
        return lengths

    def max_size(self):
        if len(self.data) > 0:
            maximum = np.max([np.max([len(buffer) for buffer in buffers.values()]) for buffers in self.data.values()])
            return maximum
        else:
            return 0

    def min_size(self):
        if len(self.data) > 0:
            maximum = np.min([np.min([len(buffer) for buffer in buffers.values()]) for buffers in self.data.values()])
            return maximum
        else:
            return 0

    def _init_buffers(self, key: int):
        self.data[key] = {b: list() for b in self.buffer_names if self.data.get(key) is None or b not in self.data[key]}

    def _reset_buffers(self, key: int):
        self.data[key] = {b: list() for b in self.buffer_names}

    def _crop_buffers(self, key: int, max_length: int):
        for name, buffer in self.data[key].items():
            if len(buffer) > max_length:
                self.data[key][name] = buffer[-max_length:]

    def _deque_buffers(self, key: int, to_del: list[int]):
        to_del = sorted(set(to_del), reverse=True)
        for name, buffer in self.data[key].items():
            buffer = list(buffer)
            for idx in to_del:
                del buffer[idx]
            self.data[key][name] = buffer # [item for idx, item in enumerate(buffer) if idx not in to_del]

    def update_mapping(self, mapping: dict[int, int]):
        """
        Map of every genome key to their index
        :param mapping:
        :return:
        """
        new_keys, new_indices = [], []
        # Add keys
        for key, index in mapping.items():
            # Update mapping
            self.mapping[key] = index
            self.reverse[index] = key
            # Initialize buffers if already existing
            if key not in self.data:
                self._init_buffers(key)
            new_keys.append(key)
            new_indices.append(index)
        # Delete keys that are not in the new mapping
        for key in list(self.mapping.keys()):
            if key not in new_keys:
                del self.mapping[key]
                del self.data[key]
        # Delete indices that are not in the new mapping
        for index in list(self.reverse.keys()):
            if index not in new_indices:
                del self.reverse[index]
        self.sort()

    def add_buffers(self, *buffers: str):
        if len(buffers) == 1 and isinstance(buffers[0], (list, tuple)):
            buffers = buffers[0]
        for buffer in buffers:
            if buffer not in self.buffer_names:
                self.buffer_names.append(buffer)
                for key in self.data.keys():
                    self.data[key][buffer] = list()
        self.reset()

    def update(self, add_dims: Union[int, list[int]] = None, validation: int = True, **inputs):
        if validation:
            buffer_names = inputs.keys()
            for name in buffer_names:
                if name not in self.buffer_names:
                    raise ValueError(f"buffer name '{name}' has not been added.")
            if validation == 2 and len(buffer_names) != len(self.buffer_names):
                raise ValueError(f"Update all buffers at once")

        def expand_dims(var, dims: Union[int, list[int]]):
            if not isinstance(dims, (list, tuple)):
                dims = [dims]
            for dim in dims:
                if isinstance(var, Tensor):
                    var = var.unsqueeze(dim)
                elif isinstance(var, ndarray):
                    var = np.expand_dims(var, dim)
                # elif isinstance(var, (list, int, float)) and dim == 0:
                #     var = [var]
            return var

        def check(var, label):
            try:
                error = False
                if not isinstance(var, (ndarray, Tensor, list)):
                    raise ValueError(f"Variable '{label}' is not a Tensor, Array or List; type = '{type(var)}'")
                if isinstance(var, (ndarray, Tensor)) and var.shape[0] != len(self.data):
                    error = True
                elif isinstance(var, list) and len(var) != len(self.data):
                    error = True
                if error:
                    out = var.shape if isinstance(var, (ndarray, Tensor)) else len(var)
                    raise ValueError(f"Shape of data '{label}' cannot be broadcast onto buffer; shape = {out}")
            except ValueError as e:
                raise e
            except IndexError as e:
                print(f"\nvar '{label}' shape = {var.shape if isinstance(var, (ndarray, Tensor)) else len(var)}\n")
                raise e
            except Exception as e:
                raise e

        def get_value(key: int, var, ctn: bool):
            index = self.mapping[key]
            if ctn:
                value = var[index]
            elif isinstance(var, Tensor):
                value = torch.select(var, dim=0, index=index)
            elif isinstance(var, list):
                value = var[index]
            else:
                raise ValueError(f"Cannot get the value from a variable of dtype '{type(var)}'")
            return value

        for name, data in inputs.items():
            if isinstance(data, Tensor):
                data = data.detach().cpu()
            if isinstance(data, (int, float, bool)):
                data = [data for _ in range(len(self.data))]
            check(data, name)
            if isinstance(data, ndarray):
                data = torch.tensor(data, device='cpu')
                to_numpy = True
            else:
                to_numpy = False
            if add_dims:
                expand_dims(data, add_dims)
            for k in self.data.keys():
                self.data[k][name].append(get_value(k, data, to_numpy))

    def reset(self, keys: list[int] = None):
        for key in self.data.keys():
            if keys is None or (keys is not None and key in keys):
                self._reset_buffers(key)

    def crop(self, max_length: int, keys: list[int] = None):
        for key in self.data.keys():
            if keys is None or (keys is not None and key in keys):
                self._crop_buffers(key, max_length)

    def deque(self, to_del: dict[int, list[int]], keys: list[int] = None):
        for key in self.data.keys():
            if keys is None or (keys is not None and key in keys):
                self._deque_buffers(key, to_del[key])

    def deque_episodes(self, episodes: int, keys: list[int] = None):
        """
        Deletes the episodes before the last n episodes.
        Used to compensate for the long data collection times of the NEAT evaluation functions.
        :param episodes: (int) Number of recent episodes to keep.
        :param keys: (list[int])
        :return: (none)
        """
        assert episodes >= 0
        if 'ep_map' in self.buffer_names and self.max_size() > 0:
            filters = {}
            episode_mapping: dict[int, list[int]] = self.rollout(buffers='ep_map', as_list=True)[0]
            for key in self.mapping.keys():
                episodes_done = max(episode_mapping[key] + [0])
                episodes_to_del = torch.tensor([ep for ep in range(episodes_done) if ep < (episodes_done-episodes)])
                mapping = torch.tensor(episode_mapping[key])
                episode_filter  = torch.isin(mapping, episodes_to_del)
                record_filter   = torch.nonzero(episode_filter, as_tuple=True)[0].tolist()
                # record_filter   = [elem.cpu().item() if elem.numel() == 1 else None for elem in record_filter]
                filters[key] = record_filter
            self.deque(filters, keys)

    def deque_steps(self, steps: int, keys: list[int] = None):
        """
        Deletes the last n steps.
        Used to compensate for the large data sizes of the NEAT evaluation functions.
        :param steps: (int) Number of steps to keep.
        :param keys: (list[int])
        :return: (none)
        """
        assert steps >= 0
        if 'ep_map' in self.buffer_names and self.max_size() > 0:
            filters = {}
            episode_mapping: dict[int, list[int]] = self.rollout(buffers='ep_map', as_list=True)[0]
            for key in self.mapping.keys():
                records = len(episode_mapping[key])
                limit = max(0, records - steps)
                filters[key] = [i for i in range(records) if i < limit]
            self.deque(filters, keys)

    def sort(self):
        self.data    = dict(sorted(self.data.items(), key=lambda item: len(item[1])))
        self.mapping = dict(sorted(self.mapping.items(), key=lambda item: item[1]))
        self.reverse = dict(sorted(self.reverse.items(), key=lambda item: item[0]))

    def rollout(self, buffers: Union[str, list[str]] = None, sequence_length: int = None, keys: list[int] = None,
                as_list=False, stack=False):
        if len(self.mapping) == 0:
            raise ValueError(f"No mapping set")
        self.sort()
        if len(list(self.data.values())[0]) == 0:
            raise ValueError(f"No data in buffers")

        if isinstance(buffers, str):
            buffers = [buffers]
        elif buffers is None:
            buffers = self.buffer_names
        if keys is None:
            keys = list(self.mapping.keys())

        res = {}
        for label in buffers:
            group = {}
            for key in keys:
                # Get key's buffer
                buffer = self.data[key][label]
                # Limit length
                if sequence_length is not None:
                    buffer = buffer[-min(len(buffer), sequence_length):]
                    # if sequence_length == 1:
                    #     buffer = buffer[0]
                # Stack buffer
                if stack:
                    if isinstance(buffer[0], ndarray):
                        buffer = np.stack(buffer, axis=0)
                    elif isinstance(buffer[0], Tensor):
                        buffer = torch.stack(buffer, dim=0)
                    elif isinstance(buffer[0], (int, float, bool)):
                        pass
                group[key] = buffer
            res[label] = group
        if as_list:
            res = list(res.values())
        return res


# ============================================================================================================ #
#                       Dense, array based buffers (Numba / NumPy / later CuPy friendly)                       #
# ============================================================================================================ #
# ``ReplayBuffer`` above keeps python lists per genome and is kept for backwards compatibility. The classes below
# split the old single buffer in two on purpose:
#   * ``RolloutBuffer`` : the data trainers compute on (rollout records, per-episode scores). One dense array per
#                         buffer of shape (genomes, capacity, *features) so kernels can run over all genomes at once
#                         and deleting / dequeing is a cheap in-place compaction, not a python list rebuild.
#   * ``LogBuffer``     : scalar time series for logging / reporting (a few floats per update, no genome axis).
# Both can be stored to / loaded from a single ``<name>.replay`` file.

REPLAY_EXT = '.replay'
_MAGIC = b'MNREPLAY'
_ALIGN = 64


def replay_path(path: str) -> str:
    """Appends the ``.replay`` extension when missing."""
    path = os.fspath(path)
    return path if path.endswith(REPLAY_EXT) else path + REPLAY_EXT


def write_replay(path: str, arrays: dict[str, np.ndarray], meta: dict) -> str:
    """
    Writes a ``.replay`` file: magic | header length | JSON header | 64-byte aligned raw arrays.
    The raw layout allows ``read_replay(..., mmap=True)`` to map the arrays directly from disk.
    """
    path = replay_path(path)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    entries, offset = {}, 0
    arrays = {name: np.ascontiguousarray(a) for name, a in arrays.items()}
    for name, array in arrays.items():
        offset = -(-offset // _ALIGN) * _ALIGN
        entries[name] = {'dtype': array.dtype.str, 'shape': list(array.shape), 'offset': offset}
        offset += array.nbytes
    header = json.dumps({'meta': meta, 'arrays': entries}).encode('utf-8')
    data_start = -(-(len(_MAGIC) + 8 + len(header)) // _ALIGN) * _ALIGN
    with open(path, 'wb') as file:
        file.write(_MAGIC)
        file.write(np.uint64(len(header)).tobytes())
        file.write(header)
        file.write(b'\0' * (data_start - file.tell()))
        for name, array in arrays.items():
            file.write(b'\0' * (data_start + entries[name]['offset'] - file.tell()))
            file.write(array.tobytes())
    return path


def read_replay(path: str, mmap: bool = False) -> tuple[dict[str, np.ndarray], dict]:
    """Reads a ``.replay`` file. With ``mmap=True`` arrays are copy-on-write memory maps (nothing is loaded eagerly)."""
    path = replay_path(path)
    with open(path, 'rb') as file:
        if file.read(len(_MAGIC)) != _MAGIC:
            raise ValueError(f"'{path}' is not a replay file")
        header_len = int(np.frombuffer(file.read(8), dtype=np.uint64)[0])
        header = json.loads(file.read(header_len).decode('utf-8'))
        data_start = -(-(len(_MAGIC) + 8 + header_len) // _ALIGN) * _ALIGN
        arrays = {}
        for name, entry in header['arrays'].items():
            shape, dtype, offset = tuple(entry['shape']), np.dtype(entry['dtype']), data_start + entry['offset']
            count = int(np.prod(shape, dtype=np.int64))
            if mmap and count > 0:
                arrays[name] = np.memmap(path, dtype=dtype, mode='c', offset=offset, shape=shape)
            else:
                file.seek(offset)
                arrays[name] = np.fromfile(file, dtype=dtype, count=count).reshape(shape)
    return arrays, header['meta']


# ----------------------------------------- Kernels (in-place edits) ----------------------------------------- #

@njit(parallel=True, cache=True)
def _unsorted_rows(ep, lengths, rows):
    bad = np.zeros(ep.shape[0], np.bool_)
    for i in prange(rows.shape[0]):
        g = rows[i]
        for t in range(1, lengths[g]):
            if ep[g, t] < ep[g, t - 1]:
                bad[g] = True
                break
    return bad


@njit(parallel=True, cache=True)
def _sort_order(ep, lengths, rows):
    order = np.zeros(ep.shape, np.int64)
    for i in prange(rows.shape[0]):
        g = rows[i]
        n = lengths[g]
        order[g, :n] = np.argsort(ep[g, :n], kind='mergesort')  # stable: ties keep their recording order
    return order


@njit(parallel=True, cache=True)
def _apply_order(arr, src, order, lengths, rows, width):
    """
    ``arr[g, t] = src[i, order[g, t]]`` for ``g = rows[i]``. ``src`` is a copy of the rows taken before the call. One flat
    ``prange`` over (row, record) cells, the CPU form of a ``cuda.grid(1)`` launch: every cell is independent.
    """
    F = arr.shape[2]
    for idx in prange(rows.shape[0] * width):
        i = idx // width
        t = idx % width
        g = rows[i]
        if t >= lengths[g]:
            continue
        o = order[g, t]
        for f in range(F):
            arr[g, t, f] = src[i, o, f]


@njit(parallel=True, cache=True)
def _compact(arr, keep, lengths, rows):
    """Moves the kept records of every row to the front (order preserved). ``lengths`` are the OLD lengths."""
    F = arr.shape[2]
    for i in prange(rows.shape[0]):
        g = rows[i]
        w = 0
        for t in range(lengths[g]):
            if keep[g, t]:
                if w != t:
                    for f in range(F):
                        arr[g, w, f] = arr[g, t, f]
                w += 1


def _to_host(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    if hasattr(x, 'get'):  # cupy
        return x.get()
    return np.asarray(x)


def _flat3(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(arr.shape[0], arr.shape[1], -1)


class RolloutBuffer(object):
    """
    Dense rollout / per-episode data store. Row ``i`` of every array belongs to the genome whose consolidated mapping
    index is ``i`` (``update_mapping``); ``lengths[i]`` records of it are valid, so genomes may hold different amounts
    of data (eg. children that were just born) while the arrays stay rectangular.

    ``data[name]`` has shape ``(genomes, capacity, *features)``; ``view(name)`` is the ``[:, :max_size()]`` slice.
    Every ``append`` must provide all buffers, so a single length per row is enough.
    """

    def __init__(self, buffers: list[str] = None, dtype=np.float32, capacity: int = 0):
        self.buffer_names: list[str] = list(buffers) if buffers else []
        self.dtype = np.dtype(dtype)
        self.min_capacity = int(capacity)
        self.data: dict[str, np.ndarray] = {}
        self.meta: dict[str, tuple[tuple, np.dtype]] = {}
        self.lengths = np.zeros(0, np.int64)
        self.capacity = 0
        self.mapping: dict[int, int] = {}
        self.reverse: dict[int, int] = {}

    # ----------------------------------------------- Layout ----------------------------------------------- #
    def __len__(self):
        return len(self.lengths)

    def rows_of(self, keys=None) -> np.ndarray:
        """Row indices of genome keys (all rows when ``keys`` is None)."""
        if keys is None:
            return np.arange(len(self.lengths), dtype=np.int64)
        if isinstance(keys, (int, np.integer)):
            keys = [keys]
        return np.asarray([self.mapping[int(k)] for k in keys], dtype=np.int64)

    def add_buffers(self, *buffers: str):
        if len(buffers) == 1 and isinstance(buffers[0], (list, tuple)):
            buffers = buffers[0]
        for buffer in buffers:
            if buffer not in self.buffer_names:
                self.buffer_names.append(buffer)
        self.reset()

    def reserve(self, records: int):
        """Capacity (records per genome) future allocations start from; avoids regrowing during a rollout."""
        self.min_capacity = max(self.min_capacity, int(records))

    def update_mapping(self, mapping: dict[int, int]):
        """
        ``{genome key: row}``. Rows of keys that persist keep their data, new keys start empty, removed keys are dropped.
        """
        mapping = {int(k): int(v) for k, v in mapping.items()}
        if mapping == self.mapping:
            return
        rows = len(mapping)
        if sorted(mapping.values()) != list(range(rows)):
            raise ValueError("Mapping indices must be exactly 0..genomes-1")
        old_map, old_len, old_data = self.mapping, self.lengths, self.data
        new_len = np.zeros(rows, np.int64)
        src = np.asarray([old_map[k] for k in mapping if k in old_map and old_len[old_map[k]] > 0], np.int64)
        dst = np.asarray([mapping[k] for k in mapping if k in old_map and old_len[old_map[k]] > 0], np.int64)
        self.mapping, self.reverse = mapping, {v: k for k, v in mapping.items()}
        self.lengths = new_len
        if len(src) == 0 or not old_data:
            self.data, self.capacity = {}, 0
            return
        longest = int(old_len[src].max())
        cap = max(longest, self.min_capacity)
        self.data = {name: self._alloc(name, rows, cap) for name in old_data}
        for name, old in old_data.items():
            self.data[name][dst, :longest] = old[src, :longest]
        new_len[dst] = old_len[src]
        self.capacity = cap

    def _alloc(self, name: str, rows: int, cap: int) -> np.ndarray:
        feat, dtype = self.meta[name]
        return np.empty((rows, cap, *feat), dtype)

    def _store_dtype(self, dtype: np.dtype) -> np.dtype:
        return self.dtype if np.issubdtype(dtype, np.floating) else dtype

    def _ensure(self, need: int):
        rows = len(self.lengths)
        missing = [n for n in self.meta if n not in self.data]
        if need <= self.capacity and not missing:
            return
        cap = self.capacity
        if need > cap:
            cap = max(need, self.min_capacity, cap * 2, 16)
        longest = self.max_size()
        data = {}
        for name in self.meta:
            new = self._alloc(name, rows, cap)
            if name in self.data and longest > 0:
                new[:, :longest] = self.data[name][:, :longest]
            data[name] = new
        self.data, self.capacity = data, cap

    # ----------------------------------------------- Writing ----------------------------------------------- #
    def _check(self, inputs: dict) -> dict[str, np.ndarray]:
        rows = len(self.lengths)
        if rows == 0:
            raise ValueError("No mapping set")
        unknown = [n for n in inputs if n not in self.buffer_names]
        missing = [n for n in self.buffer_names if n not in inputs]
        if unknown:
            raise ValueError(f"buffer name '{unknown[0]}' has not been added.")
        if missing:
            raise ValueError(f"Update all buffers at once; missing {missing}")
        return inputs

    def append(self, **inputs):
        """
        Adds ONE record to every genome. Values are arrays whose first dim is the number of genomes (Tensor / ndarray /
        cupy), or scalars that are broadcast to every genome.
        """
        rows = len(self.lengths)
        values = {}
        for name, value in self._check(inputs).items():
            value = _to_host(value)
            if value.ndim == 0:
                value = np.full((rows,), value.item(), dtype=value.dtype)
            if value.shape[0] != rows:
                raise ValueError(f"Shape of data '{name}' cannot be broadcast onto buffer; shape = {value.shape}")
            if name not in self.meta:
                self.meta[name] = (tuple(value.shape[1:]), self._store_dtype(value.dtype))
            values[name] = value
        pos = self.lengths
        self._ensure(int(pos.max()) + 1)
        uniform = bool((pos == pos[0]).all())
        for name, value in values.items():
            if uniform:
                self.data[name][:, pos[0]] = value
            else:
                self.data[name][np.arange(rows), pos] = value
        self.lengths = pos + 1

    def append_block(self, **blocks):
        """Adds ``E`` records at once to every genome; every block has shape ``(genomes, E, *features)``."""
        rows = len(self.lengths)
        values = {}
        for name, block in self._check(blocks).items():
            block = _to_host(block)
            if block.shape[0] != rows or block.ndim < 2:
                raise ValueError(f"Shape of data '{name}' cannot be broadcast onto buffer; shape = {block.shape}")
            if name not in self.meta:
                self.meta[name] = (tuple(block.shape[2:]), self._store_dtype(block.dtype))
            values[name] = block
        E = next(iter(values.values())).shape[1]
        pos = self.lengths
        self._ensure(int(pos.max()) + E)
        uniform = bool((pos == pos[0]).all())
        for name, block in values.items():
            if uniform:
                self.data[name][:, pos[0]:pos[0] + E] = block
            else:
                for g in range(rows):
                    self.data[name][g, pos[g]:pos[g] + E] = block[g]
        self.lengths = pos + E

    # ----------------------------------------------- Reading ----------------------------------------------- #
    def get(self, name: str) -> np.ndarray:
        """Full array ``(genomes, capacity, *features)``; only ``[:lengths[row]]`` of each row is valid."""
        return self.data[name]

    def flat(self, name: str) -> np.ndarray:
        """``get`` as ``(genomes, capacity, F)`` (a view)."""
        return _flat3(self.data[name])

    def view(self, name: str) -> np.ndarray:
        return self.data[name][:, :self.max_size()]

    def row(self, key: int, name: str) -> np.ndarray:
        i = self.mapping[key]
        return self.data[name][i, :self.lengths[i]]

    def size(self, key: int) -> int:
        return int(self.lengths[self.mapping[key]])

    def buffer_sizes(self, normalize=False) -> dict[int, float]:
        lengths = {key: float(self.lengths[row]) for key, row in self.mapping.items()}
        if normalize and lengths:
            lo, hi = min(lengths.values()), max(lengths.values())
            lengths = {k: ((v - lo) / (hi - lo) if hi > lo else 1.0) for k, v in lengths.items()}
        return lengths

    def max_size(self) -> int:
        return int(self.lengths.max()) if len(self.lengths) else 0

    def min_size(self) -> int:
        return int(self.lengths.min()) if len(self.lengths) else 0

    def episodes(self) -> list[int]:
        """Sorted unique episode ids currently stored."""
        if 'ep_map' not in self.data or self.max_size() == 0:
            return []
        valid = np.arange(self.capacity)[None, :] < self.lengths[:, None]
        return np.unique(self.data['ep_map'][valid]).tolist()

    # ----------------------------------------------- Editing ----------------------------------------------- #
    def reset(self, keys=None):
        """Empties the given genomes (all by default). O(genomes): memory is reused by the next appends."""
        if keys is None:
            self.lengths = np.zeros_like(self.lengths)
        else:
            self.lengths[self.rows_of(keys)] = 0

    def _compact_rows(self, keep: np.ndarray, rows: np.ndarray):
        new_len = self.lengths.copy()
        new_len[rows] = (keep & (np.arange(self.capacity)[None, :] < self.lengths[:, None]))[rows].sum(axis=1)
        for name in self.buffer_names:
            if name in self.data:
                _compact(_flat3(self.data[name]), keep, self.lengths, rows)
        self.lengths = new_len

    def keep_last_steps(self, steps: int, keys=None):
        """Keeps only the last ``steps`` records of every (given) genome."""
        assert steps >= 0
        rows = self.rows_of(keys)
        if steps == 0:
            self.lengths[rows] = 0
        elif self.max_size() > steps and len(rows):
            keep = np.arange(self.capacity)[None, :] >= (self.lengths - steps)[:, None]
            self._compact_rows(keep, rows)

    def keep_last_episodes(self, episodes: int, keys=None, latest: int | None = None):
        """
        Keeps the records of the last ``episodes`` episodes. ``latest`` is the number of episodes ever started
        (episode ids are ``< latest``); defaults to each genome's own newest stored episode + 1.
        """
        assert episodes >= 0
        if 'ep_map' not in self.data or self.max_size() == 0:
            return
        rows = self.rows_of(keys)
        ep = self.data['ep_map']
        if latest is None:
            valid = np.arange(self.capacity)[None, :] < self.lengths[:, None]
            newest = np.where(valid, ep, -1).max(axis=1) + 1
            threshold = (newest - episodes)[:, None]
        else:
            threshold = latest - episodes
        self._compact_rows(ep >= threshold, rows)

    # Names of the former ReplayBuffer API
    deque_steps = keep_last_steps
    deque_episodes = keep_last_episodes
    crop = keep_last_steps

    def sort_by(self, name: str = 'ep_map', keys=None) -> bool:
        """Stable in-place sort of every buffer by ``name`` per genome. Returns whether anything moved."""
        if name not in self.data or self.max_size() == 0:
            return False
        rows = self.rows_of(keys)
        ep = self.data[name]
        bad = np.nonzero(_unsorted_rows(ep, self.lengths, rows))[0].astype(np.int64)
        if len(bad) == 0:
            return False
        order = _sort_order(ep, self.lengths, bad)
        width = int(self.lengths[bad].max())
        for buffer_name in self.buffer_names:
            if buffer_name in self.data:
                flat = _flat3(self.data[buffer_name])
                _apply_order(flat, flat[bad, :width].copy(), order, self.lengths, bad, width)
        return True

    # ----------------------------------------------- Storage ----------------------------------------------- #
    def save(self, path: str) -> str:
        """Writes the buffer (valid records only) to ``<path>.replay``."""
        longest = self.max_size()
        arrays = {f"data/{n}": a[:, :longest] for n, a in self.data.items()}
        arrays['lengths'] = self.lengths
        meta = {
            'kind': 'rollout', 'names': self.buffer_names, 'dtype': self.dtype.str,
            'mapping': {str(k): v for k, v in self.mapping.items()},
            'meta': {n: [list(f), np.dtype(d).str] for n, (f, d) in self.meta.items()},
        }
        return write_replay(path, arrays, meta)

    @classmethod
    def load(cls, path: str, mmap: bool = False) -> 'RolloutBuffer':
        arrays, meta = read_replay(path, mmap)
        if meta.get('kind') != 'rollout':
            raise ValueError(f"'{path}' does not hold a RolloutBuffer")
        buffer = cls(meta['names'], np.dtype(meta['dtype']))
        buffer.mapping = {int(k): v for k, v in meta['mapping'].items()}
        buffer.reverse = {v: k for k, v in buffer.mapping.items()}
        buffer.meta = {n: (tuple(f), np.dtype(d)) for n, (f, d) in meta['meta'].items()}
        buffer.lengths = np.array(arrays['lengths'], dtype=np.int64)
        buffer.data = {n[len('data/'):]: a for n, a in arrays.items() if n.startswith('data/')}
        buffer.capacity = next((a.shape[1] for a in buffer.data.values()), 0)
        return buffer


class LogBuffer(object):
    """
    Scalar time series for logging / reporting (one value per name per update), kept apart from the rollout data
    so it can be saved, cropped and queried without touching the genome arrays.
    """

    def __init__(self, buffers: list[str] = None, max_length: int | None = None):
        self.buffer_names: list[str] = list(buffers) if buffers else []
        self.max_length = max_length
        self.data: dict[str, list[Any]] = {n: [] for n in self.buffer_names}

    def __len__(self):
        return max((len(v) for v in self.data.values()), default=0)

    def add_buffers(self, *buffers: str):
        if len(buffers) == 1 and isinstance(buffers[0], (list, tuple)):
            buffers = buffers[0]
        for buffer in buffers:
            if buffer not in self.data:
                self.buffer_names.append(buffer)
                self.data[buffer] = []

    def update(self, **values):
        for name, value in values.items():
            if name not in self.data:
                raise ValueError(f"buffer name '{name}' has not been added.")
            value = _to_host(value) if isinstance(value, (Tensor, np.ndarray)) else value
            if isinstance(value, np.ndarray) and value.size == 1:
                value = value.item()
            series = self.data[name]
            series.append(value)
            if self.max_length is not None and len(series) > self.max_length:
                del series[:len(series) - self.max_length]

    def get(self, name: str) -> np.ndarray:
        return np.asarray(self.data[name])

    def last(self, name: str, default=None):
        series = self.data.get(name)
        return series[-1] if series else default

    def mean(self, name: str, window: int | None = None) -> float:
        series = self.data[name] if window is None else self.data[name][-window:]
        return float(np.mean(series)) if len(series) else float('nan')

    def reset(self):
        for series in self.data.values():
            series.clear()

    def crop(self, max_length: int):
        for series in self.data.values():
            if len(series) > max_length:
                del series[:len(series) - max_length]

    def save(self, path: str) -> str:
        arrays = {n: np.asarray(v, dtype=np.float64) for n, v in self.data.items()}
        return write_replay(path, arrays, {'kind': 'log', 'names': self.buffer_names, 'max_length': self.max_length})

    @classmethod
    def load(cls, path: str) -> 'LogBuffer':
        arrays, meta = read_replay(path)
        if meta.get('kind') != 'log':
            raise ValueError(f"'{path}' does not hold a LogBuffer")
        buffer = cls(meta['names'], meta['max_length'])
        for name, array in arrays.items():
            buffer.data[name] = array.tolist()
        return buffer