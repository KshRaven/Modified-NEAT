"""
Numeric backend of the RL trainers.

Everything that used to loop over genomes / episodes / records with Torch tensors lives here as Numba
(``prange`` over genomes) kernels working on plain NumPy arrays of shape ``(genomes, capacity, features)``. Only the
actual model passes stay in Torch (see ``Algorithm.forward_pass``).

The device is the one ``ModifiedNEAT.base`` uses (``ModifiedNEAT.base.get_device()``), so there is no second device
setting. Only the CPU backend is implemented; ``CudaRLBackend`` is the slot for the CuPy / numba.cuda version, until
then a 'cuda' device falls back to the CPU backend for this (post-rollout) bookkeeping with a warning.

Conventions
-----------
* ``rows``    : int64 array of the buffer rows (genomes) a kernel works on.
* ``lengths`` : int64 array (genomes,), records stored per row. Anything at ``index >= lengths[row]`` is never read.
* ``ep``      : int64 array (genomes, capacity), episode id of every record. Must be sorted per row
                (``RolloutBuffer.sort_by('ep_map')``) before returns / episode reductions.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

__all__ = ['EpisodeRuns', 'CpuRLBackend', 'CudaRLBackend', 'get_rl_backend']

_JIT = dict(parallel=True, cache=True)


# ============================================================================================================ #
#                                                  Kernels                                                     #
# ============================================================================================================ #

@njit(cache=True)
def _grid(idx, width):
    """
    CPU counterpart of ``numba.cuda.grid(1)`` for a 2-D launch: one flat thread id -> ``(item, position)``.
    Nested ``for item / for position`` ranges are flattened into ONE ``prange(items * width)`` so every cell is an
    independent work item (no per-row load imbalance, nothing to synchronize). The CUDA kernels reuse the bodies as is
    with ``idx = cuda.grid(1)`` and a ``if idx >= items * width: return`` guard.
    """
    return idx // width, idx % width


@njit(cache=True)
def _norm(v, mode, gmean, gstd, gmin, gmax):
    if mode == 1:
        return (v - gmean) / (gstd + 1e-9)
    elif mode == 2:
        return (v - gmin) / (gmax - gmin)
    elif mode == 3:
        return -1.0 + 2.0 * ((v - gmin) / (gmax - gmin))
    elif mode == 4:
        return v - gmin
    return v


@njit(**_JIT)
def _run_flags(ep, lengths, rows, width):
    """1 where a record opens a new episode run (flattened (row, record), every cell independent)."""
    flags = np.zeros(ep.shape, np.int64)
    for idx in prange(rows.shape[0] * width):
        i, t = _grid(idx, width)
        g = rows[i]
        if t >= lengths[g]:
            continue
        if t == 0 or ep[g, t] != ep[g, t - 1]:
            flags[g, t] = 1
    return flags


@njit(**_JIT)
def _run_starts(ep, lengths, rows, width, flags, slot, starts, ids):
    """Every run start writes its own (row, run) slot: no two work items touch the same cell."""
    for idx in prange(rows.shape[0] * width):
        i, t = _grid(idx, width)
        g = rows[i]
        if t < lengths[g] and flags[g, t] == 1:
            starts[g, slot[g, t]] = t
            ids[g, slot[g, t]] = ep[g, t]


@njit(**_JIT)
def _run_ends(starts, counts, lengths, rows, ends):
    """A run ends where the next one starts (or at the row's length); flattened (row, run)."""
    E = starts.shape[1]
    for idx in prange(rows.shape[0] * E):
        i, r = _grid(idx, E)
        g = rows[i]
        if r >= counts[g]:
            continue
        ends[g, r] = starts[g, r + 1] if r + 1 < counts[g] else lengths[g]


def _find_runs(ep, lengths, rows, width):
    """Contiguous episode runs of every row: ``(counts, starts, ends, ids)``; three flattened passes + one cumsum."""
    flags = _run_flags(ep, lengths, rows, width)
    counts = flags.sum(axis=1).astype(np.int64)
    emax = int(counts[rows].max()) if len(rows) else 0
    slot = np.cumsum(flags, axis=1) - 1
    G = ep.shape[0]
    starts = np.zeros((G, emax), np.int64)
    ends = np.zeros((G, emax), np.int64)
    ids = np.full((G, emax), -1, np.int64)
    _run_starts(ep, lengths, rows, width, flags, slot, starts, ids)
    _run_ends(starts, counts, lengths, rows, ends)
    return counts, starts, ends, ids


@njit(**_JIT)
def _record_mean(arr, lengths, rows, width):
    G, cap, F = arr.shape
    out = np.zeros((G, cap), np.float64)
    for idx in prange(rows.shape[0] * width):                       # flattened (row, record)
        i, t = _grid(idx, width)
        g = rows[i]
        if t >= lengths[g]:
            continue
        s = 0.0
        for f in range(F):
            s += float(arr[g, t, f])
        out[g, t] = s / F
    return out


@njit(**_JIT)
def _segment_mean(vals, starts, ends, counts, rows):
    G = vals.shape[0]
    E = starts.shape[1]
    out = np.full((G, E), np.nan, np.float64)
    for idx in prange(rows.shape[0] * E):                           # flattened (row, episode)
        i, r = _grid(idx, E)
        g = rows[i]
        if r >= counts[g]:
            continue
        s = 0.0
        for t in range(starts[g, r], ends[g, r]):
            s += vals[g, t]
        out[g, r] = s / (ends[g, r] - starts[g, r])
    return out


@njit(**_JIT)
def _row_mean_std(mat, counts, rows, ddof):
    G = mat.shape[0]
    mean = np.full(G, np.nan, np.float64)
    std = np.zeros(G, np.float64)
    for i in prange(rows.shape[0]):
        g = rows[i]
        n = counts[g]
        if n == 0:
            continue
        s = 0.0
        for t in range(n):
            s += mat[g, t]
        m = s / n
        mean[g] = m
        if n > 1 and n - ddof > 0:
            v = 0.0
            for t in range(n):
                v += (mat[g, t] - m) ** 2
            std[g] = np.sqrt(v / (n - ddof))
    return mean, std


@njit(**_JIT)
def _record_stats(arr, lengths, rows, width):
    """Per record (flattened (row, record)): sum, min and max over the features."""
    G, cap, F = arr.shape
    rs = np.zeros((G, cap), np.float64)
    lo = np.full((G, cap), np.inf, np.float64)
    hi = np.full((G, cap), -np.inf, np.float64)
    for idx in prange(rows.shape[0] * width):
        i, t = _grid(idx, width)
        g = rows[i]
        if t >= lengths[g]:
            continue
        for f in range(F):
            v = float(arr[g, t, f])
            rs[g, t] += v
            if v < lo[g, t]: lo[g, t] = v
            if v > hi[g, t]: hi[g, t] = v
    return rs, lo, hi


@njit(**_JIT)
def _record_sq_dev(arr, lengths, rows, width, mean):
    """Per record (flattened (row, record)): sum over the features of the squared deviation from the row mean."""
    G, cap, F = arr.shape
    out = np.zeros((G, cap), np.float64)
    for idx in prange(rows.shape[0] * width):
        i, t = _grid(idx, width)
        g = rows[i]
        if t >= lengths[g]:
            continue
        for f in range(F):
            out[g, t] += (float(arr[g, t, f]) - mean[g]) ** 2
    return out


def _flat_stats(arr, lengths, rows):
    """Per row: mean, unbiased std, min, max over every element (records x features)."""
    G, cap, F = arr.shape
    width = int(lengths[rows].max()) if len(rows) else 0
    rs, lo, hi = _record_stats(arr, lengths, rows, width)
    n = lengths.astype(np.float64) * F
    mean = np.zeros(G, np.float64)
    std = np.zeros(G, np.float64)
    mn = np.full(G, np.inf, np.float64)
    mx = np.full(G, -np.inf, np.float64)
    for g in rows:  # 1-D reductions over the records, cheap next to the passes above
        if n[g] > 0:
            mean[g] = rs[g, :lengths[g]].sum() / n[g]
            mn[g] = lo[g, :lengths[g]].min()
            mx[g] = hi[g, :lengths[g]].max()
    sq = _record_sq_dev(arr, lengths, rows, width, mean)
    for g in rows:
        if n[g] > 1:
            std[g] = np.sqrt(sq[g, :lengths[g]].sum() / (n[g] - 1))
    return mean, std, mn, mx


@njit(**_JIT)
def _returns_kernel(rew, starts, ends, counts, rows, out, norm_mode, gmean, gstd, gmin, gmax,
                    run_factors, mult, do_alpha, gamma, kappa):
    """
    Normalize -> per-episode alpha scaling -> gamma (reward-to-go) and/or kappa (forward) discounting, per element.
    ``run_factors[g, r]`` is the alpha exponent of the r-th episode (run) of row g. The discounting recurrences only run
    along time inside one episode, so every (row, episode, feature) triple is an independent work item.
    ``norm_mode`` / ``gmean`` / ``gstd`` / ``gmin`` / ``gmax`` are arrays (genomes,): the normalization statistics of the
    group (genus, or the whole population) each row belongs to.
    """
    E = starts.shape[1]
    F = rew.shape[2]
    use_g = gamma > 0.0
    use_k = kappa > 0.0
    for idx in prange(rows.shape[0] * E * F):                       # flattened (row, episode, feature)
        f = idx % F
        i, r = _grid(idx // F, E)
        g = rows[i]
        if r >= counts[g]:
            continue
        s = starts[g, r]
        e = ends[g, r]
        w = 1.0
        if do_alpha:
            w = mult[g] ** run_factors[g, r]
        nm, mu, sd, lo, hi = norm_mode[g], gmean[g], gstd[g], gmin[g], gmax[g]
        if use_k:
            acc = 0.0
            for t in range(s, e):
                acc = _norm(float(rew[g, t, f]), nm, mu, sd, lo, hi) * w + acc * kappa
                out[g, t, f] = acc
        if use_g:
            acc = 0.0
            for t in range(e - 1, s - 1, -1):
                acc = _norm(float(rew[g, t, f]), nm, mu, sd, lo, hi) * w + acc * gamma
                if use_k:
                    out[g, t, f] = (acc + out[g, t, f]) / 2.0
                else:
                    out[g, t, f] = acc
        if not use_g and not use_k:
            for t in range(s, e):
                out[g, t, f] = _norm(float(rew[g, t, f]), nm, mu, sd, lo, hi) * w


@njit(**_JIT)
def _gae_kernel(rew, val, starts, ends, counts, rows, adv, ret, gamma, lam, alpha, reverse):
    E = starts.shape[1]
    F = rew.shape[2]
    for idx in prange(rows.shape[0] * E * F):                       # flattened (row, episode, feature)
        f = idx % F
        i, r = _grid(idx // F, E)
        g = rows[i]
        if r >= counts[g]:
            continue
        factor = r if not reverse else counts[g] - 1 - r
        w = alpha ** factor
        fv = 0.0
        fa = 0.0
        for t in range(ends[g, r] - 1, starts[g, r] - 1, -1):
            v = float(val[g, t, f])
            delta = float(rew[g, t, f]) * w + gamma * fv - v
            fv = v
            fa = delta + gamma * lam * fa
            adv[g, t, f] = fa
            ret[g, t, f] = fa + v


@njit(cache=True)
def _bipolar_zone(v, thr):
    """The 3 zones of a bipolar output: 2 when ``v >= thr``, 0 when ``v <= -thr``, 1 in between."""
    if v >= thr:
        return 2
    if v <= -thr:
        return 0
    return 1


@njit(**_JIT)
def _accuracy_kernel(pred, target, lengths, rows, err, mode, width):
    """Per record accuracy (mean over features). mode: 0 continuous, 1 binary, 2 discrete, 3 bipolar."""
    G, cap, Fp = pred.shape
    Ft = target.shape[2]
    F = max(Fp, Ft)
    out = np.zeros((G, cap), np.float64)
    bipolar_thr = max(0.0, 1.0 - 2.0 * err)                         # 0.8 for err=0.1
    for idx in prange(rows.shape[0] * width):                       # flattened (row, record)
        i, t = _grid(idx, width)
        g = rows[i]
        if t >= lengths[g]:
            continue
        s = 0.0
        for f in range(F):
            p = float(pred[g, t, f if Fp > 1 else 0])
            a = float(target[g, t, f if Ft > 1 else 0])
            if mode == 0:
                ok = (p <= a * (1.0 + err)) and (p >= a * (1.0 - err))
            elif mode == 1:
                ok = (p >= (1.0 - err)) == (a >= (1.0 - err))
            elif mode == 2:
                ok = p == a
            else:
                ok = _bipolar_zone(p, bipolar_thr) == _bipolar_zone(a, bipolar_thr)
            if ok:
                s += 1.0
        out[g, t] = s / F
    return out


@njit(**_JIT)
def _ppo_terms(new_lp, old_lp, adv, ent, lengths, rows, width):
    """Per record: mean(ratio * mean(adv)), mean((p_new - p_old)^2), mean(entropy)."""
    G, cap, A = new_lp.shape
    Fa = adv.shape[2]
    Fe = ent.shape[2]
    o_ppo = np.zeros((G, cap), np.float64)
    o_sq = np.zeros((G, cap), np.float64)
    o_ent = np.zeros((G, cap), np.float64)
    for idx in prange(rows.shape[0] * width):                       # flattened (row, record)
        i, t = _grid(idx, width)
        g = rows[i]
        if t >= lengths[g]:
            continue
        am = 0.0
        for f in range(Fa):
            am += float(adv[g, t, f])
        am /= Fa
        s1 = 0.0
        s2 = 0.0
        for a in range(A):
            n = float(new_lp[g, t, a])
            o = float(old_lp[g, t, a])
            s1 += np.exp(n - o) * am
            s2 += (np.exp(n) - np.exp(o)) ** 2
        o_ppo[g, t] = s1 / A
        o_sq[g, t] = s2 / A
        e = 0.0
        for f in range(Fe):
            e += float(ent[g, t, f])
        o_ent[g, t] = e / Fe
    return o_ppo, o_sq, o_ent


@njit(**_JIT)
def _value_terms(values, rets, lengths, rows, eps, width):
    G, cap, Fv = values.shape
    Fr = rets.shape[2]
    F = max(Fv, Fr)
    out = np.zeros((G, cap), np.float64)
    for idx in prange(rows.shape[0] * width):                       # flattened (row, record)
        i, t = _grid(idx, width)
        g = rows[i]
        if t >= lengths[g]:
            continue
        s = 0.0
        for f in range(F):
            d = float(values[g, t, f if Fv > 1 else 0]) - float(rets[g, t, f if Fr > 1 else 0])
            s += np.log10(d * d + eps)
        out[g, t] = s / F
    return out


@njit(**_JIT)
def _ghost_reduction(means, src_rows, use_src, lengths, rows, discrete, width):
    """
    Per record partial sums of the squared distance between each row's policy means and its ghost sources' policy
    means (all sources, all action dims; softmax over the last dim first when ``discrete``). Returns
    ``(sum, count)`` shaped (genomes, capacity); the caller reduces them over the records.
    """
    G, cap, A = means.shape
    K = src_rows.shape[1]
    psum = np.zeros((G, cap), np.float64)
    pcnt = np.zeros((G, cap), np.float64)
    for idx in prange(rows.shape[0] * width):                       # flattened (row, record)
        i, t = _grid(idx, width)
        g = rows[i]
        if t >= lengths[g] or not use_src[g]:
            continue
        v = np.zeros(A, np.float64)
        s = np.zeros(A, np.float64)
        for a in range(A):
            v[a] = float(means[g, t, a])
        if discrete:
            mx = v.max()
            z = 0.0
            for a in range(A):
                v[a] = np.exp(v[a] - mx)
                z += v[a]
            for a in range(A):
                v[a] /= z
        total = 0.0
        cnt = 0.0
        for k in range(K):
            sr = src_rows[g, k]
            if sr < 0 or t >= lengths[sr]:
                continue
            for a in range(A):
                s[a] = float(means[sr, t, a])
            if discrete:
                mx2 = s.max()
                z2 = 0.0
                for a in range(A):
                    s[a] = np.exp(s[a] - mx2)
                    z2 += s[a]
                for a in range(A):
                    s[a] /= z2
            for a in range(A):
                total += (v[a] - s[a]) ** 2
                cnt += 1.0
        psum[g, t] = total
        pcnt[g, t] = cnt
    return psum, pcnt


@njit(**_JIT)
def _rank_order(scores, lengths, rows, descending):
    G, cap = scores.shape
    order = np.zeros((G, cap), np.int64)
    for i in prange(rows.shape[0]):
        g = rows[i]
        n = lengths[g]
        if descending:
            order[g, :n] = np.argsort(-scores[g, :n], kind='mergesort')
        else:
            order[g, :n] = np.argsort(scores[g, :n], kind='mergesort')
    return order


@njit(**_JIT)
def _gather_rows(arr, order, lengths, rows, out, width):
    F = arr.shape[2]
    for idx in prange(rows.shape[0] * width):                       # flattened (row, record)
        i, t = _grid(idx, width)
        g = rows[i]
        if t >= lengths[g]:
            continue
        o = order[g, t]
        for f in range(F):
            out[g, t, f] = arr[g, o, f]


@njit(**_JIT)
def _scatter_rows(arr, order, lengths, rows, out, width):
    F = arr.shape[2]
    for idx in prange(rows.shape[0] * width):                       # flattened (row, record)
        i, t = _grid(idx, width)
        g = rows[i]
        if t >= lengths[g]:
            continue
        o = order[g, t]
        for f in range(F):
            out[g, o, f] = arr[g, t, f]


# ============================================================================================================ #
#                                                  Backend                                                     #
# ============================================================================================================ #

@dataclass
class EpisodeRuns:
    """Contiguous episode segments of every row: records ``starts[g, r]:ends[g, r]`` belong to episode ``ids[g, r]``."""
    starts: np.ndarray
    ends: np.ndarray
    ids: np.ndarray
    counts: np.ndarray

    @property
    def emax(self) -> int:
        return int(self.starts.shape[1])

    @property
    def lengths(self) -> np.ndarray:
        """(rows, emax) episode lengths (0 where an episode does not exist)."""
        return self.ends - self.starts


class CpuRLBackend(object):
    """Numba (prange over genomes) implementation of the RL bookkeeping, running on NumPy arrays."""
    name = 'cpu'
    xp = np

    # -------------------------------- Arrays -------------------------------- #
    @staticmethod
    def to_host(array) -> np.ndarray:
        """Torch (cpu / cuda) tensors and CuPy arrays -> NumPy (zero copy when already on the host)."""
        if isinstance(array, np.ndarray):
            return array
        if hasattr(array, 'detach'):  # torch
            return array.detach().cpu().numpy()
        if hasattr(array, 'get'):  # cupy
            return array.get()
        return np.asarray(array)

    @staticmethod
    def width(lengths: np.ndarray, rows: np.ndarray) -> int:
        """Grid width of the (row, record) launches: the longest valid row (no work items spent on padding)."""
        return int(lengths[rows].max()) if len(rows) else 0

    @staticmethod
    def flat3(arr: np.ndarray) -> np.ndarray:
        """View a ``(genomes, capacity, *features)`` array as ``(genomes, capacity, F)`` (never copies C arrays)."""
        if arr.ndim == 3:
            return arr
        return arr.reshape(arr.shape[0], arr.shape[1], -1)

    # -------------------------------- Episodes -------------------------------- #
    def episode_runs(self, ep: np.ndarray, lengths: np.ndarray, rows: np.ndarray) -> EpisodeRuns:
        counts, starts, ends, ids = _find_runs(ep, lengths, rows, self.width(lengths, rows))
        return EpisodeRuns(starts, ends, ids, counts)

    def record_mean(self, arr: np.ndarray, lengths: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """(genomes, capacity, *features) -> (genomes, capacity) mean over the features of every record."""
        return _record_mean(self.flat3(arr), lengths, rows, self.width(lengths, rows))

    def segment_mean(self, per_record: np.ndarray, runs: EpisodeRuns, rows: np.ndarray) -> np.ndarray:
        """(genomes, capacity) -> (genomes, episodes) mean per episode (NaN where the episode does not exist)."""
        return _segment_mean(per_record, runs.starts, runs.ends, runs.counts, rows)

    def row_mean_std(self, mat: np.ndarray, counts: np.ndarray, rows: np.ndarray, ddof: int = 0):
        """Mean and std of the first ``counts[g]`` entries of every row. ``std`` is 0 for single entries."""
        return _row_mean_std(np.ascontiguousarray(mat, dtype=np.float64), counts, rows, ddof)

    # -------------------------------- Returns -------------------------------- #
    @staticmethod
    def _run_factors(counts: np.ndarray, emax: int, rows: np.ndarray, order: int) -> np.ndarray:
        G = len(counts)
        base = np.broadcast_to(np.arange(emax, dtype=np.float64), (G, emax)).copy()
        if order == 1:
            base = counts[:, None].astype(np.float64) - 1.0 - base
        elif order in (4, 5, 6):
            for g in rows:
                R = int(counts[g])
                remaining = list(range(R))[::-1]
                popped = []
                for _ in range(R):
                    if len(remaining) == 1:
                        pos = 0
                    else:
                        ef = np.arange(len(remaining))
                        ef_ = ef[::-1] + 1
                        pos = int(np.random.choice(ef, p=(ef_ / ef_.sum()) if order in (4, 5) else None))
                    popped.append(remaining.pop(pos))
                # The first popped factor belongs to the last episode, the last popped to the first
                for r in range(R):
                    base[g, r] = popped[R - 1 - r]
        return base

    def compute_returns(
            self, rewards: np.ndarray, ep: np.ndarray, lengths: np.ndarray, rows: np.ndarray, row_genus: np.ndarray,
            gamma: float = 0.97, kappa: float = 0.0, alpha: float = 1.0, order: int = 0, inverse: bool = False,
            normalize: int = 0, genus_filter=None, beta: float = 1.0, per_genus: bool = True,
    ) -> np.ndarray:
        """
        Same maths as the former ``Algorithm.compute_returns_static`` on arrays: reward normalization (per genus by default,
        else global over ``rows``; see ``per_genus``), per-episode ``alpha`` / ``beta`` scaling, then gamma (reward-to-go) and kappa (forward) discounting.
        Returns float64 of the shape of ``rewards``; the input is never modified.
        """
        if not (-1 <= order <= 6):
            raise ValueError(f"Invalid alpha order: '{order}'")
        normalize = int(normalize)
        G = rewards.shape[0]
        r3 = self.flat3(rewards)
        out = np.zeros(r3.shape, np.float64)
        if len(rows) == 0:
            return out.reshape(rewards.shape)
        genus_filter = [] if genus_filter is None else list(genus_filter)

        # Normalization statistics (mean of means, std of per-genome stds, min of mins, max of maxes), one set per
        # group: every genus on its own (``per_genus``, so a weak genus is ranked against itself and not against a
        # stronger one) or the whole population. Stored per row so the kernel just indexes its own row.
        nmode = np.zeros(G, np.int64)
        nmean = np.zeros(G, np.float64)
        nstd = np.ones(G, np.float64)
        nmin = np.zeros(G, np.float64)
        nmax = np.ones(G, np.float64)
        if normalize in (1, 2, 3, 4):
            means, stds, mins, maxs = _flat_stats(r3, lengths, rows)
            groups = [rows[row_genus[rows] == g] for g in np.unique(row_genus[rows])] if per_genus else [rows]
            for grp in groups:
                if normalize == 1:
                    gmean = float(np.mean(means[grp]))
                    gstd = float(np.std(stds[grp])) if len(grp) > 1 else 0.0
                    nmean[grp], nstd[grp] = gmean, gstd
                    nmode[grp] = 1 if gstd != 0.0 else 0
                else:
                    gmin, gmax = float(np.min(mins[grp])), float(np.max(maxs[grp]))
                    nmin[grp], nmax[grp] = gmin, gmax
                    nmode[grp] = 4 if normalize == 4 else (normalize if (gmax - gmin) != 0.0 else 0)

        width = self.width(lengths, rows)

        # Score based ordering of the records (episodes) before applying the episode factors
        perm = None
        ep_used = ep
        if order in (2, 3, 4, 5):
            scores = _record_mean(r3, lengths, rows, width)
            perm = _rank_order(scores, lengths, rows, order in (3, 5))
            r_sorted = np.zeros_like(r3, dtype=r3.dtype)
            _gather_rows(r3, perm, lengths, rows, r_sorted, width)
            e_sorted = np.zeros_like(ep)
            _gather_rows(ep.reshape(G, ep.shape[1], 1), perm, lengths, rows, e_sorted.reshape(G, ep.shape[1], 1), width)
            r3, ep_used = r_sorted, e_sorted

        counts, starts, ends, _ = _find_runs(ep_used, lengths, rows, width)
        emax = starts.shape[1]
        factors = self._run_factors(counts, emax, rows, order)

        mult = np.full(G, float(alpha), np.float64)
        if beta != 1.0 and genus_filter:
            mult[np.isin(row_genus, genus_filter)] = float(beta)
        if inverse:
            mult = 1.0 / mult
        do_alpha = (alpha is not None and alpha != 1.0) or beta != 1.0

        _returns_kernel(r3, starts, ends, counts, rows, out, nmode, nmean, nstd, nmin, nmax,
                        factors, mult, do_alpha, float(gamma), float(kappa))
        if perm is not None:  # back to the original record positions
            back = np.zeros_like(out)
            _scatter_rows(out, perm, lengths, rows, back, width)
            out = back
        return out.reshape(rewards.shape)

    def compute_gae(self, rewards, values, ep, lengths, rows, gamma=0.95, gae_lambda=0.91, alpha=1.10, reverse=False):
        """Returns ``(returns, advantages)`` (float64, shaped like ``rewards``) with per-episode ``alpha`` scaling."""
        r3, v3 = self.flat3(rewards), self.flat3(values)
        adv, ret = np.zeros(r3.shape, np.float64), np.zeros(r3.shape, np.float64)
        runs = self.episode_runs(ep, lengths, rows)
        _gae_kernel(r3, v3, runs.starts, runs.ends, runs.counts, rows, adv, ret,
                    float(gamma), float(gae_lambda), float(alpha), bool(reverse))
        return ret.reshape(rewards.shape), adv.reshape(rewards.shape)

    # -------------------------------- Policy metrics -------------------------------- #
    def accuracy(self, pred, target, lengths, rows, error=0.10, kind='continuous') -> np.ndarray:
        """
        Per record accuracy (mean over features), shaped (genomes, capacity).

        :param kind: ``'continuous'``: prediction within ``error`` (relative) of the target. ``'binary'``: prediction
            and target on the same side of ``1 - error``. ``'bipolar'``: three zones split at ``+-(1 - 2 * error)``
            (above, within, below); correct when prediction and target fall in the same zone. ``'discrete'``: exact match.
        """
        modes = {'continuous': 0, 'binary': 1, 'discrete': 2, 'bipolar': 3}
        if kind not in modes:
            raise ValueError(f"Unsupported accuracy type '{kind}'")
        return _accuracy_kernel(self.flat3(pred), self.flat3(target), lengths, rows, float(error), modes[kind],
                                self.width(lengths, rows))

    def ppo_terms(self, new_lp, old_lp, adv, ent, lengths, rows):
        if new_lp.shape[2:] != old_lp.shape[2:]:
            raise ValueError(f"log_prob shape mismatch: new {new_lp.shape[2:]} vs old {old_lp.shape[2:]}")
        return _ppo_terms(self.flat3(new_lp), self.flat3(old_lp), self.flat3(adv), self.flat3(ent), lengths, rows,
                          self.width(lengths, rows))

    def value_terms(self, values, rets, lengths, rows, eps) -> np.ndarray:
        return _value_terms(self.flat3(values), self.flat3(rets), lengths, rows, float(eps), self.width(lengths, rows))

    def ghost_reduction(self, means, src_rows, use_src, lengths, rows, discrete=False) -> np.ndarray:
        psum, pcnt = _ghost_reduction(self.flat3(means), src_rows, use_src, lengths, rows, bool(discrete),
                                      self.width(lengths, rows))
        total, count = psum.sum(axis=1), pcnt.sum(axis=1)
        return np.where(count > 0, total / np.maximum(count, 1.0), 0.0)


class CudaRLBackend(CpuRLBackend):
    """
    TODO(cuda): same interface on CuPy arrays + numba.cuda kernels (one block per genome, like ``base.gpu``).
    Every method of :class:`CpuRLBackend` is the contract: shapes are ``(genomes, capacity, features)`` and ``rows`` /
    ``lengths`` select what is valid. ``xp`` must become ``cupy`` and ``to_host`` must call ``.get()``.
    """
    name = 'cuda'

    def __init__(self):
        raise NotImplementedError("The CUDA RL backend is not implemented yet; the CPU backend is used instead.")


_BACKENDS: dict[str, CpuRLBackend] = {}


def get_rl_backend() -> CpuRLBackend:
    """Backend for the device ``ModifiedNEAT.base`` currently uses (cached per device name)."""
    try:
        from ModifiedNEAT.base import get_device
        device = get_device()
    except Exception:  # base not importable (eg. tests): CPU
        device = 'cpu'
    if device not in _BACKENDS:
        if device == 'cuda':
            try:
                _BACKENDS[device] = CudaRLBackend()
            except NotImplementedError:
                warnings.warn("RL bookkeeping has no CUDA backend yet; using the CPU backend for it "
                              "(models still run on their torch device).", RuntimeWarning)
                _BACKENDS[device] = _BACKENDS.setdefault('cpu', CpuRLBackend())
        else:
            _BACKENDS[device] = CpuRLBackend()
    return _BACKENDS[device]