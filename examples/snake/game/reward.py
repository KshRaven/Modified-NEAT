"""Reward shaping and the learning curriculum. Per-frame lives, score and fitness live in player.py.

The road reward modes (see `RewardConfig`) grade every move against a Hamiltonian-cycle "ring road" the way the
dhcr and cell-tree solvers reason, so a learner is pushed to the same strategy: keep the body in road order,
never box yourself in, and only then cut corners. The grading (`kernels.road.scan`) runs once per frame for all
snakes; its arrays are shared with the strategy state.
"""
from __future__ import annotations

import warnings

import numpy as np
from numpy import ndarray as Array

from . import kernels
from .config import RewardConfig
from .constants import NP_FLOAT, NP_INT
from .grid import Grid
from .kernels.road import K_FREE, ROAD_DHCR, ROAD_NONE, ROAD_TREE

__all__ = ["RewardShaper"]

_ROAD_OF_MODE = {"basic": ROAD_NONE, "dhcr": ROAD_DHCR, "cell_tree": ROAD_TREE}
_WARNED: set[str] = set()


class RewardShaper:
    """Dense shaping terms for every snake plus the skill curriculum (survive first, then fast)."""

    def __init__(self, config: RewardConfig, max_hiatus: int | None = 100, grid: Grid | None = None):
        self.config = config
        self.max_hiatus = None if max_hiatus is None else int(max_hiatus)   # None = no limit
        self.grid = grid
        self.skill_value = float(config.skill_start)
        self.road = ROAD_NONE                   # road of the current episode (resolved on allocate)
        self.want_scan = False                  # set by Snakes when the strategy state needs the move grades
        self.total = 0
        self.allocate(1)

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    @property
    def skill(self) -> float:
        """Curriculum progress in [0, 1] (1 when the curriculum is off)."""
        return float(self.skill_value) if self.config.curriculum else 1.0

    @property
    def crash_scale(self) -> float:
        """Multiplier on the death / unsafe / trap costs, larger early in the curriculum."""
        return 1.0 + self.config.early_crash_mult * (1.0 - self.skill)

    def grade_episode(self, food: Array, deaths: Array) -> None:
        """Update the skill from the finished episode. Call on reset, before the stats are cleared."""
        if food is None or deaths is None or food.size == 0 or food.size != deaths.size:
            return
        self.skill_value = kernels.next_skill(self.skill_value, food, deaths, self.config)

    def _resolve_road(self) -> int:
        """Road of the next episode from the mode, the skill (staged) and what the board allows."""
        mode = self.config.mode
        if mode == "staged":
            mode = "cell_tree" if self.skill >= self.config.staged_switch else "dhcr"
        road = _ROAD_OF_MODE[mode]
        g = self.grid
        if road != ROAD_NONE and g is not None and not kernels.road_available(road, g.width, g.height):
            fallback = ROAD_DHCR if (road == ROAD_TREE and kernels.road_available(ROAD_DHCR, g.width, g.height)) else ROAD_NONE
            key = f"{mode}:{g.width}x{g.height}"
            if key not in _WARNED:
                _WARNED.add(key)
                warnings.warn(f"reward mode '{mode}' cannot build a ring road on a {g.width}x{g.height} board; "
                              f"using '{'dhcr' if fallback == ROAD_DHCR else 'basic'}'", stacklevel=3)
            road = fallback
        return road

    @property
    def mode_name(self) -> str:
        return {ROAD_NONE: "basic", ROAD_DHCR: "dhcr", ROAD_TREE: "cell_tree"}[self.road]

    @property
    def scanning(self) -> bool:
        return self.grid is not None and (self.road != ROAD_NONE or self.want_scan)

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def allocate(self, total: int) -> None:
        """Per-snake arrays, recreated on every reset (the road of the episode is chosen here)."""
        n = (total,)
        self.total = int(total)
        self.road = self._resolve_road() if self.grid is not None else ROAD_NONE
        self.shape_pos = np.zeros(n, dtype=NP_FLOAT)
        self.shape_neg = np.zeros(n, dtype=NP_FLOAT)
        self.prev_dist = np.zeros(n, dtype=NP_FLOAT)    # food distance of the previous frame, in cells
        self.last_turn = np.zeros(n, dtype=NP_INT)
        self.last_in = np.ones(n, dtype=bool)           # was the body in road order on the previous frame
        # Labels of the move that was taken (filled by pre_step)
        self.act = np.ones(n, dtype=NP_INT)
        self.act_kind = np.full(n, K_FREE, dtype=NP_INT)
        self.act_gain = np.zeros(n, dtype=NP_FLOAT)
        self.act_trap = np.zeros(n, dtype=bool)
        self.act_synced = np.ones(n, dtype=bool)
        if self.grid is None:
            return
        g = self.grid
        N, T = g.cell_total, max(1, (g.width // 2) * (g.height // 2))
        i32 = np.int32
        self.succ = np.zeros((total, N), dtype=i32)
        self.rank = np.zeros((total, N), dtype=i32)
        self.tr = np.zeros((total, T), dtype=np.uint8)
        self.td = np.zeros((total, T), dtype=np.uint8)
        # Scratch of the scan
        self.ring = np.zeros((total, N), dtype=i32)
        self.pos = np.zeros((total, N), dtype=i32)
        self.occ = np.zeros((total, N), dtype=np.uint8)
        self.queue = np.zeros((total, N), dtype=i32)
        self.vis = np.zeros((total, N), dtype=i32)
        self.stamp = np.zeros(total, dtype=NP_INT)
        self.tq = np.zeros((total, T), dtype=i32)
        self.tpc = np.zeros((total, T), dtype=i32)
        self.tvis = np.zeros((total, T), dtype=i32)
        self.tnb = np.zeros((total, 4), dtype=i32)
        self.tpath = np.zeros((total, T), dtype=i32)
        self.tcells = np.zeros((total, 4), dtype=i32)
        self.tgen = np.zeros((total, 1), dtype=i32)
        # Grades of the three relative moves
        self.kind = np.zeros((total, 3), dtype=NP_INT)
        self.gain = np.zeros((total, 3), dtype=NP_FLOAT)
        self.space = np.zeros((total, 3), dtype=NP_INT)
        self.tail_ok = np.zeros((total, 3), dtype=NP_INT)
        self.trap = np.zeros((total, 3), dtype=bool)
        self.cand_rem = np.full((total, 3), -1, dtype=NP_INT)
        self.cand_add = np.full((total, 3), -1, dtype=NP_INT)
        self.nbr = np.full((total, 3), -1, dtype=NP_INT)
        self.phi = np.zeros(total, dtype=NP_FLOAT)       # road distance head -> food
        self.synced = np.ones(total, dtype=bool)         # body in road order

    # ------------------------------------------------------------------
    # Road: build, spawn, scan, commit
    # ------------------------------------------------------------------

    def _hiatus_ref(self) -> int:
        """Frames the hiatus terms are measured against: the limit, or one lap of the board when there is none."""
        if self.max_hiatus is not None:
            return int(self.max_hiatus)
        return int(self.grid.cell_total if self.grid else 100)

    @property
    def has_road(self) -> bool:
        return self.road != ROAD_NONE

    def setup_roads(self, mask: Array, rng: np.random.Generator) -> Array:
        """New road for the masked snakes. Returns the road square where each snake's tail starts."""
        g, rw = self.grid, self.config
        N, T = g.cell_total, max(1, (g.width // 2) * (g.height // 2))
        prio = rng.random((self.total, 2 * T)) if (self.road == ROAD_TREE and rw.random_road) else np.zeros((self.total, 1))
        kernels.build_roads(mask, self.road, g.width, g.height, bool(rw.random_road), prio,
                            self.tr, self.td, self.succ, self.rank)
        return rng.integers(0, N, self.total) if rw.random_road else np.zeros(self.total, dtype=NP_INT)

    def scan(self, snakes, active: Array) -> None:
        """Grade the three moves of every snake (call after every spawn / move, before the state is read)."""
        if not self.scanning:
            return
        g, rw = self.grid, self.config
        kernels.scan(
            snakes.body_x, snakes.body_y, snakes.head_idx, snakes.length, snakes.direction,
            g.food_x, g.food_y, active,
            g.width, g.height, bool(g.wrap), self.road, self.succ, self.rank, self.tr, self.td,
            float(rw.shortcut_max_frac), int(rw.tree_any_direction),
            self.ring, self.pos, self.occ, self.queue, self.vis, self.stamp,
            self.tq, self.tpc, self.tvis, self.tnb, self.tpath, self.tcells, self.tgen,
            self.kind, self.gain, self.space, self.tail_ok, self.trap,
            self.cand_rem, self.cand_add, self.nbr, self.phi, self.synced,
        )

    def pre_step(self, actions: Array, active: Array, head_idx: Array) -> None:
        """Take the labels of the chosen moves (and apply their tree swaps). Call before the snakes move."""
        idx = np.arange(self.total)
        self.act = actions.astype(NP_INT)
        if not self.scanning:
            self.act_kind[:] = K_FREE
            self.act_gain[:] = 0.0
            self.act_trap[:] = False
            self.act_synced[:] = True
            return
        self.act_kind = self.kind[idx, actions]
        self.act_gain = self.gain[idx, actions]
        self.act_trap = self.trap[idx, actions]
        self.act_synced = self.synced.copy()
        g = self.grid
        kernels.commit(actions.astype(NP_INT), active, self.road, g.width, g.height, self.kind, self.cand_rem,
                       self.cand_add, self.nbr, self.ring, head_idx, self.tr, self.td, self.succ, self.rank)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def update(
        self, dist: Array, ate: Array, moved: Array, died: Array, hiatus: Array, low_usage: Array,
        active: Array, length: Array,
    ) -> None:
        """Compute `shape_pos` / `shape_neg` for this frame. Call after the move, before `Players.update`.
        `moved` = the food was eaten or moved (no progress term this frame)."""
        rw = self.config
        if not rw.shaping:
            self.shape_pos[:] = 0.0
            self.shape_neg[:] = 0.0
            return
        kernels.reward_shape(
            self.shape_pos, self.shape_neg,
            dist, self.prev_dist, ate, moved, died, hiatus, low_usage, active, length, self.act, self.last_turn, self.last_in,
            self.act_kind, self.act_gain, self.act_trap, self.act_synced,
            bool(self.has_road), self._hiatus_ref(), int(self.grid.cell_total if self.grid else 100),
            float(self.skill), float(self.crash_scale),
            float(rw.closer_weight), float(rw.away_weight), float(rw.time_penalty),
            float(rw.hiatus_weight), float(rw.loop_weight),
            float(rw.road_weight), float(rw.progress_weight), float(rw.shortcut_mult), float(rw.slow_cap),
            float(rw.unsafe_weight), float(rw.desync_weight), float(rw.trap_weight),
            float(rw.nascar_weight), float(rw.fear_weight), float(rw.fear_min_frac),
            float(rw.alive_weight), self._hiatus_ref(), float(rw.resync_refund),
        )