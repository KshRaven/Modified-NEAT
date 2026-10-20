import numpy as np

from numba import njit, prange
from numpy import ndarray as Array

from .constants import NP_FLOAT, NP_INT
from .util import grid
from .config import RewardConfig


# ---------------------------------------------------------------------------
# Numba-accelerated helpers
# ---------------------------------------------------------------------------

@njit(parallel=True, fastmath=True, cache=True)
def _update(
    # in-place lifecycle / fitness state (mutated on the same buffers)
    lives: Array, deaths: Array, scores: Array, true_scores: Array,
    disqualified: Array, completed: Array, frames_done: Array,
    fitness: Array, prev_fitness: Array,
    # per-frame signals (read only)
    checked: Array, finished: Array, out_of_bounds: Array, hiatus: Array,
    moved_forward: Array, tile_stag: Array, vel_norm: Array, shape_pos: Array, shape_neg: Array,
    corner_tile_curr: Array, straight_tile_curr: Array,
    on_road: Array, turn_steps: Array, straight_steps: Array,
    # scalars
    lives_total: int, max_hiatus: int, min_turn_steps: int, min_straight_steps: int,
    shaping: bool, crash_penalty: float, reward_offset: float,
):
    """
    Per-player lifecycle / fitness update.

    Mirrors the vectorised ``Players.update`` exactly, but as an explicit per-index
    loop so numba can parallelise across players (same structure as
    ``backtest.agent._update``).  Every player is independent, so there are no
    cross-iteration dependencies.

    Returns the ``eliminated`` mask (players that lost a life this frame).
    """
    players_total = lives.shape[0]

    eliminated = np.zeros(players_total, dtype=np.bool_)

    total, indices = grid((players_total,))

    for i in prange(total):
        t = indices[i]

        # A player with no lives left is out: its fitness is frozen. (Otherwise a parked car is
        # "eliminated" every frame, and the deaths multiplier makes its penalty grow quadratically.)
        if lives[t] <= 0:
            prev_fitness[t] = fitness[t]
            frames_done[t] += 1
            continue

        # ------------------------------------------------------------------
        # 0.  Derived masks
        # ------------------------------------------------------------------
        elim = out_of_bounds[t] or (hiatus[t] >= max_hiatus) or tile_stag[t] or (not moved_forward[t])
        eliminated[t] = elim

        # ------------------------------------------------------------------
        # 1.  State / counter updates
        # ------------------------------------------------------------------
        if elim:        disqualified[t] = True
        if finished[t]: completed[t]    = True
        if elim:        lives[t]       -= 1
        if lives[t] < 0:           lives[t] = 0
        if lives[t] > lives_total: lives[t] = lives_total
        if elim:        deaths[t]      += 1

        if checked[t]:  scores[t] += 1
        if finished[t]: scores[t] += 10
        if elim:        scores[t] -= 1
        if scores[t] > true_scores[t]:
            true_scores[t] = scores[t]

        # ------------------------------------------------------------------
        # 2.  Convenience scalars
        # ------------------------------------------------------------------
        _lives  = lives[t]  + 1   # >= 1; scales rewards (more lives -> more reward)
        _deaths = deaths[t] + 1   # >= 1; scales penalties (more deaths -> heavier hit)
        if on_road[t]: _lives  *= 2
        else:          _deaths *= 2

        # ------------------------------------------------------------------
        # 3.  Snapshot prev-fitness
        # ------------------------------------------------------------------
        prev_fitness[t] = fitness[t]

        # ==================================================================
        # 4.  REWARD SHAPING
        # ==================================================================

        # -- 4a. Road-keeping reward/penalty ---------------------------------
        #   ON road:  +5/frame * lives   -- grows with survival
        #   OFF road: -2/frame * deaths  -- escalates for repeat offenders
        fitness[t] += reward_offset
        if shaping:
            # Dense time-trial shaping from Cars: progress / speed-vs-target / smoothness / surfaces
            fitness[t] += shape_pos[t] * _lives
            fitness[t] -= shape_neg[t] * _deaths
        else:
            if on_road[t]: fitness[t] += 5.0 * _lives
            else:          fitness[t] -= 2.0 * _deaths

        # -- 4e. Turning into corners ----------------------------------------
        if corner_tile_curr[t]:
            if turn_steps[t] == 1:              fitness[t] += 300.0 * _lives
            if turn_steps[t] == min_turn_steps: fitness[t] += 300.0 * _lives

        # -- 4.  Straight tile alignment reward ------------------------------
        if straight_tile_curr[t]:
            if straight_steps[t] == 1:                  fitness[t] += 400.0 * _lives
            if straight_steps[t] == min_straight_steps: fitness[t] += 400.0 * _lives

        # -- 4g. Checkpoint and lap rewards ----------------------------------
        #   Corner checkpoints are worth 20% more than straight ones.
        if checked[t]:
            if straight_tile_curr[t]: fitness[t] += 250.0 * _lives
            if corner_tile_curr[t]:   fitness[t] += 300.0 * _lives
        if finished[t]: fitness[t] += 2000.0 * _lives

        # -- 4h. Elimination (death) one-off penalty -------------------------
        if elim:
            if shaping:
                # Heavier the faster the car was going: botching a corner must never beat slowing down
                fitness[t] -= crash_penalty * (1.0 + abs(vel_norm[t])) * _deaths
            else:
                if straight_tile_curr[t]: fitness[t] -= 30.0 * _deaths
                if corner_tile_curr[t]:   fitness[t] -= 30.0 * _deaths

        # ------------------------------------------------------------------
        # 5.  Housekeeping
        # ------------------------------------------------------------------
        frames_done[t] += 1

    return eliminated


class Players:
    """
    Tracks per-player lifecycle (lives / deaths), score and fitness.

    Plain Python class whose per-frame logic lives in the cached, parallel
    ``_update`` kernel above (it used to be a ``jitclass``, which cannot be cached and
    re-compiled on every process start).  All state is held in numpy arrays that the
    kernel mutates in place.
    """

    def __init__(self, lives: int = 3, max_hiatus: int = 100, reward: RewardConfig | None = None):
        self.reward: RewardConfig = reward if reward is not None else RewardConfig()
        # Scale applied to the per-frame reward returned by get_reward(); offset added to every living
        # player's raw per-frame reward. Both are plain attributes, so they can be changed at runtime.
        self.reward_scale: float  = float(self.reward.reward_scale)
        self.reward_offset: float = float(self.reward.reward_offset)
        self.total: int       = 1
        self.lives_total: int = int(lives)
        self.max_hiatus: int  = int(max_hiatus)

        self._init_arrays()

    # ------------------------------------------------------------------
    # Array initialisation helpers
    # ------------------------------------------------------------------

    def _init_arrays(self) -> None:
        n = self.total
        self.lives          = np.full((n,), self.lives_total, dtype=NP_INT)
        self.deaths         = np.full((n,), 0, dtype=NP_INT)
        self.scores         = np.full((n,), 0, dtype=NP_INT)
        self.true_scores    = np.full((n,), 0, dtype=NP_INT)
        self.disqualified   = np.full((n,), False, dtype=bool)
        self.completed      = np.full((n,), False, dtype=bool)
        self.frames_done    = np.full((n,), 0, dtype=NP_INT)
        self.fitness        = np.full((n,), 0, dtype=NP_FLOAT)
        self.prev_fitness   = np.full((n,), 0, dtype=NP_FLOAT)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, total: int):
        """Full reset for a new episode with *total* parallel players."""
        self.total = int(total)
        self._init_arrays()

    def restart(self):
        """Re-activate disqualified players (soft reset within an episode)."""
        self.disqualified[:] = False

    @property
    def active(self) -> Array:
        """Boolean mask: players that still have lives remaining."""
        return self.lives > 0

    @property
    def active_total(self) -> int:
        return int(np.count_nonzero(self.active))

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self,
        # ── progress / elimination signals ────────────────────────────────
        checked: Array,            # bool (N,)  passed a checkpoint this frame
        finished: Array,           # bool (N,)  completed a lap this frame
        out_of_bounds: Array,      # bool (N,)  left the track region
        hiatus: Array,             # int  (N,)  frames since last forward move
        moved_forward: Array,      # bool (N,)  heading within acceptable arc
        tile_stag: Array,          # bool (N,)  stuck on same tile too long
        # ── tile context ──────────────────────────────────────────────────
        corner_tile_curr: Array,   # bool (N,)  current tile is a LEFT/RIGHT turn
        corner_tile_next: Array,   # bool (N,)  next tile in sequence is a turn
        straight_tile_curr: Array, # bool (N,)  current tile is a straight/start
        # ── continuous driving signals ────────────────────────────────────
        vel_norm: Array,           # float (N,)  speed / max_speed  ∈ [0, 1]
        phase_align: Array,        # float (N,)  cos(car_angle - tile_angle) ∈ [-1, 1]
        on_road: Array,            # bool (N,)  car body touching road surface
        turning: Array,            # bool (N,)  driver steering into expected direction
        turn_steps: Array,
        min_turn_steps: int,
        braking: Array,            # bool (N,)  driver applying negative acceleration
        brake_steps: Array,
        min_brake_steps: int,
        brake_check: Array,
        # ── straight alignment signals ──────────────────────────────
        straight_aligned: Array,   # bool (N,)  car aligned with straight tile orientation
        straight_steps: Array,
        min_straight_steps: int,
        # ── turn-direction and corner-entry signals ───────────────────
        turn_align: Array,         # float (N,)  ∈ [-1, +1]
                                   #   How correctly the car is steering on a corner tile.
                                   #   +1 = turning hard the right way
                                   #    0 = not steering at all
                                   #   -1 = actively steering the wrong way
                                   #   Zeros for non-corner tiles (unused there).
        corner_entry_speed: Array, # float (N,)  ∈ [0, 1]
                                   #   Normalised speed recorded the frame the car
                                   #   entered the current corner tile.
        # ── crash signal ──────────────────────────────────────────────────
        tile_crash: Array,         # bool (N,)  car crashed on this tile;
                                   #   prevents checkpoint rewards on crash tile
        # ── optional / legacy ─────────────────────────────────────────────
        distances: Array | None = None,
        verbose=False,
        shape_pos: Array | None = None,   # float (N,) dense positive shaping (weights, pre-scaling)
        shape_neg: Array | None = None,   # float (N,) dense negative shaping
    ):
        # NOTE: ``corner_tile_next, vel_norm, phase_align, turning, braking, brake_steps,
        #       min_brake_steps, brake_check, straight_aligned, turn_align,
        #       corner_entry_speed, tile_crash, distances`` only feed reward terms that are
        #       currently disabled (see the commented block below). They are still accepted
        #       so the call signature is unchanged; to re-enable a term, add the array to
        #       ``_update`` and uncomment its scalar version there.
        if shape_pos is None: shape_pos = np.zeros(self.total, dtype=NP_FLOAT)
        if shape_neg is None: shape_neg = np.zeros(self.total, dtype=NP_FLOAT)
        eliminated = _update(
            self.lives, self.deaths, self.scores, self.true_scores,
            self.disqualified, self.completed, self.frames_done,
            self.fitness, self.prev_fitness,
            checked, finished, out_of_bounds, hiatus, moved_forward, tile_stag,
            vel_norm, shape_pos, shape_neg,
            corner_tile_curr, straight_tile_curr,
            on_road, turn_steps, straight_steps,
            self.lives_total, self.max_hiatus, int(min_turn_steps), int(min_straight_steps),
            bool(self.reward.shaping), float(self.reward.crash_penalty), float(self.reward_offset),
        )

        # # ------------------------------------------------------------------
        # # Original vectorised implementation (disabled reward terms kept for reference)
        # # ------------------------------------------------------------------
        # # ── 4b. Forward progress — STRAIGHT tile ──────────────────────────
        # fwd_straight = moved_forward & straight_tile_curr & on_road
        # self.fitness[fwd_straight] += (
        #     1.0
        #     + 2.5 * (vel_n * align_pos)[fwd_straight] # speed × heading  (anti-veer)
        #     + 1.0 * align_pos[fwd_straight]           # heading alone    (anti-veer)
        # ) * _lives[fwd_straight]
        #
        # # ── 4c. Forward progress — CORNER tile ────────────────────────────
        # # (i)  slow-in bonus: entry=0.0 → +3.0/frame   entry=1.0 → +1.0/frame
        # # (ii) turn-direction alignment: perfect turn → +3.0/frame   wrong-way → +0.0/frame
        # # (iii) heading alignment: lighter than on straights
        # fwd_corner = moved_forward & corner_tile_curr & on_road
        # entry_speed   = np.clip(corner_entry_speed, 0.0, 1.0)
        # turn_align_pos = np.clip(turn_align, 0.0, 1.0)  # wrong-way → 0, not penalty
        # self.fitness[fwd_corner] += (
        #     (2.0 - entry_speed[fwd_corner])        # (i)  slow-in bonus
        #     + 3.0 * turn_align_pos[fwd_corner]     # (ii) correct turn direction
        #     + 0.5 * align_pos[fwd_corner]          # (iii) heading arc bonus
        # )
        #
        # # ── 4d. Braking BEFORE corners ────────────────────────────────────
        # started_braking = (brake_steps == 1) & corner_tile_next & on_road
        # self.fitness[started_braking] += 50 * _lives[started_braking]
        # braking_lim_hit = (brake_steps == min_brake_steps) & corner_tile_next & on_road
        # self.fitness[braking_lim_hit] += 150 * _lives[braking_lim_hit]
        # # Part B — Corner-entry speed bonus (one-off at the moment of entry)
        # just_entered_corner = checked & corner_tile_curr & on_road
        # self.fitness[just_entered_corner] += (
        #     150.0 * (1.0 - entry_speed[just_entered_corner]) * _lives[just_entered_corner]
        # )
        #
        # # ── 4f. Not-moving-forward penalties ──────────────────────────────
        # self.fitness[moved_backward & straight_tile_curr] -= 10.0 * _deaths[moved_backward & straight_tile_curr]
        # self.fitness[moved_backward & corner_tile_curr] -= 5.0 * _deaths[moved_backward & corner_tile_curr]

        if verbose:
            breached_hiatus = hiatus >= self.max_hiatus
            for index, flag in enumerate(checked):
                if flag: print(f"Index {index} passed checkpoint")
            for index, flag in enumerate(finished):
                if flag: print(f"Index {index} completed a lap")
            for index, flag in enumerate(out_of_bounds):
                if flag: print(f"Index {index} went out of bounds")
            for index, flag in enumerate(breached_hiatus):
                if flag: print(f"Index {index} stagnated too long")

        return eliminated

    # ------------------------------------------------------------------
    # Selection / reporting
    # ------------------------------------------------------------------

    @property
    def best_index(self) -> int:
        total_score = self.true_scores.astype(NP_FLOAT) * (self.lives + 1)
        total_score[~self.active] = -np.inf
        return int(total_score.argmax())

    def get_reward(self) -> Array:
        """Delta fitness since the previous ``update()`` call, times ``reward_scale``."""
        return (self.fitness - self.prev_fitness) * self.reward_scale

    @property
    def scaled_fitness(self) -> Array:
        """Accumulated fitness in the same units the learner sees (fitness x reward_scale)."""
        return self.fitness * self.reward_scale

    def __str__(self) -> str:
        return (
            f"Players(players={self.total}, "
            f"active={self.active_total}, "
            f"best_index={self.best_index})"
        )