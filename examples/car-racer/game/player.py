from .constants import INT, BOOL, FLOAT, MASK_1D, COUNTER_1D, ARRAY_1D

from numba.experimental import jitclass
from numpy import ndarray as Array

import numpy as np


INF_MIN = FLOAT(-np.finfo(np.float32).max.item())


@jitclass([
    ('total', INT),
    ('lives_total', INT),
    ('lives', COUNTER_1D),
    ('deaths', COUNTER_1D),
    ('scores', COUNTER_1D),
    ('true_scores', COUNTER_1D),
    ('disqualified', MASK_1D),
    ('completed', MASK_1D),
    ('frames_done', COUNTER_1D),
    ('fitness', ARRAY_1D),
    ('prev_fitness', ARRAY_1D),
    ('max_hiatus', INT),
])
class Players(object):
    def __init__(self, lives=3, max_hiatus=100):
        self.total = 1
        self.lives_total = lives
        self.max_hiatus = max_hiatus

        self.lives          = np.full((self.total,), self.lives_total, dtype=INT)
        self.deaths         = np.full((self.total,), 0, dtype=INT)
        self.scores         = np.full((self.total,), 0, dtype=INT)
        self.true_scores    = np.full((self.total,), 0, dtype=INT)
        self.disqualified   = np.full((self.total,), False, dtype=BOOL)
        self.completed      = np.full((self.total,), False, dtype=BOOL)
        self.frames_done    = np.full((self.total,), 0, dtype=INT)
        self.fitness        = np.full((self.total,), 0, dtype=FLOAT)
        self.prev_fitness   = np.full((self.total,), 0, dtype=FLOAT)

    def reset(self, total: int):
        self.total = total

        self.lives          = np.full((self.total,), self.lives_total, dtype=INT)
        self.deaths         = np.full((self.total,), 0, dtype=INT)
        self.scores         = np.full((self.total,), 0, dtype=INT)
        self.true_scores    = np.full((self.total,), 0, dtype=INT)
        self.disqualified   = np.full((self.total,), False, dtype=BOOL)
        self.completed      = np.full((self.total,), False, dtype=BOOL)
        self.frames_done    = np.full((self.total,), 0, dtype=INT)
        self.fitness        = np.full((self.total,), 0, dtype=FLOAT)
        self.prev_fitness   = np.full((self.total,), 0, dtype=FLOAT)

    def restart(self):
        self.disqualified[self.disqualified] = False

    @property
    def active(self):
        return self.lives > 0

    @property
    def active_total(self):
        return np.count_nonzero(self.active).item()

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
        # ── NEW: turn-direction and corner-entry signals ───────────────────
        turn_align: Array,         # float (N,)  ∈ [-1, +1]
                                   #   How correctly the car is steering on a corner tile.
                                   #   Computed in car.py as dot(angular_vel_sign, turn_sign),
                                   #   where turn_sign = -1 for LEFT, +1 for RIGHT.
                                   #   +1 = turning hard the right way
                                   #    0 = not steering at all
                                   #   -1 = actively steering the wrong way
                                   #   Pass zeros for non-corner tiles (unused there).
        corner_entry_speed: Array, # float (N,)  ∈ [0, 1]
                                   #   Normalised speed recorded the frame the car
                                   #   entered the current corner tile.  This is
                                   #   fixed for the duration of the tile transit so
                                   #   the one-off entry bonus (4d-B) has a stable target.
                                   #   Pass current vel_norm for non-corner tiles.
        # ── optional / legacy ─────────────────────────────────────────────
        distances: Array | None = None,
        verbose=False
    ):
        # ------------------------------------------------------------------
        # 0.  Derived masks
        # ------------------------------------------------------------------
        breached_hiatus = hiatus >= self.max_hiatus
        moved_backward  = ~moved_forward
        eliminated      = out_of_bounds | breached_hiatus | tile_stag | moved_backward
        straight_tile_curr = ~corner_tile_curr
        straight_tile_next = ~corner_tile_next

        # On a straight whose next tile is a corner: ideal place to brake
        approaching_corner = corner_tile_next & straight_tile_curr

        # ------------------------------------------------------------------
        # 1.  State / counter updates
        # ------------------------------------------------------------------
        self.disqualified[eliminated] = True
        self.completed[finished]      |= True
        self.lives[eliminated]        -= 1
        self.lives                    = np.clip(self.lives, 0, self.lives_total)
        self.deaths[eliminated]       += 1

        self.scores[checked]  += 1
        self.scores[finished] += 10
        self.scores[eliminated] -= 1
        better = self.scores > self.true_scores
        self.true_scores[better] = self.scores[better]

        # ------------------------------------------------------------------
        # 2.  Convenience arrays
        # ------------------------------------------------------------------
        _lives  = self.lives  + 1 # ≥ 1; scales rewards (more lives → more reward)
        _deaths = self.deaths + 1 # ≥ 1; scales penalties (more deaths → heavier hit)
        off_road = ~on_road
        multiplier = np.full_like(on_road, 2, INT)
        _lives[on_road] *= multiplier[on_road]
        _deaths[off_road] *= multiplier[off_road]

        # Clamp continuous signals to their valid range
        vel_n     = np.clip(vel_norm,    0.0, 1.0)   # [0, 1]
        align_pos = np.clip(phase_align, 0.0, 1.0)   # [0, 1]  positive-only heading

        # ------------------------------------------------------------------
        # 3.  Snapshot prev-fitness, reset on lap completion
        # ------------------------------------------------------------------
        # self.fitness[finished] = 0. # TODO: Might be ruining fitness rankings hence disabled
        self.prev_fitness      = self.fitness.copy()

        # ==================================================================
        # 4.  REWARD SHAPING
        # ==================================================================

        # ── 4a. Road-keeping reward/penalty ───────────────────────────────
        #
        # New asymmetric scheme:
        #   ON road:  +3/frame * lives   — grows with survival (good drivers richer)
        #   OFF road: -15/frame * deaths — escalates for repeat offenders
        #
        # The 5:1 penalty ratio vs reward means a car cannot recoup losses by
        # driving slightly faster on grass — it simply must stay on the road.
        self.fitness[on_road]  += 20.0 * _lives[on_road]
        self.fitness[off_road] -= 10.0 * _deaths[off_road]

        # # ── 4b. Forward progress — STRAIGHT tile ──────────────────────────
        # #
        # # Reward = base  +  speed×alignment  +  alignment-only
        # #
        # # The speed×alignment term is the key anti-veer signal:
        # #   • Perfect heading, full speed    → +4.5 /frame
        # #   • 45° off heading, full speed    → ≈+3.2 /frame  (0.71 alignment)
        # #   • 90° off heading (perpendicular)→ +1.0 /frame  (0 alignment)
        # # A car that swerves to "sniff" a corner that doesn't exist loses
        # # the speed bonus proportional to how far it drifts off-axis.
        # fwd_straight = moved_forward & straight_tile_curr & on_road
        # self.fitness[fwd_straight] += (
        #     1.0
        #     + 2.5 * (vel_n * align_pos)[fwd_straight] # speed × heading  (anti-veer)
        #     + 1.0 * align_pos[fwd_straight]           # heading alone    (anti-veer)
        # ) * _lives[fwd_straight]

        # ── 4c. Forward progress — CORNER tile ────────────────────────────
        #
        # Three orthogonal sub-signals replace the previous single formula.
        #
        # (i)  Speed-in-corner penalty using ENTRY speed, not instantaneous.
        #      Using entry speed means the gradient is fully felt even after
        #      the car has already braked inside the corner.
        #      entry=0.0 → +3.0/frame   entry=1.0 → +1.0/frame
        #
        # (ii) Turn-direction alignment — did the car actually steer the
        #      correct way? turn_align is +1 when steering matches the
        #      corner's required direction, -1 when steering the wrong way.
        #      Clamped to [0,1] so wrong-direction gives zero (not negative),
        #      making this a pure reward signal, not a double-punishment.
        #      perfect turn → +3.0/frame   wrong-way turn → +0.0/frame
        #
        # (iii) Heading alignment — still aligned with the track arc?
        #       Lighter than on straights since mid-corner heading naturally
        #       deviates from the tile's base rotation angle.
        # fwd_corner = moved_forward & corner_tile_curr & on_road
        entry_speed   = np.clip(corner_entry_speed, 0.0, 1.0)
        # turn_align_pos = np.clip(turn_align, 0.0, 1.0)  # wrong-way → 0, not penalty

        # self.fitness[fwd_corner] += (
        #     (2.0 - entry_speed[fwd_corner])        # (i)  slow-in bonus
        #     + 3.0 * turn_align_pos[fwd_corner]            # (ii) correct turn direction
        #     + 0.5 * align_pos[fwd_corner]                 # (iii) heading arc bonus
        # )

        # ── 4d. Braking BEFORE corners (two complementary parts) ──────────
        #
        # Part A — Braking *action* on the approach straight.
        #   Provides a per-frame gradient so the car learns that pressing
        #   brake on the straight before a corner is good, especially when
        #   still fast.  Halved from the old 40 → 20 because Part B below
        #   carries the heavier signal.
        #   braking + vel_n=0.0 → +20  braking + vel_n=1.0 → +0
        # brake_action = approaching_corner & braking & on_road
        # self.fitness[brake_action] += 20.0 * (1.0 - vel_n)[brake_action]
        started_braking = (brake_steps == 1) & corner_tile_next & on_road
        self.fitness[started_braking] += 1500 * _lives[started_braking]
        braking_lim_hit = (brake_steps == min_brake_steps) & corner_tile_next & on_road
        self.fitness[braking_lim_hit] += 3000 * _lives[braking_lim_hit]

        # # Part B — Corner-entry speed bonus (one-off at the moment of entry).
        # #   `checked & corner_tile` fires exactly once as the car crosses
        # #   into a new corner tile.  The bonus is large and inversely scaled
        # #   by entry speed, creating a clear, unambiguous signal:
        # #     entry at  0% speed → +150 * lives
        # #     entry at 50% speed →  +75 * lives
        # #     entry at 100% speed →  +0
        # #   This closes the temporal credit-assignment gap: the car receives
        # #   a big reward *at* the corner for the braking it did *before* it.
        # just_entered_corner = checked & corner_tile_curr & on_road
        # self.fitness[just_entered_corner] += (
        #     150.0 * (1.0 - entry_speed[just_entered_corner])
        #     * _lives[just_entered_corner]
        # )

        # ── 4e. Turning into corners ─────────────────────────────────────
        started_turning = (turn_steps == 1) # & (corner_tile_next | corner_tile_curr)
        self.fitness[started_turning] += 500 * _lives[started_turning]
        turning_lim_hit = (turn_steps == min_turn_steps) # & (corner_tile_next | corner_tile_curr)
        self.fitness[turning_lim_hit] += 1500 * _lives[turning_lim_hit]

        # # ── 4f. Not-moving-forward penalties ──────────────────────────────
        # #
        # # Straight is punished more harshly because going backwards on a
        # # straight is a clear failure.  Corner is slightly lighter because
        # # a wide exit can look like "backward" near the tile boundary.
        # # Only really applies when players are not reset on the instance they move incorrectly
        # self.fitness[moved_backward & straight_tile_curr] -= 10.0 * _deaths[moved_backward & straight_tile_curr]
        # self.fitness[moved_backward & corner_tile_curr] -= 5.0 * _deaths[moved_backward & corner_tile_curr]

        # ── 4g. Checkpoint and lap rewards ────────────────────────────────
        #
        # Corner checkpoints worth 50% more to reward clean cornering.
        self.fitness[checked & straight_tile_curr] += 100.0 * _lives[checked & straight_tile_curr]
        self.fitness[checked & corner_tile_curr] += 1000.0 * _lives[checked & corner_tile_curr]
        self.fitness[finished] += 10000.0 * _lives[finished]

        # ── 4h. Elimination (death) one-off penalty ───────────────────────
        self.fitness[eliminated & straight_tile_curr] -= 500.0 * _deaths[eliminated & straight_tile_curr]
        self.fitness[eliminated & corner_tile_curr] -= 500.0 * _deaths[eliminated & corner_tile_curr]

        # ------------------------------------------------------------------
        # 5.  Housekeeping
        # ------------------------------------------------------------------
        self.frames_done += 1

        if verbose:
            for index, flag in enumerate(checked):
                if flag: print(f"Index {index} passed checkpoint")
            for index, flag in enumerate(finished):
                if flag: print(f"Index {index} completed a lap")
            for index, flag in enumerate(out_of_bounds):
                if flag: print(f"Index {index} went out of bounds")
            for index, flag in enumerate(breached_hiatus):
                if flag: print(f"Index {index} stagnated too long")

    # ------------------------------------------------------------------
    # Selection / reporting
    # ------------------------------------------------------------------

    @property
    def best_index(self):
        total_score = self.true_scores * (self.lives + 1)
        total_score[~self.active] = -np.inf
        return total_score.argmax().item()

    def __str__(self):
        return (
            f"Players(players={self.total}, "
            f"active={self.active_total}, "
            f"best_index={self.best_index})"
        )

    def get_reward(self):
        return self.fitness - self.prev_fitness


PLAYERS_TYPE = Players.class_type.instance_type
