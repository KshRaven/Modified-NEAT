import numpy as np
from numpy import ndarray as Array

from .config import RewardConfig
from .constants import NP_INT, NP_FLOAT


class Players:
    """Vectorized lives/score/fitness bookkeeping for every genome/player."""

    def __init__(self, lives: int = 3, max_hiatus: int = 150, reward_config: RewardConfig | None = None):
        self.total = 1
        self.lives_total = lives
        self.max_hiatus = max_hiatus

        if reward_config is None:
            reward_config = RewardConfig()
        self.reward_config = reward_config

        self.lives: Array = np.full((self.total,), self.lives_total, dtype=NP_INT)
        self.deaths: Array = np.full((self.total,), 0, dtype=NP_INT)
        self.scores: Array = np.full((self.total,), 0, dtype=NP_INT)
        self.true_scores: Array = np.full((self.total,), 0, dtype=NP_INT)
        self.disqualified: Array = np.full((self.total,), False, dtype=bool)
        self.completed: Array = np.full((self.total,), False, dtype=bool)
        self.frames_done: Array = np.full((self.total,), 0, dtype=NP_INT)
        self.fitness: Array = np.full((self.total,), 0.0, dtype=NP_FLOAT)
        self.prev_fitness: Array = np.full((self.total,), 0.0, dtype=NP_FLOAT)
        self.hiatus: Array = np.full((self.total,), 0, dtype=NP_INT)
        self.best_dist: Array = np.full((self.total,), np.inf, dtype=NP_FLOAT)

    def reset(self, total: int, start_dist: float):
        self.total = total
        self.lives = np.full((self.total,), self.lives_total, dtype=NP_INT)
        self.deaths = np.full((self.total,), 0, dtype=NP_INT)
        self.scores = np.full((self.total,), 0, dtype=NP_INT)
        self.true_scores = np.full((self.total,), 0, dtype=NP_INT)
        self.disqualified = np.full((self.total,), False, dtype=bool)
        self.completed = np.full((self.total,), False, dtype=bool)
        self.frames_done = np.full((self.total,), 0, dtype=NP_INT)
        self.fitness = np.full((self.total,), 0.0, dtype=NP_FLOAT)
        self.prev_fitness = np.full((self.total,), 0.0, dtype=NP_FLOAT)
        self.hiatus = np.full((self.total,), 0, dtype=NP_INT)
        self.best_dist = np.full((self.total,), float(start_dist), dtype=NP_FLOAT)

    def restart(self):
        """Clear the per-life disqualified flag for genomes that still have lives left
        (used when reset_on_death respawns a genome instead of ending its episode).
        Genomes that already completed the maze are never un-disqualified this way —
        reaching the goal is permanent, not a one-frame pause like a trap respawn."""
        self.disqualified[self.disqualified & (self.lives > 0) & (~self.completed)] = False

    @property
    def active(self) -> Array:
        """A player can act iff it still has lives left AND hasn't been
        disqualified (this includes genomes that already reached the goal —
        they stop moving but keep receiving bookkeeping/reward updates)."""
        return (self.lives > 0) & (~self.disqualified)

    @property
    def active_total(self) -> int:
        return int(np.count_nonzero(self.lives > 0))

    def update(
        self,
        dist_to_goal: Array,       # float (N,) current BFS distance to goal, in cells
        reached_goal: Array,       # bool  (N,) stepped onto the goal cell this step
        hit_trap: Array,           # bool  (N,) stepped onto a trap cell this step
        bumped_wall: Array,        # bool  (N,) attempted a move blocked by a wall
        moved: Array,              # bool  (N,) actually changed cell this step
        reached_checkpoint: Array | None = None,  # bool (N,) touched a NEW checkpoint cell this step
    ):
        """Orchestrates one step of bookkeeping: lives/deaths/disqualification/stagnation
        first (fitness math depends on the *pre-update* lives/deaths counts), then the
        fitness signal itself, then score bookkeeping."""
        self.prev_fitness = self.fitness.copy()

        if reached_checkpoint is None:
            reached_checkpoint = np.full((self.total,), False, dtype=bool)

        # Genomes that reached the goal on a PREVIOUS step are just "holding":
        # they no longer move, and get a flat reward instead of the usual
        # per-step math (see _update_fitness).
        holding = self.completed & ~reached_goal

        improved = dist_to_goal < self.best_dist
        stagnated = self._update_lives_and_status(dist_to_goal, improved, reached_goal, hit_trap)
        self._update_fitness(
            dist_to_goal, improved, stagnated, reached_goal, hit_trap, bumped_wall,
            reached_checkpoint, holding,
        )
        self._update_scores(reached_goal, hit_trap)

        self.best_dist = np.minimum(self.best_dist, dist_to_goal)
        self.frames_done += 1

    # ------------------------------------------------------------------
    # Lives / deaths / disqualification (non-fitness state)
    # ------------------------------------------------------------------

    def _update_lives_and_status(
        self, dist_to_goal: Array, improved: Array, reached_goal: Array, hit_trap: Array,
    ) -> Array:
        """Updates hiatus/lives/deaths/disqualified/completed. Returns the `stagnated`
        mask so _update_fitness can penalize it without recomputing it."""
        self.hiatus[improved] = 0
        self.hiatus[~improved] += 1
        stagnated = self.hiatus >= self.max_hiatus
        self.disqualified[stagnated] = True
        self.hiatus[stagnated] = 0

        self.lives[hit_trap] -= 1
        self.lives = np.clip(self.lives, 0, self.lives_total)
        self.deaths[hit_trap] += 1
        self.disqualified[hit_trap & (self.lives <= 0)] = True

        self.completed[reached_goal] = True
        self.disqualified[reached_goal] = True

        return stagnated

    # ------------------------------------------------------------------
    # Fitness (the RL reward signal) — isolated from score/lives bookkeeping
    # ------------------------------------------------------------------

    def _update_fitness(
        self,
        dist_to_goal: Array, improved: Array, stagnated: Array,
        reached_goal: Array, hit_trap: Array, bumped_wall: Array,
        reached_checkpoint: Array, holding: Array,
    ) -> None:
        rc = self.reward_config
        _lives = self.lives + 1     # >=1, scales rewards
        _deaths = self.deaths + 1   # >=1, scales penalties

        # Players that are just "holding" after an earlier goal completion skip
        # every step-based term below (progress/step/wall/stagnation/trap) —
        # none of it applies since they can't move anymore — and instead get a
        # flat `goal_hold_reward` every step. This keeps early finishers from
        # bleeding fitness (via step_penalty etc.) while later finishers rack up
        # fewer such idle frames, which would otherwise give late finishers an
        # unfair edge once rewards are propagated backwards with a gamma factor.
        active_calc = ~holding

        # ── 1. Potential-based progress reward: getting closer to the goal than
        #        ever before is rewarded once per new best distance. ──────────
        prog_mask = improved & active_calc
        self.fitness[prog_mask] += (
            rc.progress_multiplier * _lives[prog_mask] * (self.best_dist[prog_mask] - dist_to_goal[prog_mask])
        )

        # ── 2. Small per-step penalty (encourages speed) ───────────────────
        self.fitness[active_calc] -= rc.step_penalty

        # ── 3. Wall-bump penalty ─────────────────────────────────────────
        self.fitness[bumped_wall & active_calc] -= rc.wall_bump_penalty

        # ── 4. Stagnation penalty ───────────────────────────────────────
        self.fitness[stagnated & active_calc] -= rc.stagnation_penalty

        # ── 5. Trap penalty ──────────────────────────────────────────────
        trap_mask = hit_trap & active_calc
        self.fitness[trap_mask] -= rc.trap_penalty * _deaths[trap_mask]

        # ── 6. Checkpoint reward — given once per genome, the first time it
        #        touches each checkpoint cell along the best path. ──────────
        chk_mask = reached_checkpoint & active_calc
        self.fitness[chk_mask] += rc.checkpoint_reward * _lives[chk_mask]

        # ── 7. Goal reward — scaled by lives remaining and speed (fewer
        #        frames used = more reward). Given exactly once, on the step
        #        the genome steps onto the goal cell. ───────────────────────
        time_bonus = np.clip(
            1.0 - (self.frames_done.astype(NP_FLOAT) / rc.goal_time_bonus_frame_norm),
            rc.goal_time_bonus_min, rc.goal_time_bonus_max,
        )
        self.fitness[reached_goal] += (
            (rc.goal_base_reward + rc.goal_time_bonus_weight * time_bonus[reached_goal]) * _lives[reached_goal]
        )

        # ── 8. Goal-hold reward — every step after that, in place of 1-6. ────
        self.fitness[holding] += rc.goal_hold_reward

    # ------------------------------------------------------------------
    # Score bookkeeping (display/UX score, independent of fitness)
    # ------------------------------------------------------------------

    def _update_scores(self, reached_goal: Array, hit_trap: Array) -> None:
        rc = self.reward_config
        self.scores[reached_goal] += rc.score_goal
        self.scores[hit_trap] += rc.score_trap
        better = self.scores > self.true_scores
        self.true_scores[better] = self.scores[better]

    @property
    def best_index(self) -> int:
        total_score = self.true_scores.astype(NP_FLOAT) * (self.lives + 1)
        total_score = np.where(self.lives > 0, total_score, -np.inf)
        return int(total_score.argmax())

    def ranking(self) -> Array:
        """Indices sorted by fitness, descending (top-3 used for visual highlighting)."""
        return np.argsort(-self.fitness)

    def get_reward(self) -> Array:
        return self.fitness - self.prev_fitness

    def __str__(self):
        return (
            f"Players(players={self.total}, active={self.active_total}, "
            f"best_index={self.best_index})"
        )
