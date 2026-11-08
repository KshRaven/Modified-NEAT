"""Bird state, movement and sensing for every player (the `Cars` / `Snakes` of this environment)."""
from __future__ import annotations

import numpy as np
from numpy import ndarray as Array

from . import kernels
from .config import BirdConfig
from .constants import NP_FLOAT, NP_INT
from .kernels.sensors import MODE_BASIC, MODE_EGO, MODE_FULL
from .player import Players
from .reward import RewardShaper
from .world import World

__all__ = ["Birds"]


class Birds:
    """All birds as arrays. They share one world (the same pipes); each has its own flight, lives and score."""

    def __init__(
        self, world: World, players: Players, config: BirdConfig | None = None,
        shaper: RewardShaper | None = None, goal: int | None = None,
    ) -> None:
        self.world = world
        self.players = players
        self.config = config if config is not None else BirdConfig()
        self.shaper = shaper if shaper is not None else RewardShaper(players.reward, players.max_hiatus, world)
        self.goal = None if goal is None or int(goal) <= 0 else int(goal)
        self.bw, self.bh = (float(v) for v in self.config.bird_size)
        state_type = self.config.state_type
        if state_type == "auto":                            # the original game's state: 5, or 7 with moving pipes
            state_type = "full"
        if state_type not in ("basic", "full", "ego"):
            raise ValueError(f"state_type must be 'auto', 'ego', 'basic' or 'full', got {self.config.state_type!r}")
        self.state_type = state_type
        self.mode = {"basic": MODE_BASIC, "full": MODE_FULL, "ego": MODE_EGO}[state_type]
        self.features = {
            "basic": kernels.N_FEATURES_BASIC,
            "full": kernels.N_FEATURES_FULL_MOVING if world.moving else kernels.N_FEATURES_FULL,
            "ego": kernels.N_FEATURES_EGO,
        }[state_type]
        self.total = 0
        self.frame = 0
        self.labels: list[str] | None = None                # optional display name per bird (HUD)
        self.allocate(1)

    # ------------------------------------------------------------------
    # Allocation / reset
    # ------------------------------------------------------------------

    def allocate(self, total: int) -> None:
        n = (int(total),)
        self.total = int(total)
        self.x = np.zeros(n, dtype=NP_FLOAT)                # top-left of the sprite box
        self.y = np.zeros(n, dtype=NP_FLOAT)
        self.vel = np.zeros(n, dtype=NP_FLOAT)
        self.tick = np.zeros(n, dtype=NP_INT)               # frames since the last flap
        self.height = np.zeros(n, dtype=NP_FLOAT)           # y at the last flap
        self.tilt = np.zeros(n, dtype=NP_FLOAT)
        self.disp = np.zeros(n, dtype=NP_FLOAT)             # displacement of the last frame
        self.hiatus = np.zeros(n, dtype=NP_INT)             # frames since the last pipe
        self.action_count = np.zeros((self.total, 2), dtype=NP_INT)
        # Per-frame signals
        self.flap = np.zeros(n, dtype=bool)
        self.passed = np.zeros(n, dtype=bool)
        self.hit_pipe = np.zeros(n, dtype=bool)
        self.hit_bound = np.zeros(n, dtype=bool)
        self.completed = np.zeros(n, dtype=bool)
        self.low_usage = np.zeros(n, dtype=bool)
        self.err = np.zeros(n, dtype=NP_FLOAT)              # |bird centre - next gap centre|, pixels
        self.cy = np.zeros(n, dtype=NP_FLOAT)
        self.dist_x = 0.0                                   # distance from the birds to the next pipe's left edge

    def reset(self, total: int) -> None:
        """New episode with `total` birds."""
        if self.frame > 0:                                  # grade the finished episode: survive first, then be precise
            self.shaper.grade_episode(self.players.true_scores, self.players.deaths)
        self.world.reset()
        self.allocate(total)
        self.players.reset(total)
        self.shaper.allocate(total)                         # also picks the mode of this episode
        self.frame = 0
        self._spawn(np.ones(self.total, dtype=bool))
        self._measure()
        self.shaper.prev_err[:] = self.err

    def _spawn(self, mask: Array) -> None:
        c = self.config
        y0 = self.world.safe_y(self.bw, self.bh, c.init_y)  # a pipe may overlap the spawn point: start in its gap
        kernels.spawn(
            self.x, self.y, self.vel, self.tick, self.height, self.tilt, self.disp,
            self.hiatus, self.action_count, mask, float(c.init_x), float(y0),
        )

    def restart(self, mask: Array) -> None:
        """Respawn the masked birds."""
        if mask.any():
            self._spawn(mask)

    def _measure(self) -> None:
        w = self.world
        t = w.t1
        self.cy = self.y + 0.5 * self.bh
        self.err = np.abs(self.cy - 0.5 * (w.gap_top[t] + w.gap_bot[t]))
        self.dist_x = float(w.px[t] - (self.config.init_x + self.bw))

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def to_actions(self, actions: Array) -> Array:
        """Accept (N,) bools / ints (>= 1 flaps), (N,) or (N, 1) scores (flap above `action_threshold`) or (N, 2)
        scores (argmax) -> (N,) bool flap flags."""
        actions = np.asarray(actions)
        if actions.ndim == 2:
            if actions.shape[1] == 2:
                actions = actions.argmax(axis=-1)
            elif actions.shape[1] == 1:
                actions = actions[:, 0]
            else:
                raise ValueError(f"Expected action shape (N,), (N, 1), or (N, 2), got {actions.shape}")
        if actions.ndim != 1:
            raise ValueError(f"Unsupported action shape '{actions.shape}'")
        if not np.issubdtype(actions.dtype, np.number) and actions.dtype != bool:
            raise ValueError("Actions must be numeric")
        if not np.all(np.isfinite(actions)):
            raise ValueError("Actions must contain finite numeric values")
        if actions.dtype == bool or np.issubdtype(actions.dtype, np.integer):
            return actions >= 1
        return actions > float(self.config.action_threshold)

    def update(self, actions: Array, verbose: bool = False) -> Array:
        """Advance every bird one frame, update lives / fitness and respawn eliminated birds.

        Returns the mask of birds that lost a life this frame.
        """
        w, pl, c = self.world, self.players, self.config
        w.update_draft()                                    # may announce / start / end a draft for this frame
        flap = self.to_actions(actions)
        if flap.shape[0] != self.total:
            raise ValueError(f"Expected {self.total} actions, got {flap.shape[0]}")
        active = pl.active & ~pl.completed
        self.flap = flap & active

        kernels.step(
            self.y, self.vel, self.tick, self.height, self.tilt, self.disp, self.hiatus, self.action_count,
            flap, active, float(c.fall_coeff), float(c.terminal_vel), float(c.jump_vel), float(c.jump_offset),
            float(c.max_rotation), float(c.ang_vel), float(w.draft_y),
        )
        kernels.collide(
            self.x, self.y, self.bw, self.bh, float(c.hitbox_scale),
            w.px, w.gap_top, w.gap_bot, w.pw, w.floor_y, active, self.hit_pipe, self.hit_bound,
        )

        prev_target = int(w.gid[w.t1])
        crossed = w.step()                                  # pipes scroll, the target pipe may change
        crash = self.hit_pipe | self.hit_bound
        self.passed = active & ~crash & (crossed > 0)
        moved = active & (int(w.gid[w.t1]) != prev_target)   # target changed: no progress term this frame
        self.hiatus[self.passed] = 0
        if self.goal is not None:
            self.completed = active & ~crash & (pl.scores + self.passed >= self.goal)
        else:
            self.completed = np.zeros(self.total, dtype=bool)
        if c.var_thresh is not None:
            kernels.low_usage(self.action_count, float(c.var_thresh), 20, self.low_usage)
        else:
            self.low_usage[:] = False

        self._measure()
        died = crash | (self.hiatus >= pl.hiatus_limit)
        self.shaper.max_hiatus = pl.max_hiatus              # follows runtime changes of players.max_hiatus
        t = w.t1
        self.shaper.update(
            self.err, moved, died, self.hiatus, self.low_usage, active, self.flap, self.cy,
            float(w.gap_top[t]), float(w.gap_bot[t]), self.dist_x,
        )
        eliminated = pl.update(
            self.passed, self.hit_pipe, self.hit_bound, self.completed, self.hiatus,
            self.shaper.shape_pos, self.shaper.shape_neg, verbose=verbose, crash_scale=self.shaper.crash_scale,
        )

        # Respawn birds that still have lives, then measure from the new position
        respawn = eliminated & pl.active
        self.restart(respawn)
        pl.restart(respawn)
        self._measure()
        self.shaper.prev_err[:] = self.err
        self.frame += 1
        return eliminated

    # ------------------------------------------------------------------
    # Sensing
    # ------------------------------------------------------------------

    def get_state(self) -> Array:
        """State per bird, (N, features): see `kernels.sensors` for the layouts ('ego' 10, 'basic' 3, 'full' 5 / 7)."""
        w, c = self.world, self.config
        out = np.zeros((self.total, self.features), dtype=NP_FLOAT)
        kernels.get_state(
            out, self.mode, self.x, self.y, self.disp, self.tick, self.players.active,
            self.bw, self.bh, float(w.width), float(w.height), w.floor_y, float(c.terminal_vel),
            w.px, w.gap_top, w.gap_bot, w.vdir, w.pw, int(w.t1), int(w.t2), float(w.pipe_speed),
        )
        return out

    # ------------------------------------------------------------------
    # Accessors for rendering
    # ------------------------------------------------------------------

    def set_labels(self, labels: list[str] | None) -> None:
        self.labels = list(labels) if labels is not None else None

    def label(self, index: int) -> str | None:
        if self.labels is not None and 0 <= index < len(self.labels):
            return self.labels[index]
        return None

    def center(self, index: int) -> tuple[float, float]:
        return float(self.x[index] + 0.5 * self.bw), float(self.y[index] + 0.5 * self.bh)

    def ranking(self, count: int) -> list[int]:
        """Indices of the `count` best active birds (by score, then fitness)."""
        pl = self.players
        score = pl.scores.astype(NP_FLOAT) * 1e6 + pl.fitness
        score[~pl.active] = -np.inf
        order = np.argsort(-score)[:max(0, count)]
        return [int(i) for i in order if np.isfinite(score[i])]