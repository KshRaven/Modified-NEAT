"""The world: pipe ring, scroll distance and Env-private randomness. Shared by every bird (they all fly the same level)."""
from __future__ import annotations

import math

import numpy as np
from numpy import ndarray as Array

from . import kernels
from .config import BirdConfig, DraftConfig, WorldConfig
from .constants import NP_FLOAT, NP_INT

__all__ = ["World"]


class World:
    """Pipes scroll left at `velocity`; the birds keep their x. Coordinates are world pixels, y grows downwards.

    Pipe i has its left edge at `px[i]`, the gap spans `gap_top[i]` .. `gap_bot[i]`, and `vdir[i]` is its vertical
    direction (0 static, +1 down, -1 up). `t1` / `t2` are the indices of the next two pipes whose right edge has not yet cleared the birds (the pipe a bird is
    inside of stays the target until it is out).
    """

    def __init__(
        self, config: WorldConfig | None = None, bird: BirdConfig | None = None, draft: DraftConfig | None = None,
    ) -> None:
        self.config = config if config is not None else WorldConfig()
        self.bird = bird if bird is not None else BirdConfig()
        self.draft = draft if draft is not None else DraftConfig()
        c = self.config
        self.width, self.height = int(c.width), int(c.height)
        self.floor_y = float(self.height - c.floor)
        self.pw = float(c.pipe_width)
        self.gap_lo, self.gap_hi = int(min(c.gap_size)), int(max(c.gap_size))
        # Highest / lowest the gap may reach (offsets are (lower, upper))
        self.lo = float(c.gap_offset[1])
        self.hi = float(self.floor_y - c.gap_offset[0])
        if self.gap_lo < 1:
            raise ValueError(f"gap_size must be positive, got {c.gap_size}")
        if self.hi - self.lo < self.gap_hi:
            raise ValueError(
                f"gap_size {c.gap_size} does not fit between the offsets {c.gap_offset} in a {self.width}x{self.height} "
                f"world with floor {c.floor}"
            )
        self.rng = np.random.default_rng()      # Env-private RNG (pipes); seeded by Game.seed
        self.count = int(math.ceil(self.width / (self.pw + c.spawn_width))) + 3
        self.distance = 0.0                     # total scroll distance (floor / cloud animation)
        self.next_gid = 0
        self.crossed = 0                        # pipes whose pass point crossed the birds this frame
        self.draft_rng = np.random.default_rng()    # drafts have their own stream so they never change the pipe sequence
        self.reset_draft()
        self.allocate()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def bird_x(self) -> float:
        return float(self.bird.init_x)

    @property
    def pass_off(self) -> float:
        """Offset from a pipe's left edge to its pass point."""
        return self.pw * float(self.bird.pass_threshold)

    @property
    def pipe_speed(self) -> float:
        """Vertical pipe speed actually used: 0 when `pipe_movement` is off."""
        return float(self.config.pipe_speed_y) if self.config.pipe_movement else 0.0

    @property
    def moving(self) -> bool:
        """Can pipes move vertically at all?"""
        return self.pipe_speed > 0.0 and self.config.moving_chance > 0.0 and self.hi - self.lo > self.gap_lo

    # ------------------------------------------------------------------
    # Drafts (wind): idle -> warning -> blowing -> idle, never more than one at a time
    # ------------------------------------------------------------------

    IDLE, WARN, BLOW = 0, 1, 2
    AXIS_NONE, AXIS_X, AXIS_Y = 0, 1, 2

    def reset_draft(self) -> None:
        self.draft_phase = self.IDLE
        self.draft_axis = self.AXIS_NONE
        self.draft_dir = 1              # x: +1 forward / -1 backward;  y: +1 up / -1 down
        self.draft_timer = int(self.draft.draft_cooldown)
        self.draft_total = 0            # length of the current phase, for the warning progress
        self.draft_clock = 0            # animation clock of the wind streaks

    @property
    def draft_x(self) -> float:
        """Relative change of the horizontal displacement right now (+0.3 = 30 % more ground per frame)."""
        if self.draft_phase == self.BLOW and self.draft_axis == self.AXIS_X:
            return self.draft_dir * float(self.draft.draft_x_strength)
        return 0.0

    @property
    def draft_y(self) -> float:
        """Relative change of the vertical displacement right now (+ = up: lift, - = down: push)."""
        if self.draft_phase == self.BLOW and self.draft_axis == self.AXIS_Y:
            return self.draft_dir * float(self.draft.draft_y_strength)
        return 0.0

    @property
    def draft_progress(self) -> float:
        """0 -> 1 while a draft is announced (how close it is to blowing); 1 while it blows; 0 when idle."""
        if self.draft_phase == self.BLOW:
            return 1.0
        if self.draft_phase == self.WARN:
            return 1.0 - self.draft_timer / max(self.draft_total, 1)
        return 0.0

    def _start_blow(self) -> None:
        lo, hi = (int(v) for v in self.draft.draft_duration)
        lo, hi = max(1, min(lo, hi)), max(1, max(lo, hi))
        self.draft_phase = self.BLOW
        self.draft_total = self.draft_timer = int(self.draft_rng.integers(lo, hi + 1))

    def update_draft(self) -> None:
        """Advance the draft state machine by one frame (called once per step, before the birds move)."""
        d = self.draft
        if not d.enabled:
            return
        if self.draft_phase != self.IDLE:
            self.draft_clock += 1
        if self.draft_phase == self.IDLE:
            self.draft_timer -= 1
            if self.draft_timer <= 0 and self.draft_rng.random() < d.draft_chance:
                axes = ([self.AXIS_X] if d.draft_x else []) + ([self.AXIS_Y] if d.draft_y else [])
                self.draft_axis = int(axes[int(self.draft_rng.integers(len(axes)))])    # only one kind at a time
                self.draft_dir = 1 if self.draft_rng.random() < 0.5 else -1
                self.draft_clock = 0
                if d.draft_warning > 0:
                    self.draft_phase = self.WARN
                    self.draft_total = self.draft_timer = int(d.draft_warning)
                else:
                    self._start_blow()
        elif self.draft_phase == self.WARN:
            self.draft_timer -= 1
            if self.draft_timer <= 0:
                self._start_blow()
        else:                                                   # blowing
            self.draft_timer -= 1
            if self.draft_timer <= 0:
                self.draft_phase = self.IDLE
                self.draft_axis = self.AXIS_NONE
                self.draft_timer = int(d.draft_cooldown)

    @property
    def draft_label(self) -> str:
        """Short HUD text of the current draft ('' when calm)."""
        if self.draft_phase == self.IDLE:
            return ""
        if self.draft_axis == self.AXIS_X:
            arrow = "->" if self.draft_dir > 0 else "<-"
            name = "TAILWIND" if self.draft_dir > 0 else "HEADWIND"
        else:
            arrow = "^" if self.draft_dir > 0 else "v"
            name = "UPDRAFT" if self.draft_dir > 0 else "DOWNDRAFT"
        return f"{'INCOMING ' if self.draft_phase == self.WARN else ''}{name} {arrow}"

    @property
    def pipe_spacing(self) -> float:
        """Horizontal distance between two pipes (left edge to left edge)."""
        return self.pw + self.config.spawn_width

    @property
    def gap_center(self) -> Array:
        return 0.5 * (self.gap_top + self.gap_bot)

    # ------------------------------------------------------------------
    # Allocation / reset
    # ------------------------------------------------------------------

    def allocate(self) -> None:
        n = self.count
        self.px = np.zeros(n, dtype=NP_FLOAT)
        self.gap_top = np.zeros(n, dtype=NP_FLOAT)
        self.gap_bot = np.zeros(n, dtype=NP_FLOAT)
        self.vdir = np.zeros(n, dtype=NP_INT)
        self.gid = np.zeros(n, dtype=NP_INT)
        self.t1, self.t2 = 0, 0

    def reset(self) -> None:
        """New level: pipes from the Env-private generator."""
        c = self.config
        self.allocate()
        first = c.first_pipe_x if c.first_pipe_x is not None else 3.0 * self.bird_x + c.spawn_width
        self.next_gid = kernels.init_pipes(
            self.px, self.gap_top, self.gap_bot, self.vdir, self.gid, self.rng.random((self.count, 4)),
            float(first), self.pw, float(c.spawn_width), self.lo, self.hi, self.gap_lo, self.gap_hi,
            self.pipe_speed, float(c.moving_chance),
        )
        self.reset_draft()
        if self.draft.enabled:                                  # own stream: pipes stay identical with or without drafts
            self.draft_rng = np.random.default_rng(int(self.rng.integers(2 ** 32)))
        self.distance = 0.0
        self.crossed = 0
        self.t1, self.t2 = kernels.targets(self.px, self.pw, self.bird_x)

    def step(self) -> int:
        """Scroll one frame. Returns how many pipes the birds passed."""
        c = self.config
        vel = float(c.velocity) * (1.0 + self.draft_x)       # x draft: more / less ground covered per frame
        self.crossed, self.next_gid = kernels.scroll(
            self.px, self.gap_top, self.gap_bot, self.vdir, self.gid, self.rng.random((self.count, 4)),
            vel, self.pipe_speed, self.pw, float(c.spawn_width), self.lo, self.hi,
            self.gap_lo, self.gap_hi, float(c.moving_chance), int(self.next_gid), self.pass_off, self.bird_x,
        )
        self.distance += vel
        self.t1, self.t2 = kernels.targets(self.px, self.pw, self.bird_x)
        return int(self.crossed)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def safe_y(self, bw: float, bh: float, default_y: float, margin: float = 24.0) -> float:
        """Top edge for a (re)spawning bird: the default height, or the gap centre of a pipe that overlaps its x."""
        x0, x1 = self.bird_x - margin, self.bird_x + bw + margin
        for k in np.argsort(self.px):
            if self.px[k] < x1 and self.px[k] + self.pw > x0:
                return float(0.5 * (self.gap_top[k] + self.gap_bot[k]) - 0.5 * bh)
        return float(default_y)

    def order(self) -> Array:
        """Pipe indices from left to right."""
        return np.argsort(self.px)