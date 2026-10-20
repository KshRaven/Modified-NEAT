"""
Race-mode example: starting grid, slipstream, collisions and a 3-2-1-GO! countdown.

    python race_demo.py                  # watch the bots race
    python race_demo.py --human          # drive the last car on the grid with WASD
    python race_demo.py --racers 8 --laps 2
    python race_demo.py --headless --max-frames 3000   # no window (CPU benchmark / smoke test)

All race settings live in .config/race.json under "race".
"""
import argparse
import os
import sys
import time

import numpy as np

EXMP_DIR = os.path.dirname(os.path.abspath(__file__))
if EXMP_DIR not in sys.path:
    sys.path.insert(0, EXMP_DIR)

from game import Game, EnvironmentConfig
from game.race import heading_to_angle, wrap180


class Autopilot:
    """Tiny pure-pursuit driver: aim a little way down the centre line, slow for corners."""

    def __init__(self, cars, seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.cars = cars
        n = cars.total
        self.skill = rng.uniform(0.86, 0.98, n)                    # fraction of top speed it aims for
        self.lane = rng.uniform(-0.20, 0.20, n) * cars.grid.cell_size * 0.5
        self.closing = np.zeros(n, dtype=bool)

    def _pass_offset(self, i: int) -> float:
        """Pull out sideways when a slower car is right ahead (so a tow can become an overtake),
        and lift a little when closing in too fast."""
        c = self.cars
        rad = np.radians(c.angle[i])
        fx, fy = -np.sin(rad), -np.cos(rad)
        dx, dy = c.x - c.x[i], c.y - c.y[i]
        ahead = dx * fx + dy * fy
        lat = dx * fy - dy * fx                      # > 0: other car is to my left
        near = (ahead > 0) & (ahead < 3.0 * c.height) & (np.abs(lat) < 1.3 * c.width) & c.players.active
        near[i] = False
        self.closing[i] = near.any() and ahead[near].min() < 1.6 * c.height
        if not near.any():
            return 0.0
        j = np.flatnonzero(near)[np.argmin(ahead[near])]
        side = -1.0 if lat[j] > 0 else 1.0           # go round the other side
        return side * 0.30 * c.grid.cell_size * 0.5 * 0.9

    def act(self) -> np.ndarray:
        c, tr = self.cars, self.cars.track
        out = np.zeros((c.total, 2))
        for i in range(c.total):
            tile = c.tile[i]
            d = tr.dist_in_tile(tile, c.x[i], c.y[i])
            speed = c.vel_lin[i]
            look = max(0.40 * tr.size, 9.0 * speed)
            lane = self.lane[i] + self._pass_offset(i)
            tx, ty, _ = tr.point(tile, d + look, lane)
            err = wrap180(heading_to_angle(tx - c.x[i], ty - c.y[i]) - c.angle[i])
            out[i, 1] = np.clip(0.05 * err - 0.35 * c.vel_ang[i], -1, 1)

            # Corner ahead within the braking window?
            nxt = tr.next[tile]
            corner_here = tile.type.name in ('LEFT', 'RIGHT')
            corner_next = nxt.type.name in ('LEFT', 'RIGHT')
            target = self.skill[i]
            if corner_here or (corner_next and d > 0.35 * tr.size):
                target = min(target, 0.52)
            # never lift while in a tow
            if c.draft_level[i] > 0.1:
                target = 1.0 + 0.12
            if self.closing[i] and speed > 0.4 * c.max_vel_lin:   # (never at a standstill: that costs lives)
                target = min(target, speed / c.max_vel_lin - 0.02)   # feather the throttle
            out[i, 0] = np.clip((target - speed / c.max_vel_lin) * 6.0, -1, 1)
        return out


def human_input(cars, actions):
    import pygame
    keys = pygame.key.get_pressed()
    me = cars.total - 1
    actions[me] = 0.0
    if keys[pygame.K_w]: actions[me, 0] = 1
    if keys[pygame.K_s]: actions[me, 0] = -1
    if keys[pygame.K_a]: actions[me, 1] = 1
    if keys[pygame.K_d]: actions[me, 1] = -1
    if cars.vel_lin[me] <= 0.0 and actions[me, 0] == 0:
        actions[me, 0] = 0.05   # tiny creep so standing still does not cost lives


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(EXMP_DIR, '.config', 'race.json'))
    ap.add_argument('--racers', type=int, default=7)
    ap.add_argument('--laps', type=int, default=None)
    ap.add_argument('--human', action='store_true')
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()

    cfg = EnvironmentConfig() # args.config)
    cfg.race.enabled = True
    if args.laps: cfg.min_laps = args.laps
    if args.max_frames: cfg.max_frames = args.max_frames
    if args.seed is not None: cfg.seed = args.seed
    if args.headless:
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        cfg.window.fps = None

    env = Game(render_mode=None if args.headless else 'human', config=cfg)
    env.cars.usr_engaged = args.human
    racers = args.racers or cfg.race.racers
    env.reset(keys=racers)             # should show the countdown when rendering in human mode
    pilot = Autopilot(env.cars, seed=cfg.seed)

    t0, frame, info = time.perf_counter(), 0, {}
    done = False
    while not done:
        actions = pilot.act()
        if args.human:
            human_input(env.cars, actions)
        _, _, done, _, info = env.step(actions)
        env.render()
        frame += 1

    dt = time.perf_counter() - t0
    order = env.cars.race_order
    print(f"\nFinished after {frame} frames ({frame / dt:.0f} fps simulated) - {info}")
    print("Final order:", ", ".join(f"P{p + 1}=car{c}" for p, c in enumerate(order)))
    env.window.close()


if __name__ == '__main__':
    main()