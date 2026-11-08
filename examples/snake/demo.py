"""Rule-based Snake autopilot demo; does not depend on ModifiedNEAT.

The bot plans toward food with a breadth-first search that respects walls/wrap mode and the
current body occupancy. It is a demo controller, not a learned policy.

Examples:
    python demo.py
    python demo.py --snakes 8 --wrap --size 25 --seed 7
    python demo.py --headless --max-frames 2000
"""
from __future__ import annotations

import argparse
from collections import deque
import os
import sys
import time

import numpy as np

EXMP_DIR = os.path.dirname(os.path.abspath(__file__))
if EXMP_DIR not in sys.path:
    sys.path.insert(0, EXMP_DIR)

from game import EnvironmentConfig, Game
from game.constants import DIR_DX, DIR_DY


class SnakeAutopilot:
    """Small per-snake food-seeking controller using the real environment state."""

    def __init__(self, env: Game) -> None:
        self.env = env

    @staticmethod
    def _advance(pos: tuple[int, int], direction: int, width: int, height: int, wrap: bool):
        x = pos[0] + int(DIR_DX[direction])
        y = pos[1] + int(DIR_DY[direction])
        if wrap:
            return x % width, y % height
        if x < 0 or y < 0 or x >= width or y >= height:
            return None
        return x, y

    @staticmethod
    def _food_distance(a: tuple[int, int], b: tuple[int, int], width: int, height: int, wrap: bool) -> int:
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        if wrap:
            dx, dy = min(dx, width - dx), min(dy, height - dy)
        return dx + dy

    def _route_distance(
        self,
        start: tuple[int, int],
        target: tuple[int, int],
        heading: int,
        blocked: set[tuple[int, int]],
    ) -> int | None:
        """Shortest legal relative-turn route on the currently free cells; None means unreachable."""
        if start == target:
            return 0
        env = self.env
        g = env.grid
        queue = deque([(start[0], start[1], heading, 0)])
        seen = {(start[0], start[1], heading)}
        while queue:
            x, y, direction, dist = queue.popleft()
            for action in (0, 1, 2):
                # Match kernels.movement.step: left=0, straight=1, right=2.
                next_direction = (direction + 1 - action) % 4
                nxt = self._advance((x, y), next_direction, g.width, g.height, g.wrap)
                if nxt is None:
                    continue
                if nxt == target:
                    return dist + 1
                if nxt in blocked:
                    continue
                state = (nxt[0], nxt[1], next_direction)
                if state not in seen:
                    seen.add(state)
                    queue.append((nxt[0], nxt[1], next_direction, dist + 1))
        return None

    def _choose_one(self, index: int) -> int:
        env, sn, g = self.env, self.env.snakes, self.env.grid
        head = sn.head(index)
        heading = int(sn.direction[index])
        body = sn.body(index)
        tail = tuple(map(int, body[-1]))
        # The tail vacates its cell on a non-food move, so that cell is legal as the next step.
        blocked = {tuple(map(int, cell)) for cell in body[:-1]}
        fx, fy = int(g.food_x[index]), int(g.food_y[index])
        target = (fx, fy) if fx >= 0 and fy >= 0 else None

        candidates = []
        for action in (0, 1, 2):
            direction = (heading + 1 - action) % 4
            nxt = self._advance(head, direction, g.width, g.height, g.wrap)
            if nxt is None or nxt in blocked:
                continue

            # Count next-step options as a safety tie-breaker; the route itself uses legal relative turns.
            open_neighbours = 0
            for action2 in (0, 1, 2):
                d2 = (direction + 1 - action2) % 4
                pos2 = self._advance(nxt, d2, g.width, g.height, g.wrap)
                if pos2 is not None and (pos2 not in blocked or pos2 == target):
                    open_neighbours += 1

            if target is None:
                # A full board is normally terminal. Favor the move with the most room if reached here.
                rank = (0, -open_neighbours, 0 if action == 1 else 1, action)
            else:
                route = self._route_distance(nxt, target, direction, blocked)
                if route is None:
                    route = 10_000 + self._food_distance(nxt, target, g.width, g.height, g.wrap)
                rank = (route, -open_neighbours, 0 if action == 1 else 1, action)
            candidates.append((rank, action))

        if not candidates:
            # If no safe action exists, continuing straight is deterministic; collision is unavoidable.
            return 1
        if target is None:
            return min(candidates, key=lambda item: item[0])[1]
        return min(candidates, key=lambda item: item[0])[1]

    def act(self) -> np.ndarray:
        actions = np.ones(self.env.snakes.total, dtype=np.int64)  # straight for inactive slots
        active = self.env.players.active & ~self.env.players.completed
        for i in np.flatnonzero(active):
            actions[i] = self._choose_one(int(i))
        return actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a rule-based Snake autopilot")
    parser.add_argument("--snakes", type=int, default=1, help="number of independent autopilot snakes")
    parser.add_argument("--size", type=int, default=20, help="square board width and height, in cells")
    parser.add_argument("--max-frames", type=int, default=2000, help="episode frame limit")
    parser.add_argument("--max-hiatus", type=int, default=150, help="frames without food before losing a life")
    parser.add_argument("--lives", type=int, default=3, help="lives per snake")
    parser.add_argument("--seed", type=int, default=7, help="random seed; use a different value to vary food")
    parser.add_argument("--wrap", action="store_true", help="allow crossing board boundaries")
    parser.add_argument("--headless", action="store_true", help="run without opening a window")
    parser.add_argument("--pixel", action="store_true", help="enable the optional retro pixel renderer")
    parser.add_argument("--fps", type=int, default=30, help="rendered frame rate cap")
    parser.add_argument("--debug", action="store_true", help="show sensors and debug panel")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.snakes < 1:
        raise SystemExit("--snakes must be at least 1")
    if args.size < 3:
        raise SystemExit("--size must be at least 3")
    if args.max_frames < 1 or args.max_hiatus < 1:
        raise SystemExit("--max-frames and --max-hiatus must be positive")

    config = EnvironmentConfig(params={
        "grid_size": (args.size, args.size),
        "wrap": args.wrap,
        "seed": args.seed,
        "lives": args.lives,
        "max_hiatus": args.max_hiatus,
        "max_frames": args.max_frames,
        "window": {
            "title": "Snake — Autopilot Demo",
            "fps": None if args.headless else args.fps,
            "pixel": args.pixel,
            "verbose_level": 1 if args.debug else 0,
            "show_controls_hint": True,
        },
    })
    env = Game(render_mode=None if args.headless else "human", config=config)
    env.reset(keys=args.snakes)
    pilot = SnakeAutopilot(env)

    started, frames, done = time.perf_counter(), 0, False
    try:
        while not done and frames < args.max_frames:
            _, _, terminated, truncated, info = env.step(pilot.act())
            done = bool(terminated or truncated)
            env.render()
            frames += 1
        elapsed = max(time.perf_counter() - started, 1e-9)
        print(f"Completed {frames} frames in {elapsed:.2f}s ({frames / elapsed:.1f} steps/s)")
        print(f"Reason/status: {info if frames else 'no steps executed'}")
        for i in range(env.snakes.total):
            print(
                f"Snake {i + 1}: best_score={int(env.players.true_scores[i])}, "
                f"current_score={int(env.players.scores[i])}, lives={int(env.players.lives[i])}, "
                f"length={int(env.snakes.length[i])}, fitness={float(env.players.fitness[i]):.2f}"
            )
    except KeyboardInterrupt:
        print("\nAutopilot demo interrupted.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
