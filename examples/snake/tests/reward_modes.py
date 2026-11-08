"""Check the road reward modes: a bot that always takes the best-graded move must never die, never leave road
order, and earn far more than a bot that only follows the road, which in turn must beat a random bot.

    python tests/reward_modes.py --size 10 --snakes 16

Every bot gets the same frame budget (6 x cells by default, i.e. before any snake can fill the board: an episode
ends at the first completed board, which would cut the fastest bot's reward short and make the comparison unfair).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game import EnvironmentConfig, Game
from game.kernels.road import K_ROAD, K_SHORT, K_SLOW


def play(mode: str, bot: str, size: int, n: int, frames: int, seed: int = 0, wrap: bool = False):
    cfg = EnvironmentConfig(params={"grid_size": (size, size), "seed": seed, "lives": 1, "max_frames": frames,
                                    "wrap": wrap, "reward": {"mode": mode, "skill_rate": 0.0}})
    env = Game(render_mode=None, config=cfg)
    env.reset(keys=n)
    sh, rng, done, desync, steps, t0 = env.snakes.shaper, np.random.default_rng(1), False, 0, 0, time.time()
    while not done:
        if bot == "random":
            a = rng.integers(0, 3, n)
        elif bot == "road":
            a = (sh.kind == K_ROAD).argmax(1)
        else:
            a = np.where(np.isin(sh.kind, (K_ROAD, K_SHORT, K_SLOW)), sh.gain, -1e9).argmax(1)
        _, _, _, done, _ = env.step(a)
        steps += 1
        desync += int((~sh.synced & env.players.active).sum()) if sh.has_road else 0
    pl = env.players
    print(f"{mode:9s} {bot:6s} steps={steps:5d} meals/snake={pl.true_scores.mean():5.1f} deaths={int(pl.deaths.sum()):2d} "
          f"completed={int(pl.completed.sum()):2d} out-of-order frames={desync:4d} fitness={pl.fitness.mean():9.0f} {time.time() - t0:5.1f}s")
    return pl.fitness.mean(), int(pl.deaths.sum()), desync


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--snakes", type=int, default=16)
    ap.add_argument("--frames", type=int, default=0, help="0 = 6 x cells")
    ap.add_argument("--wrap", action="store_true")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    a.frames = a.frames or 6 * a.size * a.size
    ok = True
    for mode in ("dhcr", "cell_tree"):
        rnd, road, best = (play(mode, b, a.size, a.snakes, a.frames, wrap=a.wrap) for b in ("random", "road", "best"))
        good = best[1] == 0 and best[2] == 0 and best[0] > road[0] > rnd[0]
        ok &= good
        print(f"  -> {mode}: {'OK' if good else 'FAILED'}")
    sys.exit(0 if ok else 1)
