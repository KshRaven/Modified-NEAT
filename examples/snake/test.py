"""Manual Snake environment run.

Examples:
    python test.py
    python test.py --width 30 --height 20 --fps 12
    python test.py --wrap

Controls are handled by ``Game.run``: arrows/WASD steer, T toggles wrap mode and restarts,
N starts a new game, Q quits, Space pauses, V toggles sensor/debug overlays, and P toggles
pixel rendering.
"""
from __future__ import annotations

import argparse
import os
import sys
import math

EXMP_DIR = os.path.dirname(os.path.abspath(__file__))
if EXMP_DIR not in sys.path:
    sys.path.insert(0, EXMP_DIR)

from game import EnvironmentConfig, Game


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Snake manually")
    parser.add_argument("--width", type=int, default=20, help="board width in cells")
    parser.add_argument("--height", type=int, default=20, help="board height in cells")
    parser.add_argument("--fps", type=int, default=10, help="game speed (frames per second)")
    parser.add_argument("--seed", type=int, default=None, help="reproducible spawn/food seed")
    parser.add_argument("--wrap", action="store_true", help="start with boundary wrap enabled")
    parser.add_argument("--lives", type=int, default=3, help="lives before game over")
    parser.add_argument("--max-hiatus", type=int, default=None, help="frames without food before losing a life (default: no limit)")
    parser.add_argument("--init-length", type=int, default=3, help="starting snake length")
    parser.add_argument("--random-heading", action="store_true", help="randomize starting direction")
    parser.add_argument("--pixel", action="store_true", help="start with the retro pixel renderer enabled")
    parser.add_argument("--debug", action="store_true", help="start with sensor/debug display enabled")
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    if args.width < 3 or args.height < 3:
        raise SystemExit("Board width and height must both be at least 3.")
    if args.fps < 1:
        raise SystemExit("--fps must be a positive integer.")

    config = EnvironmentConfig(params={
        "grid_size": (args.width, args.height),
        "grid": {
            "wrap": args.wrap,
        },
        "wrap": args.wrap,
        "seed": args.seed,
        "lives": args.lives,
        "max_hiatus": args.max_hiatus,
        "snake": {
            "init_length": args.init_length,
            "init_direction": None if args.random_heading else 1,
        },
        "window": {
            "title": "Snake — Manual Test",
            "fps": args.fps,
            "pixel": args.pixel,
            "verbose_level": 1 if args.debug else 0,
            "show_controls_hint": True,
        },
    })
    env = Game(render_mode="human", config=config)
    print(
        f"Manual Snake test: board={args.width}x{args.height}, wrap={args.wrap}, "
        f"lives={args.lives}. Use arrows/WASD; T toggles wrap; N restarts; Q quits."
    )
    try:
        env.run(fps=args.fps)
    finally:
        env.close()


if __name__ == "__main__":
    run()