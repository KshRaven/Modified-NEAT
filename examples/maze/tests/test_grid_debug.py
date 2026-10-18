from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game import EnvironmentConfig, Game


def main() -> None:
    config = EnvironmentConfig(params={
        "seed": 0,
        "grid": {"grid_size": (5, 5)},
        "trap_ratio": 0.35,
        "braid_ratio": 0.15,
        "agent": {"discrete_states": True, "return_grid": True},
    })
    env = Game(render_mode=None, config=config)
    env.reset(keys=1, seed=0)

    cell_types = np.array(
        [[env.grid.cell((row, col)).type.value for col in range(env.grid.cols)]
         for row in range(env.grid.rows)],
        dtype=int,
    )

    print("5x5 cell types (0=empty, 1=start, 2=goal, 3=trap):")
    print(cell_types)
    print("Expanded _grid (0=passage, 1=wall/boundary, 2..5=cell types):")
    print(env.grid._grid)


if __name__ == "__main__":
    main()