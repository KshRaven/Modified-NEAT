# Maze Runner

Maze Runner is a Gymnasium environment where genomes move one cell at a time
through a generated maze, avoid traps, and find a goal in the center region.
The underlying world is a discrete `(row, col)` grid. There is no continuous
position in the simulation.

## Files

| File | Role |
|---|---|
| `game/constants.py` | Colors and NumPy dtypes |
| `game/functional.py` | Directions, cell types, and movement helpers |
| `game/config.py` | Maze, agent, window, player, and reward configuration |
| `game/tile.py` | A maze cell with walls and a semantic type |
| `game/grid.py` | Maze generation, path mapping, checkpoints, and expanded grids |
| `game/player.py` | Vectorized lives, death, score, and fitness bookkeeping |
| `game/agent.py` | Movement and compact/full observations |
| `game/window.py` | Pygame rendering, best-path overlay, trails, and agents |
| `game/env.py` | Gymnasium `reset`, `step`, and `render` API |
| `sequential.py` | Sequential PPO/NEAT runner |

## State configuration

State behavior is controlled by two independent flags in `AgentConfig`:

- `return_grid=False`: compact LiDAR state.
- `return_grid=True`: full expanded maze state.
- `discrete_states=False`: normalized floating-point state.
- `discrete_states=True`: integer state IDs suitable for embeddings.

### Compact state

With `return_grid=False`, each genome receives a state of shape `(N, 10)`:

1. Four cardinal LiDAR distances: up, right, down, left.
2. Four corresponding target types.
3. The type of the cell currently occupied by the genome.
4. Distance to the goal using the configured distance metric.

The LiDAR stops at walls, checkpoints, bends, junctions, dead ends, traps, the
start, or the goal. It does not expose absolute position or BFS distance to the
goal, so the compact policy can learn a layout-invariant navigation strategy.

Continuous compact states use `[-1, 1]`. Discrete compact states use distance
bins `0..9`, where `0` means no distance because of a wall. Both LiDAR distance
features and the goal-distance feature use these bins in discrete mode. LiDAR type IDs are:

```text
0 wall
1 path
2 alternate path
3 checkpoint
4 start
5 goal
6 trap
7 dead end
```

The default goal distance is the maze-aware BFS distance: the shortest number of
valid cell moves from the current cell to the goal, including walls and detours.
Its normalization maximum is the largest BFS distance in the current maze. Set
`agent.use_euclidean_distance = True` to use straight-line cell-coordinate
distance instead; its maximum is computed from every cell in the current grid.

### Expanded grid state

With `return_grid=True`, the maze is expanded from `rows x cols` to:

```text
(2 * rows + 1, 2 * cols + 1)
```

Original cell centers are at odd/odd indices. The remaining positions represent
passages or walls/boundaries.

Discrete expanded-grid observations have shape `(N, 2*rows+1, 2*cols+1)` and use:

```text
0 passage
1 wall or boundary
2 empty cell
3 start cell
4 goal cell
5 trap cell
6 current player position
```

Continuous expanded-grid observations have shape
`(N, 1, 2*rows+1, 2*cols+1)` and contain normalized values in `[-1, 1]`.
The player position is stamped into each genome's copy of the grid.

After `Agents.reset()`, `Agents.max_options` contains the discrete embedding
vocabulary size.

## Actions

`discrete_actions` controls how `Game.step()` interprets actions:

| Setting | Space | Shape | Values |
|---|---|---|---|
| `True` | `MultiDiscrete` | `(N,)` | `0=up`, `1=right`, `2=down`, `3=left` |
| `False` | `Box` | `(N, 4)` | Unbounded direction scores |

In continuous mode, each row contains four unbounded scores in this order:
`[up, right, down, left]`. The largest score selects the direction. The old
`[-1, 1]` limit and `action_cutoff` are not used for continuous actions. State
settings do not change the action format.

## Paths and checkpoints

`Grid.all_paths` stores mapped simple start-to-goal paths, with the shortest path
first in `Grid.best_path`. Enumeration is bounded by `grid.max_paths` (default
`4096`) to prevent unbounded work in highly braided mazes.

Structural checkpoints are placed at bends, junctions, dead ends, traps, start,
and goal. The configured `checkpoints` value adds more evenly spaced checkpoints
along the best path. Each checkpoint reward is awarded at most once per genome.

## Rendering

The best path is highlighted by default with a cyan line and checkpoint markers.
It is rendered behind genome trails and player markers. Configure it with:

```python
"window": {
    "draw_best_path": True,
    "best_path_color": (40, 220, 220),
    "best_path_width": 4,
}
```

## Lifecycle and rewards

`max_hiatus` counts steps without reaching a new checkpoint. When enabled, a
hiatus consumes a life, increments `deaths`, applies the hiatus penalty, and
resets the hiatus counter. Trap hits also consume a life. A player is
disqualified only after all lives are lost.

The episode ends when one of these conditions is met:

- `max_frames` is reached by any player.
- All players are disqualified.
- `first_winners` players reach the goal.

Reward defaults are configured under `EnvironmentConfig.player.reward`:

- Goal: `600` base plus up to `900` speed bonus.
- Best-path checkpoint: `25`; alternate checkpoint: `5`.
- New-best distance progress: `3` per cell.
- Path deviation: `-3`; leaving all mapped paths: `-20`.
- Trap: `-75` scaled by death count; hiatus death: `-40`.
- Wall bump: `-0.5`; step cost: `-0.1`.
- Goal hold: `+2` per step after reaching the goal.

`reward_scale` defaults to `0.01`. It scales only the reward returned by
`Game.step()`; internal fitness remains unscaled for population ranking.

## Quickstart

```python
from game import Game, EnvironmentConfig
import numpy as np

config = EnvironmentConfig(params={
    "grid": {"grid_size": (21, 21), "cell_size": 28},
    "trap_ratio": 0.35,
    "goal_region_ratio": 0.34,
    "max_frames": 1000,
    "agent": {
        "discrete_states": False,
        "discrete_actions": True,
        "return_grid": False,
        "reset_on_death": True,
        "use_euclidean_distance": False,
    },
    "player": {
        "lives": 5,
        "max_hiatus": 20,
        "reward": {"reward_scale": 0.01},
    },
})

env = Game(render_mode="human", config=config)
states, info = env.reset(keys=20, seed=0)

done = False
while not done:
    actions = np.random.randint(0, 4, size=(20,))
    states, rewards, terminated, truncated, info = env.step(actions)
    env.render()
    done = terminated or truncated
```

## Manual play

```python
from game import Game, EnvironmentConfig

Game(config=EnvironmentConfig(
    params={"agent": {"discrete_actions": True}}
)).run(total=3)
```

Arrow keys and WASD move genome 0. Press `N` for a new maze and `Q` to quit.

## Debug output

The test folder contains a deterministic expanded-grid printer:

```powershell
python examples/maze/tests/test_grid_debug.py
```

It prints the base cell-type matrix followed by `Grid._grid`, including passages,
walls, boundaries, and typed cell centers.
