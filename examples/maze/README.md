# Maze Runner — Gymnasium Environment

Same package layout/conventions as the Car-Racer environment. A "genome"/player is a
dot that must find a goal hidden in the **center region** of an arbitrary `cols x rows`
grid, as fast as possible, while avoiding **traps** placed at some dead ends.

The grid is used internally as a plain grid: every genome occupies a single integer
`(row, col)` cell and moves exactly one cell — up/down/left/right — per step (e.g.
`(1,'A') -> (2,'A')`). There is no continuous/pixel position anywhere in the simulation.
Visually, agents render as small blobs snapped to cell centers on a board of flat
colored squares.

## Files

| File | Role |
|---|---|
| `game/constants.py` | Colors & numpy dtypes |
| `game/functional.py` | `Direction`, `CellType` enums + pure grid-movement helpers |
| `game/config.py` | `TileConfig`, `GridConfig`, `GenConfig`, `AgentConfig`, `WindowConfig`, `PlayerConfig`, `EnvironmentConfig` |
| `game/tile.py` | `Cell` — one grid cell (4 walls + type), flat-square rendering |
| `game/grid.py` | `Grid` — maze generation, goal/trap placement, BFS distance map, `return_grid` array builder |
| `game/player.py` | `Players` — vectorized lives/score/fitness bookkeeping |
| `game/agent.py` | `Agents` — the dots: grid-cell movement + observations for all 4 modes |
| `game/window.py` | `Window` — pygame rendering (flat squares, path trails, ranked blobs) |
| `game/env.py` | `Game(gymnasium.Env)` — `reset`/`step`/`render`/`run` |

## The 4 modes (and where they live)

**Modes are NOT an integer in `EnvironmentConfig`.** They're set directly as two
independent booleans inside `AgentConfig` (`config.agent`), exactly where the logic
that uses them lives:

- `discrete_states` — controls the *observation* built in `agent.py`'s `get_state()`
  (and is forwarded into `grid.py` to pick the `return_grid` array's dtype/encoding).
- `discrete_actions` — controls how `agent.py`'s `update()` interprets the raw
  `actions` array.

| Mode | `discrete_states` | `discrete_actions` | Observation | Action |
|---|---|---|---|---|
| 1 | True | True | int cell/wall/goal-direction features (Embedding-ready) | 1-of-4 discrete move |
| 2 | True | False | same int features | continuous 2D vector, snapped to a cardinal step |
| 3 | False | True | continuous `[-1,1]` position/sensor features | 1-of-4 discrete move |
| 4 | False | False | continuous `[-1,1]` position/sensor features | continuous 2D vector, snapped to a cardinal step |

### `return_grid` (new)

Set `agent.return_grid = True` to make `get_state()` return the **entire grid array**,
once per genome/player, instead of the compact feature vector:

- `discrete_states=False` → `(N, rows, cols, 8)` float32 in `[-1, 1]`: per-cell
  `[wall_up, wall_right, wall_down, wall_left, is_start, is_goal, is_trap, is_player_here]`
  — for convolutional models.
- `discrete_states=True` → `(N, rows, cols, 3)` int64: per-cell
  `[wall_bitmask(0-15), cell_type_code(0-3), is_player_here(0/1)]`
  — small non-negative ints, directly usable as `nn.Embedding` indices.

### `reset_on_death` (new)

Set `agent.reset_on_death` (default `True`) to control what happens when a genome
loses a life to a trap:
- `True` — the genome respawns at the maze's start cell.
- `False` — the genome keeps playing from the cell it died in.

Either way, the life is still spent — this flag only changes *where* play continues.

## Quickstart

```python
from game import Game, EnvironmentConfig
import numpy as np

cfg = EnvironmentConfig(params={
    "grid_size": (21, 21),       # arbitrary cols x rows
    "cell_size": 28,
    "trap_ratio": 0.35,          # fraction of dead-ends turned into traps
    "goal_region_ratio": 0.34,   # goal confined to the center third of the grid
    "max_frames": 1000,
    "agent": {
        "discrete_states": False,
        "discrete_actions": True,
        "return_grid": False,
        "reset_on_death": True,
    },
})
env = Game(render_mode="human", config=cfg)
states, info = env.reset(keys=20, seed=0)   # 20 genomes/players

done = False
while not done:
    actions = np.random.randint(0, 4, size=(20,))   # or np.random.uniform(-1,1,(20,2)) for continuous actions
    states, rewards, terminated, truncated, info = env.step(actions)
    env.render()
    done = terminated
```

## Manual play (testing)

```python
from game import Game, EnvironmentConfig
Game(config=EnvironmentConfig(params={"agent": {"discrete_actions": True}})).run(total=3)
# Arrow keys / WASD move genome 0. N = new maze. Q = quit.
```

## Reward shaping (`player.py`)

- `+5 * lives * improvement` when a genome's best-ever BFS distance-to-goal improves (potential-based).
- `-0.2` per step (encourages speed).
- `-1` on a wall bump.
- `-100 * (deaths+1)` and a life lost on hitting a trap.
- `-50` and elimination on stagnation (`max_hiatus` steps without a new best distance).
- `+1000..2000 * lives` on reaching the goal, scaled by steps remaining (faster = more reward).
