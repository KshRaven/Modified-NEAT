# Snake Environment

A vectorized Snake environment split into simulation, accelerated kernels, player/reward accounting, rendering, and entry-point scripts. It deliberately shares the **package organization** of Car-Racer, not its driving physics or race rules.

## Run files

Install the runtime dependencies first:

```bash
pip install -r requirements.txt
```

Manual keyboard play:

```bash
python test.py
python test.py --width 30 --height 20 --fps 12
python test.py --wrap
```

Controls: arrows/WASD steer, `T` toggles boundary wrapping and restarts, `N` starts a new game, `Q` quits, Space pauses, `V` toggles sensors/debug display, `P` toggles pixel rendering.

Rule-based autopilot demo (does not require ModifiedNEAT):

```bash
python demo.py
python demo.py --snakes 8 --wrap --size 25 --seed 7
python demo.py --headless --max-frames 2000
```

ModifiedNEAT training:

```bash
python main.py --genomes 100 --generations 100
python main.py --genomes 100 --generations 100 --wrap
python main.py --load true --generations 50
python main.py --load true --eval-only
python main.py --no-render --no-test --genomes 100 --generations 10
```

`main.py` requires the project's `ModifiedNEAT` package to be importable. The script adds the Snake directory and its project root (two directory levels above the package) to `sys.path`, matching the Car-Racer layout. Install PyTorch and make the ModifiedNEAT source available alongside the project before training.

## Structure

- `game/env.py`: top-level `Game` / Gymnasium-style API, reset/step, mode switching, manual-play loop
- `game/config.py`, `game/constants.py`, `game/functional.py`: configuration, shared constants, action/direction helpers
- `game/grid.py`: board occupancy and food placement
- `game/snake.py`: per-snake state, movement coordination, sensors and observations
- `game/kernels/`: Numba-accelerated movement/spawn, food distance, sensors, reward shaping and `road.py` (ring road, move grading, flood fill)
- `game/player.py`, `game/reward.py`: lives, scores, fitness, event rewards, dense shaping
- `game/window.py`, `game/renderer.py`, `game/pixel.py`, `game/util.py`: Pygame window and normal/retro rendering
- `test.py`: manual keyboard run
- `demo.py`: path-search autopilot demo
- `main.py`: ModifiedNEAT training and best-population evaluation

## Reward modes (road shaping)

`reward.mode` picks what the dense reward imitates. The two fastest solvers in `tests/` both rest on one idea: a
one-way **ring road** through every cell (a Hamiltonian cycle). A snake whose body stays in road order cannot box
itself in, and it may cut corners only in ways that keep that order. The reward grades every move against exactly
that logic, so a learner is pushed toward the same strategy instead of just "go for the food".

| mode | road | what a move off the road needs |
|---|---|---|
| `basic` | none | the original food-distance shaping (default) |
| `dhcr` | fixed zigzag cycle | hop to a free neighbour further along the road, not past the food (tier-1 shortcut of `dhcr_fast_v3`) |
| `cell_tree` | outline of a random spanning tree over 2x2 cells | a tree swap that makes the move a road link while the body stays one piece (`cell_tree.py`); legal swaps are applied, so the road bends around the snake |
| `staged` | `dhcr`, then `cell_tree` | switches when the skill passes `staged_switch` |

Every move of every snake is graded each frame by `kernels/road.py::scan`: **road** (next road square), **short**
(safe hop that saves road cells), **slow** (safe, not shorter), **unsafe** (breaks road order / no legal swap),
**dead**. The scan also flood-fills the room behind each move (is it a pocket smaller than the snake with no way to the tail?).

Curriculum, like the car racer: a `skill` in [0, 1] is graded at every `reset` from meals per death of the last episode.
Low skill = survive first (a bonus for staying in road order, crash / unsafe / trap costs up to `1 + early_crash_mult`
times larger). High skill = go fast (the extra cells a shortcut saves are credited at `shortcut_mult x skill`).
The dhcr "mods" are small terms: `nascar_weight` (reversing the turn direction) and `fear_weight` (swinging off the road
into the food on a fuller board). The A*/repair tier of dhcr is deliberately not used: it cost 25 s per 1000 games
for the same move count as the cheap tiers.

Living must never be worth less than dying (the "survive first" lesson). A snake with no skill leaves road order within
a few moves, so per-frame costs bigger than the death cost make "ram the wall" the best policy. Hence `alive_weight`
(bonus per living frame x (1 - skill), fading to zero as the hiatus grows so stalling pays nothing), a mild
`desync_weight`, and road order treated as a potential: `unsafe_weight` is charged when the body leaves road order and
`resync_refund` of it is paid back when it is in order again.

Everything is Numba-parallel over snakes; the cell-tree swap check costs about a microsecond per move.

```python
env = Game(render_mode=None, grid_size=(10, 10), mode="cell_tree")        # also: reward={"mode": "staged"}
config = EnvironmentConfig(params={"reward": {"mode": "staged", "staged_switch": 0.6}})
```

Notes:
* Road modes need an even number of cells (`dhcr`) or even width and height >= 4 (`cell_tree`); other boards fall back
  (cell_tree -> dhcr -> basic) with a warning. `wrap=True` works (the road itself never crosses the boundary).
* They spawn the snake along the road (random tail square, `random_road`); `init_direction` is ignored.
* `state_type="auto"` is the `ego` state (see "State" below). The old 20-value `strategy` state is still selectable
  (`state_type="strategy"`) but it contains the road solver's verdict on every move, so it only serves as an oracle
  baseline for comparison, not as something to train on.
* `max_hiatus` is raised to the cell count (following the road takes up to cells-1 moves per meal; `auto_hiatus`),
  and `max_frames` (default 1000) should be above the ~1200-1700 moves a 10x10 game takes.
* `python tests/reward_modes.py` plays the graded-best-move bot in every mode (0 deaths expected) and checks the reward ordering.
* In `main.py` set `REWARD_MODE`.

## Boundary mode

`wrap=False` is the default: hitting a boundary is a wall collision. With `wrap=True`, movement and sensor rays cross to the opposite side; food distance uses the shorter wrapped offset; the renderer marks the boundary as crossable. The setting is available in `Game(..., wrap=True)`, can be changed with `env.set_wrap(...)`, and is toggled during manual play with `T` (which restarts the game).

## State (`state_type="ego"`, the default)

The network gets what a driver would see, not what a solver would conclude. Nothing is a compass direction or an
absolute coordinate: every value is measured from the head in the heading frame (forward / left), so the same
situation produces the same inputs wherever and however the snake is facing. No value grades a move (safe, shortcut,
trap); the reward uses those grades, the state does not.

All values are roughly in [-1, 1]. `w`, `h` = board size, `N = w * h`. With `wrap` distances take the shorter way round.

| # | name | how it is calculated |
|---|---|---|
| 0 | food bearing | `atan2(left, forward) / pi`, where `forward` / `left` are the food offset projected on the heading and on its left. 0 = dead ahead, +0.5 left, -0.5 right, +-1 behind |
| 1 | food distance | `(abs(dx) + abs(dy)) / (w + h)` from head to food |
| 2 | tail bearing | same as 0, for the last body cell |
| 3 | tail distance | same as 1, for the last body cell |
| 4 | ray left | cells from the head to the first wall or body part going left, `(cells - 1) / ray_length`: 0 = blocked right next to the head, 1 = open |
| 5 | ray ahead | same, straight ahead |
| 6 | ray right | same, to the right |
| 7 | length | snake length / N |

The first 8 values are pure sensing. With `ego_road=True` (default, road reward modes only) five values are added:
the position of the neighbour cells on the one-way ring road, the way a track tile gives a car its entry and exit.

| # | name | how it is calculated |
|---|---|---|
| 8 | road offset, left cell | `(rank[left cell] - rank[head]) mod N / N`: road steps from the head to that cell. 1/N = the road itself goes there, close to 1 = it lies just behind the head, 0 = off the board |
| 9 | road offset, cell ahead | same |
| 10 | road offset, right cell | same |
| 11 | road offset, food | `(rank[food] - rank[head]) mod N / N` |
| 12 | road offset, tail | `(rank[tail] - rank[head]) mod N / N`: how much free road lies ahead of the head |

These are positions only. The rule "a hop is safe if the cell is free (rays) and its offset is below the tail offset,
and a shortcut if it is also not past the food offset" is not given; the network has to find it through the reward.
`ego_road=False` removes values 8-12 so the snake must learn from sensing alone (use it as an ablation: if it still
learns, the road is not doing the work). For `cell_tree` the ring changes when the tree is rewired; the offsets always
follow the current ring.

Where the oracle is used: only in `RewardShaper` (`game/reward.py`, `kernels/reward.py`), which grades the move that
was taken (road / shortcut / unsafe / dead / trap) to produce reward. That is the critic. It never reaches the inputs.