# Car Racer

A top-down racing environment for training many cars at once (neuroevolution or RL). Each reset generates a new random track, and every car gets a state vector, a reward and two continuous actions (throttle/brake, steering). Game logic runs in Numba, rendering in pygame.

## Install

Python 3.10+ with:

```
pip install numpy numba pygame gymnasium
```

`torch` is only needed if you pass tensors to `step()` or run the training scripts (`main.py`, `sequential.py`), which also need `ModifiedNEAT`.

## Quick start

```python
from game import Game, EnvironmentConfig

config = EnvironmentConfig(file=".config/main.json")   # or EnvironmentConfig(params={...})
env = Game(render_mode="human", config=config)         # render_mode=None for headless training

state, _ = env.reset(keys=16)                          # one car per key
done = False
while not done:
    actions = policy(state)                            # shape (cars, 2), values in [-1, 1]
    state, reward, done, _, info = env.step(actions)
    env.render()
```

- **Actions** per car: `[throttle/brake, steering]`.
- **State** per car: distances and phase to the next tiles, speed, surface under the car, lidar beams. The layout is documented in `Cars.get_state()`.
- `info` has `completed`, `disqualified` and `time_out`.

Other entry points: `test.py` (race with default config), `race_demo.py` (race with autopilot bots, `--human` to drive with WASD, `--headless` for a benchmark).

## Configuration

Every key is optional and every default is described in `game/config.py`. Settings are nested JSON sections: `tile`, `grid`, `car`, `window`, `generation`, `player`, `race`, `reward`. Two ready-made files are in `.config/`: `main.json` (time trial) and `race.json` (race mode).

### Movement restriction (`car.restrict_movement`)

Takes a bool or an int. `true` means the default (2), `false` means 0.

| Level | Behaviour |
|-------|-----------|
| 0 | Free movement, no position restarts. For testing sensors only. |
| 1 | Solid walls at the track edge and around every tile that is not the current or next one. A car that hits a wall slides along it in the direction it was heading and its nose turns towards that direction (`car.wall_align`, `car.wall_damping`), so it can steer back onto the track. The previous tile stays open: a car can reverse into it but is restarted once `max_hiatus` runs out. |
| 2 | Like 1, but the previous tile is walled off too. **Default.** |
| 3 | No walls. Leaving the track or entering a wrong tile restarts the car. |
| 4 | Like 3, plus an approach-angle limit (`car.max_approach_angle`) and no reversing. |
| 5 | Like 4, plus leaving road and curb (onto gravel or grass) restarts the car. |

### Slipstream

In race mode, cars drafting behind another car get extra acceleration and top speed. `max_linear_velocity` itself never changes, so the speed feature goes above 1.0 while towed.

To train for this alone, set `race.slipstream.random_enabled`. Each car then gets `random_events` tows per life, each lasting `random_frames` frames, starting when it reaches a random tile `random_min_tiles` to `random_max_tiles` ahead. It works outside race mode too.

## Reward

Each car gets a dense per-frame reward (`reward` section, see `RewardConfig`). It works in two phases that blend into each other:

1. **Survive first.** Grass, gravel and crashes cost extra and speed is not regulated, so a fresh population quickly learns to stay on the track.
2. **Then go fast.** Progress along the centre line is credited up to a target speed. The target looks `lookahead_tiles` ahead, grades each corner and follows a braking profile, so it falls gradually towards a corner and rises again after the apex (`exit_gain`). Going over it costs little at first (`overspeed_tolerance`, quadratic) and more the further over you are. Being far below it costs a little, so crawling does not pay.

Surfaces: the curb is expensive on straights (`curb_straight_mult`) but cheap in corners, where the target speed is raised to what the curb's better handling allows. Grass and gravel cost more the closer a corner is (`offroad_corner_mult`).

The phase is controlled by `skill` (0 to 1), measured at every `reset()` from the previous episode: clean tiles per crash, smoothed with `skill_rate`. Set `reward.curriculum` to `false` to use the full reward from the first episode, or `reward.shaping` to `false` for the old flat reward.

Raw rewards are large; `reward_scale` brings them down to what the learner sees.

## Pixel mode

An optional retro look for the track, cars and slipstream. The finished frame is averaged down, its colours are reduced, and it is scaled back up with nearest-neighbour. HUD and text stay sharp. It changes nothing in the simulation.

```json
{ "window": { "pixel_mode": "race", "pixel_size": 4, "pixel_color_bits": 3 } }
```

`pixel_mode` is `"off"`, `"on"` or `"race"` (only in race mode, the default). A larger `pixel_size` is blockier and cheaper; `pixel_color_bits` 3 gives 512 colours, 4 gives 4096.

## Race mode

Turn it on with `race.enabled` or `with env.race_mode():`. It adds a staggered starting grid, a countdown, car-to-car collisions, slipstream, a leaderboard, and lap counting at the finish line. Up to `race.max_racers` cars (12 by default).

## Window and controls

With `window.cinematic` on, the window has a fixed size and a camera follows the action. The default view follows P1 at a constant zoom.

| Key | Action |
|-----|--------|
| `1`-`9` | Follow the car in that race position |
| `Tab` | Follow a random car |
| `0` | Toggle auto-fit of the top `fit_cars` cars (zoom changes with the pack) |
| `-` / `+`, mouse wheel | Zoom |
| `L` | Fit the whole grid |
| `V` | Cycle the debug overlay (stats, lidar, tile arrows) |
| `Space` | Pause |
| `Shift` + `-` / `+` | Lower / raise the frame rate cap |
| `Esc` | End the episode |

If the frame rate is low, set `window.fps` to `null`, keep `window.smooth_scaling` off and use a smaller `window.cinematic_size`. Drawing is the main cost, not the simulation.

## Project layout

```
game/
  env.py        Game: reset / step / render
  tile.py       Tile drawing (road, curb, grass, gravel)
  grid.py       Track generation and surface masks
  car.py        Physics, sensors, walls, signals, reward shaping (Numba kernels)
  player.py     Lives, hiatus and per-frame fitness
  race.py       Starting grid, slipstream, collisions
  window.py     Window, camera, overlays
  pixel.py      Pixelization
  config.py     All settings and their defaults
.config/        Example JSON configs
```