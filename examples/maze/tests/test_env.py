from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game import EnvironmentConfig, Game


if __name__ == "__main__":
    genomes = 3
    cfg = EnvironmentConfig(params={
        "grid": {
            "grid_size": (21, 21),
            "cell_size": 28,
        },
        "trap_ratio": 0.35,
        "goal_region_ratio": 0.34,
        "max_frames": 1000,
        "agent": {
            "discrete_states": False,
            "discrete_actions": True,
            "return_grid": False,
            "reset_on_death": True,
        },
    })
    env = Game(render_mode="human", config=cfg)
    states, info = env.reset(keys=genomes, seed=0)

    debug_steps = 100
    done = False
    step = 0
    while not done:
        debug = step % debug_steps == 0
        if debug:
            print(f"State[{step}] => {states.shape}\n{states}")
        actions = np.random.randint(0, 4, size=(genomes,))
        if debug:
            print(f"Action[{step}] => {actions.shape}\n{actions}")
        states, rewards, terminated, truncated, info = env.step(actions)
        if debug:
            print(f"Reward[{step}] => {rewards.shape}\n{rewards}")
        env.render()
        done = terminated
        step += 1