from game import Game, EnvironmentConfig
import numpy as np

if __name__ == "__main__":
    GENOMES = 3
    
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
    states, info = env.reset(keys=GENOMES, seed=0)   # 20 genomes/players

    DEBUG_STEPS = 100
    done = False
    step = 0
    while not done:
        debug = step % DEBUG_STEPS == 0
        if debug: print(f"State[{step}] => {states.shape}\n{states}")
        actions = np.random.randint(0, 4, size=(GENOMES,))   # or np.random.uniform(-1,1,(20,2)) for continuous actions
        if debug: print(f"Action[{step}] => {actions.shape}\n{actions}")
        states, rewards, terminated, truncated, info = env.step(actions)
        if debug: print(f"Reward[{step}] => {rewards.shape}\n{rewards}")
        env.render()
        done = terminated
        step += 1