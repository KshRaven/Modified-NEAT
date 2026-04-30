import numpy as np
import pygame

from gymnasium import Env, spaces
from gymnasium.core import RenderFrame
from numpy import ndarray as Array
from torch import Tensor
from typing import Any

from .config import EnvironmentConfig, overwrite_config
from .grid import Grid
from .player import Players
from .car import Cars
from .window import Window
from .functional import Type


class Game(Env):
    """
    To use this class simply initialize and instance and call the .loop() method
    inside of a pygame event loop (i.e while loop). Inside of your event loop
    you can call the .draw() and .move_paddle() methods according to your use case.
    Use the information returned from .loop() to determine when to end the game by calling
    .reset().
    """

    def __init__(
            self,
            render_mode: str | None = "human",
            config: EnvironmentConfig | None = None,
            **params: dict[str, Any]
    ):
        if config is None:
            config = EnvironmentConfig()
        
        overwrite_config(config, params)
        overwrite_config(config.grid, vars(config))
        overwrite_config(config.player, vars(config))
        overwrite_config(config.generation, vars(config))
        self.config = config
        
        # Create grid with GridConfig and TileConfig
        self.grid = Grid(config=config.grid, tile_config=config.tile)
        self.grid.random(config.max_attempts, config.min_tiles, config.seed)
        
        # Create players
        self.players = Players(config.lives, config.max_hiatus)
        
        # Create cars with CarConfig
        self.cars = Cars(self.grid, self.players, config=config.car)
        self.cars.reset(1)
        
        # Create window with WindowConfig
        self.window = Window(self.grid, self.players, self.cars, config=config.window)

        self.previous_rewards: Array | None = None
        self.rewards: Array | None = None
        
        self.render_mode = render_mode
        self.max_frames = config.max_frames
        self.min_laps = config.min_laps
        
        features_total = self.cars.get_state().shape[-1]
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1, features_total), dtype=np.float64)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1, 2), dtype=np.float64) # spaces.Discrete(3)

        self.terminated = False

    def reset(self, **params) -> tuple[Array, dict[str, Any]]:
        """
        Resets the entire game.
        :kwarg seed: optional(int) - Seed for random number generator.
        :kwarg keys: list(int) | int - Keys used to create players
        """
        self.terminated = False
        
        if self.render_mode != 'human':
            self.window.close()
        else:
            self.window.initialize()
        
        seed = params.get('seed', self.config.seed)
        keys: list[int] | int | None = params.get('keys')
        if keys is None:
            raise ValueError(f"Must inputs list of key or keys_total to reset environment")
        elif isinstance(keys, int):
            keys = list(range(keys))

        total = len(keys)    
        
        features_total = self.cars.get_state().shape[-1]
        self.observation_space = spaces.Box(low=-1, high=1, shape=(total, features_total), dtype=np.float64)
        self.action_space = spaces.Box(low=-1, high=1, shape=(total, 2), dtype=np.float64) # spaces.Discrete(3)

        self.grid.random(self.config.max_attempts, self.config.min_tiles, seed)
        self.cars.reset(len(keys))

        self.rewards = self.players.get_reward()
        self.previous_rewards = self.rewards.copy()

        states = self.cars.get_state()

        return states, {}

    def step(self, actions: Array | Tensor) -> tuple[Array, Array, list[bool] | bool, bool, dict[str, Any]]:
        if not self.terminated:
            # Handle input shapes and type
            if isinstance(actions, Tensor):
                actions = actions.cpu().numpy()
            # assert actions.ndim <= 2
            if actions.ndim == 2:
            #     if actions.shape[1] == 3:
            #         actions = actions.argmax(axis=-1)
            #     # elif actions.shape[1] == 1:
            #     #     actions = np.floor(actions[..., 0] * 3).clip(min=0, max=2).astype(int)
            #     else:
            #         raise ValueError(f"Unsupported shape '{actions.shape}'")
            # elif actions.ndim == 1:
                pass
            else:
                raise ValueError(f"Unsupported number of dimension '{actions.ndim}'")
            
            # Update the environment
            self.cars.update(actions)

            # Get the next state to be used
            states = self.cars.get_state() # shape(players, features=5)

            # Get the rewards
            self.previous_rewards = self.rewards.copy()
            self.rewards = self.players.get_reward()
            rewards = np.expand_dims(self.rewards, -1) # shape(players, features=1)

            # Check env status before restarting and games to avoid full reset
            any_complete = np.any(self.cars.laps_done >= self.min_laps).item()
            all_disqualified = np.all(~self.players.active).item()
            timed_out = np.any(self.players.frames_done >= self.max_frames).item()
            done = any_complete or all_disqualified or timed_out

            if self.render_mode == 'human':
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
            if done:
                self.terminated = True

            return states, rewards, done, done, {
                'completed': any_complete, 
                'disqualified': all_disqualified,
                'time_out': timed_out,
            }
        else:
            raise RuntimeError("Environment is terminated!")

    def render(self, **options) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode is not None:
            if self.render_mode == 'human':
                if self.window.screen is None:
                    self.window.initialize()
                self.window.render()
            else:
                if self.window.screen is not None:
                    self.window.close()
                # TODO: Implement
                pass

    def run(self):
        self.min_laps = np.inf
        self.cars.usr_engaged = True
        self.render_mode = 'human'
        self.window.fps = 30
        
        self.window.initialize()

        print(f"Starting run!")
        try:
            frame = 0
            enable_bots = False
            started = False
            self.cars.reset(total=3)
            print(self.players)
            while self.window.running:
                done = False
                randomize = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.terminated = True
                        break
                    # Only check .key if it's a KEYDOWN event
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_b:
                            enable_bots = True
                            break
                        if event.key == pygame.K_n:
                            randomize = True
                            break
                        if event.key == pygame.K_q:
                            done = True
                            break

                self.cars.update_on_event(verbose=False)

                state = np.round(self.cars.get_state(), decimals=4)
                print(f"\rstate = {state[-1]}", end='')

                if randomize or np.all(self.players.lives <= 0):
                    self.grid.random(self.config.max_attempts, self.config.min_tiles, self.config.seed)
                    self.cars.reset(total=3)

                self.window.render()
                frame += 1

                if done:
                    print(f"\nRun has ended!")
                    break
        finally:
            self.window.close()

        self.min_laps = 2
        self.cars.usr_engaged = False
        self.render_mode = None
        self.window.fps = None
        self.window.clock = None


# Export for easy importing
__all__ = ['Game']
