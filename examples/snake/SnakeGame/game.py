

from .functional import GridEnum
from .window import Window
from .player import Players
from .grid import Grid
from ModifiedNEAT.util.fancy_text import CM, Fore

from torch import Tensor
from numpy import ndarray as CPUArray
from gymnasium import Env, spaces
from gymnasium.core import RenderFrame
from typing import Any

import pygame
import numpy as np
import random
import math


class Backend(object):
    WHITE   = (255, 255, 255)
    BLACK   = (0, 0, 0)
    RED     = (255, 0, 0)
    BLUE1   = (0, 0, 255)
    BLUE2   = (0, 100, 255)
    PURPLE  = (128, 0, 128)
    BROWN   = (165, 42, 42)
    ORANGE  = (255, 165, 0)
    YELLOW  = (255, 255, 0)
    LIME    = (0, 255, 0)

    def __init__(self, window: Window, players: Players, grid: Grid, blob_size: int = 20):
        pygame.init()
        self.SCORE_FONT = pygame.font.SysFont("comicsans", 50)
        self.window = window
        self.players = players
        self.grid = grid
        self.BLOB_SIZE = blob_size

    def reset(self):
        pygame.init()

    def render(self):
        window = self.window.surface
        window.fill(self.BLACK)

        best_index = self.players.best_index

        # grid = np.rot90(self.grid.grid[best_index].copy(), 1)
        grid = self.grid.grid[best_index].copy()
        for index, value in np.ndenumerate(grid):
            try:
                # value = round(value / 255 * 4)
                pos_x, pos_y = index[::-1]
                x, y = pos_x * self.BLOB_SIZE, pos_y * self.BLOB_SIZE
                if value == GridEnum.Boundary.value:
                    color = self.ORANGE
                elif value == GridEnum.Food.value:
                    color = self.RED
                elif value == GridEnum.SnakeHead.value:
                    color = self.LIME
                elif value >= GridEnum.SnakeBody.value:
                    color = self.BLUE2
                elif value == GridEnum.Empty.value:
                    color = self.BLACK
                else:
                    color = self.YELLOW
                    pass
                pygame.draw.rect(window, color, pygame.Rect(x, y, self.BLOB_SIZE, self.BLOB_SIZE))
            except ValueError as e:
                print(f"Array = {grid.shape}")
                print(f"Test index = {CM(index, Fore.LIGHTCYAN_EX)}, Value = {CM(index, Fore.LIGHTYELLOW_EX)}")
                print(grid)
                raise e

        text = self.SCORE_FONT.render(f"Score: {self.players.scores[best_index]}", True, self.WHITE)
        window.blit(text, [0, 0])

        pygame.display.update()


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
            window_shape: tuple[int, int], state_type='grid',
            render_mode: str | None = 'human', **options
    ):
        self.render_mode = render_mode
        self.clock = pygame.time.Clock()

        self.shape = window_shape
        self.window = Window(*window_shape, options.get('blob', 20))

        self.max_frames = options.get('max_frames', 500)
        self.players = Players(options.get('lives', 3), options.get('max_hiatus', 3))

        self.grid = Grid(
            self.players, self.shape,
            options.get('init_dir', 1), options.get('init_len', 3),
            options.get('timeout', None), options.get('var_thresh', 0.05)
        )

        self.backend = Backend(self.window, self.players, self.grid, options.get('blob', 20))

        self.previous_rewards: CPUArray | None = None

        self.state_type = state_type
        self.convolutional = state_type == 'grid'
        normal = state_type != 'grid'
        self.observation_space = spaces.Box(
            low=-np.inf if normal else 0, high=np.inf if normal else int(math.prod(window_shape)),
            shape=(10,) if normal else (1, *self.grid.shape[1:]),
            dtype=np.float64 if normal else np.int64
        )
        self.action_space = spaces.Discrete(3)
        self.terminated = False

    def reset(self, **params) -> tuple[CPUArray, dict[str, Any]]:
        """
        Resets the entire game.
        :kwarg seed: optional(int) - Seed for random number generator.
        :kwarg keys: list(int) | int - Keys used to create players
        """
        self.terminated = False
        keys: list[int] | int | None = params.get('keys')
        if keys is None:
            raise ValueError(f"Must inputs list of key or keys_total to reset environment")
        elif isinstance(keys, int):
            keys = list(range(keys))

        self.players.reset(len(keys))
        self.grid.reset()
        self.backend.reset()

        self.previous_rewards = self.players.fitness.copy()

        raw_states = self.grid.move(np.random.randint(low=0, high=3, size=(self.players.total,)), self.convolutional)
        states: CPUArray = np.expand_dims(raw_states, axis=1) if self.convolutional else np.stack(raw_states, axis=-1)

        return states, {}

    def step(self, actions: CPUArray | Tensor) -> tuple[CPUArray, CPUArray, list[bool] | bool, bool, dict[str, Any]]:
        if not self.terminated:
            # Handle input shapes and type
            if isinstance(actions, Tensor):
                actions = actions.cpu().numpy()
            assert actions.ndim <= 2
            if actions.ndim == 2:
                if actions.shape[1] == 3:
                    actions = actions.argmax(axis=-1)
                # elif actions.shape[1] == 1:
                #     actions = np.floor(actions[..., 0] * 3).clip(min=0, max=2).astype(int)
                else:
                    raise ValueError(f"Unsupported shape '{actions.shape}'")
            elif actions.ndim == 1:
                pass
            else:
                raise ValueError(f"Unsupported number of dimension '{actions.ndim}'")

            # Get the next state to be used
            raw_states = self.grid.move(actions, self.convolutional)
            states: CPUArray = np.expand_dims(raw_states, axis=1) if self.convolutional else np.stack(raw_states, axis=-1)
            # Restart any games that have sufficient lives to continue
            self.players.restart()

            # Calculate rewards
            rewards = self.players.fitness - self.players.prev_fitness
            rewards = np.expand_dims(rewards, -1)

            # Check env status before restarting and games to avoid full reset
            any_complete = np.any(self.players.completed).item()
            all_disqualified = np.all(~self.players.active).item()
            timed_out = np.any(self.players.frames_done >= self.max_frames).item()
            done = any_complete or all_disqualified or timed_out
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            if done:
                self.terminated = True

            return states, rewards, done, done, {}
        else:
            raise RuntimeError("Environment is terminated!")

    def render(self, **options) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode is not None:
            if self.render_mode == 'human':
                self.backend.render()

    def run(self):
        self.goal = np.inf
        print(self.players)
        _lives = self.players.lives_total
        _paddle_speed = self.paddles.VEL
        self.players.lives_total = 99
        self.paddles.VEL *= 5
        state = self.reset(keys=2)[0]

        started = False
        print(f"starting loop")
        self.backend.render(True, True)
        frame = 0
        while not self.terminated:
            self.clock.tick(3 if not started else 20)
            action_user0 = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.terminated = True
                    break

                # Only check .key if it's a KEYDOWN event
                elif event.type in [pygame.KEYDOWN]:
                    if not started and event.key in [pygame.K_SPACE, pygame.K_w, pygame.K_s, pygame.K_UP, pygame.K_DOWN]:
                        started = True
                        print("\rGame has started!")

                    if event.key in [pygame.K_w, pygame.K_UP]:
                        action_user0 = 1
                    elif event.key in [pygame.K_s, pygame.K_DOWN]:
                        action_user0 = 2

            if started:
                action_user1 = random.randint(0, 4-1) % 2 + 1
                actions = np.array([action_user0, action_user1], dtype=np.int64)
            else:
                actions = np.zeros(2, dtype=np.int64)
                if frame == 0:
                    state, reward, _, self.terminated, _ = self.step(actions)
                print(f"\rwaiting{'.' * (frame % 10)}", end='')
            if started:
                next_state, reward, _, self.terminated, _ = self.step(actions)
            self.backend.render(True, True)
            frame += 1
            if self.terminated:
                print(f"\rGame has ended!")
                break

        self.players.lives_total = _lives
        self.paddles.VEL = _paddle_speed

        pygame.display.quit()
        pygame.quit()


if __name__ == "__main__":
    import time as clock
    game = Game((500, 500))
    game.run()

    clock.sleep(10)






