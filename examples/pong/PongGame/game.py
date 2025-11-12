

from .window import Window
from .player import Players
from .paddle import Paddles
from .ball import Balls

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
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)

    def __init__(self, window: Window, players: Players, paddles: Paddles, balls: Balls):
        pygame.init()
        self.SCORE_FONT = pygame.font.SysFont("comicsans", 50)
        self.window = window
        self.players = players
        self.paddles = paddles
        self.balls = balls

    def _draw_score(self, index: int):
        scores = self.players.scores
        window = self.window.surface
        li = index if index % 2 == 0 else index - 1
        ri = index if index % 2 == 1 else index + 1
        ls, rs = scores[li], scores[ri]
        left_score_text = self.SCORE_FONT.render(f"{li}={ls}", 1, self.WHITE)
        right_score_text = self.SCORE_FONT.render(f"{ri}={rs}", 1, self.WHITE)
        window.blit(left_score_text, (self.window.width // 4 - left_score_text.get_width()//2, 20))
        window.blit(right_score_text, (self.window.width * (3/4) - right_score_text.get_width()//2, 20))

    def _draw_hits(self, index: int):
        hits = self.players.hits
        window = self.window.surface
        li = index if index % 2 == 0 else index - 1
        ri = index if index % 2 == 1 else index + 1
        lh, rh = hits[li], hits[ri]
        hits_text = self.SCORE_FONT.render(f"{lh + rh}", 1, self.RED)
        window.blit(hits_text, (self.window.width // 2 - hits_text.get_width()//2, 10))

    def _draw_divider(self):
        for i in range(10, self.window.height, self.window.height//20):
            if i % 2 == 1:
                continue
            pygame.draw.rect(
                self.window.surface, self.WHITE, (self.window.width//2 - 5, i, 10, self.window.height//20)
            )

    def _draw_objects(self, index: int):
        window = self.window.surface
        li = index if index % 2 == 0 else index - 1
        ri = index if index % 2 == 1 else index + 1
        # Render paddles
        for i in [li, ri]:
            pygame.draw.rect(
                window, self.WHITE, (self.paddles.x[i], self.paddles.y[i], self.paddles.WIDTH, self.paddles.HEIGHT)
            )
        # Render ball. Render both in order to check that they are working properly
        for i in [index]: # [li, ri]:
            pygame.draw.circle(
                window, self.WHITE, (self.balls.x[i], self.balls.y[i]), self.balls.RADIUS
            )

    def reset(self):
        pygame.init()

    def render(self, draw_score=True, draw_hits=False):
        self.window.surface.fill(self.BLACK)

        self._draw_divider()

        best_index = self.players.best_index

        if draw_score:
            self._draw_score(best_index)

        if draw_hits:
            self._draw_hits(best_index)

        self._draw_objects(best_index)
        pygame.display.update()


class Game(Env):
    """
    To use this class simply initialize and instance and call the .loop() method
    inside of a pygame event loop (i.e while loop). Inside of your event loop
    you can call the .draw() and .move_paddle() methods according to your use case.
    Use the information returned from .loop() to determine when to end the game by calling
    .reset().
    """

    def __init__(self, window_shape: tuple[int, int], goal=20, max_factor=3, lives=3,
                 render_mode: str | None = 'human', **options):
        self.render_mode = render_mode
        self.clock = pygame.time.Clock()

        self.window = Window(*window_shape)

        self.players = Players(lives)

        self.paddles = Paddles(
            self.players, 10, self.window.height // 2, self.window.shape,
            options.get('paddle_vel', 4.5), *options.get('paddle_shape', (20, 100))
        )

        self.balls = Balls(
            self.players, self.window.width // 2, self.window.height // 2, self.window.shape,
            max_vel=2.75
        )

        self.backend = Backend(self.window, self.players, self.paddles, self.balls)

        self.goal = goal
        self.max_factor = max_factor
        self.base =  math.exp(math.log(self.max_factor) / self.goal)
        self.previous_rewards: CPUArray | None = None

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64)
        self.action_space = spaces.Discrete(3)
        self.terminated = False

    def _handle_collision(self):
        # Get the ball and paddle
        left, right = self.players.left, self.players.right
        balls = self.balls
        paddles = self.paddles

        # Handle deflection from base or roof
        base_deflection = (balls.y + balls.RADIUS) >= self.window.height
        roof_deflection = (balls.y - balls.RADIUS) <= 0
        balls.y_vel[base_deflection | roof_deflection] *= -1

        # Handle collision with paddles
        lp_ball = (balls.x_vel < 0) # [left]
        rp_ball = (balls.x_vel > 0) # [right]

        within_x_bounds = np.full_like(balls.x, False, dtype=bool)
        within_x_bounds[left] = (balls.x[left] - balls.RADIUS) <= (paddles.x[left] + paddles.WIDTH)
        within_x_bounds[right] = (balls.x[right] + balls.RADIUS) >= paddles.x[right]
        within_y_bounds = (balls.y >= paddles.y) & (balls.y <= (paddles.y + paddles.HEIGHT))
        deflection = within_x_bounds & within_y_bounds
        strike = within_x_bounds & ~within_y_bounds

        self.players.update(deflection, strike)
        # within_bounds_lp = within_x_bounds_lp & within_y_bounds_lp
        # within_bounds_rp = within_x_bounds_rp & within_y_bounds_rp

        # Handle ball deflection, relative to paddle center (for optional spin). TODO: Might want to move this outside or to a njit function
        middle_y = paddles.y + (paddles.HEIGHT / 2)
        offset = (balls.y - middle_y) / (paddles.HEIGHT / 2) # normalized: -1 at top, +1 at bottom
        SPIN_FACTOR = random.random() # 0.10  # Apply small spin influence (optional). Tweak between 0 (pure reflection) and 1 (more control)
        y_vel = balls.y_vel * (1 - SPIN_FACTOR) + offset * balls.MAX_VEL * SPIN_FACTOR
        # difference_in_y = middle_y - balls.y
        # reduction_factor = (paddles.HEIGHT / 2) / balls.MAX_VEL
        # y_vel = difference_in_y / reduction_factor
        y_vel[left & rp_ball] = y_vel[right & rp_ball]
        y_vel[right & lp_ball] = y_vel[left & lp_ball]
        try:
            assert np.all(y_vel[left] == y_vel[right])
        except AssertionError as e:
            print(y_vel)
            print(left)
            print(right)
            raise e
        # lp_deflection = lp_ball & deflection
        # rp_deflection = rp_ball & deflection

        paddle_deflection = deflection.copy()
        paddle_deflection[left] |= deflection[right]
        paddle_deflection[right] |= deflection[left]
        balls.x_vel[paddle_deflection] *= -1
        # balls.x_vel[right][paddle_deflection] *= -1
        balls.y_vel[paddle_deflection] = y_vel[paddle_deflection] # * -1
        # balls.y_vel[left][lp_deflection] = balls.y_vel[right][lp_deflection] = -1 * y_vel[left]
        # balls.y_vel[left][rp_deflection] = balls.y_vel[right][rp_deflection] = -1 * y_vel[right]

        # Normalize velocity to maintain constant speed
        speed = np.sqrt(balls.x_vel**2 + balls.y_vel**2) # Compute speed magnitude
        norm = np.sqrt(speed)
        balls.x_vel = (balls.x_vel / norm) * balls.MAX_VEL
        balls.y_vel = (balls.y_vel / norm) * balls.MAX_VEL

        scored = strike.copy()
        scored[left] = strike[right]
        scored[right] = strike[left]
        return deflection, scored

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
        self.paddles.reset()
        self.balls.reset()
        self.backend.reset()

        self.previous_rewards = self.players.fitness.copy()

        states: CPUArray = np.stack([
            self.paddles.y / self.window.height,
            np.abs(self.paddles.x - self.balls.x) / self.window.width,
            self.balls.y / self.window.height,
        ], axis=-1)

        return states, {}

    def step(self, actions: CPUArray | Tensor) -> tuple[CPUArray, CPUArray, list[bool] | bool, bool, dict[str, Any]]:
        if not self.terminated:
            # Handle input shapes and type
            if isinstance(actions, Tensor):
                actions = actions.cpu().numpy()
            assert actions.ndim <= 2
            if actions.ndim == 2:
                actions = actions.argmax(axis=-1)

            # Update frame / environment
            self.paddles.move(actions)
            self.balls.move()
            deflected, scored = self._handle_collision()

            # Get the next state to be used
            states: CPUArray = np.stack([
                self.paddles.y / self.window.height,
                np.abs(self.paddles.x - self.balls.x) / self.window.width,
                self.balls.y / self.window.height,
            ], axis=-1)

            # Calculate rewards
            fitness         = self.players.fitness
            frames          = self.players.frames_done # Used to make certain rewards scale exponentially
            disqualified    = self.players.disqualified
            active          = self.players.active
            # Punish when player is scored
            fitness[disqualified] -= 100
            # Punish for not moving
            stationary = self.paddles.stationary
            fitness[stationary] -= 0.01 * frames[stationary]
            # Punish for being fully disqualified
            fitness[~active] -= 0.02
            # Reward for deflecting or scoring
            COEFF_POINT = 0.1
            fitness[deflected] += COEFF_POINT * frames[deflected]
            fitness[scored] += COEFF_POINT * frames[scored]
            # Actual reward is the change in reward
            current_rewards = self.players.fitness.copy()
            rewards = current_rewards - self.previous_rewards
            self.previous_rewards = current_rewards
            rewards = np.expand_dims(rewards, -1)

            # Check env status before restarting and games to avoid full reset
            all_complete = np.any((self.players.hits + self.players.scores) >= self.goal).item() # TODO: Might want to change to .any
            all_disqualified = np.all(~self.players.active).item()
            done = all_complete or all_disqualified
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            if done:
                self.terminated = True

            # Restart any games that have sufficient lives to continue
            # self.paddles.restart()
            self.balls.restart()
            self.players.restart()

            return states, rewards, done, done, {}
        else:
            raise RuntimeError("Environment is terminated!")

    def render(self, **options) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode is not None:
            if self.render_mode == 'human':
                self.backend.render(True, True)

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






