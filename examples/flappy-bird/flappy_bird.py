
from ModifiedNEAT.util.fancy_text import CM, Fore
from ModifiedNEAT.nn.base import Model
from ModifiedNEAT.nn.modules.sub import Linear, Conv1d, Transpose, ResidualBlock, Sequential, GroupNorm, ConverBase, SequenceEncoding
# from ModifiedNEAT.nn.modules import Reformer
from ModifiedNEAT.util.datetime import unix_to_datetime_file

import ModifiedNEAT as neat
import ModifiedNEAT.nn as mn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gymnasium import Env, spaces
from torch import Tensor
from numba import njit, prange
from numba.typed import List, Dict
from numba.core.errors import NumbaPerformanceWarning
from typing import Union, Any, Iterable
# from gymnasium.core import ObsType, ActType

import pygame
import random
import os
import time as clock
import numpy as np
import warnings
import multiprocessing as mp

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
torch.set_printoptions(threshold=10)

DEVICE = 'gpu' if torch.cuda.is_available() else 'cpu'
DTYPE  = torch.bfloat16


class Window(object):
    image: pygame.Surface = None

    def __init__(self, height: int = 800, width: int = 600, width_ext: int = 300, hitbox: int = 10):
        # -------------------- Attributes -------------------- #
        self.height     = height
        self.width      = width + width_ext
        self.width_ext  = width_ext
        self.hitbox     = hitbox # TODO: Find its original effect
        # -------------------- PyGame -------------------- #
        self.display    = pygame.display.set_mode((width, height))
        pygame.font.init()  # init font
        pygame.display.set_caption("Flappy Bird")
        self.stat_font  = pygame.font.SysFont("comicsans", 50)
        self.end_font   = pygame.font.SysFont("comicsans", 70)
        if self.image is None:
            self.image      = pygame.transform.scale(pygame.image.load(
                os.path.join("imgs", "bg.png")).convert_alpha(), (width, height))
        # -------------------- States -------------------- #
        self.draw_lines = False
        self.initialized = True

    def initialize(self):
        self.display    = pygame.display.set_mode((self.width, self.height))
        pygame.font.init()  # init font
        pygame.display.set_caption("Flappy Bird")
        self.stat_font  = pygame.font.SysFont("comicsans", 50)
        self.end_font   = pygame.font.SysFont("comicsans", 70)
        self.image      = pygame.transform.scale(pygame.image.load(
            os.path.join("imgs", "bg.png")
        ).convert_alpha(), (self.width, self.height))
        self.initialized = True

    def close(self):
        # TODO: Implement so that there are no errors
        # pygame.font.quit()
        # pygame.display.quit()
        self.initialized = False

    def render(self):
        self.display.blit(self.image, (0, 0))


class Floor(object):
    image: pygame.Surface = None

    def __init__(self, window: Window, x: int, level: int, velocity: int = 6):
        # -------------------- PyGame -------------------- #
        if self.image is None:
            self.image = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")).convert_alpha())
        # -------------------- Attributes -------------------- #
        self.window     = window
        self.height     = self.image.get_height()
        self.width      = self.image.get_width()
        self.x          = x
        self.y          = window.height - level
        self.velocity   = velocity

    def move(self):
        self.x -= self.velocity

    def render(self):
        self.window.display.blit(self.image, (self.x, self.y))


class FloorHandler(object):
    def __init__(self, window: Window, level: int, velocity = 6):
        # -------------------- Attributes -------------------- #
        self.window         = window
        self.level          = level
        self.velocity       = velocity
        self.floors         = [Floor(self.window, 0, self.level, self.velocity)]

        self.reset()

    def add(self):
        floor = self.floors[-1]
        x = floor.x + floor.width
        while x <= self.window.width:
            floor = Floor(self.window, x, self.level, self.velocity)
            self.floors.append(floor)
            x = floor.x + floor.width

    def delete(self):
        index = 0
        while index < len(self.floors):
            floor = self.floors[index]
            if floor.x + floor.width < 0:
                self.floors.pop(index)
            else:
                index += 1

    def update(self):
        self.delete()
        self.add()

    def reset(self):
        self.floors = [Floor(self.window, 0, self.level, self.velocity)]
        self.update()

    def move(self):
        self.update()
        for floor in self.floors:
            floor.move()

    def render(self):
        for pipe in self.floors:
            pipe.render()

    @property
    def y(self):
        for floor in self.floors:
            return floor.y
        raise RuntimeError(f"No floors available")


class Pipe(object):
    image: pygame.Surface = None

    def __init__(self, x: int, window: Window, floors: FloorHandler,
                 l_offset=100, u_offset=100, gap_l_lim=200, gap_u_lim=200,
                 velocity_x=6, velocity_y=3):
        # -------------------- Attributes -------------------- #
        self.x              = x
        self.window         = window
        self.floors         = floors
        self.l_lim, self.u_lim = u_offset, self.floors.y - l_offset
        self.gap_l_lim, self.gap_u_lim = gap_l_lim, gap_u_lim
        assert abs(self.u_lim - self.l_lim) >= max(self.gap_l_lim, self.gap_u_lim)
        self.velocity_x, self.velocity_y = velocity_x, velocity_y
        self.gap: int       = None
        self.gap_top: int   = None
        self.gap_bot: int   = None
        self.y_top: int     = None
        self.y_bot: int     = None
        # -------------------- PyGame -------------------- #
        if self.image is None:
            self.image = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")).convert_alpha())
        self.pxt = 1.0
        self.pipe_top       = pygame.transform.flip(self.image, False, True)
        self.pipe_bottom    = self.image
        # -------------------- States -------------------- #
        self.passed         = False
        self.collision      = False
        self.direction_y    = random.randint(0, 1)
        self.can_move_vertically = False

        self.set_height()

    def set_height(self):
        self.gap = random.randint(self.gap_l_lim, self.gap_u_lim)
        self.can_move_vertically = self.gap < self.u_lim - self.l_lim and random.random() < 0.50
        self.gap_top = random.randrange(self.l_lim, self.u_lim - self.gap)
        self.gap_bot = self.gap_top + self.gap
        self.y_top = float(self.gap_top - self.pipe_top.get_height())
        self.y_bot = float(self.gap_bot)

    @property
    def width(self):
        return self.image.get_width() * self.pxt

    @property
    def vertical_direction(self):
        return (-1. + 2. * self.direction_y) if self.can_move_vertically else 0.

    @property
    def vertical_offset(self):
        return self.gap_top / self.window.height, (self.floors.y - self.gap_bot) / self.window.height

    def move(self):
        self.x -= self.velocity_x

        def shift(displacement: int, direction: int):
            velocity = displacement * (-1) ** direction
            self.gap_top += velocity
            self.gap_bot += velocity
            self.y_top += velocity
            self.y_bot += velocity

        if self.can_move_vertically:
            # Moving downwards
            if self.direction_y == 0:
                position = self.gap_bot + self.velocity_y
                if position < self.u_lim:
                    shift(self.velocity_y, self.direction_y)
                else:
                    self.direction_y = 1
                    shift(self.velocity_y, self.direction_y)
            # Moving upwards
            if self.direction_y == 1:
                position = self.gap_top - self.velocity_y
                if position > self.l_lim:
                    shift(self.velocity_y, self.direction_y)
                else:
                    self.direction_y = 0
                    shift(self.velocity_y, self.direction_y)

    def render(self):
        # render top
        self.window.display.blit(self.pipe_top, (self.x, self.y_top))
        # render bottom
        self.window.display.blit(self.pipe_bottom, (self.x, self.y_bot))


class PipesHandler(object):
    def __init__(self, window: Window, floors: FloorHandler, init_x: int, spawn_width: int,
                 offset: int | tuple[int, int], gap: int | tuple[int, int], velocity_x = 6, velocity_y = 3,
                 ):
        if not isinstance(offset, Iterable):
            offset = (offset, offset)
        if not isinstance(gap, Iterable):
            gap = (gap, gap)
        # -------------------- Attributes -------------------- #
        self.window         = window
        self.floors         = floors
        self.init_x         = init_x
        self.spawn_width    = spawn_width
        self.offset         = offset
        self.gap            = gap
        self.velocity_x     = velocity_x
        self.velocity_y     = velocity_y
        self.pipes          = [Pipe(self.init_x + self.spawn_width, *self.get_params())]

        self.reset()

    def get(self):
        for pipe in self.pipes:
            if not pipe.passed:
                return pipe
        raise RuntimeError(f"No valid pipes available")

    def get_params(self):
        return self.window, self.floors, *self.offset, *self.gap, self.velocity_x, self.velocity_y

    def reset(self):
        self.pipes  = [Pipe(self.init_x + self.spawn_width, *self.get_params())]
        self.update()

    def add(self):
        pipe = self.pipes[-1]
        x = pipe.x + pipe.width + self.spawn_width
        while x <= self.window.width:
            pipe = Pipe(x, *self.get_params())
            self.pipes.append(pipe)
            x = pipe.x + pipe.width + self.spawn_width

    def delete(self):
        index = 0
        while index < len(self.pipes):
            pipe = self.pipes[index]
            if pipe.x + pipe.width < 0:
                self.pipes.pop(index)
            else:
                index += 1

    def update(self):
        self.delete()
        self.add()

    def move(self):
        for pipe in self.pipes:
            pipe.move()

    def render(self):
        for pipe in self.pipes:
            pipe.render()


def blitRotateCenter(surf: pygame.Surface, image: pygame.Surface, topleft: tuple[int, int], tilt: float):
    rotated_image = pygame.transform.rotate(image, tilt)
    new_rect      = rotated_image.get_rect(center=image.get_rect(topleft=topleft).center)

    surf.blit(rotated_image, new_rect.topleft)
    return rotated_image


class Birds(object):
    images: list[pygame.Surface] = None
    images_anti: list[pygame.Surface] = None
    TERMINAL_VEL    = 14
    MAX_ROTATION    = 25
    ANG_VEL         = 30
    ANIME_TIME      = 4
    PASS_THRESHOLD = 0.5

    def __init__(self, window: Window, floors: FloorHandler, pipes: PipesHandler,
                 num: int, init_x: int = 200, init_y: int = 200, velocity: float = 10.5,
                 type2count: int = None, type2offset: int = 0,
                 threshold: float = 0.9, full_state=False,
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float32):
        # -------------------- PyGame -------------------- #
        if self.images is None:
            self.images: list       = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", f"bird{x}.png"))) for x in range(1, 4)]
            self.images_anti: list  = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", f"anti{x}.png"))) for x in range(1, 4)]
        self.image_num          = len(self.images)
        assert self.image_num == 3
        self.ANIMATIONS: list[int] = list(range(self.image_num)) + list(reversed(list(range(self.image_num-1))))
        self.ANIME_MULT_MAX     = len(self.ANIMATIONS)
        # -------------------- Attributes -------------------- #
        self.window     = window
        self.floors     = floors
        self.pipes      = pipes
        self.init_x, self.init_y = init_x, init_y
        self.x          = torch.full((num,), init_x, device=device, dtype=dtype)
        self.y          = torch.full((num,), init_y, device=device, dtype=dtype)
        self.tilt       = torch.full((num,), 0, device=device, dtype=dtype)
        self.tick_count = torch.full((num,), 0, device=device, dtype=torch.int32)
        self.vel        = torch.full((num,), 0, device=device, dtype=dtype)
        self.height     = self.y.clone()
        self.img_count  = torch.full((num,), 0, device=device, dtype=torch.int32)
        self.img_ref    = torch.full((num,), 0, device=device, dtype=torch.int32)
        self.score      = torch.full((num,), 0, device=device, dtype=dtype)
        self.prev_score = torch.zeros_like(self.score)
        self.dead       = torch.full((num,), False, device=device, dtype=torch.bool)
        self.is_anti    = torch.zeros(num, device=device, dtype=torch.bool)

        # use mapping to reduce calculation on dead birds
        self.mapping: dict[int, int] = Dict([(idx, idx) for idx in range(num)])
        if type2count:
            self.is_anti[-type2count:] = True
        self.x[self.is_anti] -= type2offset

        self.bird_num = num
        self.velocity = velocity
        self.threshold = threshold
        self.full_state = full_state

    def reset(self):
        self.x[:] = self.init_x
        self.y[:] = self.init_y
        self.height = self.y.clone()
        self.tilt[:] = self.tick_count[:] = self.vel[:] = 0.0
        self.img_count[:] = self.img_ref[:] = self.score[:] = 0.0
        self.prev_score = torch.zeros_like(self.score)
        self.dead[:] = False

    @property
    def alive(self):
        return ~self.dead

    def jump(self, activation: Tensor):
        if len(activation) != self.bird_num:
            raise ValueError(f"Activation num do not match; Got {len(activation)}, expected {self.bird_num}")
        if activation.ndim == 3:
            activation = activation[:, -1]
        elif activation.ndim >= 4:
            raise ValueError(f"Unsupported activation shape; Got {activation.shape}, expected 3 or 2")
        # activation shape (batch_size / seq_len, genomes, features)
        activation                  = ~self.dead & (activation[:, 0] >= self.threshold)
        self.vel[activation]        = -self.velocity
        self.tick_count[activation] = 0
        self.height[activation]     = self.y[activation].clone()
        # sys.exit(1)

    def move(self):
        self.tick_count += 1
        mask = ~self.dead

        # For downward acceleration +y direction is downwards in PyGame
        FALL_COEFF = 1.5
        displacement = (self.vel * self.tick_count) + (FALL_COEFF * self.tick_count ** 2)

        # Terminal velocity clip
        tv_mask = displacement >= self.TERMINAL_VEL
        displacement[tv_mask] = ((displacement / torch.abs(displacement)) * self.TERMINAL_VEL)[tv_mask]

        # No displacement clip
        JUMP_VEL = 2
        nd_mask = displacement < 0
        displacement[nd_mask] -= JUMP_VEL

        # Add displacement
        self.y[mask] += displacement[mask]

        # tilt up
        fall_hold = (displacement < 0) | (self.y < (self.height + 50))
        tu = fall_hold & (self.tilt < self.MAX_ROTATION)
        self.tilt[mask & tu] = self.MAX_ROTATION

        # tilt down
        td = ~fall_hold & (self.tilt > -90)
        self.tilt[mask & td] -= self.ANG_VEL
        pass

    def render(self):
        alive = ~self.dead

        self.img_count[alive] += 1

        # For animation of bird, loop through three images
        prev_mask = torch.zeros_like(self.img_count, dtype=torch.bool)
        for idx, anime_idx in enumerate(self.ANIMATIONS):
            if idx != len(self.ANIMATIONS)-1:
                multiplier = idx + 1
                ref_mask = alive & (self.img_count < self.ANIME_TIME * multiplier) & ~prev_mask
            else:
                multiplier = idx
                ref_mask = alive & (self.img_count >= self.ANIME_TIME * multiplier) & ~prev_mask
            self.img_ref[ref_mask] = anime_idx
            if idx == len(self.ANIMATIONS)-1:
                self.img_count[ref_mask] = 0
            prev_mask = ref_mask

        # so when bird is nose diving it isn't flapping
        nd_mask = alive & (self.tilt <= -80)
        self.img_ref[nd_mask] = 1
        self.img_count[nd_mask] = self.ANIME_TIME * 2

        # tilt the bird
        for anti, ref, x, y, tilt, dead in zip(
                self.is_anti[alive], self.img_ref[alive], self.x[alive], self.y[alive], self.tilt[alive], self.dead[alive]
        ):
            if not dead:
                _ = blitRotateCenter(
                    surf=self.window.display,
                    image=(self.images if not anti.item() else self.images_anti)[ref],
                    topleft=(x.item(), y.item()),
                    tilt=tilt.item()
                )

                if anti.item():
                    pass
                pass

    def get_mask(self):
        return [pygame.mask.from_surface(self.images[ref]) if not dead else None
                for ref, dead in zip(self.img_ref, self.dead)]

    def check_collision(self):
        pipe = self.pipes.get()
        # Pipe mask
        top_mask = pygame.mask.from_surface(pipe.pipe_top)
        bot_mask = pygame.mask.from_surface(pipe.pipe_bottom)
        # pipe_mid = (pipe.gap_top + pipe.gap_bot) / 2

        zip_ = zip(
            self.get_mask(), self.x.detach().cpu(),
            self.y.detach().cpu(), self.dead.detach().cpu()
        )
        # max_distance = np.sqrt(self.window.height ** 2 + (self.window.width / 2) ** 2).item()
        for idx, (bird_mask, x, y, dead) in enumerate(zip_):
            x, y, dead = x.item(), y.item(), dead.item()
            if not dead:
                top_offset = (int(pipe.x - x), int(pipe.y_top - round(y)))
                bot_offset = (int(pipe.x - x), int(pipe.y_bot - round(y)))

                top_pipe_collision = bird_mask.overlap(bot_mask, bot_offset)
                bot_pipe_collision = bird_mask.overlap(top_mask, top_offset)
                pipe_collision = top_pipe_collision or bot_pipe_collision
                out_of_bounds = (y <= 0.0) or (y + bird_mask.get_size()[1] >= self.floors.y)

                # distance_left = np.sqrt((x - pipe.x) ** 2 + (y - pipe_mid) ** 2).item() / max_distance
                if out_of_bounds:
                    self.dead[idx] = True
                    # self.score[idx] -= distance_left * 1000 * 3
                elif pipe_collision:
                    self.dead[idx] = True
                    # self.score[idx] -= distance_left * 1000

    def check_passed(self):
        self.score[~self.dead] += 1
        self.score[self.dead] += -1
        triggered = False
        res = False
        for pipe in self.pipes.pipes:
            if not pipe.passed:
                passed = (pipe.x + (pipe.image.get_width() * pipe.pxt * self.PASS_THRESHOLD) < self.x) & ~self.dead
                if not triggered and torch.any(passed):
                    self.score[passed] += 100
                    self.score[~passed] += -100
                    # centre = pipe.top + (pipe.top - pipe.bottom)/2
                    # self.score[~self.dead] -= ((centre - self.y[~self.dead])/WIN_HEIGHT)**2 + \
                    #                           ((pipe.x+pipe.image.get_width()*pipe.pxt - self.x[~self.dead])/WIN_WIDTH)**2
                    pipe.passed = True
                    res = True
                    triggered = True
        return res

    def get_state(self):
        pipe = self.pipes.get()
        # Get state(genomes, seq_len=1, action_features)
        state_features = [
            self.y / self.window.height,
            (self.y - pipe.gap_top) / self.window.height,
            (self.y - pipe.gap_bot) / self.window.height,
        ]
        if self.full_state:
            state_features.extend([
                (pipe.x - self.x) / self.window.width,
                (pipe.x + pipe.width - self.x) / self.window.width,
            ])
            if self.pipes.velocity_y != 0:
                state_features.extend([
                    torch.tensor([data], device=self.y.device, dtype=self.y.dtype).expand(self.bird_num)
                    for data in pipe.vertical_offset
                ])
        tensor = torch.stack(state_features, dim=-1) # .unsqueeze(0)
        # disabled = torch.full_like(tensor, -1).to(tensor.device, tensor.dtype)
        tensor[self.dead, :] = -1
        return tensor

    def get_reward(self):
        return self.score # - self.prev_score

    def get_images(self, index: int):
        return (self.images if not self.is_anti[index] else self.images_anti)[self.img_ref[index]]

    @property
    def active_num(self):
        return torch.sum(self.dead == 0).item()


class Renderer(object):
    def __init__(self, obj, timeout=10):
        self.obj = obj
        self.timeout = timeout
        self.data = {'obj': obj, 'timeout': timeout}
        self.states = {'terminated': False, 'started': False, 'running': False}
        self.process: mp.Process = None

    @staticmethod
    def run(data, states):
        states['started'] = True
        while not states['terminated']:
            states['running'] = True
            # data['obj'].draw()
        # clock.sleep(data['timeout'])
        states['running'] = False

    def running(self):
        return self.states['running']

    def start(self):
        if self.process is not None:
            print(f"\n")
            ts = clock.perf_counter()
            while self.process.is_alive():
                print(f"\r... waiting for process to end", end='')
                if clock.perf_counter() - ts >= self.timeout:
                    print(f"\n... forcefully terminating the process")
                    self.process.terminate()
            print(f"\n")

        self.process = mp.Process(target=self.run, args=(self.data, self.states), daemon=True)
        self.process.start()

    def stop(self):
        self.states['terminated'] = True
        # if self.process.is_alive():
        #     raise RuntimeError(f"Failed to stop process.")
        self.states['started'] = self.states['running'] = False


class Game(Env):
    COLOR_TEXT = (255, 255, 255)
    COLOR_LINE = (255, 0, 0)

    def __init__(self, count: int, goal: int = 20, seq_len: int = None, type2count: int = None, type2offset: int = 0,
                 height: int = 800, width: int = 500, width_ext: int = 0, floor: int = 100, hitbox: int = 10,
                 spawn_width: int = 200, gap_offset: int | tuple[int, int] = 100, gap_size: int | tuple[int, int] = 200, velocity: int = 6,
                 init_x=100, init_y=300, init_pipe_x: int = None, pipe_y_velocity = 3,
                 full_state = False, tick: int = 256,
                 threshold=0.9, device: torch.device = 'cpu', dtype: torch.dtype = torch.float32):
        super(Game, self).__init__()
        if init_pipe_x is None:
            init_pipe_x = init_x * 3
        # -------------------- Attributes -------------------- #
        self.window     = Window(height, width, width_ext, hitbox,)
        self.floor      = FloorHandler(self.window, floor, velocity)
        self.pipes      = PipesHandler(self.window, self.floor, init_pipe_x, spawn_width, gap_offset, gap_size,
                                       velocity, pipe_y_velocity)
        self.birds      = Birds(self.window, self.floor, self.pipes,
                                count, init_x, init_y, 10.0, type2count, type2offset, threshold, full_state,
                                device, dtype)
        self.renderer   = Renderer(self)
        self.clock      = pygame.time.Clock()
        self.goal       = goal
        self.sequential = seq_len is not None and seq_len > 0
        self.seq_len    = seq_len
        self.tick_value = tick
        # -------------------- Gym -------------------- #
        inputs = (3 if not full_state else 5 + (2 if pipe_y_velocity != 0 else 0))
        self.observation_space = spaces.Box(
            low=-3, high=+3, shape=(count, inputs) if not self.sequential else (count, seq_len, inputs)
        )
        self.action_space = spaces.Box(
            low=-3, high=+3, shape=(count, 1) if not self.sequential else (count, seq_len, 1)
        )
        # -------------------- States -------------------- #
        self.buffer = torch.zeros(self.observation_space.shape, dtype=dtype, device=device)
        # self.prev_score: Tensor = None
        self.generation = 0
        self.score = 0
        self.terminated = False
        self.device = device
        self.dtype = dtype
        self.steps = 0

        self.window.close()
        self.initialized = False

    def tick(self, val=240):
        self.clock.tick(val)

    def initialize(self, keys: list[int] = None):
        if not self.initialized:
            self.window.initialize()
            self.initialized = True
        self.floor.reset()
        self.pipes.reset()
        self.birds.reset()
        self.buffer[:] = 0.0
        self.score = 0
        self.terminated = False
        self.steps = 0
        self.window.close()

    def update(self, activation: Tensor):
        self.birds.score[:] = 0 # prev_score = self.birds.score
        self.birds.jump(activation)
        self.birds.move()
        self.birds.check_collision()
        if self.birds.check_passed():
            self.score += 1
        self.floor.update()
        self.pipes.update()
        self.floor.move()
        self.pipes.move()

    def get_state(self):
        return self.birds.get_state()

    def get_reward(self):
        return self.birds.get_reward()

    def reset(self, keys: list[int] = None, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Tensor, dict[str, Any]]:
        self.initialize()
        state = self.get_state()
        if self.sequential:
            self.buffer[:, :-1] = self.buffer[:, 1:].clone()
            self.buffer[:, -1] = state
            state = self.buffer.cpu().clone().to(self.device)
        return state, {}

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, bool, bool, dict[str, Any]]:
        if not self.terminated:
            self.update(action)

            state = self.get_state()
            if self.sequential:
                self.buffer[:, :-1] = self.buffer[:, 1:].clone()
                self.buffer[:, -1] = state
                state = self.buffer.cpu().clone().to(self.device)

            self.terminated = self.score >= self.goal or self.birds.active_num <= 0
            if self.terminated:
                self.birds.score[~self.birds.dead] += 10000
                self.birds.score[self.birds.dead] -= 2000
                self.generation += 1

            reward = self.get_reward()

            return state, reward, self.terminated, self.terminated, {}
        else:
            raise RuntimeError(f"Game has ended")

    def _draw(self, debug=False, testing=False):
        if not self.window.initialized:
            self.window.initialize()
        if self.window.initialize:
            if not testing and self.steps % 20 == 0: # and self.steps % 30 == 0:
                pygame.display.quit()
                self.window.initialize()
                pass

            if self.tick_value is not None:
                self.tick(self.tick_value)
            # Display window
            self.window.render()

            # Draw pipes
            self.pipes.render()

            # Draw gpu
            self.floor.render()

            # Draw birds or debug
            self.birds.render()
            for bird_index, dead in enumerate(self.birds.dead):
                # render lines from bird to pipe
                if debug and not dead:
                    pipe = self.pipes.get()
                    try:
                        x                  = self.birds.x[bird_index].item()
                        y                  = self.birds.y[bird_index].item()
                        img                = self.birds.get_images(bird_index)
                        pxt = pipe.pxt
                        bird_center        = (x + img.get_width() / 2, y + img.get_height() / 2)
                        pipe_top_center    = (pipe.x + pipe.pipe_top.get_width() * pxt, pipe.gap_top)
                        pipe_bottom_center = (pipe.x + pipe.pipe_bottom.get_width() * pxt, pipe.gap_bot)

                        pygame.draw.line(self.window.display, self.COLOR_LINE, bird_center, pipe_top_center, 5)
                        pygame.draw.line(self.window.display, self.COLOR_LINE, bird_center, pipe_bottom_center, 5)
                    except KeyboardInterrupt:
                        pass

            # score
            score_label = self.window.stat_font.render(f"Score: {self.score:.2f}", 1, self.COLOR_TEXT)
            self.window.display.blit(score_label, (self.window.width - score_label.get_width() - 15, 10))

            # generations
            score_label = self.window.stat_font.render(f"Gens: {self.generation}", 1, self.COLOR_TEXT)
            self.window.display.blit(score_label, (10, 10))

            # alive
            score_label = self.window.stat_font.render(f"Alive: {self.birds.active_num}", 1, self.COLOR_TEXT)
            self.window.display.blit(score_label, (10, 50))

            pygame.display.update()

    def render(self, debug=False, testing=False):
        if not self.terminated:
            # if not self.renderer.running():
            #     self.renderer.start()
            self._draw(testing=testing)
            self.steps += 1
        if self.terminated:
            self.renderer.stop()

    def test(self):
        state = self.reset()[0]

        states: list[Tensor] = []
        rewards: list[Tensor] = []

        self.render(debug=True, testing=True)
        started = False
        while not self.terminated:
            self.clock.tick(20)
            action = torch.zeros(self.action_space.shape, device=self.device, dtype=self.dtype)
            if not started:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.terminated = True
                    if event.type in [pygame.K_SPACE, pygame.KEYUP]:
                        action[:] = 1.0
                        started = True
            if started:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.terminated = True
                    if event.type in [pygame.K_SPACE, pygame.KEYUP]:
                        action[:] = 1.0
                next_state, reward, _, self.terminated, _ = self.step(action)
                states.append(state.cpu())
                rewards.append(reward.cpu())
                state = next_state

            self.render(debug=True, testing=True)
        self.window.close()

        return states, rewards


# test_game = Game(1, 10, height=800, width=500, floor=70, device=DEVICE, dtype=DTYPE)
# for _ in range(10):
#     test_game.test()


class BaseModel(Model):
    def __init__(self, inputs: int, outputs: int, dim_size: int, layers: int, norm_groups=1, activation=nn.SiLU(),
                 probabilistic=False, bias=True, device: torch.device = 'cpu', dtype: torch.device = torch.float32, **options):
        super().__init__()
        # Attributes
        self.inputs         = inputs
        self.outputs        = outputs
        self.dim_size       = dim_size
        self.layers         = layers
        self.distribution   = options.get('distribution', 'normal')
        self.stride         = 1
        self.norm_groups    = norm_groups
        self.probabilistic  = probabilistic
        self.clip_min       = options.get('clip_min', -4)
        self.clip_max       = options.get('clip_max', 0)
        self.clip_range     = self.clip_max - self.clip_min

        # Build
        self.projection = mn.Sequential(*[
            mn.Linear(inputs, dim_size, True, device, dtype),
            *sum([
                [
                    # mn.LayerNorm(dim_size, bias=False, device=device, dtype=dtype),
                    activation,
                    nn.Linear(dim_size, dim_size, bias, device, dtype),
                ]
                for _ in range(layers)
            ], []),
        ])
        self.pol_proj = mn.Sequential(*[
            # mn.Linear(dim_size, dim_size, bias, device, dtype),
            # mn.LayerNorm(dim_size, bias=False, device=device, dtype=dtype),
            activation,
            mn.Linear(dim_size, 2*outputs, True, device, dtype)
        ])

    def extra_repr(self) -> str:
        return f"probabilistic={self.probabilistic}, distro='{self.distribution}'"

    def forward(self, state: Tensor, keys: Union[int, list[int]] = None, **kwargs):
        return self.get_policy(state, keys=keys, **kwargs)

    def get_mean_std(self, latent: Tensor, keys: Union[int, list[int]] = None) -> Tensor:
        mean_std        = self.pol_proj(latent, keys=keys)
        mean, log_std   = torch.chunk(mean_std, 2, -1)
        mean            = F.sigmoid(mean) * 6 + -3
        std             = torch.pow(10, F.sigmoid(log_std) * self.clip_range + self.clip_min)
        return mean, std

    def get_action(self, state: Tensor, keys: Union[int, list[int]] = None) -> tuple[Tensor, Tensor]:
        latent      = self.projection(state, keys=keys)
        mean, std   = self.get_mean_std(latent, keys=keys)
        dist        = torch.distributions.Normal(mean, std)
        action      = torch.sigmoid((dist.sample() if self.probabilistic else mean) * torch.pi)
        log_prob    = dist.log_prob(action)
        return action, log_prob

    def evaluate_action(self, state: Tensor, action: Tensor, keys: Union[int, list[int]] = None) -> [Tensor, Union[Tensor, None]]:
        latent      = self.projection(state, keys=keys)
        mean, std   = self.get_mean_std(latent, keys=keys)
        dist        = torch.distributions.Normal(mean, std)
        log_prob    = dist.log_prob(action)
        entropy     = dist.entropy()
        return log_prob, entropy

    def get_policy(self, state: Tensor, keys: Union[int, list[int]] = None, **options) -> Tensor:
        latent      = self.projection(state, keys=keys)
        mean, std   = self.get_mean_std(latent, keys=keys)
        dist        = torch.distributions.Normal(mean, std)
        action      = torch.sigmoid((dist.sample() if options.get('normal', self.probabilistic) else mean) * torch.pi)
        return action

    # def get_value(self, state: Tensor, keys: Union[int, list[int]] = None) -> Tensor:
    #     latent      = self.projection(state, keys=keys)
    #     value       = self.val_proj(latent, keys=keys)
    #     return value


def fix(value: float, default: float = 1):
    if np.isinf(value) or np.isnan(value):
        return default
    else:
        return value


# GAME SETTINGS
PIPE_Y_VELOCITY = 0
FULL_STATES = False

# Architecture
GENOMES             = 100
INPUTS              = (5 + (2 if PIPE_Y_VELOCITY else 0) if FULL_STATES else 3)
OUTPUTS             = 1
EMBED_SIZE          = 16
NORM_GROUPS         = 1
LAYERS              = 3
ENABLE_BIAS         = True
PROBABILISTIC       = False
MEMORY_SIZE         = 10
GAMMA               = np.exp(np.log(0.33) / 512)
ALPHA               = fix(np.exp(np.log(1.50) / (MEMORY_SIZE - 1)), 1.0)
ALPHA_ORDER         = 4
REW_NORM            = 2
LOSS_REG            = 0.
ACTIVATION          = nn.Tanh()
CLIP_MIN            = -5
CLIP_MAX            = -0
DISTRIBUTION        = 'mult_var_normal'

MODEL0 = BaseModel(INPUTS, OUTPUTS, EMBED_SIZE, LAYERS, NORM_GROUPS, ACTIVATION, PROBABILISTIC, ENABLE_BIAS, DEVICE, DTYPE, clip_min=CLIP_MIN, clip_max=CLIP_MAX, distribution=DISTRIBUTION)
MODEL1 = BaseModel(INPUTS, OUTPUTS, EMBED_SIZE, LAYERS, NORM_GROUPS, nn.ReLU(), PROBABILISTIC, ENABLE_BIAS, DEVICE, DTYPE, clip_min=CLIP_MIN, clip_max=CLIP_MAX, distribution=DISTRIBUTION)
MODEL2 = BaseModel(INPUTS, OUTPUTS, EMBED_SIZE, LAYERS, NORM_GROUPS, nn.SiLU(), PROBABILISTIC, ENABLE_BIAS, DEVICE, DTYPE, clip_min=CLIP_MIN, clip_max=CLIP_MAX, distribution=DISTRIBUTION)

INIT_GEN: int = None

RUNS = 1
GOAL = 20
STEPS = GOAL * 100 * RUNS


print(f"creating config")
config = neat.Config()

config.genome.init_type                     = 'normal'
config.genome.weight_init_mean              = 0.0
config.genome.weight_init_std               = 1.0
config.genome.weight_min_value              = -np.inf
config.genome.weight_max_value              = +np.inf
config.genome.weight_mutate_power           = 3e-1
config.genome.weight_mutate_rate            = 0.70
config.genome.weight_replace_rate           = 0.01
config.genome.weight_add_prob               = 0.0
config.genome.weight_del_prob               = 0.0
config.genome.single_structural_mutation    = False

config.reproduction.min_species_size        = GENOMES
config.reproduction.purge                   = 1
config.reproduction.clone_threshold         = 0.00
config.reproduction.survival_threshold      = 0.20
config.reproduction.cross_threshold         = 0.00
config.reproduction.elitism                 = 30
config.species.compatibility_threshold      = np.inf
config.stagnation.max_stagnation            = 1
config.stagnation.species_elitism           = 2
config.reproduction.darwin_multiplier       = 0.50
config.reproduction.cross_multiplier        = 0.50
config.reproduction.preserve_elite          = False
config.save()
config.load(2)


def evaluate(population: neat.Population, **options):
    trainer: neat.rl.NEAT = options['trainer']
    mapping0, mapping1, mapping2 = population.get_mapping()
    cons_mapping = population.get_mapping(consolidated=True)
    trainer.update_mapping(cons_mapping)
    # BUFFER = torch.zeros(SEQ_LEN, population.pop_size, INPUTS).to(DEVICE, DTYPE)
    global INIT_GEN
    if INIT_GEN is None:
        INIT_GEN = population.generation

    for genome in population.genomes.values():
        genome.fitness = 0

    terminate = False
    start = 0
    run_step = 0
    game_step = 0
    DEBUG = True
    DEBUG_STEP = 0
    MODEL0.train()
    while not terminate:
        env = Game(
            population.size, goal=GOAL,
            height=800, width=800, full_state=FULL_STATES,
            spawn_width=200, tick=None, gap_offset=30, pipe_y_velocity=PIPE_Y_VELOCITY,
            type2count=GENOMES, type2offset=0, device=DEVICE, dtype=DTYPE
        )
        # print(f"Anti Count = {game.birds.}")

        gts = clock.perf_counter()
        step = 0
        reverse_mapping0 = {index: key for key, index in mapping0.items()}
        reverse_mapping1 = {index: key for key, index in mapping1.items()}
        reverse_mapping2 = {index: key for key, index in mapping2.items()}

        states = env.reset()[0]
        done = False
        while not done:
            with torch.no_grad():
                # Get Inputs ~ send bird location, top pipe location and bottom pipe location
                # and determine from network whether to jump or not
                if DEBUG and population.generation == INIT_GEN and step == DEBUG_STEP:
                    print(f"\nobservations =>\n{states}\n\tshape = {states.shape}")

                # Filter dead birds from calculation
                ts = clock.perf_counter()
                keys0, keys1, keys2 = [], [], []
                indices0, indices1, indices2 = [], [], []
                for index, dead in enumerate(env.birds.dead):
                    if not dead:
                        if index in reverse_mapping0:
                            keys0.append(reverse_mapping0[index])
                            indices0.append(index)
                        elif index-len(mapping0) in reverse_mapping1:
                            keys1.append(reverse_mapping1[index-len(mapping0)])
                            indices1.append(index-len(mapping0))
                        elif index-(len(mapping0)+len(mapping1)) in reverse_mapping2:
                            keys2.append(reverse_mapping2[index-(len(mapping0)+len(mapping1))])
                            indices2.append(index-(len(mapping0)+len(mapping1)))
                        else:
                            print(f"\nPopulation size {population.size}"
                                  f"\nReverse mapping \n{reverse_mapping0} \n{reverse_mapping1}"
                                  f"\nIndex = {index}, birds_shape = {env.birds.dead.shape}")
                            raise KeyError()
                if len(indices0) == 0:
                    keys0 = list(mapping0.keys())
                    indices0 = list(mapping0.values())
                if len(indices1) == 0:
                    keys1 = list(mapping1.keys())
                    indices1 = list(mapping1.values())
                if len(indices2) == 0:
                    keys2 = list(mapping2.keys())
                    indices2 = list(mapping2.values())
                if DEBUG and population.generation == INIT_GEN and step == DEBUG_STEP:
                    print(f"keys =>\n{keys0}")
                    print(f"indices =>\n{indices0}")

                # Get actions
                observations0, observations1, observations2 = torch.split(
                    states, [len(mapping0), len(mapping1), len(mapping2)], dim=0
                )
                actions0, probs = MODEL0.get_action(observations0[indices0].unsqueeze(1), keys=keys0)
                actions1, probs = MODEL1.get_action(observations1[indices1].unsqueeze(1), keys=keys1)
                actions2, probs = MODEL2.get_action(observations2[indices2].unsqueeze(1), keys=keys2)
                # shape(seq_len=1, genomes, features_out)
                actions0, actions1, actions2, probs = actions0.squeeze(1), actions1.squeeze(1), actions2.squeeze(1), probs.squeeze(1)
                if DEBUG and population.generation == INIT_GEN and step == DEBUG_STEP:
                    print(f"actions =>\n{actions0}\n\tshape = {actions0.shape}")
                    print(f"probs =>\n{probs}\n\tshape = {probs.shape}")

                # Pad dead bird actions
                if True:
                    padding = env.birds.bird_num - actions0.shape[0] - actions1.shape[0] - actions2.shape[0]
                    if padding > 0:
                        def fill_up(tensor: Tensor, indices: list[int], total: int):
                            fill = tensor.clone()
                            tensor = torch.zeros(
                                total, *tensor.shape[1:], device=DEVICE, dtype=DTYPE
                                )
                            tensor[indices] = fill
                            return tensor
                        actions0 = fill_up(actions0, indices0, len(reverse_mapping0))
                        actions1 = fill_up(actions1, indices1, len(reverse_mapping1))
                        actions2 = fill_up(actions2, indices2, len(reverse_mapping2))
                    actions = torch.cat([actions0, actions1, actions2[..., :OUTPUTS]], dim=0)
                if DEBUG and population.generation == INIT_GEN and step == DEBUG_STEP:
                    print(f"filled actions =>\n{actions}\n\tshape = {actions.shape}")
                calc_time = clock.perf_counter() - ts

                # Get rewards
                next_states, rewards, _, done, _ = env.step(actions)
                if DEBUG and population.generation == INIT_GEN and step == DEBUG_STEP:
                    print(f"rewards =>\n{rewards}\n\tshape = {rewards.shape}")
                    # v = MODEL.get_value(observations[:len(reverse_mapping0)].unsqueeze(1), ).squeeze(1)
                    # print(f"values =>\n{v}\n\tshape = {v.shape}")
                    # del v

                # Updated buffers
                terminate = trainer.update(states, actions, rewards, done, done)

                alive = round(env.birds.active_num)
                max_score = round(rewards.max().item(), 2)
                if alive > 0:
                    best_index = torch.argmax(env.birds.score).cpu().item()
                    if best_index in reverse_mapping0:
                        best_key = reverse_mapping0[best_index]
                    elif best_index in reverse_mapping1:
                        best_key = reverse_mapping1[best_index]
                    elif best_index in reverse_mapping2:
                        best_key = reverse_mapping2[best_index]
                    else:
                        # raise KeyError()
                        best_key = None
                print(f"\r{CM('Executing', Fore.GREEN)}: time_elapsed = {round(clock.perf_counter()-gts)}s, "
                      f"alive = {alive}, max_rew = {max_score}, best_key={best_key}, ct={calc_time:.2e}, sd={trainer.steps_done} "
                      f"bl={trainer.primary.max_size()}", end='')

                # Render display
                if population.generation % 10 == 0:
                    env.render()

                states = next_states

            # break if score gets large enough
            if done or terminate:
                # pickle.dump(population.genomes, open(".\\best.pickle", "wb"))
                terminate = True
                break

            step += 1
            run_step += 1
            if step == DEBUG_STEP:
                DEBUG = False

        print("\n------------------------------ Post debugging ------------------------------")
        print(f"Buffer multiplier = {MEMORY_SIZE}")
        print(f"Mapping = {trainer.episode_mapping}")
        print(f"Lengths = {trainer.episode_lengths}")
        print(f"Episodes = {trainer.primary.episodes()}")
        print("----------------------------------------------------------------------------")

        l_lim, u_lim = 10, env.floor.y * 1.05
        # print(f"u lim = {u_lim}, l lim = {l_lim}")
        for idx, (score, genome) in enumerate(zip(env.birds.get_reward(), population.genomes.values())):
            genome.fitness = score.item()
            y_position = env.birds.y[idx]
            died_beyond_limits = y_position <= l_lim or y_position >= u_lim
            if died_beyond_limits:
                population.to_delete.append(genome.key)

        game_step += 1
    print(f"\n")

    population.save_dict('flappy_bird', replace=population.generation != INIT_GEN)
    # trainer.save('flappy_bird', replace=population.generation != INIT_GEN)


def genome_debug(algorithm: neat.rl.NEAT):
    population = algorithm.population
    COUNT = 5

    def fill(text: str, space: int):
        if not isinstance(text, str):
            text = str(text)
        amount = max(0, int(space - len(text)))
        return f" {' ' * amount}{text} "

    print(f"Population Summary:")
    ks = max(1, np.max(np.log10([g.key for g in population.genomes.values()])).item()) + 1
    gs = 3
    fs = max(1, np.max(np.log10([g.actual_fitness for g in population.genomes.values()])).item()) + 4
    for i, genus in enumerate(population.genera):
        genomes = sorted([g for g in population.genomes.values() if g.genus == genus], key=lambda g: g.fitness, reverse=True)[:COUNT]
        for genome in genomes:
            print(f"\t{fill(genome.key, ks)}|{fill(genome.genus, gs)}|{fill(round(genome.actual_fitness, 4), fs)}")
        if i < len(population.genera) - 1:
            print(f"\t...")


def run():
    # MODEL.single_mode(True)
    # Configuration
    # Create the population, which is the top-level object for a NEAT run.
    print(f"creating population")
    population = neat.Population(GENOMES, MODEL0, config, init_reporter=True)
    population1 = neat.Population(GENOMES, MODEL1, config, init_reporter=True)
    population2 = neat.Population(GENOMES, MODEL2, config, init_reporter=True)
    population.absorb_population(population1)
    population.absorb_population(population2)
    print(MODEL0.pol_proj)
    # population.load_dict(name='flappy_bird', file_no=None)

    TRAIN = True
    if TRAIN:
        trainer = neat.rl.NEAT(
            population,
            schedulers=[
                # neat.optim.scheduler.RandomAnnealing(config, 1e-1, 1e-0, 5, ['weight_init_std', 'weight_mutate_power'], True),
                # neat.optim.scheduler.CosineAnnealing(config, 10, 0.1, 'weight_mutate_rate', True, True),
                # neat.optim.scheduler.CosineAnnealing(config, 10, 0.1, 'weight_replace_rate', True, True),
                # neat.optim.scheduler.CosineAnnealing(config, 15, 0.05, 'weight_add_prob', True, True),
                # neat.optim.scheduler.CosineAnnealing(config, 15, 0.05, 'weight_del_prob', True, True),
            ],
            device=DEVICE, dtype=DTYPE,
            log_sub_dir='flappy_bird\\',
            log_name=f"{unix_to_datetime_file(clock.time())}_"
                     f"e{EMBED_SIZE}-l{LAYERS}--b{int(ENABLE_BIAS)}-"
                     f"g{round(GAMMA, 4)}-a{round(ALPHA, 4)}-ao{ALPHA_ORDER}-"
                     f"rn{REW_NORM}-p{round(LOSS_REG, 4)}",
            gamma=GAMMA, alpha=ALPHA, order=ALPHA_ORDER, normalize=REW_NORM,
            rew_reg=1.0, pol_reg=0.0, validate=True, groups=None,
            max_episodes=MEMORY_SIZE,
        )
        trainer.set_report_hook(genome_debug)

        print(f"starting evaluation: population={len(population.genomes)}")
        # trainer.load(name='flappy_bird', file_no=None)
        try:
            trainer.learn(evaluate, STEPS, 100, 2048, 0.1, 'binary', 2)
        except KeyboardInterrupt:
            pass
    else:
        population.load_dict(name='flappy_bird', file_no=None)

    env = Game(
        population.size, goal=100,
        height=800, width=1100, full_state=FULL_STATES,
        spawn_width=200, tick=None, gap_offset=30, pipe_y_velocity=PIPE_Y_VELOCITY,
        type2count=GENOMES, type2offset=0, device=DEVICE, dtype=DTYPE
    )
    mapping0, mapping1, mapping2 = population.get_mapping()
    cons_mapping = population.get_mapping(consolidated=True)
    for i in range(10):
        done = False
        step = 0
        states = env.reset()[0]
        ts = clock.perf_counter()
        while not done:
            # Get actions
            with torch.no_grad():
                observations0, observations1, observations2 = torch.split(
                    states, [len(mapping0), len(mapping1), len(mapping2)], dim=0
                )
                actions0, probs = MODEL0.get_action(observations0.unsqueeze(1))
                actions1, probs = MODEL1.get_action(observations1.unsqueeze(1))
                actions2, probs = MODEL2.get_action(observations2.unsqueeze(1))
                actions0, actions1, actions2, probs = \
                    actions0.squeeze(1), actions1.squeeze(1), actions2.squeeze(1), probs.squeeze(1)
            actions = torch.cat([actions0, actions1, actions2[..., :OUTPUTS]], dim=0)

            # Get rewards
            next_states, rewards, _, done, _ = env.step(actions)

            # Get next states
            states = next_states

            # Render
            env.render()

            print(f"\rTime elapsed: {clock.perf_counter() - ts:.2f}s, "
                  f"Alive = {env.birds.active_num} "
                  f"Score = {env.score} ",
                  end='')
        print(f" ")

    # while True:
    #     evaluate(population, trainer=trainer)

    # for p in population.get(winner):
    #     print(p)

    # show final stats
    # print('\nBest genome:\n{!s}'.format(winner.key))


if __name__ == '__main__':
    print(MODEL0)
    run()
