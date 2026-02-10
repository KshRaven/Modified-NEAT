
from gymnasium import Env, spaces
from torch import Tensor
# from numba import njit, prange
from numba.typed import List, Dict
from numba.core.errors import NumbaPerformanceWarning
from typing import Any, Iterable
# from gymnasium.core import ObsType, ActType

import torch
import pygame
import random
import os
import time as clock
# import numpy as np
import warnings
import multiprocessing as mp


warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
torch.set_printoptions(threshold=10)


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
    FALL_COEFF      = 1.75 # 1.5
    TERMINAL_VEL    = 20 # 14
    JUMP_VEL        = 7.5 # 2
    MAX_ROTATION    = 25
    ANG_VEL         = 30
    ANIME_TIME      = 4
    PASS_THRESHOLD = 0.5

    def __init__(self, window: Window, floors: FloorHandler, pipes: PipesHandler,
                 num: int, init_x: int = 200, init_y: int = 200, velocity: float = 10.0,
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
        displacement = (self.vel * self.tick_count) + (self.FALL_COEFF * self.tick_count ** 2)

        # Terminal velocity clip
        tv_mask = displacement >= self.TERMINAL_VEL
        displacement[tv_mask] = ((displacement / torch.abs(displacement)) * self.TERMINAL_VEL)[tv_mask]

        # No displacement clip
        nd_mask = displacement < 0
        displacement[nd_mask] -= self.JUMP_VEL

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
                 full_state = False, tick: int | None = 256, delay: int | None = None,
                 threshold=0.9, device: torch.device | str = 'cpu', dtype: torch.dtype = torch.float32):
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
        self.offset     = min(seq_len - 1, max(0, delay)) if delay is not None and seq_len is not None else 0
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
        self.offset_select = - (torch.arange(self.offset, dtype=torch.long, device=device) + 1)
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
        # TODO: Make the keys initialization work
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
            if self.offset > 0:
                state[:, self.offset_select] = 0.0
        return state, {}

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, bool, bool, dict[str, Any]]:
        if not self.terminated:
            self.update(action)

            state = self.get_state()
            if self.sequential:
                self.buffer[:, :-1] = self.buffer[:, 1:].clone()
                self.buffer[:, -1] = state
                state = self.buffer.clone().to(self.device)
                if self.offset > 0:
                    state[:, self.offset_select] = 0.0


            self.terminated = self.score >= self.goal or self.birds.active_num <= 0
            if self.terminated:
                self.birds.score[~self.birds.dead] += 10000
                self.birds.score[self.birds.dead] -= 2000
                self.generation += 1

            reward = self.get_reward().clone()

            return state, reward, self.terminated, self.terminated, {}
        else:
            raise RuntimeError(f"Game has ended")

    def _draw(self, debug=False, testing=False):
        if not self.window.initialized:
            self.window.initialize()
        if self.window.initialize:
            # if not testing and self.steps % 20 == 0: # and self.steps % 30 == 0:
            #     pygame.display.quit()
            #     self.window.initialize()
            #     pass

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
                        pipe_top_center    = (pipe.x + (pipe.image.get_width() * pipe.pxt * self.birds.PASS_THRESHOLD), pipe.gap_top)
                        pipe_bottom_center = (pipe.x + (pipe.image.get_width() * pipe.pxt * self.birds.PASS_THRESHOLD), pipe.gap_bot)

                        pygame.draw.line(self.window.display, self.COLOR_LINE, bird_center, pipe_top_center, 5)
                        pygame.draw.line(self.window.display, self.COLOR_LINE, bird_center, pipe_bottom_center, 5)
                    except KeyboardInterrupt:
                        pass
                    except Exception as e:
                        print(f"Unable to debug")
                        raise e

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
            self._draw(debug=debug, testing=testing)
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


if __name__ == '__main__':
    test_game = Game(
        1, 10, height=800, width=800, floor=70,
        gap_size=200, velocity=6,
        device=torch.device('cpu'), dtype=torch.float32
    )
    while True:
        test_game.test()