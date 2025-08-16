
from ModifiedNEAT.util.fancy_text import CM, Fore
from ModifiedNEAT.nn.base import Model
from ModifiedNEAT.nn.modules.sub import Linear, Conv1d, Transpose, ResidualBlock, Sequential, GroupNorm, ConverBase, SequenceEncoding
from ModifiedNEAT.nn.modules import Reformer
from ModifiedNEAT.util.datetime import unix_to_datetime_file

import ModifiedNEAT as neat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch import Tensor
from numba import njit, prange
from numba.typed import List, Dict
from numba.core.errors import NumbaPerformanceWarning
from typing import Union

import pygame
import random
import os
import time as clock
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
torch.set_printoptions(threshold=10)
pygame.font.init()  # init font

DEVICE = 'cpu' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32

# Define window
THRESHOLD = 0.9
WIN_HEIGHT = 800
WIN_WIDTH  = 450
WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

FLOOR = 730
STAT_FONT = pygame.font.SysFont("comicsans", 50)
# END_FONT = pygame.font.SysFont("comicsans", 70)
DRAW_LINES = False

# PIPE_IMG    = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")).convert_alpha())
# BG_IMG      = pygame.transform.scale(pygame.image.load(os.path.join("imgs", "bg.png")).convert_alpha(), (600, 900))
# BIRD_IMGS   = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird" + str(x) + ".png")))
#                for x in range(1, 4)]
# BASE_IMG    = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "cuda.png")).convert_alpha())

gen = 0


class Pipe(object):
    image = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")).convert_alpha())
    u_lim: int = 20
    l_lim: int = 40
    pxt: float = 1.0

    def __init__(self, x: int, gap_l_lim=200, gap_u_lim=200, velocity=6):
        self.gap_l_lim = gap_l_lim
        self.gap_u_lim = gap_u_lim
        self.velocity = velocity
        self.x = x
        self.height = 0

        # where the top and bottom of the pipe is
        self.top = 0
        self.bottom = 0

        self.pipe_top = pygame.transform.flip(self.image, False, True)
        self.pipe_bottom = self.image

        self.passed = False
        self.collision = False

        self.set_height()

    def set_height(self,):
        self.height = float(random.randrange(self.u_lim, self.l_lim))
        self.top    = self.height - self.pipe_top.get_height()
        self.bottom = self.height + random.randint(self.gap_l_lim, self.gap_u_lim)

    def move(self):
        self.x -= self.velocity

    def draw(self, win):
        # draw top
        win.blit(self.pipe_top, (self.x, self.top))
        # draw bottom
        win.blit(self.pipe_bottom, (self.x, self.bottom))

    def collide(self, bird, win):
        """
        returns if a point is colliding with the pipe
        :param bird: Bird object
        :return: Bool
        """
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.pipe_top)
        bottom_mask = pygame.mask.from_surface(self.pipe_bottom)
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if b_point or t_point:
            return True

        return False


class Base:
    def __init__(self, y: int):
        self.image = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")).convert_alpha())
        self.velocity = 6
        self.width = self.image.get_width()

        self.y = y
        self.x1 = 0
        self.x2 = self.width

    def move(self):
        self.x1 -= self.velocity
        self.x2 -= self.velocity

        if self.x1 + self.width < 0:
            self.x1 = self.x2 + self.width

        if self.x2 + self.width < 0:
            self.x2 = self.x1 + self.width

    def draw(self, win):
        win.blit(self.image, (self.x1, self.y))
        win.blit(self.image, (self.x2, self.y))


def blitRotateCenter(surf: pygame.Surface, image: pygame.Surface, topleft: tuple[int, int], tilt: float):
    rotated_image = pygame.transform.rotate(image, tilt)
    new_rect      = rotated_image.get_rect(center=image.get_rect(topleft=topleft).center)

    surf.blit(rotated_image, new_rect.topleft)
    return rotated_image


class Window(object):
    def __init__(self, height: int = 800, width: int = 500, hitbox: int = 20):
        self.height = height
        self.width = width
        self.display = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))

        self.ceil  = hitbox
        self.floor = 730

        self.stat_font = pygame.font.SysFont("comicsans", 50)
        self.end_font = pygame.font.SysFont("comicsans", 70)
        self.draw_lines = False

        self.background_image = pygame.transform.scale(pygame.image.load(
            os.path.join("imgs", "bg.png")).convert_alpha(), (600, 900))


class Pipes(object):
    def __init__(self, gen_pos: int):
        self.gen_pos = gen_pos
        self.pipes = [Pipe(self.gen_pos)]
        self.to_del = []

    def get(self):
        for pipe in self.pipes:
            if not pipe.passed:
                return pipe
        return self.pipes[0]

    def add(self):
        self.pipes.append(Pipe(self.gen_pos))

    def delete(self):
        for pipe in self.to_del:
            self.pipes.remove(pipe)
        self.to_del = []

    def move(self):
        for pipe in self.pipes:
            if pipe.x + pipe.pipe_top.get_width() < 0:
                self.to_del.append(pipe)
            pipe.move()

    def draw(self, win):
        for pipe in self.pipes:
            pipe.draw(win)


class Birds(object):
    def __init__(self, num: int, x: int = 200, y: int = 200, type2count: int = None, offset=0,
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float32):
        self.x          = torch.full((num,), x, device=device, dtype=dtype)
        self.y          = torch.full((num,), y, device=device, dtype=dtype)
        self.tilt       = torch.full((num,), 0, device=device, dtype=dtype)
        self.tick_count = torch.full((num,), 0, device=device, dtype=torch.int32)
        self.vel        = torch.full((num,), 0, device=device, dtype=dtype)
        self.height     = self.y.clone()
        self.img_count  = torch.full((num,), 0, device=device, dtype=torch.int32)
        self.img_ref    = torch.full((num,), 0, device=device, dtype=torch.int32)
        self.score      = torch.full((num,), 0, device=device, dtype=dtype)
        self.dead       = torch.full((num,), False, device=device, dtype=torch.bool)
        # use mapping to reduce calculation on dead birds
        self.mapping: dict[int, int] = Dict([(idx, idx) for idx in range(num)])

        self.images: list = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", f"bird{x}.png"))) for x in range(1, 4)]
        self.images_anti: list = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", f"anti{x}.png"))) for x in range(1, 4)]
        self.is_anti = torch.zeros(num, dtype=torch.bool)
        if type2count:
            self.is_anti[-type2count:] = True
        self.x[self.is_anti] -= offset
        self.MAX_ROTATION = 25
        self.ANG_VEL = 20
        image_num = len(self.images)
        assert image_num == 3
        self.ANIMATIONS: list[int] = list(range(image_num)) + list(reversed(list(range(image_num-1))))
        self.ANIME_MULT_MAX = len(self.ANIMATIONS)
        self.ANIME_TIME = 5

        self.bird_num = num
        self.active_num = num

    def get_alive(self):
        return self.dead

    def order(self):
        pass

    def jump(self, activation: Tensor):
        if len(activation) != len(self.dead):
            raise ValueError(f"Activation num do not match; Got {len(activation)}, expected {len(self.dead)}")
        if activation.ndim == 2:
            activation = activation.squeeze(-1)
        # activation shape (batch_size / seq_len, genomes, features)
        activation = (~self.dead & (activation >= THRESHOLD))
        self.vel[activation]        = -10.5
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
        TERMINAL_VEL = 16
        tv_mask = displacement >= TERMINAL_VEL
        displacement[tv_mask] = ((displacement / torch.abs(displacement)) * TERMINAL_VEL)[tv_mask]

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

    def draw(self, win):
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
        for anti, ref, x, y, tilt, dead in zip(self.is_anti[alive], self.img_ref[alive], self.x[alive], self.y[alive], self.tilt[alive], self.dead[alive]):
            if not dead:
                rot_image = blitRotateCenter(
                    surf=win,
                    image=(self.images if not anti.item() else self.images_anti)[ref],
                    topleft=(x.item(), y.item()),
                    tilt=tilt.item()
                )

                if anti.item():
                    pass
                pass

    def get_mask(self):
        return [pygame.mask.from_surface(self.images[ref]) for ref in self.img_ref]

    def check_collision(self, pipe: Pipe, window: Window):
        # Pipe mask
        top_mask = pygame.mask.from_surface(pipe.pipe_top)
        bot_mask = pygame.mask.from_surface(pipe.pipe_bottom)
        pipe_mid = (pipe.height - pipe.bottom) / 2

        zip_ = zip(self.get_mask(), self.x.detach().cpu().numpy(),
                   self.y.detach().cpu().numpy(), self.dead.detach().cpu().numpy())
        for idx, (bird_mask, x, y, dead) in enumerate(zip_):
            if not dead:
                top_offset = (int(pipe.x - x), int(pipe.top - round(y)))
                bot_offset = (int(pipe.x - x), int(pipe.bottom - round(y)))

                b_point = bird_mask.overlap(bot_mask, bot_offset)
                t_point = bird_mask.overlap(top_mask, top_offset)

                if b_point or t_point:
                    self.dead[idx] = True
                    pipe.collision = True
                    self.score[idx] -= (x - pipe.x) * (pipe_mid - y) / 100
                elif y < window.ceil or y > window.floor:
                    self.dead[idx] = True
                    self.score[idx] -= (x - pipe.x) * (pipe_mid - y) * 2 / 100

    def check_passed(self, pipes: Pipes):
        self.score[~self.dead] += 1
        self.score[self.dead] += -1
        add_pipe = False
        for pipe in pipes.pipes:
            passed = (pipe.x + pipe.image.get_width() * pipe.pxt < self.x) & ~self.dead
            if not pipe.passed and torch.any(passed):
                self.score[passed] += 50
                pipe.passed = True
                add_pipe = True
            elif pipe.passed:
                pass
            if not pipe.passed:
                # centre = pipe.top + (pipe.top - pipe.bottom)/2
                # self.score[~self.dead] -= ((centre - self.y[~self.dead])/WIN_HEIGHT)**2 + \
                #                           ((pipe.x+pipe.image.get_width()*pipe.pxt - self.x[~self.dead])/WIN_WIDTH)**2
                self.score[self.dead] -= 1

        return add_pipe

    def get_state(self, pipe: Pipe):
        # Get state(genomes, seq_len=1, action_features)
        tensor = torch.stack([
            self.y,
            (self.y - pipe.height),
            (self.y - pipe.bottom),
            (pipe.x - self.x),
            (pipe.x + pipe.image.get_width()*pipe.pxt - self.x),
        ], dim=-1) # .unsqueeze(0)
        # disabled = torch.full_like(tensor, -1).to(tensor.device, tensor.dtype)
        tensor[self.dead] = -1
        return tensor

    def get_reward(self):
        return self.score

    def get_image(self, index: int):
        return (self.images if not self.is_anti[index] else self.images_anti)[self.img_ref[index]]

    def active(self):
        self.active_num = torch.sum(self.dead == 0).item()
        return self.active_num


class Game(object):
    def __init__(self, birds: int, type2count: int = None, offset: int = 0,
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float32, render=False):
        if render:
            pygame.display.set_caption("Flappy Bird")
        self.window: Window = Window()
        Pipe.u_lim = 70
        Pipe.l_lim = 430
        def_height = round(self.window.height * 2 / 5)
        def_width = round(self.window.width * 2 / 5)
        self.pipes: Pipes = Pipes(round(self.window.width * 0.80))
        self.birds: Birds = Birds(birds, def_width, def_height, type2count, offset, device, dtype)
        self.base: Base = Base(self.window.floor)
        self.clock = pygame.time.Clock()

        self.generation = 0
        self.score = 0

        self.text_color = (255, 255, 255)
        self.line_color = (255, 0, 0)

    def tick(self, val=100):
        self.clock.tick(val)

    def update(self, activation: Tensor):
        self.birds.jump(activation)
        self.birds.move()
        self.birds.check_collision(self.pipes.get(), self.window)
        # if self.pipes.pipes[-1].x + self.pipes.gen_pos < self.window.width:
        #     self.pipes.add()
        if self.birds.check_passed(self.pipes):
            self.pipes.add()
            self.score += 1
        self.pipes.move()
        self.pipes.delete()
        self.base.move()

    def draw(self, debug=False):
        # Display window
        self.window.display.blit(self.window.background_image, (0, 0))

        # Draw pipes
        self.pipes.draw(self.window.display)

        # Draw cuda
        self.base.draw(self.window.display)

        # Draw birds or debug
        self.birds.draw(self.window.display)
        for bird_index, dead in enumerate(self.birds.dead):
            # draw lines from bird to pipe
            if debug and not dead:
                pipe = self.pipes.get()
                try:
                    x                  = self.birds.x[bird_index].item()
                    y                  = self.birds.y[bird_index].item()
                    img                = self.birds.get_image(bird_index)
                    pxt = pipe.pxt
                    bird_center        = (x + img.get_width() / 2, y + img.get_height() / 2)
                    pipe_top_center    = (pipe.x + pipe.pipe_top.get_width() * pxt, pipe.height)
                    pipe_bottom_center = (pipe.x + pipe.pipe_bottom.get_width() * pxt, pipe.bottom)

                    pygame.draw.line(self.window.display, self.line_color, bird_center, pipe_top_center, 5)
                    pygame.draw.line(self.window.display, self.line_color, bird_center, pipe_bottom_center, 5)
                except KeyboardInterrupt:
                    pass

        # score
        score_label = STAT_FONT.render(f"Score: {self.score:.2f}", 1, self.text_color)
        self.window.display.blit(score_label, (self.window.width - score_label.get_width() - 15, 10))

        # generations
        score_label = STAT_FONT.render(f"Gens: {gen}", 1, self.text_color)
        self.window.display.blit(score_label, (10, 10))

        # alive
        score_label = STAT_FONT.render(f"Alive: {self.birds.active()}", 1, self.text_color)
        self.window.display.blit(score_label, (10, 50))

        pygame.display.update()

    def get_state(self):
        return self.birds.get_state(self.pipes.get())


class RModel(Model):
    def __init__(self, inputs: int, outputs: int, seq_len: int, dim_size: int, layers: int,
                 kernel_size=1, heads: int = None, kv_heads: int = None, differential: int = False, norm_groups=1,
                 bias=False, device: torch.device = 'cpu', dtype: torch.device = torch.float32):
        super().__init__()
        # Attributes
        self.inputs         = inputs
        self.outputs        = outputs
        self.dim_size       = dim_size
        self.layers         = layers
        self.distribution   = 'normal'
        self.kernel_size    = kernel_size
        self.stride         = 1
        self.norm_groups    = norm_groups
        self.differential   = differential
        padding_mode   = 'reflect'

        # Build
        self.pri_actv = nn.SiLU()
        self.pol_proj = neat.nn.Sequential(*[
            Transpose(),
            Conv1d(inputs, dim_size, self.kernel_size, self.stride, -1, bias=bias, device=device, dtype=dtype, padding_mode=padding_mode),
            ResidualBlock(dim_size, dim_size, self.kernel_size, self.norm_groups,
                          bias, device, dtype, image_ndim=1, actv=self.pri_actv, padding_mode=padding_mode),
            ConverBase((seq_len,), dim_size, self.kernel_size, self.norm_groups, layers, heads, kv_heads,
                       self.differential, True, bias, device, dtype, actv=self.pri_actv, auto_single=True, padding_mode=padding_mode),
            ResidualBlock(dim_size, dim_size, 1, self.norm_groups,
                          bias, device, dtype, image_ndim=1, actv=self.pri_actv, padding_mode=padding_mode),
            nn.Flatten(-2, -1),
            self.pri_actv,
        ])
        self.mean_log_std = Linear(dim_size, 2*outputs, bias, device, dtype)
        self.sec_actv   = None
        # self.val_proj   = neat.nn.Sequential(*[
        #     Transpose(),
        #     Conv1d(inputs, dim_size, self.kernel_size, self.stride, -1, bias=bias, device=device, dtype=dtype),
        #     ResidualBlock(dim_size, dim_size, self.kernel_size, self.norm_groups,
        #                   bias, device, dtype, image_ndim=1, actv=self.pri_actv),
        #     SequenceEncoding(seq_len, dim_size, bias, device, dtype),
        #     ConverBase((seq_len,), dim_size, self.kernel_size, self.norm_groups, layers, heads, kv_heads,
        #                self.differential, True, bias, device, dtype, actv=self.pri_actv, auto_single=True),
        #     # ResidualBlock(dim_size, dim_size, 1, self.norm_groups,
        #     #               bias, device, dtype, image_ndim=1, actv=self.pri_actv),
        #     nn.Flatten(-2, -1),
        #     self.pri_actv,
        # ])
        # self.decode     = Linear(dim_size, 1, bias, device, dtype)

    def forward(self, state: Tensor, keys: Union[int, list[int]] = None, **kwargs):
        return self.get_policy(state, keys=keys, **kwargs)

    def get_mean_std(self, latent: Tensor, keys: Union[int, list[int]] = None) -> Tensor:
        mean_std        = self.mean_log_std(latent, keys=keys)
        mean, log_std   = torch.chunk(mean_std, 2, -1)
        mean            = F.sigmoid(mean)
        std             = torch.pow(10, F.sigmoid(log_std) * 3 + -4)
        return mean, std

    def get_action(self, state: Tensor, keys: Union[int, list[int]] = None) -> tuple[Tensor, Tensor]:
        latent      = self.pol_proj(state, keys=keys)
        mean, std   = self.get_mean_std(latent, keys=keys)
        dist        = torch.distributions.Normal(mean, std)
        action      = mean # dist.sample()
        if self.sec_actv is not None:
            action = self.sec_actv(action)
        log_prob    = dist.log_prob(action)
        return action, log_prob

    def evaluate_action(self, state: Tensor, action: Tensor, keys: Union[int, list[int]] = None) -> [Tensor, Union[Tensor, None]]:
        latent      = self.pol_proj(state, keys=keys)
        mean, std   = self.get_mean_std(latent, keys=keys)
        dist        = torch.distributions.Normal(mean, std)
        log_prob    = dist.log_prob(action)
        entropy     = dist.entropy()
        return log_prob, entropy

    def get_policy(self, state: Tensor, keys: Union[int, list[int]] = None, **options) -> Tensor:
        latent      = self.pol_proj(state, keys=keys)
        mean, std   = self.get_mean_std(latent, keys=keys)
        dist        = torch.distributions.Normal(mean, std)
        action      = mean # dist.sample()
        if self.sec_actv is not None:
            action = self.sec_actv(action)
        return action

    # def get_value(self, state: Tensor, keys: Union[int, list[int]] = None) -> Tensor:
    #     latent      = self.val_proj(state, keys=keys)
    #     value       = self.decode(latent, keys=keys)
    #     return value


def fix(value: float, default: float = 1):
    if np.isinf(value) or np.isnan(value):
        return default
    else:
        return value


# Network
GENOMES     = 50
INPUTS      = 5
OUTPUTS     = 1
EMBED_SIZE  = 16
KERNEL_SIZE = 1
NORM_GROUPS = 1
SEQ_LEN     = 2
LAYERS      = 1
HEADS       = 1
KV_HEADS    = HEADS
ENABLE_BIAS = True
DIFFERENTIAL = False
MEMORY_SIZE = 5
GAMMA       = np.exp(np.log(0.33) / 4)
ALPHA       = fix(np.exp(np.log(2.00) / (MEMORY_SIZE - 1)), 1.0)
BETA        = None # fix(np.exp(np.log(1.5) / (5 - 1)), 1.0)
LOSS_REG    = 0.

# MODEL = Reformer(INPUTS, OUTPUTS, 1, EMBED_SIZE, SEQ_LEN, LAYERS, HEADS, KV_HEADS, True, 0.1, ENABLE_BIAS,
#                  False, DEVICE, DTYPE,
#                  constant=2 ** np.floor(np.log2(SEQ_LEN * EMBED_SIZE)) // 2,
#                  pri_actv=nn.SiLU(), sec_actv=nn.Sigmoid())

MODEL = RModel(INPUTS, OUTPUTS, SEQ_LEN, EMBED_SIZE, LAYERS, KERNEL_SIZE, HEADS, KV_HEADS, DIFFERENTIAL, NORM_GROUPS,
               ENABLE_BIAS, DEVICE, DTYPE)
MODEL1 = RModel(INPUTS, OUTPUTS, SEQ_LEN, EMBED_SIZE, LAYERS, KERNEL_SIZE, HEADS, KV_HEADS, DIFFERENTIAL, NORM_GROUPS,
                ENABLE_BIAS, DEVICE, DTYPE)

INIT_GEN: int = None

RUNS = 1
LIMIT = 40
STEPS = LIMIT * 100 * RUNS


def evaluate(population: neat.Population, **options):
    trainer: neat.rl.NEAT = options['trainer']
    mapping0, mapping1 = population.get_mapping()
    cons_mapping = population.get_mapping(consolidated=True)
    trainer.update_mapping(cons_mapping)
    # BUFFER = torch.zeros(SEQ_LEN, population.pop_size, INPUTS).to(DEVICE, DTYPE)
    global INIT_GEN
    if INIT_GEN is None:
        INIT_GEN = population.generation

    def extend(array: Tensor, policy: Tensor):
        # buffer(seq_len, genomes, features), policy(1, genomes, features)
        array[:, :-1] = array[:, 1:].clone()
        array[:, SEQ_LEN-1:SEQ_LEN] = policy
        return array

    for genome in population.genomes.values():
        genome.fitness = 0

    terminate = False
    start = 0
    run_step = 0
    game_step = 0
    DEBUG = True
    DEBUG_STEP = 0
    MODEL.train()
    while not terminate:
        game = Game(population.size, len(mapping1), 0, DEVICE, DTYPE)
        # print(f"Anti Count = {game.birds.}")
        action_buffer = torch.zeros(population.size, SEQ_LEN, INPUTS).to(DEVICE, DTYPE)
        reward_buffer = torch.zeros(population.size, SEQ_LEN, 1).to(DEVICE, DTYPE)
        ph, mh, pw = FLOOR, FLOOR-70, WIN_WIDTH
        global_mean = torch.tensor([ph/2, mh/2, mh/2, pw/2, pw/2], device=DEVICE, dtype=DTYPE).unsqueeze(0)
        global_std = torch.tensor([ph/4, mh/6, mh/6, pw/4, pw/4], device=DEVICE, dtype=DTYPE).unsqueeze(0)

        limit = 30
        exe = True
        gts = clock.perf_counter()
        step = 0
        reverse_mapping0 = {index: key for key, index in mapping0.items()}
        reverse_mapping1 = {index: key for key, index in mapping1.items()}

        while exe and game.birds.active() > 0:
            # BUFFER[:] = 0
            game.tick(1000)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exe = False

            with torch.no_grad():
                # Get Inputs ~ send bird location, top pipe location and bottom pipe location
                # and determine from network whether to jump or not
                observations = game.get_state() # shape(features_in)
                observations = (observations - global_mean) / global_std
                observations = observations.unsqueeze(-2) # .expand(population.pop_size, *observations.shape)
                if DEBUG and population.generation == INIT_GEN and step == DEBUG_STEP:
                    print(f"\nobservations =>\n{observations}\n\tshape = {observations.shape}")
                    print(action_buffer.shape)
                observations = extend(action_buffer, observations)
                if DEBUG and population.generation == INIT_GEN and step == DEBUG_STEP:
                    print(f"extended inputs =>\n{observations}\n\tshape = {observations.shape}")
                ts = clock.perf_counter()
                keys0, keys1 = [], []
                indices0, indices1 = [], []
                for index, dead in enumerate(game.birds.dead):
                    if not dead:
                        if index in reverse_mapping0:
                            keys0.append(reverse_mapping0[index])
                            indices0.append(index)
                        elif index-len(mapping0) in reverse_mapping1:
                            keys1.append(reverse_mapping1[index-len(mapping0)])
                            indices1.append(index-len(mapping0))
                        else:
                            print(f"\nPopulation size {population.size}"
                                  f"\nReverse mapping \n{reverse_mapping0} \n{reverse_mapping1}"
                                  f"\nIndex = {index}, birds_shape = {game.birds.dead.shape}")
                            raise KeyError()
                if len(indices0) == 0:
                    keys0 = list(mapping0.keys())
                    indices0 = list(mapping0.values())
                if len(indices1) == 0:
                    keys1 = list(mapping1.keys())
                    indices1 = list(mapping1.values())
                if DEBUG and population.generation == INIT_GEN and step == DEBUG_STEP:
                    print(f"keys =>\n{keys0}")
                    print(f"indices =>\n{indices0}")
                observations0, observations1 = torch.split(observations, len(mapping0), dim=0)
                actions0, probs = MODEL.get_action(observations0[indices0].unsqueeze(1), keys=keys0)
                actions1, probs = MODEL1.get_action(observations1[indices1].unsqueeze(1), keys=keys1)
                # actions, probs = MODEL.get_action(observations.unsqueeze(1))
                # shape(seq_len=1, genomes, features_out)
                actions0, actions1, probs = actions0.squeeze(1), actions1.squeeze(1), probs.squeeze(1)
                # actions = (actions >= 0.95).float()
                if DEBUG and population.generation == INIT_GEN and step == DEBUG_STEP:
                    print(f"actions =>\n{actions0}\n\tshape = {actions0.shape}")
                    print(f"probs =>\n{probs}\n\tshape = {probs.shape}")
                if True:
                    padding = game.birds.bird_num - actions0.shape[0] - actions1.shape[1]
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
                    actions = torch.cat([actions0, actions1[..., :OUTPUTS]], dim=0)
                if DEBUG and population.generation == INIT_GEN and step == DEBUG_STEP:
                    print(f"filled actions =>\n{actions}\n\tshape = {actions.shape}")
                calc_time = clock.perf_counter() - ts
                # game.update(actions[:, 0])
                game.update(actions)
                if (population.generation+1) % 5 == 0:
                    game.draw(False)
                rewards = torch.softmax(game.birds.score.unsqueeze(-1), 0)
                # rewards += (rewards - rewards.min(dim=0, keepdim=True)[0])
                # extend(reward_buffer, game.birds.score.unsqueeze(-1))
                if DEBUG and population.generation == INIT_GEN and step == DEBUG_STEP:
                    print(f"rewards =>\n{rewards}\n\tshape = {rewards.shape}")
                    # v = MODEL.get_value(observations[:len(reverse_mapping0)].unsqueeze(1), ).squeeze(1)
                    # print(f"values =>\n{v}\n\tshape = {v.shape}")
                    # del v
                # print(f"\rO = {outputs.flatten().cpu().numpy()} SCORE: = {game.birds.score.cpu().numpy()}", end='')

                round_end = (game.birds.active() == 0 and game_step == RUNS-1) or game.score >= LIMIT
                terminate = trainer.update(observations, actions, rewards, round_end, round_end)

                alive = round((torch.sum(~game.birds.dead) / game.birds.dead.numel() * 100).item(), 2)
                max_score = round(rewards.max().item(), 2)
                if alive > 0:
                    best_index = torch.argmax(game.birds.score).cpu().item()
                    if best_index in reverse_mapping0:
                        best_key = reverse_mapping0[best_index]
                    elif best_index in reverse_mapping1:
                        best_key = reverse_mapping1[best_index]
                    else:
                        # raise KeyError()
                        beskt_key = None
                print(f"\r{CM('Executing', Fore.GREEN)}: time_elapsed = {round(clock.perf_counter()-gts)}s, "
                      f"alive = {alive}, max_rew = {max_score}, best_key={best_key}, ct={calc_time:.2e}, sd={trainer.steps_done} "
                      f"bl={trainer.primary.max_size()}", end='')

            # break if score gets large enough
            if round_end or terminate:
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

        u_lim, l_lim = game.window.height * 0.95, (game.window.height - game.window.floor) * 1.05
        # print(f"u lim = {u_lim}, l lim = {l_lim}")
        for idx, (score, genome) in enumerate(zip(game.birds.get_reward(), population.genomes.values())):
            genome.fitness = score.item()
            y_position = game.birds.y[idx]
            died_beyond_limits = y_position > u_lim or y_position < l_lim
            if died_beyond_limits:
                population.to_delete.append(genome.key)

        game_step += 1
    print(f"\n")

    population.save_dict('flappy_bird', replace=population.generation != INIT_GEN)
    # trainer.save('flappy_bird', replace=population.generation != INIT_GEN)


def run():
    # MODEL.single_mode(True)
    # Configuration
    print(f"creating config")
    config = neat.Config()

    config.genome.init_type                     = 'normal'
    config.genome.weight_init_mean              = 0.0
    config.genome.weight_init_std               = 1.0
    config.genome.weight_min_value              = -np.inf
    config.genome.weight_max_value              = +np.inf
    config.genome.weight_mutate_power           = 1e-1
    config.genome.weight_mutate_rate            = 0.65
    config.genome.weight_replace_rate           = 0.10
    config.genome.weight_add_prob               = 0.33
    config.genome.weight_del_prob               = 0.33
    config.genome.single_structural_mutation    = False

    config.reproduction.min_species_size        = GENOMES
    config.reproduction.purge                   = 1
    config.reproduction.survival_threshold      = 0.10
    config.reproduction.cross_threshold         = 0.05
    config.reproduction.elitism                 = 15
    config.species.compatibility_threshold      = np.inf
    config.stagnation.max_stagnation            = 1
    config.stagnation.species_elitism           = 2
    config.reproduction.darwin_multiplier       = 0.50
    config.save()
    config.load(2)

    # Create the population, which is the top-level object for a NEAT run.
    print(f"creating population")
    population = neat.Population(GENOMES, MODEL, config, init_reporter=True)
    population1 = neat.Population(GENOMES, MODEL1, config, init_reporter=True)
    population.absorb_population(population1)
    print(MODEL.pol_proj)
    population.load_dict(name='flappy_bird', file_no=None)

    trainer = neat.rl.NEAT(
        population,
        schedulers=[
            neat.optim.scheduler.RandomAnnealing(config, 1e-1, 1e-0, 5, ['weight_init_std', 'weight_mutate_power'], True),
            # neat.optim.scheduler.CosineAnnealing(config, 10, 10, 'weight_mutate_power', True, True),
            # neat.optim.scheduler.CosineAnnealing(config, 15, 0.2, 'weight_mutate_rate', True, True),
            # neat.optim.scheduler.CosineAnnealing(config, 15, 5, 'weight_replace_rate', True, True),
        ],
        device=DEVICE, dtype=DTYPE,
        log_sub_dir='flappy_bird\\',
        log_name=f"{unix_to_datetime_file(clock.time())}-"
                 f"s{SEQ_LEN}-e{EMBED_SIZE}-l{LAYERS}-h{HEADS}-b{int(ENABLE_BIAS)}-"
                 f"g{round(GAMMA, 4)}-r{round(LOSS_REG, 4)}",
        gamma=GAMMA, alpha=ALPHA, reverse=False, best=False, normalize=True,
        rew_reg=1.0, pol_reg=0.0, validate=True, groups=None,
        max_episodes=MEMORY_SIZE,
    )

    # Run for up to 50 generations.
    print(f"starting evaluation: population={len(population.genomes)}")
    # trainer.load(name='flappy_bird', file_no=None)
    trainer.learn(evaluate, STEPS, 100, 2048, 0.1, 'binary', 2)

    # while True:
    #     evaluate(population, trainer=trainer)

    # for p in population.get(winner):
    #     print(p)

    # show final stats
    # print('\nBest genome:\n{!s}'.format(winner.key))


if __name__ == '__main__':
    print(MODEL)
    run()
