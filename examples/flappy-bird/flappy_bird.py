
from ModifiedNEAT.util.fancy_text import CM, Fore
from ModifiedNEAT.nn.base import Model
from ModifiedNEAT.nn.modules.sub import Linear, Conv1d, Transpose, ResidualBlock, Sequential, GroupNorm, ConverBase
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
from typing import Union

import pygame
import random
import os
import time as clock
import numpy as np

torch.set_printoptions(threshold=10)
pygame.font.init()  # init font

DEVICE = 'cpu' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float64

# Define window
THRESHOLD = 0.9
WIN_HEIGHT = 800
WIN_WIDTH  = 550
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
    u_lim: int = 0
    l_lim: int = 50
    pxt: float = 1.0

    def __init__(self, x: int, gap_l_lim=180, gap_u_lim=230, velocity=6):
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
        self.height = random.randrange(self.u_lim, self.l_lim)
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
        self.velocity = 5
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


def blitRotateCenter(surf, image, topleft, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect      = rotated_image.get_rect(center=image.get_rect(topleft=topleft).center)

    surf.blit(rotated_image, new_rect.topleft)


class Window(object):
    def __init__(self, height: int = 800, width: int = 600, hitbox: int = 20):
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
        self.pipes = [Pipe(gen_pos)]
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
    def __init__(self, num: int, x: int = 200, y: int = 200,
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float32):
        self.x          = torch.full((num,), x, device=device, dtype=dtype)
        self.y          = torch.full((num,), y, device=device, dtype=dtype)
        self.tilt       = torch.full((num,), 0, device=device, dtype=dtype)
        self.tick_count = torch.full((num,), 0, device=device, dtype=torch.int64)
        self.vel        = torch.full((num,), 0, device=device, dtype=dtype)
        self.height     = self.y.clone()
        self.img_count  = torch.full((num,), 0, device=device, dtype=torch.int64)
        self.img_ref    = torch.full((num,), 0, device=device, dtype=torch.int64)
        self.score      = torch.full((num,), 0, device=device, dtype=dtype)
        self.dead       = torch.full((num,), False, device=device, dtype=torch.bool)
        # use mapping to reduce calculation on dead birds
        self.mapping: dict[int, int] = Dict([(idx, idx) for idx in range(num)])

        self.images: list = [pygame.transform.scale2x(pygame.image.load(
            os.path.join("imgs", "bird" + str(x) + ".png"))) for x in range(1, 4)]
        self.max_rot = 25
        self.ang_vel = 20
        image_num = len(self.images)
        self.animations = [i for i in range(image_num-1)] + [image_num-1] + [i for i in reversed(range(image_num-1))]
        self.animation_mult_max = len(self.animations)
        self.animation_time = 5

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

        # for downward acceleration
        displacement = self.vel[mask] * self.tick_count[mask] + 0.5 * 3 * self.tick_count[mask] ** 2

        # terminal velocity
        tv = displacement >= 16
        displacement[tv] = ((displacement / torch.abs(displacement)) * 16)[tv]

        nd = displacement < 0
        displacement[nd] -= 2

        # print(displacement.shape)
        self.y[mask] += displacement
        # print(self.y.shape)

        # tilt up
        temp = (displacement < 0) | (self.y[mask] < (self.height[mask] + 50))
        tu = temp & (self.tilt[mask] < self.max_rot)
        # print(temp.shape, tu.shape)
        self.tilt[mask][tu] = self.max_rot
        # tilt down
        td = ~temp & (self.tilt[mask] > -90)
        self.tilt[mask][td] -= self.ang_vel

    def draw(self, win):
        self.img_count += 1
        mask = ~self.dead

        # For animation of bird, loop through three images

        mult = 0
        prev_level = torch.zeros_like(self.img_count[mask], dtype=torch.bool)
        while True:
            mult += 1
            if mult != self.animation_mult_max:
                level = (self.img_count[mask] <= self.animation_time * mult) & ~prev_level
                prev_level = prev_level | level
                # print(level)
                self.img_ref[mask][level] = self.animations[mult-1]
            else:
                level = self.img_count[mask] > self.animation_time * (mult - 1)
                self.img_ref[mask][level] = 0
                self.img_count[mask][level] = 0
                break

        # so when bird is nose diving it isn't flapping
        nd = self.tilt[mask] <= -80
        self.img_ref[mask][nd]   = 1
        self.img_count[mask][nd] = self.animation_time*2

        # tilt the bird
        for ref, x, y, tilt, dead in zip(self.img_ref[mask], self.x[mask], self.y[mask], self.tilt[mask], self.dead[mask]):
            if not dead:
                blitRotateCenter(win, self.images[ref], (x.item(), y.item()), tilt.item())

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
                top_offset = (pipe.x - x, pipe.top - round(y))
                bot_offset = (pipe.x - x, pipe.bottom - round(y))

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
        return self.images[self.img_ref[index]]

    def active(self):
        self.active_num = torch.sum(self.dead == 0).item()
        return self.active_num


class Game(object):
    def __init__(self, birds: int,  device: torch.device = 'cpu', dtype: torch.dtype = torch.float32, render=False):
        if render:
            pygame.display.set_caption("Flappy Bird")
        self.window: Window = Window()
        Pipe.u_lim = 50
        Pipe.l_lim = 450
        def_height = round(self.window.height * 2 / 5)
        def_width = round(self.window.width * 2 / 5)
        self.pipes: Pipes = Pipes(round(self.window.width * 0.75))
        self.birds: Birds = Birds(birds, def_width, def_height, device, dtype)
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

        # Build
        self.pri_actv = nn.SiLU()
        self.pol_proj = neat.nn.Sequential(*[
            Transpose(),
            # GroupNorm(1, inputs, affine=True, bias=bias, device=device, dtype=dtype),
            Conv1d(inputs, dim_size, self.kernel_size, self.stride, -1, bias=bias, device=device, dtype=dtype),
            ResidualBlock(dim_size, dim_size, self.kernel_size, self.norm_groups,
                          bias, device, dtype, image_ndim=1, actv=self.pri_actv),
            ConverBase((seq_len,), dim_size, self.kernel_size, self.norm_groups, layers, heads, kv_heads,
                       self.differential, True, bias, device, dtype, actv=self.pri_actv, auto_single=True),
            ResidualBlock(dim_size, dim_size, 1, self.norm_groups,
                          bias, device, dtype, image_ndim=1, actv=self.pri_actv),
            nn.Flatten(-2, -1),
            self.pri_actv,
        ])
        self.mean_log_std = Linear(dim_size, 2*outputs, bias, device, dtype)
        self.sec_actv   = None
        self.val_proj   = neat.nn.Sequential(*[
            Transpose(),
            Conv1d(inputs, dim_size, self.kernel_size, self.stride, -1, bias=bias, device=device, dtype=dtype),
            ResidualBlock(dim_size, dim_size, self.kernel_size, self.norm_groups,
                          bias, device, dtype, image_ndim=1, actv=self.pri_actv),
            ConverBase((seq_len,), dim_size, self.kernel_size, self.norm_groups, layers, heads, kv_heads,
                       self.differential, True, bias, device, dtype, actv=self.pri_actv, auto_single=True),
            # ResidualBlock(dim_size, dim_size, 1, self.norm_groups,
            #               bias, device, dtype, image_ndim=1, actv=self.pri_actv),
            nn.Flatten(-2, -1),
            self.pri_actv,
        ])
        self.decode     = Linear(dim_size, 1, bias, device, dtype)

    def forward(self, state: Tensor, keys: Union[int, list[int]] = None, **kwargs):
        return self.get_policy(state, keys=keys, **kwargs)

    def get_mean_std(self, latent: Tensor, keys: Union[int, list[int]] = None) -> Tensor:
        mean_std        = self.mean_log_std(latent, keys=keys)
        mean, log_std   = torch.chunk(mean_std, 2, -1)
        mean            = F.tanh(mean)
        std             = torch.exp(F.hardtanh(log_std, -10, 0))
        return mean, std

    def get_action(self, state: Tensor, keys: Union[int, list[int]] = None) -> tuple[Tensor, Tensor]:
        latent      = self.pol_proj(state, keys=keys)
        mean, std   = self.get_mean_std(latent, keys=keys)
        dist        = torch.distributions.Normal(mean, std)
        action      = dist.sample()
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
        action      = dist.sample()
        if self.sec_actv is not None:
            action = self.sec_actv(action)
        return action

    def get_value(self, state: Tensor, keys: Union[int, list[int]] = None) -> Tensor:
        latent      = self.val_proj(state, keys=keys)
        value       = self.decode(latent, keys=keys)
        return value


# Network
INPUTS      = 5
OUTPUTS     = 1
GENOMES     = 200
EMBED_SIZE  = 16
KERNEL_SIZE = 1
NORM_GROUPS = 1
SEQ_LEN     = 16
LAYERS      = 1
HEADS       = 1
KV_HEADS    = 1
ENABLE_BIAS = True
DIFFERENTIAL = True
GAMMA       = np.exp(np.log(0.10) / (16 - 1))
ALPHA       = np.exp(np.log(1.5) / (2 - 1))
LOSS_REG    = 0.

# MODEL = Reformer(INPUTS, OUTPUTS, 1, EMBED_SIZE, SEQ_LEN, LAYERS, HEADS, KV_HEADS, True, 0.1, ENABLE_BIAS,
#                  False, DEVICE, DTYPE,
#                  constant=2 ** np.floor(np.log2(SEQ_LEN * EMBED_SIZE)) // 2,
#                  pri_actv=nn.SiLU(), sec_actv=nn.Sigmoid())

MODEL = RModel(INPUTS, OUTPUTS, SEQ_LEN, EMBED_SIZE, LAYERS, KERNEL_SIZE, HEADS, KV_HEADS, DIFFERENTIAL, NORM_GROUPS,
               ENABLE_BIAS, DEVICE, DTYPE)

INIT_GEN: int = None

RUNS = 1
LIMIT = 30
STEPS = LIMIT * 100 * RUNS


def evaluate(population: neat.Population, **options):
    trainer: neat.rl.NEAT = options['trainer']
    trainer.update_mapping(population.get_mapping())
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

    trainer.deque_episodes(1)
    terminate = False
    start = 0
    run_step = 0
    game_step = 0
    DEBUG = True
    DEBUG_STEP = 0
    MODEL.train()
    while not terminate:
        game = Game(population.pop_size, DEVICE, DTYPE)
        action_buffer = torch.zeros(population.pop_size, SEQ_LEN, INPUTS).to(DEVICE, DTYPE)
        reward_buffer = torch.zeros(population.pop_size, SEQ_LEN, 1).to(DEVICE, DTYPE)
        ph, mh, pw = FLOOR, FLOOR-70, WIN_WIDTH
        global_mean = torch.tensor([ph/2, mh/2, mh/2, pw/2, pw/2], device=DEVICE, dtype=DTYPE).unsqueeze(0)
        global_std = torch.tensor([ph/4, mh/6, mh/6, pw/4, pw/4], device=DEVICE, dtype=DTYPE).unsqueeze(0)

        limit = 30
        exe = True
        gts = clock.perf_counter()
        step = 0
        forward_mapping = population.get_mapping()
        reverse_mapping = {index: key for key, index in forward_mapping.items()}

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
                keys = []
                indices = []
                for index, dead in enumerate(game.birds.dead):
                    if not dead:
                        keys.append(reverse_mapping[index])
                        indices.append(index)
                if DEBUG and population.generation == INIT_GEN and step == DEBUG_STEP:
                    print(f"keys =>\n{keys}")
                    print(f"indices =>\n{indices}")
                actions, probs = MODEL.get_action(observations[indices].unsqueeze(1), keys=keys)
                # actions, probs = MODEL.get_action(observations.unsqueeze(1))
                # shape(seq_len=1, genomes, features_out)
                actions, probs = actions.squeeze(1), probs.squeeze(1)
                # actions = (actions >= 0.95).float()
                if DEBUG and population.generation == INIT_GEN and step == DEBUG_STEP:
                    print(f"actions =>\n{actions}\n\tshape = {actions.shape}")
                    print(f"probs =>\n{probs}\n\tshape = {probs.shape}")
                if True:
                    padding = game.birds.bird_num - actions.shape[0]
                    if padding > 0:
                        fill = actions.clone()
                        actions = torch.zeros(
                            game.birds.bird_num, *actions.shape[1:], device=DEVICE, dtype=DTYPE
                        )
                        actions[indices] = fill
                        fill = probs.clone()
                        probs = torch.zeros(
                            game.birds.bird_num, *probs.shape[1:], device=DEVICE, dtype=DTYPE
                        )
                        probs[indices] = fill
                    pass
                if DEBUG and population.generation == INIT_GEN and step == DEBUG_STEP:
                    print(f"filled actions =>\n{actions}\n\tshape = {actions.shape}")
                calc_time = clock.perf_counter() - ts
                # game.update(actions[:, 0])
                game.update(actions)
                game.draw(False)
                rewards = game.birds.score.unsqueeze(-1)
                # extend(reward_buffer, game.birds.score.unsqueeze(-1))
                if DEBUG and population.generation == INIT_GEN and step == DEBUG_STEP:
                    print(f"rewards =>\n{rewards}\n\tshape = {rewards.shape}")
                    v = MODEL.get_value(observations.unsqueeze(1)).squeeze(1)
                    print(f"values =>\n{v}\n\tshape = {v.shape}")
                    del v
                # print(f"\rO = {outputs.flatten().cpu().numpy()} SCORE: = {game.birds.score.cpu().numpy()}", end='')

                round_end = (game.birds.active() == 0 and game_step == RUNS-1) or game.score >= LIMIT
                terminate = trainer.update(observations, actions, rewards, round_end, round_end)

                alive = round((torch.sum(~game.birds.dead) / game.birds.dead.numel() * 100).item(), 2)
                max_score = round(rewards.max().item(), 2)
                if alive > 0:
                    best_index = torch.argmax(game.birds.score).cpu().item()
                    best_key = reverse_mapping[best_index]
                print(f"\r{CM('Executing', Fore.GREEN)}: time_elapsed = {round(clock.perf_counter()-gts)}s, "
                      f"alive = {alive}, max_rew = {max_score}, best_key={best_key}, ct={calc_time:.2e}, sd={trainer.steps_done} "
                      f"bl={trainer.replay.max_size()}", end='')

            # break if score gets large enough
            if round_end or terminate:
                # pickle.dump(population.genomes, open(".\\best.pickle", "wb"))
                terminate = True
                break

            step += 1
            run_step += 1
            if step == DEBUG_STEP:
                DEBUG = False

        print(f"\n\nEpisodes = {trainer.replay.episodes()}")

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
    config.genome.init_type             = 'normal'
    config.genome.weight_init_mean      = 0
    config.genome.weight_init_std       = 1
    config.genome.weight_min_value      = -np.inf
    config.genome.weight_max_value      = np.inf
    config.genome.weight_mutate_power   = 0.1
    config.genome.weight_mutate_rate    = 0.8
    config.reproduction.min_species_size = 100
    config.reproduction.purge           = 1
    config.reproduction.survival_threshold = 0.10
    config.reproduction.elitism         = 10
    config.species.compatibility_threshold = 1.5
    config.stagnation.max_stagnation    = 1
    config.stagnation.species_elitism   = 3
    config.save()
    config.load(2)

    # Create the population, which is the top-level object for a NEAT run.
    print(f"creating population")
    population = neat.Population(GENOMES, MODEL, config, init_reporter=True)
    print(MODEL.pol_proj)
    population.load_dict(name='flappy_bird', file_no=None)

    trainer = neat.rl.NEAT(
        MODEL, population,
        schedulers=[
            neat.optim.scheduler.CosineAnnealing(config, 10, 0.01, 'weight_mutate_power', True, True),
            neat.optim.scheduler.CosineAnnealing(config, 6, 0.7, 'weight_mutate_rate', True, True),
        ], device=DEVICE, dtype=DTYPE,
        log_sub_dir='flappy_bird\\',
        log_name=f"{unix_to_datetime_file(clock.time())}-"
                 f"s{SEQ_LEN}-e{EMBED_SIZE}-l{LAYERS}-h{HEADS}-b{int(ENABLE_BIAS)}-"
                 f"g{round(GAMMA, 4)}-r{round(LOSS_REG, 4)}",
        gamma=GAMMA, alpha=ALPHA, reverse=False,
        rew_reg=1.0, pol_reg=0.95, validate=True, groups=20,
    )

    # Run for up to 50 generations.
    print(f"starting evaluation: population={len(population.genomes)}")
    # trainer.load(name='flappy_bird', file_no=None)
    trainer.learn(evaluate, STEPS, 30, 2048, 0.1, 'binary', 2)

    # for p in population.get(winner):
    #     print(p)

    # show final stats
    # print('\nBest genome:\n{!s}'.format(winner.key))


if __name__ == '__main__':
    print(MODEL)
    run()
