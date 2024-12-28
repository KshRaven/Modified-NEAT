
from build.util.fancy_text import CM, Fore
from build.nn.base import Model
from build.models.sub import Linear
from build.models import Reformer
from build.util.datetime import unix_to_datetime_file

import build as neat
import torch
import torch.nn as nn
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

DEVICE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32

# Define window
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
# BASE_IMG    = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")).convert_alpha())

gen = 0


class Pipe(object):
    image = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")).convert_alpha())
    u_lim: int = 0
    l_lim: int = 50

    def __init__(self, x: int, gap=200, velocity=6):
        self.gap = gap
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
        self.bottom = self.height + self.gap

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
        t_point = bird_mask.overlap(top_mask,top_offset)

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
        # activation shape (batch_size / seq_len, genomes, features)
        activation = (~self.dead & (activation[:, 0] >= 0.90))
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
        nd                 = self.tilt[mask] <= -80
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
        self.score[self.dead] = -100
        add_pipe = False
        for pipe in pipes.pipes:
            passed = (pipe.x + pipe.image.get_width() * 0.9 < self.x) & ~self.dead
            if not pipe.passed and torch.any(passed):
                self.score[passed] += 50
                pipe.passed = True
                add_pipe = True
            elif pipe.passed:
                pass

        return add_pipe

    def get_state(self, pipe: Pipe):
        # Get state(seq_len=1, genomes, action_features)
        tensor = torch.cat(
            [self.y.unsqueeze(-1),
             (self.y - pipe.height).unsqueeze(-1),
             (self.y - pipe.bottom).unsqueeze(-1),
             # (self.x - pipe.x + pipe.image.get_width() * 0.9).unsqueeze(-1),
             # self.reward,
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

    def tick(self, val=30):
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

        # Draw base
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
                    bird_center        = (x + img.get_width() / 2, y + img.get_height() / 2)
                    pipe_top_center    = (pipe.x + pipe.pipe_top.get_width() / 2, pipe.height)
                    pipe_bottom_center = (pipe.x + pipe.pipe_bottom.get_width() / 2, pipe.bottom)

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
    def __init__(self, inputs, outputs, dim_size, bias, device, dtype):
        super().__init__()
        self.distribution = 'normal'
        self.act_proj   = Linear(inputs, dim_size, bias, device, dtype, nn.SiLU())
        if self.distribution == 'discrete':
            outputs = 2 ** outputs
        self.mean       = Linear(dim_size, outputs, bias, device, dtype, nn.Sigmoid())
        self.log_std    = Linear(dim_size, outputs, bias, device, dtype, nn.Sigmoid())
        self.rew_proj   = Linear(inputs, dim_size, bias, device, dtype, nn.SiLU())
        self.decode     = Linear(dim_size, 1, bias, device, dtype, None)

    def forward(self, state: Tensor):
        return self.get_policy(state)

    def get_mean(self, latent: Tensor, key: int = None) -> Tensor:
        return self.mean(latent, key=key)

    def get_std(self, latent: Tensor, key: int = None) -> Tensor:
        return 10 ** (-4 + self.log_std(latent, key=key) * 3)

    def get_action(self, state: Tensor, key: int = None) -> tuple[Tensor, Tensor]:
        latent = self.act_proj(state, key=key)
        mean, std = self.get_mean(latent, key=key), self.get_std(latent, key=key)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_action(self, state: Tensor, action: Tensor, key: int = None) -> [Tensor, Union[Tensor, None]]:
        latent = self.act_proj(state, key=key)
        mean, std = self.get_mean(latent, key=key), self.get_std(latent, key=key)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy

    def get_policy(self, state: Tensor, key: int = None, **options) -> Tensor:
        latent = self.act_proj(state, key=key)
        mean, std = self.get_mean(latent, key=key), self.get_std(latent, key=key)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action

    def get_value(self, state: Tensor, key: int = None) -> Tensor:
            latent = self.rew_proj(state, key=key)
            value = self.decode(latent, key=key)
            return value


# Network
INPUTS      = 3
OUTPUTS     = 1
GENOMES     = 200
EMBED_SIZE  = 32
SEQ_LEN     = 64
LAYERS      = 1
HEADS       = 1
KV_HEADS    = 1
ENABLE_BIAS = False
GAMMA       = 0.1
LOSS_REG    = 0.001
# MODEL = build.models.MiniFormer(INPUTS, OUTPUTS, EMBED_SIZE, SEQ_LEN, LAYERS, 1, 1, 0.1, ENABLE_BIAS, DEVICE, DTYPE,
#                                distribution='normal', pri_actv=build.nn.activations.Tanh(), sec_actv=nn.Sigmoid())
MODEL = Reformer(INPUTS, OUTPUTS, 1, EMBED_SIZE, SEQ_LEN, LAYERS, HEADS, KV_HEADS, True, 0.1, ENABLE_BIAS,
                 False, DEVICE, DTYPE,
                 constant=2 ** np.floor(np.log2(SEQ_LEN * EMBED_SIZE)) // 2,
                 pri_actv=nn.SiLU(), sec_actv=nn.Sigmoid())
INIT_GEN: int = None

RUNS = 1
LIMIT = 30
STEPS = LIMIT * 100 * RUNS


def evaluate(population: neat.Population, **options):
    trainer: neat.rl.PPO = options['trainer']
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
    while not terminate:
        game = Game(population.pop_size, DEVICE, DTYPE)
        action_buffer = torch.zeros(population.pop_size, SEQ_LEN, INPUTS).to(DEVICE, DTYPE)
        reward_buffer = torch.zeros(population.pop_size, SEQ_LEN, 1).to(DEVICE, DTYPE)

        limit = 30
        exe = True
        gts = clock.perf_counter()
        step = 0
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
                observations = observations.unsqueeze(-2) # .expand(population.pop_size, *observations.shape)
                if DEBUG and population.generation == INIT_GEN and step == DEBUG_STEP:
                    print(f"\nobservations =>\n{observations}\n\tshape = {observations.shape}")
                    print(action_buffer.shape)
                observations = extend(action_buffer, observations)
                if DEBUG and population.generation == INIT_GEN and step == DEBUG_STEP:
                    print(f"extended inputs =>\n{observations}\n\tshape = {observations.shape}")
                ts = clock.perf_counter()
                actions, probs = MODEL.get_action(observations.unsqueeze(1)) # genome_mask=~game.birds.dead) # shape(seq_len=1, genomes, features_out)
                # actions = (actions >= 0.95).float()
                if DEBUG and population.generation == INIT_GEN and step == DEBUG_STEP:
                    print(f"actions =>\n{actions}\n\tshape = {actions.shape}")
                    print(f"probs =>\n{probs}\n\tshape = {probs.shape}")
                calc_time = clock.perf_counter() - ts
                game.update(actions[:, 0])
                game.draw()
                rewards = game.birds.score.unsqueeze(-1) # extend(reward_buffer, game.birds.score.unsqueeze(-1))
                if DEBUG and population.generation == INIT_GEN and step == DEBUG_STEP:
                    print(f"rewards =>\n{rewards}\n\tshape = {rewards.shape}")
                    v = MODEL.get_value(observations.unsqueeze(1))
                    print(f"values =>\n{v}\n\tshape = {v.shape}")
                    del v
                # print(f"\rO = {outputs.flatten().cpu().numpy()} SCORE: = {game.birds.score.cpu().numpy()}", end='')

                round_end = (game.birds.active() == 0 and game_step == RUNS-1) or game.score >= LIMIT
                terminate = trainer.update(observations, actions, probs, rewards, round_end)

                alive = round((torch.sum(~game.birds.dead) / game.birds.dead.numel() * 100).item(), 2)
                max_score = round(rewards.max().item(), 2)
                print(f"\r{CM('Executing', Fore.GREEN)}: time_elapsed = {round(clock.perf_counter()-gts)}s, "
                      f"alive = {alive}, max_rew = {max_score}, ct={calc_time:.2e}, sd={trainer.steps_done}", end='')

            # break if score gets large enough
            if round_end or terminate:
                # pickle.dump(population.genomes, open(".\\best.pickle", "wb"))
                terminate = True
                break

            step += 1
            run_step += 1
            if step == DEBUG_STEP:
                DEBUG = False

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

    # population.save_dict('flappy_bird', replace=population.generation != INIT_GEN)
    # trainer.save('flappy_bird', replace=population.generation != INIT_GEN)


def run():
    MODEL.single_mode(True)
    # Configuration
    print(f"creating config")
    config = neat.Config()
    config.genome.init_type = 'normal'
    config.genome.weight_init_mean = 0
    config.genome.weight_init_std = 1
    config.genome.weight_min_value = -np.pi * 10
    config.genome.weight_max_value = np.pi * 10
    config.genome.weight_mutate_power = 1
    config.genome.weight_mutate_rate = 0.5
    config.reproduction.min_species_size = 150
    config.reproduction.purge = 1
    config.reproduction.survival_threshold = 0.033
    config.reproduction.elitism = 30
    config.species.compatibility_threshold = 2
    config.stagnation.max_stagnation = 1
    config.species.compatibility_threshold = 1000000
    config.save()
    config.load(2)

    # Create the population, which is the top-level object for a NEAT run.
    print(f"creating population")
    population = neat.Population(GENOMES, MODEL, config, init_reporter=True)
    print(MODEL.pol_proj)
    # population.load_dict(name='flappy_bird', file_no=23)

    trainer = neat.rl.PPO(MODEL, population, DEVICE, DTYPE, gamma=GAMMA,
                          scheduler=neat.scheduler.CosineAnnealing(config, 100, 10, 0.01, True, True),
                          loss_reg=LOSS_REG, pol_reg=0.5, val_reg=1.0, ent_reg=0e-6,
                          norm_rew=True, norm_adv=False,
                          log_sub_dir='flappy_bird\\',
                          log_name=f"{unix_to_datetime_file(clock.time())}-"
                                   f"s{SEQ_LEN}-e{EMBED_SIZE}-l{LAYERS}-h{HEADS}-b{int(ENABLE_BIAS)}-"
                                   f"g{round(GAMMA, 4)}-r{round(LOSS_REG, 4)}"
                          )

    # Run for up to 50 generations.
    print(f"starting evaluation: population={len(population.genomes)}")
    # trainer.load(name='flappy_bird', file_no=None)
    trainer.learn(evaluate, STEPS, 30, 1024, 0.1, 'binary', 2)

    # for p in population.get(winner):
    #     print(p)

    # show final stats
    # print('\nBest genome:\n{!s}'.format(winner.key))


if __name__ == '__main__':
    run()
