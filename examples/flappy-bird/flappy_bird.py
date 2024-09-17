
from build.util.fancy_text import CM, Fore
from build.nn.base import Model
from build.models.sub import Linear

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


pygame.font.init()  # init font

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32

# Define window
WIN_HEIGHT = 800
WIN_WIDTH  = 600
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
        activation = (~self.dead & (activation[:, 0] >= 0.75))
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
    def __init__(self, inputs, outputs, dim_size, device, dtype):
        super().__init__()
        self.act_proj   = Linear(inputs, dim_size, True, device, dtype, nn.Tanh())
        self.mean       = Linear(dim_size, outputs, True, device, dtype, nn.Sigmoid())
        self.log_std    = Linear(dim_size, outputs, True, device, dtype, nn.Tanh())
        self.rew_proj   = Linear(inputs, dim_size, True, device, dtype, nn.Tanh())
        self.decode     = Linear(dim_size, 1, True, device, dtype, None)

    def forward(self, state: Tensor):
        return self.get_policy(state)

    def get_mean(self, latent: Tensor) -> Tensor:
        return self.mean(latent)

    def get_std(self, latent: Tensor) -> Tensor:
        return torch.exp(self.log_std(latent) * 2)

    def get_action(self, state: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.act_proj(state)
        mean, std = self.get_mean(latent), self.get_std(latent)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_action(self, state: Tensor, action: Tensor) -> [Tensor, Union[Tensor, None]]:
        latent = self.act_proj(state)
        mean, std = self.get_mean(latent), self.get_std(latent)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy

    def get_policy(self, state: Tensor, **options) -> Tensor:
        latent = self.act_proj(state)
        mean, std = self.get_mean(latent), self.get_std(latent)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action

    def get_value(self, state: Tensor) -> Tensor:
        latent = self.rew_proj(state)
        value = self.decode(latent)
        return value


# Network
INPUTS      = 3
OUTPUTS     = 1
GENOMES     = 1000
EMBED_SIZE  = 64
SEQ_LEN     = 32
LAYERS      = 1
ENABLE_BIAS = True
# MODEL = build.models.MiniFormer(INPUTS, OUTPUTS, EMBED_SIZE, SEQ_LEN, LAYERS, 1, 1, 0.1, ENABLE_BIAS, DEVICE, DTYPE,
#                                distribution='normal', pri_actv=build.nn.activations.Tanh(), sec_actv=nn.Sigmoid())
MODEL = RModel(INPUTS, OUTPUTS, EMBED_SIZE, DEVICE, DTYPE)
INIT_GEN: int = None


def evaluate(population: neat.Population, **options):
    trainer: neat.rl.PPO = options['trainer']
    BUFFER = torch.zeros(SEQ_LEN, population.pop_size, INPUTS).to(DEVICE, DTYPE)
    global INIT_GEN
    if INIT_GEN is None:
        INIT_GEN = population.generation

    def extend(policy: Tensor):
        # buffer(seq_len, genomes, features), policy(1, genomes, features)
        BUFFER[:-1] = BUFFER[1:].clone()
        BUFFER[SEQ_LEN-1:SEQ_LEN] = policy
        return BUFFER

    for genome in population.genomes.values():
        genome.fitness = 0

    game = Game(population.pop_size, DEVICE, DTYPE)

    limit = 10
    exe = True
    gts = clock.perf_counter()
    run_step = 0
    while exe and game.birds.active() > 0:
        BUFFER[:] = 0
        game.tick(1000)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exe = False

        with torch.no_grad():
            # Get Inputs ~ send bird location, top pipe location and bottom pipe location
            # and determine from network whether to jump or not
            observations  = game.get_state() # shape(seq_len=1, genomes, features_in)
            if run_step == 0 and population.generation == INIT_GEN:
                print(f"\nobservations =>\n{observations}\n\tshape = {observations.shape}")
            # observations  = extend(inputs)
            # if run_step == 0:
            #     print(f"extended inputs =>\n{inputs}\n\tshape = {inputs.shape}")
            ts = clock.perf_counter()
            actions, probs = MODEL.get_action(observations) # genome_mask=~game.birds.dead) # shape(seq_len=1, genomes, features_out)
            if run_step == 0 and population.generation == INIT_GEN:
                print(f"actions =>\n{actions.transpose(-1, -2)}\n\tshape = {actions.shape}")
            calc_time = clock.perf_counter() - ts
            game.update(actions)
            game.draw()
            rewards = game.birds.score.unsqueeze(-1)
            if run_step == 0 and population.generation == INIT_GEN:
                print(f"rewards =>\n{rewards.transpose(-1, -2)}\n\tshape = {rewards.shape}")
            # print(f"\rO = {outputs.flatten().cpu().numpy()} SCORE: = {game.birds.score.cpu().numpy()}", end='')

            trainer.update(observations, actions, probs, rewards, game.birds.active() == 0 or game.score >= limit)

            alive = round((torch.sum(~game.birds.dead) / game.birds.dead.numel() * 100).item(), 2)
            max_score = round(rewards.max().item(), 2)
            print(f"\r{CM('Executing', Fore.GREEN)}: time_elapsed = {round(clock.perf_counter()-gts)}s, "
                  f"alive = {alive}, max_rew = {max_score}, ct={calc_time:.2e}", end='')

        # break if score gets large enough
        if game.score >= limit:
            # pickle.dump(population.genomes, open(".\\best.pickle", "wb"))
            break

        run_step += 1

    for idx, (score, genome) in enumerate(zip(game.birds.get_reward(), population.genomes.values())):
        genome.fitness = score.item()


def run():
    # Configuration
    print(f"creating config")
    config = neat.Config('flappy_bird', os.path.join(os.path.dirname(__file__), 'configs'))
    config.general.fitness_threshold        = 1e8
    config.reproduction.elitism             = 100
    config.reproduction.min_species_size    = 200
    config.reproduction.survival_threshold  = 0.05
    # Create the population, which is the top-level object for a NEAT run.
    print(f"creating population")
    population = neat.Population(GENOMES, MODEL, config, init_reporter=2)
    for p in population.modules:
        print(p)

    trainer = neat.rl.PPO(MODEL, population, DEVICE, DTYPE, gamma=0.50,
                          scheduler=neat.scheduler.CosineAnnealing(config, 101, 20, 0.01, True))

    # Run for up to 50 generations.
    print(f"starting evaluation")
    trainer.learn(evaluate, 100, 1, 64, 0.5, 1, True)

    # for p in population.get(winner):
    #     print(p)

    # show final stats
    # print('\nBest genome:\n{!s}'.format(winner.key))


if __name__ == '__main__':
    run()
