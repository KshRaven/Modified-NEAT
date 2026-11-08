"""Flappy Bird training with one shared model and multiple NEAT genera."""

import argparse
import time as clock

import numpy as np
import torch

import main as example


GENERA = example.MODEL_NUM
FILE_NAME = f"{example.FILE_NAME}-shared-generas"
FILE_DIR = example.FILE_DIR
FILE_NO = None
INIT_GEN = None
POPULATION = None
ENV = None
MODEL = None


def evaluate(population: example.neat.Population, **options):
    global INIT_GEN, FILE_NO

    trainer: example.neat.rl.PPO = options['trainer']
    mapping = population.get_mapping(consolidated=True)
    genome_keys = list(mapping)
    trainer.update_mapping(mapping)
    if INIT_GEN is None:
        INIT_GEN = population.generation

    for genome in population.genomes.values():
        genome.fitness = 0

    MODEL.eval()
    terminate = False
    while not terminate:
        start_time = clock.perf_counter()
        states = example._T(ENV.reset(mapping=mapping)[0])
        step = 0
        done = False
        while not done:
            with torch.no_grad():
                debug = step == 16 and (
                    population.generation == INIT_GEN
                    or (population.generation > 0 and population.generation % 50 == 0)
                )
                if debug:
                    print(f"\nstates =>\n{states}\n\tshape = {states.shape}")

                if example.FILTER_DEAD:
                    active_indices = np.flatnonzero(ENV.players.active).tolist()
                else:
                    active_indices = list(range(len(genome_keys)))
                if not active_indices:
                    active_indices = list(range(len(genome_keys)))

                active_keys = [genome_keys[index] for index in active_indices]
                indices = torch.as_tensor(active_indices, device=states.device, dtype=torch.int64)
                observations = states.index_select(0, indices)
                actions, log_probs = MODEL.get_action(
                    observations.unsqueeze(1), keys=active_keys
                )
                actions = actions.squeeze(1)
                log_probs = log_probs.squeeze(1)

                if len(active_indices) != len(genome_keys):
                    full_actions = torch.zeros(
                        len(genome_keys), *actions.shape[1:], device=actions.device, dtype=actions.dtype
                    )
                    full_log_probs = torch.zeros(
                        len(genome_keys), *log_probs.shape[1:], device=log_probs.device, dtype=log_probs.dtype
                    )
                    full_actions[indices] = actions
                    full_log_probs[indices] = log_probs
                    actions, log_probs = full_actions, full_log_probs

                env_actions = actions if example.DISCRETE else torch.tanh(actions)
                next_states, rewards, _, done, _ = ENV.step(env_actions)
                next_states, rewards = example._T(next_states), example._T(rewards)
                terminate = trainer.update(states, actions, rewards, log_probs, done, done)

                print(
                    f"\r{example.CM('Executing', example.Fore.GREEN)}: "
                    f"time_elapsed = {round(clock.perf_counter() - start_time)}s, "
                    f"alive = {ENV.players.active_total}, "
                    f"max_rew = {round(rewards.max().item(), 2)}, "
                    f"sd={trainer.steps_done}, bl={trainer.primary.max_size()}     ", end=''
                )
                states = next_states

            if done or terminate:
                terminate = True
                break
            step += 1

        print("\n------------------------------ Post debugging ------------------------------")
        print(f"Buffer multiplier = {example.MEMORY_SIZE}")
        print(f"Mapping = {trainer.episode_mapping}")
        print(f"Lengths = {trainer.episode_lengths}")
        print(f"Episodes Primary = {trainer.primary.episodes()}")
        print(f"Episodes Secondary = {trainer.secondary.episodes()}")
        print("----------------------------------------------------------------------------")

        lower_limit, upper_limit = 10, ENV.world.floor_y * 1.05
        for key, index in mapping.items():
            genome = population.genomes[key]
            genome.fitness = float(ENV.players.fitness[index])
            y_position = ENV.birds.y[index]
            if y_position <= lower_limit or y_position >= upper_limit:
                population.to_delete.append(genome.key)

    MODEL.train()
    _, FILE_NO = population.save(
        FILE_NAME, FILE_DIR, FILE_NO, replace=population.generation != INIT_GEN
    )
    if population.generation == INIT_GEN:
        print(f"Saved initial population to file number '{FILE_NO}'")
        INIT_GEN = population.generation
        FILE_NO += 1


def test_best_network():
    elites = []
    for species_index, specie in enumerate(POPULATION.species_set.species.values()):
        genomes = sorted(
            [genome for genome in specie.members.values()
             if genome.fitness is not None and not np.isnan(genome.fitness)],
            key=lambda genome: genome.fitness,
            reverse=True,
        )
        elites.extend(genomes[:example.TEST_ELITES])
        print(f"\nElites [{species_index}]: {[genome.key for genome in genomes[:example.TEST_ELITES]]}")
    if not elites:
        print("No evaluated genomes to test")
        return

    POPULATION.crop([genome.key for genome in elites], example.DEVICE)
    env = example.Game(
        render_mode='human', goal=example.TEST_GOAL, seq_len=None, height=700, width=1000,
        state_type='full' if example.FULL_STATES else 'basic',
        pipe_speed_y=example.PIPE_Y_VELOCITY * 1.0, spawn_width=example.SPAWN_WIDTH,
        gap_offset=example.GAP_OFFSET, fps=45,
    )
    mapping = POPULATION.get_mapping(consolidated=True)
    keys = list(mapping)

    MODEL.eval()
    try:
        for run_index in range(example.TEST_RUNS):
            print(f"Running test {run_index + 1}/{example.TEST_RUNS}")
            done = False
            states = example._T(env.reset(mapping=mapping)[0])
            while not done:
                with torch.no_grad():
                    actions = MODEL.get_policy(states.unsqueeze(1), keys=keys).squeeze(1)
                env_actions = actions if example.DISCRETE else torch.tanh(actions)
                states, _, _, done, _ = env.step(env_actions)
                states = example._T(states)
                env.render()
            print(f"Top score: {env.players.scores.max().item()}")
    except KeyboardInterrupt:
        print("\nTerminated")
    finally:
        MODEL.train()


def load_population(value: str | None):
    global FILE_NO
    if value is None or value.lower() in ("false", "none", "null", ""):
        return
    if value.lower() == "true":
        FILE_NO = None
    else:
        try:
            FILE_NO = int(value)
        except ValueError:
            raise SystemExit(f"--load must be 'true', 'false' or a file number, got '{value}'")
    POPULATION.load(FILE_NAME, FILE_DIR, FILE_NO)


def run():
    global ENV, FILE_NO, INIT_GEN, MODEL, POPULATION

    parser = argparse.ArgumentParser(description="Run NEAT on Flappy Bird with one shared model")
    parser.add_argument("--rm", type=str, default=None, help='Render mode for the game (e.g. "human")')
    parser.add_argument("--load", type=str, default=None,
                        help="Load this example's latest checkpoint or a file number")
    parser.add_argument("--train", type=str, default="true", help="Train the model (true or false)")
    args = parser.parse_args()

    ENV = example.Game(
        render_mode=args.rm, goal=example.GOAL, seq_len=None, height=700, width=700,
        state_type='full' if example.FULL_STATES else 'basic',
        pipe_speed_y=example.PIPE_Y_VELOCITY, spawn_width=example.SPAWN_WIDTH,
        gap_offset=example.GAP_OFFSET, gap_size=example.GAP_SIZE,
        fps=None if args.rm != 'human' else 60,
    )
    example.INPUTS = int(ENV.observation_space.shape[-1])
    example.OUTPUTS = 2 if example.DISCRETE else int(ENV.action_space.shape[-1])
    MODEL = example.BaseModel(
        example.INPUTS, example.OUTPUTS, example.EMBED_SIZE, example.LAYERS,
        example.TEST_ACTIVATION, example.PROBABILISTIC, example.BIAS, example.DEVICE, example.DTYPE,
        clip_min=example.CLIP_MIN, clip_max=example.CLIP_MAX,
        distribution=example.DISTRIBUTION, fwd_exp=example.FWD_EXP,
        normalize=example.NORMALIZE, coefficients=example.COEFFICIENTS,
    )
    example.MODELS = [MODEL]
    print(MODEL)

    print("\ncreating shared-genus population")
    POPULATION = example.neat.Population(
        example.GENOMES * GENERA, MODEL, example.CONFIG, genera=GENERA, init_reporter=True
    )
    print(POPULATION)
    load_population(args.load)
    INIT_GEN = None

    if example.str_to_bool(args.train):
        print(
            f"Calculated hyperparameters: gamma={example.GAMMA:.4f}, "
            f"kappa={example.KAPPA:.4f}, alpha={example.ALPHA:.4f}"
        )
        trainer = example.neat.rl.PPO(
            POPULATION,
            schedulers=[
                example.scheduler.CosineAnnealing(
                    example.CONFIG, 30, 0.5,
                    ["weight_mutate_power", "weight_init_std"], True, True
                ),
            ],
            device=example.DEVICE, dtype=example.DTYPE,
            log_sub_dir='flappy_bird/',
            log_name=f"shared-{clock.time():.0f}-e{example.EMBED_SIZE}-l{example.LAYERS}",
            gamma=example.GAMMA, alpha=example.ALPHA, kappa=example.KAPPA,
            order=example.ALPHA_ORDER, normalize=example.REW_NORM,
            rew_reg=example.REW_REG, loss_reg=example.LOSS_REG, val_reg=example.VAL_REG,
            pol_reg=example.POL_REG, ent_reg=example.ENT_REG, div_reg=example.DIV_REG,
            cpy_reg=example.CPY_REG, validate=True, segr_size=None, use_entropy=True,
            use_critic=False, max_episodes=example.MEMORY_SIZE, max_steps=None,
        )
        trainer.set_report_hook(example.genome_debug)
        print(f"starting evaluation: population={len(POPULATION.genomes)}")
        try:
            trainer.learn(
                evaluate, example.STEPS, example.EPOCHS, 1380,
                accuracy_error=0.1,
                accuracy_type='binary' if not example.DISCRETE else 'discrete',
                verbose=True,
            )
        except KeyboardInterrupt:
            pass

    test_best_network()


if __name__ == "__main__":
    run()