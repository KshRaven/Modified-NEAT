"""
Compare the last saved instance model (main.py) with the last saved sequential model (sequential.py) on the SAME stream.

    python compare.py                       # latest checkpoints, 5 rendered episodes, 1 elite per genus
    python compare.py --rm none             # headless table only
    python compare.py --inst-load 12 --seq-load 30 --top 3 --episodes 20 --rm none

How it works (one Game, one stream, so differences come from the models only):
  1. Every genome of both populations plays one SCREENING episode (own seed); the best `--top` per genus are kept.
  2. The elites of both models plus three scripted reference cursors play the EVALUATION episodes (other seeds):
       oracle          sees the target                        = ceiling
       memory-ideal    follows the newest clean cue in the window   = what a perfect sequential model does
       instance-ideal  follows the cue on clean frames, holds on decoys = best a no-memory model can do
     The instance model is fed only the newest frame of the window, the sequential model the whole window.
  3. Prints reward / tracking error (clean vs decoy frames) and a paired instance-vs-sequential difference per episode.
Rendered lines: orange = instance, blue = sequential, white = oracle, purple = instance-ideal, pink = memory-ideal.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.insert(0, HERE)

import main as inst              # noqa: E402  (module-level config / constants of the instance model)
import sequential as seq         # noqa: E402  (module-level config / constants of the sequential model)
from game import Game            # noqa: E402

neat, DEVICE, _T = seq.neat, seq.DEVICE, seq._T
NAMES = ["instance", "sequential", "oracle", "instance-ideal", "memory-ideal"]
SCREEN_SEED = 99991

ENV_KEYS = ("DECOY_FRAC", "DECOY_RUN", "CUE_SIGMA", "TAG_SEP", "TAG_SIGMA", "DRIFT", "OBS_NOISE", "TARGET_SPEED", "AGENT_SPEED")


def env_params() -> dict:
    """Environment of the comparison = the sequential model's constants (warns when main.py was trained on another)."""
    diff = [k for k in ENV_KEYS if getattr(inst, k) != getattr(seq, k)]
    if diff:
        print(f"WARNING: main.py and sequential.py differ in {diff}; using sequential.py's values")
    return dict(decoy_frac=seq.DECOY_FRAC, decoy_run=seq.DECOY_RUN, cue_sigma=seq.CUE_SIGMA, tag_sep=seq.TAG_SEP,
                tag_sigma=seq.TAG_SIGMA, drift=seq.DRIFT, obs_noise=seq.OBS_NOISE, target_speed=seq.TARGET_SPEED,
                speed=seq.AGENT_SPEED)


def load(mod, load_arg: str, name: str):
    """Rebuild the population of a training script exactly like its run() does, then load the checkpoint."""
    if mod.DISCRETE:
        sys.exit(f"{name}: compare.py supports continuous actions only (DISCRETE=False)")
    probe = Game(horizon=2, seq_len=None, **env_params())
    mod.INPUTS = int(probe.observation_space.shape[-1])
    mod.OUTPUTS = int(probe.action_space.shape[-1])
    models = mod.build_models()
    if mod is inst and mod.GENERA > 1:
        models = models * mod.GENERA
    G, genera = mod.GENOMES, mod.GENERA
    if genera > 1:
        pop = neat.Population(G * genera, models[0], mod.CONFIG, init_reporter=True, genera=genera)
    else:
        pop = neat.Population(G, models[0], mod.CONFIG, init_reporter=True)
        for m in models[1:]:
            pop.absorb_population(neat.Population(G, m, mod.CONFIG, init_reporter=True))
    mod.MODELS, mod.POPULATION = models, pop
    try:
        mod.load_population(load_arg)
    except Exception as e:      # missing checkpoint, changed hyper-parameters, ...
        sys.exit(f"could not load the {name} checkpoint ('{load_arg}'): {e}")
    for m in models:
        m.eval()
    return pop


def make_policy(mod, pop, sequential: bool):
    """(n, L, F) tensor of the population's rows (consolidated-mapping order) -> (n, 1) env actions in [-1, 1]."""
    split_maps = pop.get_mapping()
    if len(pop.genera) == 1:
        split_maps = (split_maps,)
    sizes = [len(m) for m in split_maps]
    cons = pop.get_mapping(consolidated=True)
    models, shared = mod.MODELS, mod.GENERA > 1

    def act(states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if sequential:
                if shared:      # one model, every genus in one call (needs the keys, see sequential.shared_call)
                    return models[0].to_env(torch.cat(mod.shared_call(models[0], 'get_policy', states.unsqueeze(1), cons), 0).squeeze(1))
                parts = [m.get_policy(o.unsqueeze(1)) for m, o in zip(models, torch.split(states, sizes, 0))]
            else:               # instance model: newest frame only
                last = states[:, -1, :]
                parts = [m.get_policy(o.unsqueeze(1), keys=list(sm.keys()) if shared else None)
                         for m, o, sm in zip(models, torch.split(last, sizes, 0), split_maps)]
            return models[0].to_env(torch.cat(parts, 0).squeeze(1))     # raw actions -> env values (tanh for LEVELS=1)
    return act, cons


def play(env: Game, acts: dict, n: int, seed: int, base: int, render: bool = False):
    """One episode of every cursor. acts: name -> (policy, row slice); rows base..base+2 are the scripted references."""
    states, _ = env.reset(keys=n, seed=seed)
    done = False
    speed = env.config.agent.speed
    pull, count, act = np.zeros((2, n)), np.zeros(2), np.zeros(n)       # [clean, decoy] x cursor
    while not done:
        s = _T(states)
        a = np.zeros(n)
        for name, (policy, sl) in acts.items():
            a[sl] = policy(s[sl]).detach().cpu().numpy().reshape(-1)
        sn, x = states, env.agents.x
        i0 = base
        a[i0] = np.clip((env.world.target - x[i0]) / speed, -1, 1)              # oracle
        last = sn[i0 + 1, -1]
        a[i0 + 1] = np.clip((last[1] - x[i0 + 1]) / speed, -1, 1) if last[2] > 0 else 0.0     # instance-ideal
        clean = np.flatnonzero(sn[i0 + 2, :, 2] > 0)                                          # memory-ideal
        a[i0 + 2] = np.clip((sn[i0 + 2, clean[-1], 1] - x[i0 + 2]) / speed, -1, 1) if clean.size else 0.0
        # pull = action * sign(cue - cursor): +1 = steers to the cue it sees, 0 = ignores it (measured BEFORE the step)
        k = int(env.world.kind)             # kind of the frame these states show: 1 = decoy
        pull[k] += np.clip(a, -1, 1) * np.sign(sn[:, -1, 1] - x)
        count[k] += 1
        act += np.abs(np.clip(a, -1, 1))
        states, _, _, done, _ = env.step(a)
        if render:
            env.render()
    env.pull = pull / np.maximum(count, 1)[:, None]
    env.activity = act / max(count.sum(), 1)
    return env


def main():
    p = argparse.ArgumentParser(description="Instance vs sequential best genomes on the same Signal Tracker stream")
    p.add_argument('--rm', default='human', help="'human' (render, default) or 'none' (table only)")
    p.add_argument('--inst-load', default='true', help="main.py checkpoint: 'true' = latest or a file number")
    p.add_argument('--seq-load', default='true', help="sequential.py checkpoint: 'true' = latest or a file number")
    p.add_argument('--top', type=int, default=1, help='elite genomes per genus kept from each model')
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--horizon', type=int, default=seq.TEST_GOAL)
    p.add_argument('--seed', type=int, default=1234, help='seed of the first evaluation episode')
    p.add_argument('--fps', type=int, default=45)
    args = p.parse_args()
    render = args.rm.lower() not in ('none', 'false', 'no', '0')
    params = env_params()
    L = seq.SEQ_LEN

    pops = {'instance': load(inst, args.inst_load, 'instance'), 'sequential': load(seq, args.seq_load, 'sequential')}
    mods = {'instance': inst, 'sequential': seq}

    # ---- 1. screening: every genome plays one episode, best `top` per genus survive
    elites = {}
    for name, pop in pops.items():
        act, cons = make_policy(mods[name], pop, name == 'sequential')
        env = Game(horizon=args.horizon, seq_len=L, **params)
        play(env, {name: (act, slice(0, len(cons)))}, len(cons) + 3, SCREEN_SEED, base=len(cons))
        fit = env.players.fitness
        keep = []
        for sp in pop.species_set.species.values():
            ranked = sorted(sp.members.values(), key=lambda g: fit[cons[g.key]], reverse=True)
            keep += [g.key for g in ranked[:args.top]]
        elites[name] = keep
        print(f"{name}: kept {len(keep)} of {len(cons)} genomes (screening best {max(fit[cons[k]] for k in keep):.3f})")
        pop.crop(keep, DEVICE)

    # ---- 2. evaluation on fresh seeds
    acts, off, sizes = {}, 0, {}
    for name, pop in pops.items():
        act, cons = make_policy(mods[name], pop, name == 'sequential')
        acts[name] = (act, slice(off, off + len(cons)))
        sizes[name] = len(cons)
        off += len(cons)
    n = off + 3
    idx = np.concatenate([np.full(sizes['instance'], 0), np.full(sizes['sequential'], 1), [2, 3, 4]])
    focus = np.concatenate([np.ones(off, bool), np.zeros(3, bool)])
    env = Game(render_mode='human' if render else None, horizon=args.horizon, seq_len=L, fps=args.fps if render else None, **params)
    env.focus = focus
    env.window.set_groups(NAMES, idx)

    rows = {g: [] for g in range(5)}
    for ep in range(args.episodes):
        seed = args.seed + ep
        if render:
            print(f"episode {ep + 1}/{args.episodes} (seed {seed}) - Esc ends the episode early")
        play(env, acts, n, seed, base=off, render=render)
        pl = env.players
        for g in range(5):
            m = idx == g
            rows[g].append([pl.fitness[m].mean(), pl.fitness[m].max(), pl.mae[m].mean(), pl.mae_clean[m].mean(), pl.mae_decoy[m].mean(),
                             env.pull[0][m].mean(), env.pull[1][m].mean(), env.activity[m].mean()])
    env.close()

    # ---- 3. report
    R = {g: np.asarray(v) for g, v in rows.items()}
    print(f"\nepisodes={args.episodes} horizon={args.horizon} seq_len={L} elites/genus={args.top}  (mean over episodes)")
    print(f"{'':16}{'reward mean':>12}{'reward best':>12}{'|err| all':>11}{'|err| clean':>12}{'|err| decoy':>12}{'decoy-clean':>12}{'pull clean':>11}{'pull decoy':>11}{'|action|':>9}")
    for g, name in enumerate(NAMES):
        r = R[g].mean(0)
        print(f"{name:16}{r[0]:12.3f}{r[1]:12.3f}{r[2]:11.3f}{r[3]:12.3f}{r[4]:12.3f}{r[4] - r[3]:12.3f}{r[5]:11.3f}{r[6]:11.3f}{r[7]:9.3f}")
    d = R[1][:, 1] - R[0][:, 1]                  # best sequential cursor - best instance cursor, per episode
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float('nan')
    print(f"\nsequential - instance (best cursor of each, same stream): {d.mean():+.4f} +/- {se:.4f} (s.e.), "
          f"sequential wins {int((d > 0).sum())}/{len(d)} episodes")
    gap = (R[0][:, 4] - R[0][:, 3]).mean(), (R[1][:, 4] - R[1][:, 3]).mean()
    print(f"decoy-clean error gap: instance {gap[0]:.3f}, sequential {gap[1]:.3f}  (smaller = ignores decoys better)")
    head = R[4][:, 0].mean() - R[3][:, 0].mean()                 # what a perfect memory could add over a perfect no-memory model
    got = R[1][:, 0].mean() - R[0][:, 0].mean()
    print(f"\nHEADROOM for memory in this env: memory-ideal - instance-ideal = {head:+.3f} reward "
          f"(sequential - instance actually achieved: {got:+.3f}, share {got / head:.0%})" if abs(head) > 1e-9 else "")
    print("pull = mean(action x sign(cue - cursor)): +1 steers to the cue it sees, 0 ignores it. Ideal: clean high, decoy ~0.\n"
          "A small decoy-clean error gap alone can just mean 'bad everywhere' (see |err| clean): read it together with pull.")
    if head < 0.05:
        print("  -> headroom is small: even a PERFECT sequential model could add only this much here. Make memory matter before blaming "
              "the model: raise DECOY_RUN (20+) / DECOY_FRAC (0.7+), lower TAG_SEP (0.3-0.8, a single frame can no longer "
              "tell clean from decoy), raise TARGET_SPEED, add DRIFT. Retrain both on the same settings.")
    print("Read: instance should sit near/below 'instance-ideal'; a sequential model that is really using its window "
          "lands between 'instance-ideal' and 'memory-ideal'. Equal to instance = the window is not being used.")


if __name__ == "__main__":
    main()