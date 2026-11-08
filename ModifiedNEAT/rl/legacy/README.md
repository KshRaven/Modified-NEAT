# RL training guide

The reinforcement-learning integration in ModifiedNEAT provides rollout-based training loops for evolving genomes against environments. The main entry points are the reusable algorithm base and the concrete trainers implemented for NEAT-style rollout updates.

## Main classes

- Algorithm: shared rollout and replay-buffer machinery
- NEAT: rollout-based trainer for episodic scoring and fitness assignment
- PPO: an alternative policy optimization implementation

## Typical workflow

```python
import torch
import ModifiedNEAT as neat

population = neat.Population(...)
trainer = neat.NEAT(
    population,
    schedulers=None,
    device=torch.device("cpu"),
    log_dir="storage/neat_rl_logs/",
)

trainer.learn(
    evaluation_function=my_eval_fn,
    steps=2048,
    epochs=10,
    batch_size=64,
    verbose=2,
)
```

## What the trainer does

The training loop typically performs:

1. rollout collection from the environment
2. buffer updates for state, action, reward, and episode metadata
3. return computation for each episode
4. genome scoring and population fitness updates

## Practical notes

- Keep rollout size consistent with the complexity of the environment.
- Tune reward regularization terms carefully for custom reward shaping.
- Use verbose output during debugging to inspect rollout and buffer behavior.

The RL components are designed to work alongside the population and config systems, so they can be used as part of a larger evolutionary experiment.
