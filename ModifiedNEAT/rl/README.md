# Reinforcement Learning

`ModifiedNEAT.rl` collects environment rollouts for a population, scores its genomes, and records training metrics. Its trainers are `NEAT` and `PPO`; both are available as `neat.rl.NEAT` / `neat.rl.PPO` and `neat.NEAT` / `neat.PPO`. `Ghost` is the `NEAT` trainer with an explicit `ghosts` mapping for cross-genus policy regularization.

## Training loop

Create a concrete `NeatModule` for your policy and a `Population`, then pass the population to a trainer:

```python
trainer = neat.rl.NEAT(population, log_dir="runs/", log_name="experiment-01")
trainer.learn(evaluate, steps=2048, epochs=10, verbose=1)
```

`evaluate` is called once per training update as `evaluate(population, trainer=trainer)`. It is responsible for running the environment and assigning each genome's fitness. Before recording steps, update the trainer's genome-to-row mapping; then call `trainer.update(...)` with the observation, action, reward, and termination status for each environment step. `PPO.update(...)` also takes the action log-probabilities produced by the policy. `update` returns whether the requested rollout length has been reached. See [the maze example](../../examples/maze/main.py) for a complete environment loop.

`steps` is the number of records collected per update, `epochs` is the number of updates (`None` runs indefinitely), and `verbose=1` prints a per-update report. The default batching mode processes genomes individually in record batches; `batching="record"` processes all genomes in a genus together, with `record_batch_size` records at a time.

## TensorBoard

The trainer creates a TensorBoard run automatically. By default, event files go under `storage/neat_rl_logs/<TrainerClass>/`; use `log_dir`, `log_sub_dir`, and `log_name` to choose another location. Built-in rollout, policy, population, timing, and scheduler metrics are logged once per update.

To add environment metrics, register a source before calling `learn`. A dictionary is useful when your environment updates the values during evaluation:

```python
env_metrics = {"goal": 0.0, "difficulty": 1.0}
trainer.add_log_source("env", env_metrics)

def evaluate(population, trainer):
    env_metrics["goal"] = current_goal
    env_metrics["difficulty"] = current_difficulty
    ...

trainer.learn(evaluate, steps=2048, epochs=10)
```

At the end of each update the trainer reads the dictionary and writes tags such as `env/goal` and `env/difficulty`. Alternatively, register a callable that returns the current values:

```python
trainer.add_log_source("env", lambda trainer: {
    "goal": current_goal,
    "difficulty": current_difficulty,
})
```

For values available only inside the evaluation function, queue them with `trainer.log_external("env", goal=current_goal)`. They are written with the next update's metrics; repeated writes to the same tag during one update keep the last value. Set `report=True` in `add_log_source` to also show the values in the terminal report.

For a one-off write, use `trainer.log_now(custom={"score": score})`; this writes immediately at the current update count. Sections become TensorBoard tag prefixes, and arrays with multiple values are written as histograms. For other TensorBoard events, `trainer.tb.call` forwards to the underlying `SummaryWriter`, for example `trainer.tb.call("add_text", "env/notes", "Evaluation started", trainer.updates_done)`.

Start TensorBoard with the parent directory containing your run folders:

```bash
runlytics --logdir runs/
```

Open the local URL printed by TensorBoard (usually `http://localhost:6006`). If using the default trainer path, set `--logdir storage/neat_rl_logs/` instead.

## Rollout data and performance

`trainer.primary` stores step-level rollout arrays and `trainer.secondary` stores per-episode values. Each array has shape `(genomes, records, *features)` with a valid-record count per genome. `trainer.logging` is a separate scalar time-series buffer. `RolloutBuffer` and `LogBuffer` can each be saved with `.save(path)` to a `.replay` file and restored with their `.load(path)` class methods; rollout loading accepts `mmap=True` for memory-mapped arrays.

Rollouts use NumPy arrays. Numba kernels handle episode sorting, returns, reductions, and other post-rollout calculations; PyTorch is used for model passes. Model passes can run on the configured Torch device, but RL bookkeeping currently uses the CPU backend, including when a CUDA device is selected.
