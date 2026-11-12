# Signal Tracker (attention-noisy environment)

A hidden target drifts slowly in [-1, 1]; each genome steers a cursor (`speed * action` per step) and is rewarded by
`1 - |cursor - target|`. Every frame is `[cursor, cue, tag]`:

| frame | cue | tag |
|-------|-----|-----|
| clean | target + `cue_sigma` noise | high (`+tag_sep/2`) |
| decoy | uniform noise | low (`-tag_sep/2`) |

Decoys arrive in bursts (`decoy_frac`, `decoy_run`), so the current frame is often useless: the model has to attend to the
recent clean frames and ignore the decoys. That is noise **between the instances of a sequence**. An instance model
(`main.py`) cannot solve it beyond ~`1 - E|err|` of the current frame; the sequential one (`sequential.py`) can.

Extra knobs: `drift` (slow offset shared by all frames = common-mode noise, what differential attention subtracts),
`obs_noise` (plain per-feature noise), `tag_sep` (smaller = harder to tell frames apart), `target_speed`.

## Run
```
python main.py                 # instance states (no memory)
python sequential.py           # sequential states + (differential) attention
python sequential.py --rm human --train false --load true
python compare.py             # last instance vs last sequential model on the same stream (see its docstring)
```
Compare in the dashboard (`attn_noise/*`, `diff_attn/*`): set `DIFFERENTIAL` 0 / 1 / 2 and `NOISE_REG` 0 / >0 in
`sequential.py`; raise `DECOY_RUN`, `DECOY_FRAC`, `DRIFT` or lower `TAG_SEP` to make the noise matter.

Rendering (`--rm human`): scrolling time plot (target green, best cursor white, mean blue, worst orange; cue dots green=clean, red=decoy), every cursor on the track, the live frame, and the model input window. Space pauses, +/- zoom the time axis, Shift +/- change fps, Esc stops. `render_mode="ansi"` prints one text line.

`NOISE_REG` comes from `examples/misc/noise_reg.py` (`NoiseRegPPO`).

## Reading the metrics
* `attn_noise/noise_*`: how far the attention maps move when the input is slightly perturbed (0 = ignores it).
* `diff_attn/*leak*`: `|1 + sum(lambda)|`, 0 = shared components fully rejected, 1 = plain attention.
* `diff_attn/*beta*` (simplex): rejection strength, 1 = used, 0 = off.

## Reading a genome (what "good" looks like here)
Reward = `1 - |cursor - target|`. The absolute numbers depend on `DECOY_FRAC`, `DECOY_RUN`, `TARGET_SPEED` and the horizon, so
judge a genome against reference cursors played on the **same stream** (`compare.py` adds and prints them):

| reference | what it is |
|-----------|------------|
| oracle | sees the target and moves at the speed limit: the ceiling |
| memory-ideal | follows the cue of the newest clean frame in the window: what a perfect sequential model does |
| instance-ideal | follows the cue on clean frames, holds still on decoys: the best a no-memory model can do |

Visually: the white line should sit on the green target line and stay on it **through red (decoy) bursts**. A genome that
jumps to every red dot is just copying the current cue; one that keeps a smooth line through the bursts has a memory.
Numerically (side panel): `mean |error|` on clean frames is similar for everyone, the **decoy - clean gap** is what
separates models. Gap ~ 0 = it ignores decoys; large = it follows them. The sequential models must beat the
instance-ideal row, instance models should land at or below it (they only ever see one frame, so a clear win over it means a bug or leaked information).

## Live perturbation panel (`--rm human`)
The bottom-right panel changes the stream while it runs (decoy fraction / burst length, cue / tag / observation noise, drift,
target speed): drag a slider, `R` resets them. Left click on the time plot kicks the hidden target to that height, right click
forces a decoy burst (12 frames). Hovering lights up the slider under the mouse and a crosshair shows the track value under the
cursor. The window only queues the events; `Game.apply_event` applies them (`Game.set_param`, `World.kick`, `World.burst`), so
the same calls work from code. Changing parameters mid-episode changes the random stream, so a perturbed run is not repeatable.