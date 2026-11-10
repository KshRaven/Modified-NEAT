# TODO

- Run the flappy-bird example with `NOISE_LOG=True, NOISE_REG=0` for the baseline noise level, then `NOISE_REG=0.25`; compare `diff_attn/m*/noise_*`.
- Attention noise probe measures sensitivity of the maps to input noise; consider a task-grounded variant (mass on tokens that do not change the action).
- `share_kv` gives exact cache/full-sequence parity but older tokens are never contextualised by later layers; evaluate whether a banded (streaming) variant with full-history evaluation is worth it.
- ConvSelfAttention / ConvCrossAttention do not have diff_mode / branch_mode / noise probe yet.
- Examples other than flappy-bird/sequential.py are not wired to the new options.
- With `SHARE_KV=True` the policy-accuracy pass is skipped (`POL_REG=0`); run it once with `pol_reg>0` as a parity audit (policy_accuracy should stay at 1.0 up to float error).
- Attention folds extra dims into one batch axis, so a KV cache needs the same extra dims on every call (a change re-initialises it). ConvSelfAttention / ConvCrossAttention still expect their original layout.
- Notebook: the rendered `test_best_network` is skipped (needs a window); add a headless replay (frames to a GIF) if wanted.
- Public GitHub copy is behind the working tree (runlytics logger, `GENERA` support); push the tree or use `modified-neat-sync.zip` via `PATCH_ZIP`.
- `GENERA > 1` in `main.py` calls the shared model once per genus with explicit keys; `sequential.py` makes one call for all genera (needed by the KV cache).
