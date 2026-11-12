# TODO

- Run a Flappy Bird baseline with `NOISE_LOG=True` and `NOISE_REG=0`, then compare it with `NOISE_REG=0.25`; review `diff_attn/m*/noise_*`.
- Evaluate a task-grounded attention-noise score based on attention mass on tokens that do not change the selected action.
- Audit policy accuracy with `SHARE_KV=True` and `pol_reg>0`; it should remain 1.0 within floating-point tolerance.
- Evaluate whether a banded streaming attention variant is useful when later layers cannot contextualize cached older tokens.
- Document and test the KV-cache requirement that extra input dimensions remain unchanged between calls.
- Add a headless `test_best_network` replay that exports frames as a GIF.
- Keep the shared-model `GENERA` behavior documented and exercised: `main.py` calls once per genus with explicit keys, while `sequential.py` calls once for all genera for KV-cache compatibility.