# TODO

## Population and reproduction
- Support reproduction between modules with unequal numbers of `NeatModule` instances or incompatible `NeatParameter` counts/shapes.
- Implement saving and loading of reinforcement-learning algorithms.

## Packaging and release
- Run `scripts/pypimgr` option **F** locally, then upload to TestPyPI and install from it to confirm the dashboard (`neatboard`) serves `dist` correctly.
- Resolve the legacy `ModifiedNEAT/cuda/` package: `cuda/mutation.py` imports a missing `ModifiedNEAT.cuda.initialization`, and the directory is not packaged or used. Delete it or complete it.
- Remove the unused `@supabase/supabase-js` dependency from the dashboard with `bun remove` so the Bun lockfile stays synchronized.
- Keep `ModifiedNEAT/dashboard/bun.lock` committed so dashboard builds are reproducible.
- Consider PyPI Trusted Publishing through GitHub Actions instead of API tokens.
- Consider moving to `pyproject.toml`-only metadata and removing the custom `setup.py` build commands.
- Complete the CUDA installer TODOs in `install_cuda.py`, including resolving compatible CUDA, PyTorch, CuPy, and numba-cuda versions.
- Add installation of operating-system dependencies required by CUDA libraries, including C++ build tools and the CUDA Toolkit.

## Dashboard
- Hide `/api/config` and `/api/info` values that expose the absolute `source_dir` and request host, or require authentication when hosted remotely.
- Address the arbitrary-code execution risk from pickle loading, for example with a restricted loader or authentication for `neatboard --host`.
- Fix `dashboard/launch.py` importing `backend.app` so it works outside the dashboard directory, or keep it explicitly as a development helper.
- Clarify the verbose levels in `cli.py` and `launch.py`, which are still subject to change.

## Tests
- Categorize the test suite by area and purpose.

## Implemented
- `Population` initialization accepts a genus count or explicit genus IDs and creates all genomes in one shared module parameter batch. Genus-specific mappings, speciation, reproduction, cropping, and save/load preserve genus identity; population absorption remains available.
