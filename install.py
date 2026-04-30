#!/usr/bin/env python3
"""
install.py — install ModifiedNEAT from pyproject.toml with optional GPU support.

Usage:
    python install.py          # CPU-only installation (default)
    python install.py --cpu    # explicitly specify CPU-only
    python install.py --gpu    # auto-detects CUDA and installs GPU dependencies
    python install.py --all    # same as --gpu

Strategy
--------
1. Install core dependencies from pyproject.toml (CPU-compatible).
2. Optionally probe for CUDA and install GPU dependencies if --gpu or --all is specified.
3. GPU packages (cuda-python, cuda-toolkit, numba-cuda, cupy) are OPTIONAL.
4. Base modules use CPU by default and auto-detect GPU if available.

Install options:
    pip install .              # CPU only (default)
    pip install .[cpu]         # CPU only (same as above)
    pip install .[gpu]         # CPU + GPU (auto-detects CUDA version)
    pip install .[all]         # Same as .[gpu]

Post-installation:
    Set the device via environment variable before importing:
        export MODIFIEDNEAT_DEVICE=cpu      # Use CPU (default)
        export MODIFIEDNEAT_DEVICE=cuda     # Use CUDA/GPU
"""

import subprocess
import sys
import re
import argparse
import warnings
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# CUDA detection
# ---------------------------------------------------------------------------

def cuda_version_from_nvcc() -> tuple[int, int] | None:
    """Return (major, minor) from nvcc --version, or None."""
    r = run(["nvcc", "--version"])
    if r.returncode != 0:
        return None
    m = re.search(r"release (\d+)\.(\d+)", r.stdout)
    return (int(m.group(1)), int(m.group(2))) if m else None


def cuda_version_from_nvidia_smi() -> tuple[int, int] | None:
    """Return (major, minor) from nvidia-smi, or None."""
    r = run(["nvidia-smi"])
    if r.returncode != 0:
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", r.stdout)
    return (int(m.group(1)), int(m.group(2))) if m else None


def cuda_version_from_cuda_python() -> tuple[int, int] | None:
    """Return (major, minor) via the cuda-python runtime API, or None."""
    try:
        from cuda import cuda  # type: ignore
        err, device_count = cuda.cuDeviceGetCount()
        if err.value != 0 or device_count == 0:
            return None
        err, ver = cuda.cuDriverGetVersion()
        if err.value != 0:
            return None
        major, minor = divmod(ver, 1000)
        minor //= 10
        return (major, minor)
    except Exception:
        return None


def detect_cuda() -> tuple[int, int] | None:
    """Try every probe in order; return first hit or None."""
    for probe in (
        cuda_version_from_nvcc,
        cuda_version_from_nvidia_smi,
        cuda_version_from_cuda_python,
    ):
        ver = probe()
        if ver:
            return ver
    return None


# ---------------------------------------------------------------------------
# GPU wheel selection
# ---------------------------------------------------------------------------

GPU_WHEEL_MAP: dict[int, str] = {
    # maps CUDA major version → PyPI cupy wheel name
    11: "cupy-cuda11x",
    12: "cupy-cuda12x",
    13: "cupy-cuda13x",
}

CUPY_EXTRA_INDEX = "https://pypi.ngc.nvidia.com"


def cupy_wheel_for(major: int) -> str | None:
    return GPU_WHEEL_MAP.get(major)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install ModifiedNEAT with optional GPU/CUDA support",
        epilog="""
Examples:
  python install.py              # CPU-only (default, recommended)
  python install.py --cpu        # Explicitly CPU-only
  python install.py --gpu        # Auto-detect CUDA and install GPU deps
  python install.py --all        # Same as --gpu
  pip install .                  # CPU only (via pip)
  pip install .[gpu]             # GPU support via pip
        """
    )
    parser.add_argument("--cpu", action="store_true",
                        help="CPU-only installation (default)")
    parser.add_argument("--gpu", action="store_true",
                        help="Enable GPU support (auto-detect CUDA version)")
    parser.add_argument("--all", action="store_true",
                        help="Install all optional dependencies (same as --gpu)")
    args = parser.parse_args()

    # Determine which mode to use
    enable_gpu = args.gpu or args.all
    
    # If both --cpu and --gpu/--all are specified, that's ambiguous
    if args.cpu and enable_gpu:
        parser.error("--cpu and --gpu/--all are mutually exclusive")

    # Step 1 — install core package
    print("=" * 70)
    if enable_gpu:
        print("Installing ModifiedNEAT (CPU core + GPU support)")
    else:
        print("Installing ModifiedNEAT (CPU-only)")
    print("=" * 70)
    
    install_cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
    
    try:
        subprocess.check_call(install_cmd)
        print("✓ ModifiedNEAT core installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install ModifiedNEAT: {e}")
        sys.exit(1)

    # Step 2 — optionally install GPU dependencies
    if not enable_gpu:
        print("\n" + "=" * 70)
        print("GPU Installation: SKIPPED (default)")
        print("=" * 70)
        print("ModifiedNEAT will use CPU. To enable GPU support:")
        print("  • Auto-detect CUDA: python install.py --gpu")
        print("  • Via pip: pip install -e .[gpu]")
        print("\nTo use GPU after installation, set environment variable before importing:")
        print("  export MODIFIEDNEAT_DEVICE=cuda")
        return

    # Step 2a — GPU mode: probe for CUDA
    print("\n" + "=" * 70)
    print("GPU Installation: Probing for CUDA runtime")
    print("=" * 70)
    
    cuda_ver = detect_cuda()

    if cuda_ver is None:
        print("[GPU] No CUDA-capable GPU / toolkit detected.")
        print("      ModifiedNEAT will run on CPU only.")
        print("      If you have an NVIDIA GPU, install CUDA Toolkit and re-run:")
        print("        python install.py --gpu")
        return

    major, minor = cuda_ver
    print(f"[GPU] ✓ Detected CUDA {major}.{minor}")

    # Step 2b — install GPU dependencies
    print("\n[GPU] Installing GPU packages (cuda-python, cuda-toolkit, numba-cuda, cupy) …")
    
    gpu_packages = [
        "cuda-python",
        "cuda-toolkit",
        "numba-cuda[cu13]",
    ]
    
    cupy_wheel = cupy_wheel_for(major)
    if cupy_wheel:
        gpu_packages.append(cupy_wheel)
        print(f"[GPU] Using {cupy_wheel} for CUDA {major}.x")
    else:
        warnings.warn(
            f"No cupy wheel known for CUDA {major}.x; skipping cupy installation.\n"
            f"      See https://docs.cupy.dev/en/stable/install.html",
            UserWarning
        )

    try:
        cmd = [sys.executable, "-m", "pip", "install"] + gpu_packages
        if cupy_wheel:
            cmd.extend(["--extra-index-url", CUPY_EXTRA_INDEX])
        subprocess.check_call(cmd)
        print("✓ GPU dependencies installed successfully.")
        print(f"  ModifiedNEAT is now configured for GPU acceleration (CUDA {major}.{minor})")
        print("\nTo use GPU, set environment variable before importing:")
        print("  export MODIFIEDNEAT_DEVICE=cuda")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install GPU dependencies: {e}")
        print("  ModifiedNEAT will continue to work on CPU.")
        sys.exit(1)


if __name__ == "__main__":
    main()
