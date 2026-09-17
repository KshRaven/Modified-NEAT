#!/usr/bin/env python3
"""
install.py — Optional helper script for ModifiedNEAT installation with GPU auto-detection.

This script is OPTIONAL. You can also install directly with pip:
    pip install .              # CPU-only (default, recommended)
    pip install .[gpu]         # GPU support (requires the correct CUDA-compatible wheel index)

DIRECT pip/uv INSTALLATION (Recommended):
    # CPU-only (safe, works everywhere):
    pip install .

    # GPU with auto-detected CUDA:
    pip install --extra-index-url https://download.pytorch.org/whl/cu132 -e .[gpu]

USING THIS SCRIPT (Advanced):
    python install.py              # Auto-detect and install CPU or GPU
    python install.py --cpu        # Force CPU-only installation
    python install.py --gpu        # Auto-detect CUDA and install GPU
    python install.py --all        # Same as --gpu

Post-installation:
    Set device via environment variable before importing:
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


# TODO: Find a way to directly install latest cuda drivers, cuda-toolkit libraries and C/C++ build libraries
#       that come from visual studio when necessary. Especially for Windows
# TODO: Instead of multiple cuda wheels, find a way to resolve and get the latest PyTorch wheel. 
#       ie pytorch.org./.../cu<version> instead of the current cu11<v>, ..., cu13<v>
# TODO: Find way to resolve latest version of cupy and supported version of numba-cuda


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

DEFAULT_PYPI_INDEX_URL = "https://pypi.org/simple"

# PyTorch wheel index URLs for different CUDA versions
PYTORCH_INDEX_MAP: dict[int | None, str] = {
    # maps CUDA major version → PyTorch wheel index URL
    None: "https://download.pytorch.org/whl/cpu",
    11: "https://download.pytorch.org/whl/cu118",
    12: "https://download.pytorch.org/whl/cu121",
    13: "https://download.pytorch.org/whl/cu132",
}

# PyTorch version mapping to ensure compatibility
PYTORCH_VERSION_MAP: dict[int | None, str] = {
    None: "torch",
    11: "torch>=2.0.0,<2.5.0",      # CUDA 11.8 TODO: Verify
    12: "torch>=2.0.0,<2.5.0",      # CUDA 12.1
    13: "torch>=2.4.0,<2.6.0",      # CUDA 13.0
}

GPU_WHEEL_MAP: dict[int, str] = {
    # maps CUDA major version → PyPI cupy wheel name
    11: "cupy-cuda11x",
    12: "cupy-cuda12x",
    13: "cupy-cuda13x",
}

CUPY_EXTRA_INDEX = "https://pypi.ngc.nvidia.com"


def pytorch_index_for(major: int | None) -> str | None:
    """Get PyTorch wheel index URL for given CUDA major version."""
    return PYTORCH_INDEX_MAP.get(major)


def pytorch_version_for(major: int | None) -> str:
    """Get PyTorch version constraint for given CUDA major version."""
    return PYTORCH_VERSION_MAP.get(major, PYTORCH_VERSION_MAP[None])


def cupy_wheel_for(major: int) -> str | None:
    return GPU_WHEEL_MAP.get(major)


def pip_extra_index_args(enable_gpu: bool, cuda_major: int | None = None) -> list[str]:
    """Return explicit extra-index-url arguments for CPU/GPU torch installs."""
    urls: list[str] = []
    seen: set[str] = set()

    for url in [DEFAULT_PYPI_INDEX_URL, pytorch_index_for(cuda_major if enable_gpu else None)]:
        if url and url not in seen:
            seen.add(url)
            urls.extend(["--extra-index-url", url])

    if enable_gpu:
        cupy_index = CUPY_EXTRA_INDEX
        if cupy_index not in seen:
            seen.add(cupy_index)
            urls.extend(["--extra-index-url", cupy_index])

    return urls


def install_with_pip(enable_gpu: bool, cuda_major: int | None = None) -> bool:
    """Perform installation using pip with proper index URLs."""
    
    # Step 1: Install PyTorch with correct index
    print("\n" + "=" * 70)
    if enable_gpu and cuda_major:
        index_url = pytorch_index_for(cuda_major)
        torch_spec = pytorch_version_for(cuda_major)
        print(f"Installing PyTorch for CUDA {cuda_major}.x")
        print(f"Index: {index_url}")
        cmd = [
            sys.executable, "-m", "pip", "install",
            *pip_extra_index_args(True, cuda_major),
            torch_spec
        ]
    else:
        print("Installing CPU-only PyTorch")
        torch_spec = pytorch_version_for(None)
        cmd = [
            sys.executable, "-m", "pip", "install",
            *pip_extra_index_args(False, None),
            torch_spec
        ]
    print("=" * 70)
    
    try:
        subprocess.check_call(cmd)
        print(f"✓ PyTorch installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install PyTorch: {e}")
        return False

    # Step 2: Install ModifiedNEAT with appropriate extras
    print("\n" + "=" * 70)
    if enable_gpu:
        print("Installing ModifiedNEAT with GPU support")
        install_cmd = [
            sys.executable, "-m", "pip", "install",
            *pip_extra_index_args(True, cuda_major),
            "-e", ".[gpu]"
        ]
    else:
        print("Installing ModifiedNEAT (CPU-only)")
        install_cmd = [
            sys.executable, "-m", "pip", "install",
            *pip_extra_index_args(False, None),
            "-e", "."
        ]
    print("=" * 70)

    try:
        subprocess.check_call(install_cmd)
        print("✓ ModifiedNEAT installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install ModifiedNEAT: {e}")
        return False

    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install ModifiedNEAT with optional GPU/CUDA support",
        epilog="""
Examples (using this script):
  python install.py              # Auto-detect CPU/GPU and install
  python install.py --cpu        # Force CPU-only installation
  python install.py --gpu        # Auto-detect CUDA and install GPU
  python install.py --all        # Same as --gpu

Direct pip installation (no script needed):
  pip install .                  # CPU-only (default)
  pip install --extra-index-url https://download.pytorch.org/whl/cu132 -e .[gpu]  # GPU with CUDA 13.0
  pip install -c constraints-gpu-cu132.txt -e .[gpu]  # Using constraints file
        """
    )
    parser.add_argument("--cpu", action="store_true",
                        help="CPU-only installation")
    parser.add_argument("--gpu", action="store_true",
                        help="Enable GPU support (auto-detect CUDA version)")
    parser.add_argument("--all", action="store_true",
                        help="Install all optional dependencies (same as --gpu)")
    args = parser.parse_args()

    enable_gpu = args.gpu or args.all
    
    if args.cpu and enable_gpu:
        parser.error("--cpu and --gpu/--all are mutually exclusive")

    print("=" * 70)
    print("ModifiedNEAT Installation Helper")
    print("=" * 70)
    print("\nNote: You can also install directly with pip without this script:")
    print("  pip install .")
    print("  pip install --index-url https://download.pytorch.org/whl/cu132 -e .[gpu]")
    print("  python setup.py develop")

    cuda_major = None
    if not args.cpu:
        # Try to detect CUDA
        cuda_ver = detect_cuda()
        if cuda_ver:
            cuda_major, cuda_minor = cuda_ver
            print(f"\n✓ Detected CUDA {cuda_major}.{cuda_minor}")
            enable_gpu = True
        elif enable_gpu:
            print("\n✗ No CUDA detected, but --gpu was specified.")
            print("Falling back to CPU-only installation.")
            enable_gpu = False
        else:
            print("\nNo CUDA detected. Installing CPU-only version.")

    if not install_with_pip(enable_gpu, cuda_major):
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Installation Complete!")
    print("=" * 70)
    
    if enable_gpu:
        print("\nModifiedNEAT is configured for GPU acceleration.")
        print("Set device via environment variable:")
        print("  export MODIFIEDNEAT_DEVICE=cuda")
    else:
        print("\nModifiedNEAT is configured for CPU.")
        print("To enable GPU support later:")
        print("  pip install --index-url https://download.pytorch.org/whl/cu132 .[gpu]")

    print("\nTo import ModifiedNEAT:")
    print("  from ModifiedNEAT import population")


if __name__ == "__main__":
    main()
