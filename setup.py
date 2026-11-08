#!/usr/bin/env python3
"""Setup script for ModifiedNEAT with dashboard build support."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from setuptools import find_packages, setup
    from setuptools.command.build_py import build_py as _build_py
    from setuptools.command.develop import develop as _develop
except Exception:
    def find_packages(where=".", include=("*",), exclude=()):
        packages = []
        root = Path(where).resolve()
        for path in root.rglob("__init__.py"):
            rel = path.parent.relative_to(root)
            if rel == Path("."):
                continue
            pkg = ".".join(rel.parts)
            if include and not any(pkg.startswith(item.rstrip(".*")) for item in include):
                continue
            if exclude and any(pkg.startswith(item.rstrip(".*")) for item in exclude):
                continue
            packages.append(pkg)
        return packages

    def setup(*args, **kwargs):
        print("setuptools is not available in this environment; install it first to perform a real build/install.")

    class _build_py:  # type: ignore[no-redef]
        def run(self) -> None:
            build_dashboard()

    class _develop:  # type: ignore[no-redef]
        def run(self) -> None:
            build_dashboard()

ROOT = Path(__file__).resolve().parent
DASHBOARD_DIR = ROOT / "ModifiedNEAT" / "dashboard"
DIST_DIR = DASHBOARD_DIR / "dist"


def read_requirements(path: Path) -> list[str]:
    requirements: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("-r "):
            continue
        requirements.append(line)
    return requirements


def should_skip_dashboard_build() -> bool:
    if "--skip-dashboard-build" in sys.argv:
        sys.argv.remove("--skip-dashboard-build")  # setuptools rejects unknown options
        return True
    return os.environ.get("MODIFIEDNEAT_SKIP_DASHBOARD_BUILD", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def run_command(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    print(f"-> {' '.join(cmd)}")
    exe = shutil.which(cmd[0])  # resolves npm.cmd etc. on Windows
    if exe:
        cmd = [exe, *cmd[1:]]
    return subprocess.run(cmd, cwd=str(cwd or ROOT), text=True, check=False)


def ensure_node_toolchain() -> str | None:
    """Detect an existing JS toolchain. Never installs system packages during a build."""
    if shutil.which("bun"):
        return "bun"
    if shutil.which("node") and shutil.which("npm"):
        return "npm"
    return None


def build_dashboard() -> None:
    if should_skip_dashboard_build():
        print("Skipping dashboard build because MODIFIEDNEAT_SKIP_DASHBOARD_BUILD was set.")
        return

    if not DASHBOARD_DIR.exists():
        return

    force = os.environ.get("MODIFIEDNEAT_FORCE_DASHBOARD_BUILD", "").lower() in {"1", "true", "yes", "on"}
    if (DIST_DIR / "index.html").exists() and not force:
        print("Dashboard build artifacts already exist; skipping rebuild.")
        return

    if not (DASHBOARD_DIR / "package.json").exists():
        print("Dashboard package manifest not found; skipping build.")
        return

    if not (DASHBOARD_DIR / "src").exists():
        print("Dashboard source files not found; skipping build.")
        return

    manager = ensure_node_toolchain()
    if not manager:
        print("Node.js tooling could not be installed; skipping dashboard build.")
        return

    print(f"Building dashboard assets with {manager}...")
    if manager == "bun":
        install_result = run_command(["bun", "install", "--frozen-lockfile"] if (DASHBOARD_DIR / "bun.lock").exists() else ["bun", "install"], DASHBOARD_DIR)
        if install_result.returncode != 0:
            print("Dashboard dependency installation failed; skipping build.")
            return
        build_result = run_command(["bun", "run", "build"], DASHBOARD_DIR)
    else:
        install_result = run_command(["npm", "ci" if (DASHBOARD_DIR / "package-lock.json").exists() else "install"], DASHBOARD_DIR)
        if install_result.returncode != 0:
            print("Dashboard dependency installation failed; skipping build.")
            return
        build_result = run_command(["npm", "run", "build"], DASHBOARD_DIR)

    if build_result.returncode == 0 and (DIST_DIR / "index.html").exists():
        print("Dashboard build completed successfully.")
    else:
        print("Dashboard build did not produce assets; installation will continue.")


class BuildPyWithDashboard(_build_py):
    def run(self) -> None:
        build_dashboard()
        super().run()


class DevelopWithDashboard(_develop):
    def run(self) -> None:
        build_dashboard()
        super().run()


setup(
    name="ModifiedNEAT",
    version="0.9.1",
    description="Modified version of python NEAT algorithm that uses PyTorch modules",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Bradley Odimmasi",
    author_email="bodimmasi@gmail.com",
    license_expression="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.11",
    packages=find_packages(include=["ModifiedNEAT", "ModifiedNEAT.*"], exclude=["*.testing", "*.testing.*"]),
    include_package_data=False,
    package_data={"ModifiedNEAT.dashboard": ["dist/*", "dist/assets/*"]},  # built assets only
    exclude_package_data={"": ["*.pyc", "*.map"]},
    install_requires=sorted(
        {
            *read_requirements(ROOT / "requirements.txt"),
            *read_requirements(ROOT / "ModifiedNEAT" / "dashboard" / "backend" / "requirements.txt"),
        },
        key=str.lower,
    ),
    extras_require={
        "cpu": [],
        "gpu": [
            "cuda-python>=12.0",
            "cuda-toolkit",
            "numba-cuda[cu13]",
            "cupy-cuda13x",
        ],
        "all": [
            "cuda-python>=12.0",
            "cuda-toolkit",
            "numba-cuda[cu13]",
            "cupy-cuda13x",
        ],
    },
    entry_points={"console_scripts": ["neatboard=ModifiedNEAT.dashboard.backend.cli:main"]},
    zip_safe=False,
    cmdclass={
        "build_py": BuildPyWithDashboard,
        "develop": DevelopWithDashboard,
    },
)
