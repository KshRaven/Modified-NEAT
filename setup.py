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


def get_dependency_links() -> list[str]:
    links = [
        "https://pypi.org/simple",
        "https://download.pytorch.org/whl/cpu",
        "https://download.pytorch.org/whl/cu118",
        "https://download.pytorch.org/whl/cu121",
        "https://download.pytorch.org/whl/cu132",
        "https://pypi.ngc.nvidia.com",
    ]
    return list(dict.fromkeys(links))


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
        return True
    return os.environ.get("MODIFIEDNEAT_SKIP_DASHBOARD_BUILD", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def run_command(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    print(f"-> {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(cwd or ROOT), text=True, check=False)


def run_install_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    if getattr(os, "geteuid", lambda: 0)() == 0:
        return run_command(cmd)
    if shutil.which("sudo"):
        return run_command(["sudo", *cmd])
    return run_command(cmd)


def ensure_node_toolchain() -> str | None:
    if shutil.which("bun"):
        return "bun"
    if shutil.which("node") and shutil.which("npm"):
        return "npm"

    if sys.platform == "darwin" and shutil.which("brew"):
        result = run_install_command(["brew", "install", "node"])
        if result.returncode == 0 and shutil.which("node") and shutil.which("npm"):
            return "npm"
    elif os.name == "nt" and shutil.which("winget"):
        result = run_install_command(["winget", "install", "OpenJS.NodeJS.LTS", "--source", "winget"])
        if result.returncode == 0 and shutil.which("node") and shutil.which("npm"):
            return "npm"
    else:
        if shutil.which("apt-get"):
            run_install_command(["apt-get", "update"])
            run_install_command(["apt-get", "install", "-y", "nodejs", "npm"])
        elif shutil.which("dnf"):
            run_install_command(["dnf", "install", "-y", "nodejs", "npm"])
        elif shutil.which("pacman"):
            run_install_command(["pacman", "-S", "--noconfirm", "nodejs", "npm"])
        elif shutil.which("brew"):
            run_install_command(["brew", "install", "node"])

    if shutil.which("node") and shutil.which("npm"):
        return "npm"
    return None


def build_dashboard() -> None:
    if should_skip_dashboard_build():
        print("Skipping dashboard build because MODIFIEDNEAT_SKIP_DASHBOARD_BUILD was set.")
        return

    if not DASHBOARD_DIR.exists():
        return

    if (DIST_DIR / "index.html").exists():
        print("Dashboard build artifacts already exist; skipping rebuild.")
        return

    if not (DASHBOARD_DIR / "package.json").exists():
        print("Dashboard package manifest not found; skipping build.")
        return

    if not (DASHBOARD_DIR / "index.html").exists() and not (DASHBOARD_DIR / "src").exists():
        print("Dashboard source files not found; skipping build.")
        return

    manager = ensure_node_toolchain()
    if not manager:
        print("Node.js tooling could not be installed; skipping dashboard build.")
        return

    print(f"Building dashboard assets with {manager}...")
    if manager == "bun":
        install_result = run_command(["bun", "install"], DASHBOARD_DIR)
        if install_result.returncode != 0:
            print("Dashboard dependency installation failed; skipping build.")
            return
        build_result = run_command(["bun", "run", "build"], DASHBOARD_DIR)
    else:
        install_result = run_command(["npm", "install"], DASHBOARD_DIR)
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
    version="0.7.0",
    description="Modified version of python NEAT algorithm that uses PyTorch modules",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Bradley Odimmasi",
    author_email="bodimmasi@gmail.com",
    license="MIT",
    python_requires=">=3.11",
    packages=find_packages(include=["ModifiedNEAT", "ModifiedNEAT.*"]),
    include_package_data=True,
    package_data={
        "ModifiedNEAT.dashboard": [
            "dist/**",
            "index.html",
            "package.json",
            "postcss.config.js",
            "tailwind.config.js",
            "vite.config.ts",
            "tsconfig*.json",
            "eslint.config.js",
            "backend/**/*",
            "src/**/*",
        ]
    },
    install_requires=read_requirements(ROOT / "requirements.txt"),
    dependency_links=get_dependency_links(),
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
