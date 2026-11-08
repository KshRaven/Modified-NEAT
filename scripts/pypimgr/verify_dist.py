#!/usr/bin/env python3
"""Inspect built sdist/wheel files for content that must never be published.

Usage: verify_dist.py <file> [<file> ...]      Exit code 1 if anything forbidden is found.
"""
from __future__ import annotations

import re
import sys
import tarfile
import zipfile
from pathlib import Path

FORBIDDEN_PATHS = [
    r"(^|/)node_modules/", r"(^|/)dashboard/src/", r"(^|/)__pycache__/", r"\.py[cod]$", r"\.map$",
    r"(^|/)\.env", r"\.pem$", r"\.key$", r"(^|/)\.pypirc$", r"(^|/)\.git(/|ignore$|attributes$)",
    r"(^|/)bun\.lock$", r"(^|/)package(-lock)?\.json$", r"(^|/)tsconfig.*\.json$",
    r"(^|/)(vite|tailwind|postcss|eslint)\.config\.[jt]s$", r"(^|/)backend/testing/",
    r"(^|/)scripts/", r"\.(pkl|pt|pth|ckpt|db|sqlite3?)$", r"(^|/)TODO\.md$",
]
SECRET_PATTERNS = {
    "private key": rb"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "PyPI token": rb"pypi-AgE[A-Za-z0-9_\-]{40,}",
    "AWS key": rb"AKIA[0-9A-Z]{16}",
    "GitHub token": rb"gh[pousr]_[A-Za-z0-9]{36,}",
    "API key (sk-)": rb"sk-[A-Za-z0-9]{32,}",
    "Supabase/JWT": rb"eyJ[A-Za-z0-9_\-]{30,}\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}",
    "hardcoded password": rb"(?i)(password|passwd|secret|api_?key)\s*[:=]\s*['\"][^'\"\s]{6,}['\"]",
}
PATH_WARN = {
    "personal path": rb"(/home/[a-z0-9_.-]+/|/Users/[A-Za-z0-9_.-]+/|[A-Za-z]:\\\\?Users\\\\?[A-Za-z0-9_.-]+)",
}
MAX_SCAN = 8 * 1024 * 1024


def members(path: Path):
    if path.suffix == ".whl" or path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            for i in z.infolist():
                if not i.is_dir():
                    yield i.filename, (lambda n=i.filename: z.read(n))
    else:
        with tarfile.open(path) as t:
            for m in t.getmembers():
                if m.isfile():
                    yield m.name, (lambda m=m: t.extractfile(m).read())


def main(files: list[str]) -> int:
    bad = warned = 0
    for f in files:
        p = Path(f)
        print(f"\n== {p.name}")
        count = 0
        for name, reader in members(p):
            count += 1
            for pat in FORBIDDEN_PATHS:
                if re.search(pat, name):
                    print(f"  [FORBIDDEN] {name}")
                    bad += 1
                    break
            if name.endswith((".png", ".ico", ".woff", ".woff2", ".gz", ".zip")):
                continue
            data = reader()[:MAX_SCAN]
            for label, rx in SECRET_PATTERNS.items():
                if re.search(rx, data):
                    print(f"  [SECRET?]   {name}: {label}")
                    bad += 1
            if name.endswith((".py", ".md", ".txt", ".cfg", ".toml", ".json", ".html")):
                for label, rx in PATH_WARN.items():
                    if re.search(rx, data):
                        print(f"  [warn]      {name}: {label}")
                        warned += 1
        print(f"  {count} files inspected")
    print(f"\nProblems: {bad}   Warnings: {warned}")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
