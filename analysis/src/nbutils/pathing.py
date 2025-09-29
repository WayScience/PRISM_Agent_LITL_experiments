# analysis/src/nbutils/pathing.py
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
from typing import Iterable, Optional
import os
import shutil

DEFAULT_MARKERS = (".git", ".env", "LICENSE")

def _default_start() -> Path:
    # Prefer the file's directory when running a .py (including nbconvert output)
    # Fall back to CWD when in a notebook/REPL (no __file__)
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()

@lru_cache(maxsize=1)
def repo_root(
    start: Optional[Path] = None,
    markers: Iterable[str] = DEFAULT_MARKERS,
    env_var: str = "NBUTILS_REPO_ROOT",
) -> Path:
    """
    Locate the repo root by walking upward from `start` (or a sensible default).
    Precedence:
      1) Explicit env var override NBUTILS_REPO_ROOT
      2) Upward search for any of `markers`
      3) Fallback: `git rev-parse --show-toplevel` if Git is available
    """
    # 1) Env override
    if (v := os.getenv(env_var)):
        p = Path(v).expanduser().resolve()
        if p.exists():
            return p

    # 2) Upward search for markers
    here = (start or _default_start()).resolve()
    for p in (here, *here.parents):
        if any((p / m).exists() for m in markers):
            return p

    # 3) Git fallback
    if shutil.which("git"):
        try:
            import subprocess
            out = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=str(here),
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            p = Path(out).resolve()
            if p.exists():
                return p
        except Exception:
            pass

    raise FileNotFoundError(f"Could not locate repo root from {here}")

def project_file(*parts: str, **kwargs) -> Path:
    """Convenience: repo_root() / parts..."""
    return repo_root(**kwargs).joinpath(*parts)
