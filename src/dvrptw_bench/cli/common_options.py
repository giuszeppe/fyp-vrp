"""Shared CLI utilities."""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path


def now_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def git_hash() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        return out
    except Exception:
        return None


def ensure_run_dir(output_root: Path, run_id: str) -> Path:
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
