"""Dataset file discovery and filtering utilities."""

from __future__ import annotations

from pathlib import Path


def is_rc_instance(path: Path) -> bool:
    stem = path.stem.upper()
    return stem.startswith("RC")
def is_c_instance(path: Path) -> bool:
    stem = path.stem.upper()
    return stem.startswith("C")


def find_c_instances(dataset_root: Path) -> list[Path]:
    files = [p for p in dataset_root.rglob("*.txt") if is_c_instance(p)]
    return sorted(files)

def find_c_solutions(dataset_root: Path) -> list[Path]:
    files = [p for p in dataset_root.rglob("*.sol") if is_c_instance(p)]
    return sorted(files)

def find_rc_instances(dataset_root: Path) -> list[Path]:
    files = [p for p in dataset_root.rglob("*.txt") if is_rc_instance(p)]
    return sorted(files)

def find_rc_solutions(dataset_root: Path) -> list[Path]:
    files = [p for p in dataset_root.rglob("*.sol") if is_rc_instance(p)]
    return sorted(files)