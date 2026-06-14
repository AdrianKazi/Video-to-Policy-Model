"""Prompt before overwriting non-empty pipeline outputs."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


def _is_effectively_empty(path: Path) -> bool:
    if not path.exists():
        return True
    if path.is_file():
        return path.stat().st_size == 0
    try:
        next(path.iterdir())
    except StopIteration:
        return True
    return False


def confirm_reset(
    path: Path,
    label: str,
    *,
    assume_yes: bool = False,
) -> bool:
    """
    If `path` exists and is non-empty, ask whether to remove it.
    Returns True if the caller may proceed (path absent or cleared).
    """
    if _is_effectively_empty(path):
        return True

    if assume_yes:
        _remove_path(path)
        return True

    msg = (
        f"\n[confirm] {label}\n"
        f"Target exists and is not empty:\n  {path}\n"
        f"Delete this and continue? [y/N]: "
    )
    try:
        ans = input(msg).strip().lower()
    except EOFError:
        print("No TTY; aborting. Use --yes to skip this prompt.", file=sys.stderr)
        return False

    if ans in ("y", "yes"):
        _remove_path(path)
        return True
    print("Aborted.")
    return False


def _remove_path(path: Path) -> None:
    if path.is_file() or path.is_symlink():
        path.unlink()
    else:
        shutil.rmtree(path)
