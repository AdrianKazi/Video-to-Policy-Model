"""Shared filesystem helpers for Student-side pipelines."""
from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def student_dir() -> Path:
    return Path(__file__).resolve().parents[1]
