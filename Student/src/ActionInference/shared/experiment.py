import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

from ActionInference.shared.paths import (
    AE_TEST_PT,
    AE_TRAIN_PT,
    EXPERIMENTS_DIR,
    MACHINE_PROBE_PT,
    SEQ_TEST_PT,
    SEQ_TRAIN_PT,
    STUDENT_ROOT,
)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return str(value)


def _run(cmd):
    try:
        return subprocess.check_output(
            cmd,
            cwd=STUDENT_ROOT.parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _file_fingerprint(path):
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "sha256": h.hexdigest(),
    }


def get_run_id(default_name):
    explicit = os.environ.get("ACTION_EXPERIMENT_ID")
    if explicit:
        return explicit

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = default_name.lower().replace(" ", "_")
    return f"{stamp}_{safe_name}"


def start_experiment(stage, config=None, hypothesis=None):
    run_id = get_run_id(stage)
    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    git_commit = _run(["git", "rev-parse", "HEAD"])
    git_diff = _run(["git", "diff"])
    git_available = git_commit is not None

    meta = {
        "run_id": run_id,
        "stage": stage,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "hypothesis": hypothesis or os.environ.get("ACTION_EXPERIMENT_HYPOTHESIS", ""),
        "seed": os.environ.get("ACTION_EXPERIMENT_SEED", ""),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "device": (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        ),
        "git_available": git_available,
        "git_commit": git_commit,
        "dataset_version": {
            "ae_train": _file_fingerprint(AE_TRAIN_PT),
            "ae_test": _file_fingerprint(AE_TEST_PT),
            "seq_train": _file_fingerprint(SEQ_TRAIN_PT),
            "seq_test": _file_fingerprint(SEQ_TEST_PT),
            "machine_probe": _file_fingerprint(MACHINE_PROBE_PT),
        },
    }

    _write_json(run_dir / f"{stage}_meta.json", meta)
    if config is not None:
        _write_json(run_dir / f"{stage}_config.json", config)
    if git_diff:
        (run_dir / "git_diff.patch").write_text(git_diff)

    summary = run_dir / "result_summary.md"
    if not summary.exists():
        summary.write_text(
            f"# {run_id}\n\n"
            f"## Hypothesis\n\n{meta['hypothesis'] or 'TODO'}\n\n"
            "## Results\n\nTODO\n\n"
            "## Next Step\n\nTODO\n"
        )

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "stage": stage,
    }


def finish_experiment(exp, metrics=None, status="complete"):
    run_dir = Path(exp["run_dir"])
    stage = exp["stage"]
    metrics = metrics or {}

    _write_json(run_dir / f"{stage}_metrics.json", metrics)
    _update_index(exp["run_id"], stage, status, metrics)


def _write_json(path, data):
    Path(path).write_text(
        json.dumps(data, indent=2, sort_keys=True, default=_json_default)
    )


def _update_index(run_id, stage, status, metrics):
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    index_path = EXPERIMENTS_DIR / "runs_index.csv"

    row = {
        "run_id": run_id,
        "stage": stage,
        "status": status,
        "video_mse": metrics.get("video_mse", ""),
        "delta_cos": metrics.get("delta_cos", ""),
        "machine_mse": metrics.get("machine_mse", ""),
        "zero_over_machine": metrics.get("zero_over_machine", ""),
        "runtime_reward": metrics.get("total_reward", ""),
        "runtime_steps": metrics.get("steps", ""),
    }

    rows = []
    if index_path.exists():
        with index_path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))

    rows = [
        r for r in rows
        if not (r.get("run_id") == run_id and r.get("stage") == stage)
    ]
    rows.append(row)

    with index_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerows(rows)
