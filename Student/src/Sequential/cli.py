"""CLI for the minimal sequential pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from Sequential.build_dataset import run_build_dataset
from Sequential.config import SequentialConfig
from Sequential.evaluate import run_evaluate
from Sequential.train import run_train


def _apply_overrides(cfg: SequentialConfig, args: argparse.Namespace) -> None:
    if getattr(args, "z_dim", None) is not None:
        cfg.z_dim = args.z_dim
    if getattr(args, "seq_len", None) is not None:
        cfg.seq_len = args.seq_len
    if getattr(args, "epochs", None) is not None:
        cfg.epochs = args.epochs
    if getattr(args, "batch_size", None) is not None:
        cfg.batch_size = args.batch_size
    if getattr(args, "lr", None) is not None:
        cfg.lr = args.lr
    if getattr(args, "device", None) is not None:
        cfg.device = args.device


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sequential latent pipeline")
    parser.add_argument("--z-dim", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("dataset", help="data/frames → runs/sequential_*/predataset.pt + seq datasets")

    p_train = sub.add_parser("train", help="Train sequential model in latest dataset run")
    p_train.add_argument("--ae-checkpoint", type=Path, default=None)

    p_eval = sub.add_parser("eval", help="Quick sequential sanity-check plot")
    p_eval.add_argument("--ae-checkpoint", type=Path, default=None)

    p_all = sub.add_parser("all", help="dataset → train → eval")
    p_all.add_argument("--ae-checkpoint", type=Path, default=None)

    args = parser.parse_args(argv)
    cfg = SequentialConfig()
    _apply_overrides(cfg, args)

    if args.cmd == "dataset":
        run_build_dataset(cfg)
    elif args.cmd == "train":
        run_train(cfg, ae_checkpoint=args.ae_checkpoint)
    elif args.cmd == "eval":
        run_evaluate(cfg, ae_checkpoint=args.ae_checkpoint)
    elif args.cmd == "all":
        run_dir = run_build_dataset(cfg)
        run_train(cfg, run_dir=run_dir, ae_checkpoint=args.ae_checkpoint)
        run_evaluate(cfg, run_dir=run_dir, ae_checkpoint=args.ae_checkpoint)
    else:
        raise AssertionError
