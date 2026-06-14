"""CLI: run extract / dataset / train / eval / all."""
from __future__ import annotations

import argparse
from pathlib import Path

from Autoencoder.build_dataset import run_build_dataset
from Autoencoder.config import AutoencoderConfig
from Autoencoder.evaluate import run_evaluate
from Autoencoder.extract_frames import run_extract_frames
from Autoencoder.train import run_train


def _add_yes_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Assume 'yes' for overwrite prompts (non-interactive / CI)",
    )


def _base_parser() -> argparse.ArgumentParser:
    # Global training knobs on the root parser; `-y` lives on each subcommand (argparse + subparsers).
    p = argparse.ArgumentParser(description="Autoencoder pipeline (Student/src/Autoencoder)")
    p.add_argument("--z-dim", type=int, default=None, help="Latent size (default: config)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", type=str, default=None, help="cpu | cuda | mps | cuda:0 ...")
    p.add_argument(
        "--limit-train-samples",
        type=int,
        default=None,
        help="Train on a random subset of frames per epoch (faster dev; default: use all)",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (0 = main process only; try 2–4 on Linux/CUDA)",
    )
    return p


def _apply_overrides(cfg: AutoencoderConfig, args: argparse.Namespace) -> None:
    if args.z_dim is not None:
        cfg.z_dim = args.z_dim
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.device is not None:
        cfg.device = args.device
    if getattr(args, "limit_train_samples", None) is not None:
        cfg.limit_train_samples = args.limit_train_samples
    if getattr(args, "num_workers", None) is not None:
        cfg.num_workers = args.num_workers


def main(argv: list[str] | None = None) -> None:
    parser = _base_parser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ext = sub.add_parser("extract", help="MP4 → data/frames/")
    _add_yes_flag(p_ext)
    p_ext.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="If frames already exist, exit without doing anything",
    )

    p_ds = sub.add_parser("dataset", help="data/frames → runs/autoencoder_*/ae_*_dataset.pt")
    _add_yes_flag(p_ds)

    p_tr = sub.add_parser("train", help="Train AE in latest dataset run folder")
    _add_yes_flag(p_tr)

    p_ev = sub.add_parser("eval", help="Regenerate reconstruction.png in latest run or --checkpoint")
    _add_yes_flag(p_ev)
    p_ev.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to model.pth (default: latest runs/autoencoder_*/model.pth)",
    )

    p_all = sub.add_parser("all", help="extract → dataset → train → eval")
    _add_yes_flag(p_all)

    args = parser.parse_args(argv)
    cfg = AutoencoderConfig()
    _apply_overrides(cfg, args)
    yes = args.yes

    if args.cmd == "extract":
        run_extract_frames(
            cfg.paths,
            skip_if_nonempty=getattr(args, "skip_if_exists", False),
            assume_yes=yes,
        )
    elif args.cmd == "dataset":
        run_build_dataset(cfg, assume_yes=yes)
    elif args.cmd == "train":
        run_train(cfg)
    elif args.cmd == "eval":
        run_evaluate(cfg, checkpoint=getattr(args, "checkpoint", None))
    elif args.cmd == "all":
        run_extract_frames(cfg.paths, skip_if_nonempty=False, assume_yes=yes)
        run_dir = run_build_dataset(cfg, assume_yes=yes)
        run_train(cfg, run_dir=run_dir)
        run_evaluate(cfg, checkpoint=run_dir / "model.pth")
    else:
        raise AssertionError


if __name__ == "__main__":
    main()
