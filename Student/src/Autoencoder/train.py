"""Train autoencoder inside an existing self-contained run folder."""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from Autoencoder.config import AutoencoderConfig
from Autoencoder.dataset import EpisodeTensorDataset, FrameDataset
from Autoencoder.device_util import pick_device
from Autoencoder.losses import weighted_mse
from Autoencoder.network import AutoEncoder
from Autoencoder.recon_plot import save_reconstruction_png


# def _latest_dataset_run_dir(cfg: AutoencoderConfig) -> Path:
#     runs = sorted(cfg.paths.runs_dir.glob("autoencoder_*"))
#     for run_dir in reversed(runs):
#         if (run_dir / "ae_train_dataset.pt").is_file() and (run_dir / "ae_test_dataset.pt").is_file():
#             return run_dir
#     raise FileNotFoundError(
#         f"No autoencoder_* run with ae_train_dataset.pt / ae_test_dataset.pt under {cfg.paths.runs_dir}. "
#         "Run `dataset` first."
#     )

def _latest_dataset_run_dir(cfg: AutoencoderConfig) -> Path:
    run_dir = cfg.paths.runs_dir
    if (run_dir / "ae_train_dataset.pt").is_file() and (run_dir / "ae_test_dataset.pt").is_file():
        return run_dir
    raise FileNotFoundError(
        f"No ae_train_dataset.pt / ae_test_dataset.pt under {cfg.paths.runs_dir}. "
        "Run `dataset` first."
    )


def run_train(cfg: AutoencoderConfig, run_dir: Path | None = None) -> tuple[Path, dict[str, list[float]]]:
    if run_dir is None:
        run_dir = _latest_dataset_run_dir(cfg)

    train_pt = run_dir / "ae_train_dataset.pt"
    test_pt = run_dir / "ae_test_dataset.pt"
    if not train_pt.is_file() or not test_pt.is_file():
        raise FileNotFoundError(f"Missing AE datasets under {run_dir}. Run `dataset` first.")

    device = pick_device(cfg.device)
    train_ds: FrameDataset | Subset = FrameDataset(train_pt)
    test_ds = FrameDataset(test_pt)
    n_full = len(train_ds)
    if cfg.limit_train_samples is not None:
        m = min(cfg.limit_train_samples, n_full)
        if m < n_full:
            g = torch.Generator().manual_seed(cfg.seed)
            idx = torch.randperm(n_full, generator=g)[:m].tolist()
            train_ds = Subset(train_ds, idx)
            print(f"[train] subsampled {m}/{n_full} frames (limit_train_samples={cfg.limit_train_samples})")

    dl_kw: dict = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if cfg.num_workers > 0:
        dl_kw["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, shuffle=True, **dl_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **dl_kw)

    ae_model = AutoEncoder(z_dim=cfg.z_dim).to(device)
    opt = torch.optim.Adam(ae_model.parameters(), lr=cfg.lr)

    train_losses: list[float] = []
    test_losses: list[float] = []
    t0 = time.time()

    best_test_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    print(f"[train] device={device} | train_samples={len(train_ds)} | test_samples={len(test_ds)}")
    print(f"[train] train_batches/epoch={len(train_loader)} | test_batches/epoch={len(test_loader)}")
    print(f"[train] run_dir={run_dir}")

    for epoch in range(1, cfg.epochs + 1):
        ae_model.train()
        train_epoch_loss = 0.0
        train_n_batches = 0
        for step, batch in enumerate(train_loader, start=1):
            x = batch.to(device, non_blocking=True)
            x_hat, _ = ae_model(x)
            loss = weighted_mse(
                x_hat,
                x,
                threshold=cfg.wmse_threshold,
                high_weight=cfg.wmse_high_weight,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae_model.parameters(), cfg.grad_clip)
            opt.step()
            train_epoch_loss += float(loss.item())
            train_n_batches += 1
            if cfg.log_every and step % cfg.log_every == 0:
                dt = time.time() - t0
                print(
                    f"[train][ae] epoch {epoch:03d}/{cfg.epochs} "
                    f"step {step:05d}/{len(train_loader)} "
                    f"loss {loss.item():.6f} | {dt:.1f}s"
                )

        ae_model.eval()
        test_epoch_loss = 0.0
        test_n_batches = 0
        with torch.no_grad():
            for batch in test_loader:
                x = batch.to(device, non_blocking=True)
                x_hat, _ = ae_model(x)
                loss = weighted_mse(
                    x_hat,
                    x,
                    threshold=cfg.wmse_threshold,
                    high_weight=cfg.wmse_high_weight,
                )
                test_epoch_loss += float(loss.item())
                test_n_batches += 1

        train_mean_loss = train_epoch_loss / max(train_n_batches, 1)
        test_mean_loss = test_epoch_loss / max(test_n_batches, 1)
        train_losses.append(train_mean_loss)
        test_losses.append(test_mean_loss)

        improved = test_mean_loss < (best_test_loss - cfg.early_stop_min_delta)
        if improved:
            best_test_loss = test_mean_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in ae_model.state_dict().items()
            }
        else:
            stale_epochs += 1

        dt = time.time() - t0
        print(
            f"[train][ae] epoch {epoch:03d}/{cfg.epochs} "
            f"train_loss {train_mean_loss:.6f} | test_loss {test_mean_loss:.6f} | "
            f"best_test {best_test_loss:.6f} @ {best_epoch:03d} | elapsed {dt:.1f}s"
        )

        if stale_epochs >= cfg.early_stop_patience:
            print(
                f"[train][ae] early stop at epoch {epoch:03d}; "
                f"best_test {best_test_loss:.6f} @ {best_epoch:03d}"
            )
            break

    if best_state is not None:
        ae_model.load_state_dict(best_state)

    ckpt_path = run_dir / "model.pth"
    torch.save(ae_model.state_dict(), ckpt_path)
    torch.save(
        {
            "ae_train_losses": train_losses,
            "ae_test_losses": test_losses,
        },
        run_dir / "losses.pt",
    )

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, color="#2d6cdf", linewidth=2, label="train")
    plt.plot(test_losses, color="#e67e22", linewidth=2, label="test")
    plt.xlabel("epoch")
    plt.ylabel("weighted MSE")
    plt.title("Autoencoder training loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    loss_png = run_dir / "loss.png"
    plt.tight_layout()
    plt.savefig(loss_png, dpi=150)
    plt.close()

    ae_model.eval()
    ep_ds = EpisodeTensorDataset(test_pt)
    x0 = ep_ds[0].to(device)
    with torch.no_grad():
        x_hat0, _ = ae_model(x0)
    save_reconstruction_png(
        run_dir / "reconstruction.png",
        x0,
        x_hat0,
        num_rows=min(cfg.recon_rows, x0.shape[0]),
    )

    i = torch.randint(0, len(test_ds), (1,)).item()
    x = test_ds[i][None].to(device)
    with torch.no_grad():
        x_hat, z = ae_model(x)

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(x[0, 0].detach().cpu(), cmap="gray")
    plt.axis("off")
    plt.title(f"orig (idx={i})")
    plt.subplot(1, 3, 2)
    plt.imshow(z.view(8, 8).detach().cpu(), cmap="gray")
    plt.axis("off")
    plt.title(f"latent (idx={i})")
    plt.subplot(1, 3, 3)
    plt.imshow(x_hat[0, 0].detach().cpu(), cmap="gray")
    plt.axis("off")
    plt.title("recon")
    plt.tight_layout()
    plt.savefig(run_dir / "sample_reconstruction.png", dpi=150)
    plt.close()

    print(f"[train] saved AE artifacts → {run_dir}")
    return run_dir, {"ae_train_losses": train_losses, "ae_test_losses": test_losses}
