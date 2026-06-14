"""Re-run AE reconstruction for an existing self-contained run."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from Autoencoder.config import AutoencoderConfig, AutoencoderPaths
from Autoencoder.dataset import EpisodeTensorDataset
from Autoencoder.device_util import pick_device
from Autoencoder.losses import weighted_mse
from Autoencoder.network import AutoEncoder
from Autoencoder.recon_plot import save_reconstruction_png


# def _latest_run_dir(paths: AutoencoderPaths) -> Path:
#     runs = sorted(paths.runs_dir.glob("autoencoder_*"))
#     for run_dir in reversed(runs):
#         if (run_dir / "model.pth").is_file() and (run_dir / "ae_test_dataset.pt").is_file():
#             return run_dir
#     raise FileNotFoundError(f"No complete autoencoder_* folders under {paths.runs_dir}. Train first.")

def _latest_run_dir(paths: AutoencoderPaths) -> Path:
    run_dir = paths.runs_dir
    if (run_dir / "model.pth").is_file() and (run_dir / "ae_test_dataset.pt").is_file():
        return run_dir
    raise FileNotFoundError(f"No complete autoencoder artifacts under {paths.runs_dir}. Train first.")

def run_evaluate(
    cfg: AutoencoderConfig,
    checkpoint: Path | None = None,
) -> Path:
    paths = cfg.paths

    if checkpoint is None:
        run_dir = _latest_run_dir(paths)
        checkpoint = run_dir / "model.pth"
    else:
        run_dir = checkpoint.parent

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    device = pick_device(cfg.device)
    ae_model = AutoEncoder(z_dim=cfg.z_dim).to(device)
    ae_model.load_state_dict(torch.load(checkpoint, map_location=device))
    ae_model.eval()

    test_pt = run_dir / "ae_test_dataset.pt"
    ds = EpisodeTensorDataset(test_pt)
    x = ds[0].to(device)

    with torch.no_grad():
        x_hat, _ = ae_model(x)

    loss_val = float(
        weighted_mse(
            x_hat,
            x,
            threshold=cfg.wmse_threshold,
            high_weight=cfg.wmse_high_weight,
        ).item()
    )
    print(f"[eval] weighted_mse on first episode: {loss_val:.6f}")

    out_path = run_dir / "reconstruction.png"
    save_reconstruction_png(
        out_path,
        x,
        x_hat,
        num_rows=min(cfg.recon_rows, x.shape[0]),
    )
    i = torch.randint(0, x.shape[0], (1,)).item()
    z = ae_model.encoder(x[i : i + 1])
    x_sample = ae_model.decode(z)
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(x[i, 0].detach().cpu(), cmap="gray")
    plt.axis("off")
    plt.title(f"orig (idx={i})")
    plt.subplot(1, 3, 2)
    plt.imshow(z.view(8, 8).detach().cpu(), cmap="gray")
    plt.axis("off")
    plt.title(f"latent (idx={i})")
    plt.subplot(1, 3, 3)
    plt.imshow(x_sample[0, 0].detach().cpu(), cmap="gray")
    plt.axis("off")
    plt.title("recon")
    plt.tight_layout()
    plt.savefig(run_dir / "sample_reconstruction.png", dpi=150)
    plt.close()
    print(f"[eval] saved → {out_path}")
    return out_path
