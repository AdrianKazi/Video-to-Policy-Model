"""Quick qualitative evaluation for the sequential model."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from Autoencoder.device_util import pick_device
from Autoencoder.network import AutoEncoder
from Sequential.config import SequentialConfig
from Sequential.dataset import build_dataloaders
from Sequential.network import LatentLSTM
from Sequential.train import _latest_ae_checkpoint, _latest_seq_run_dir


def run_evaluate(cfg: SequentialConfig, run_dir: Path | None = None, ae_checkpoint: Path | None = None) -> Path:
    if run_dir is None:
        run_dir = _latest_seq_run_dir(cfg)
    if ae_checkpoint is None:
        ae_checkpoint = _latest_ae_checkpoint(cfg)

    device = pick_device(cfg.device)
    ae_model = AutoEncoder(z_dim=cfg.z_dim).to(device)
    ae_model.load_state_dict(torch.load(ae_checkpoint, map_location=device))
    ae_model.eval()

    lstm_model = LatentLSTM(
        z_dim=cfg.z_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    lstm_model.load_state_dict(torch.load(run_dir / "model.pth", map_location=device))
    lstm_model.eval()

    train_data = torch.load(run_dir / "seq_train_dataset.pt", map_location="cpu")
    test_data = torch.load(run_dir / "seq_test_dataset.pt", map_location="cpu")
    _, seq_test_dataloader = build_dataloaders(train_data, test_data, batch_size=cfg.batch_size)

    batch = next(iter(seq_test_dataloader))
    x = batch["x"].to(device)
    lengths = batch["length"].to(device)
    i = 0
    seq_len = int(lengths[i].item())

    with torch.no_grad():
        x_valid = x[i : i + 1, :seq_len]
        x_input = x_valid[:, :-1]
        x_true = x_valid[:, -1]

        z_list = []
        for t in range(x_input.shape[1]):
            _, z_t = ae_model(x_input[:, t])
            z_list.append(z_t)

        z_input = torch.stack(z_list, dim=1)
        _, z_true = ae_model(x_true)
        z_pred = lstm_model(z_input)
        x_pred = ae_model.decode(z_pred).cpu()

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(x_true[0, 0].detach().cpu(), cmap="gray")
    plt.title("true")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(x_pred[0, 0], cmap="gray")
    plt.title("pred")
    plt.axis("off")
    plt.tight_layout()
    frame_out = run_dir / "next_frame_prediction.png"
    plt.savefig(frame_out, dpi=150)
    plt.close()

    z_true_np = z_true.squeeze(0).detach().cpu().numpy()
    z_pred_np = z_pred.squeeze(0).detach().cpu().numpy()
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(z_true_np.reshape(8, 8), cmap="viridis")
    plt.title("true")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(z_pred_np.reshape(8, 8), cmap="viridis")
    plt.title("pred")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(run_dir / "next_latent_prediction.png", dpi=150)
    plt.close()

    print(f"[eval][seq] saved → {frame_out}")
    return frame_out
