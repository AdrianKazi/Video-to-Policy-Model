"""Train the sequential latent predictor inside a self-contained run folder."""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from Autoencoder.device_util import pick_device
from Autoencoder.network import AutoEncoder
from Sequential.config import SequentialConfig
from Sequential.dataset import build_dataloaders
from Sequential.network import LatentLSTM


# def _latest_seq_run_dir(cfg: SequentialConfig) -> Path:
#     runs = sorted(cfg.paths.runs_dir.glob("sequential_*"))
#     for run_dir in reversed(runs):
#         if (run_dir / "seq_train_dataset.pt").is_file() and (run_dir / "seq_test_dataset.pt").is_file():
#             return run_dir
#     raise FileNotFoundError(f"No sequential_* run with datasets under {cfg.paths.runs_dir}. Run `dataset` first.")

def _latest_seq_run_dir(cfg: SequentialConfig) -> Path:
    run_dir = cfg.paths.runs_dir
    if (run_dir / "seq_train_dataset.pt").is_file() and (run_dir / "seq_test_dataset.pt").is_file():
        return run_dir
    raise FileNotFoundError(f"No sequential datasets under {cfg.paths.runs_dir}. Run `dataset` first.")

# def _latest_ae_checkpoint(cfg: SequentialConfig) -> Path:
#     runs = sorted(cfg.paths.ae_runs_dir.glob("autoencoder_*"))
#     for run_dir in reversed(runs):
#         ckpt = run_dir / "model.pth"
#         if ckpt.is_file():
#             return ckpt
#     raise FileNotFoundError(f"No autoencoder checkpoint found under {cfg.paths.ae_runs_dir}.")

def _latest_ae_checkpoint(cfg: SequentialConfig) -> Path:
    ckpt = cfg.paths.ae_runs_dir / "model.pth"
    if ckpt.is_file():
        return ckpt
    raise FileNotFoundError(f"No autoencoder checkpoint found under {cfg.paths.ae_runs_dir}.")

def run_train(cfg: SequentialConfig, run_dir: Path | None = None, ae_checkpoint: Path | None = None) -> tuple[Path, list[float]]:
    if run_dir is None:
        run_dir = _latest_seq_run_dir(cfg)
    if ae_checkpoint is None:
        ae_checkpoint = _latest_ae_checkpoint(cfg)

    train_data = torch.load(run_dir / "seq_train_dataset.pt", map_location="cpu")
    test_data = torch.load(run_dir / "seq_test_dataset.pt", map_location="cpu")
    seq_train_dataloader, seq_test_dataloader = build_dataloaders(train_data, test_data, batch_size=cfg.batch_size)
    device = pick_device(cfg.device)

    ae_model = AutoEncoder(z_dim=cfg.z_dim).to(device)
    ae_model.load_state_dict(torch.load(ae_checkpoint, map_location=device))
    ae_model.eval()
    for p in ae_model.parameters():
        p.requires_grad = False

    lstm_model = LatentLSTM(
        z_dim=cfg.z_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    opt = torch.optim.Adam(lstm_model.parameters(), lr=cfg.lr)

    train_losses: list[float] = []
    test_losses: list[float] = []
    t0 = time.time()

    best_test_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        lstm_model.train()
        ep_loss = 0.0
        n_b = 0

        for batch in seq_train_dataloader:
            x = batch["x"].to(device)
            lengths = batch["length"].to(device)

            z_inputs: list[torch.Tensor] = []
            z_targets: list[torch.Tensor] = []

            with torch.no_grad():
                for i in range(x.shape[0]):
                    seq_len = int(lengths[i].item())
                    if seq_len < 2:
                        continue

                    x_valid = x[i : i + 1, :seq_len]
                    x_input = x_valid[:, :-1]
                    x_target = x_valid[:, -1]

                    z_list = []
                    for t in range(x_input.shape[1]):
                        _, z_t = ae_model(x_input[:, t])
                        z_list.append(z_t)

                    z_input = torch.stack(z_list, dim=1)
                    _, z_target = ae_model(x_target)
                    z_inputs.append(z_input.squeeze(0))
                    z_targets.append(z_target.squeeze(0))

            if not z_inputs:
                continue

            input_lengths = torch.tensor([z.shape[0] for z in z_inputs], dtype=torch.int64, device=device)
            max_len = int(input_lengths.max().item())

            padded_inputs = []
            for z in z_inputs:
                if z.shape[0] < max_len:
                    pad = torch.zeros(max_len - z.shape[0], z.shape[1], device=device, dtype=z.dtype)
                    z = torch.cat([z, pad], dim=0)
                padded_inputs.append(z)

            z_inputs_t = torch.stack(padded_inputs, dim=0)
            z_targets_t = torch.stack(z_targets, dim=0)

            z_pred = lstm_model(z_inputs_t, lengths=input_lengths)
            loss = F.mse_loss(z_pred, z_targets_t)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), cfg.grad_clip)
            opt.step()

            ep_loss += float(loss.item())
            n_b += 1

            train_mean_loss = ep_loss / max(n_b, 1)

        lstm_model.eval()
        test_ep_loss = 0.0
        test_n_b = 0

        with torch.no_grad():
            for batch in seq_test_dataloader:
                x = batch["x"].to(device)
                lengths = batch["length"].to(device)

                z_inputs: list[torch.Tensor] = []
                z_targets: list[torch.Tensor] = []

                for i in range(x.shape[0]):
                    seq_len = int(lengths[i].item())
                    if seq_len < 2:
                        continue

                    x_valid = x[i : i + 1, :seq_len]
                    x_input = x_valid[:, :-1]
                    x_target = x_valid[:, -1]

                    z_list = []
                    for t in range(x_input.shape[1]):
                        _, z_t = ae_model(x_input[:, t])
                        z_list.append(z_t)

                    z_input = torch.stack(z_list, dim=1)
                    _, z_target = ae_model(x_target)
                    z_inputs.append(z_input.squeeze(0))
                    z_targets.append(z_target.squeeze(0))

                if not z_inputs:
                    continue

                input_lengths = torch.tensor([z.shape[0] for z in z_inputs], dtype=torch.int64, device=device)
                max_len = int(input_lengths.max().item())

                padded_inputs = []
                for z in z_inputs:
                    if z.shape[0] < max_len:
                        pad = torch.zeros(max_len - z.shape[0], z.shape[1], device=device, dtype=z.dtype)
                        z = torch.cat([z, pad], dim=0)
                    padded_inputs.append(z)

                z_inputs_t = torch.stack(padded_inputs, dim=0)
                z_targets_t = torch.stack(z_targets, dim=0)

                z_pred = lstm_model(z_inputs_t, lengths=input_lengths)
                loss = F.mse_loss(z_pred, z_targets_t)

                test_ep_loss += float(loss.item())
                test_n_b += 1

        test_mean_loss = test_ep_loss / max(test_n_b, 1)
        train_losses.append(train_mean_loss)
        test_losses.append(test_mean_loss)

        improved = test_mean_loss < (best_test_loss - cfg.early_stop_min_delta)
        if improved:
            best_test_loss = test_mean_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in lstm_model.state_dict().items()
            }
        else:
            stale_epochs += 1

        print(
            f"[train][seq] epoch {epoch:03d}/{cfg.epochs} "
            f"train_loss {train_mean_loss:.6f} | test_loss {test_mean_loss:.6f} | "
            f"best_test {best_test_loss:.6f} @ {best_epoch:03d} | {time.time() - t0:.1f}s"
        )

        if stale_epochs >= cfg.early_stop_patience:
            print(
                f"[train][seq] early stop at epoch {epoch:03d}; "
                f"best_test {best_test_loss:.6f} @ {best_epoch:03d}"
            )
            break

    if best_state is not None:
        lstm_model.load_state_dict(best_state)

    torch.save(lstm_model.state_dict(), run_dir / "model.pth")
    torch.save(
        {
            "seq_train_losses": train_losses,
            "seq_test_losses": test_losses,
        },
        run_dir / "losses.pt",
    )
    meta = {
        "z_dim": cfg.z_dim,
        "seq_len": cfg.seq_len,
        "predataset_path": str(run_dir / "predataset.pt"),
        "train_dataset_path": str(run_dir / "seq_train_dataset.pt"),
        "test_dataset_path": str(run_dir / "seq_test_dataset.pt"),
        "ae_checkpoint": str(ae_checkpoint),
        "num_epochs": cfg.epochs,
        "learning_rate": cfg.lr,
        "grad_clip": cfg.grad_clip,
    }
    torch.save(meta, run_dir / "meta.pt")

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, lw=2, label="train")
    plt.plot(test_losses, lw=2, label="test")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("Sequential LSTM Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "loss.png", dpi=140)
    plt.close()

    print(f"[train][seq] saved artifacts → {run_dir}")
    return run_dir, train_losses
