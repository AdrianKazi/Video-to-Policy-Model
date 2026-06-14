from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ActionInference.machine_forward.config import MACHINE_FORWARD_CONFIG
from ActionInference.machine_forward.models import MachineForwardModel
from ActionInference.shared.experiment import finish_experiment, start_experiment
from ActionInference.shared.loaders import load_ae_model
from ActionInference.shared.paths import (
    MACHINE_DEBUG_MP4,
    MACHINE_FORWARD_RUN_DIR,
    MACHINE_METRICS_PT,
    MACHINE_MODEL_PT,
    MACHINE_PROBE_PT,
)


def _device():
    return torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )


def _to_img(x):
    return x.detach().cpu().squeeze().clamp(0, 1).numpy()


def _fig_to_rgb(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[:, :, :3].copy()


def _load_model(device):
    model = MachineForwardModel(
        z_dim=MACHINE_FORWARD_CONFIG["z_dim"],
        real_a_dim=MACHINE_FORWARD_CONFIG["real_a_dim"],
        hidden_dim=MACHINE_FORWARD_CONFIG["hidden_dim"],
        num_layers=MACHINE_FORWARD_CONFIG["num_layers"],
    ).to(device)
    model.load_state_dict(torch.load(MACHINE_MODEL_PT, map_location=device))
    model.eval()
    return model


def evaluate_machine_forward(batch_size=128):
    device = _device()
    model = _load_model(device)
    probe_data = torch.load(MACHINE_PROBE_PT, map_location="cpu")

    loader = DataLoader(
        TensorDataset(
            probe_data["z_hist"].float(),
            probe_data["a_real"].float(),
            probe_data["dz"].float(),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    sum_model = 0.0
    sum_zero = 0.0
    sum_cos = 0.0
    n = 0

    with torch.no_grad():
        for z_hist_b, a_b, dz_b in loader:
            z_hist_b = z_hist_b.to(device)
            a_b = a_b.to(device)
            dz_b = dz_b.to(device)

            dz_pred = model(z_hist_b, a_b)
            loss_model = F.mse_loss(dz_pred, dz_b)
            loss_zero = F.mse_loss(torch.zeros_like(dz_b), dz_b)
            cos = F.cosine_similarity(dz_pred, dz_b, dim=-1).mean()

            bs = z_hist_b.shape[0]
            sum_model += float(loss_model.item()) * bs
            sum_zero += float(loss_zero.item()) * bs
            sum_cos += float(cos.item()) * bs
            n += bs

    metrics = {
        "n": n,
        "machine_mse": sum_model / max(n, 1),
        "zero_mse": sum_zero / max(n, 1),
        "delta_cos": sum_cos / max(n, 1),
    }
    metrics["zero_over_machine"] = metrics["zero_mse"] / max(metrics["machine_mse"], 1e-12)

    MACHINE_FORWARD_RUN_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(metrics, MACHINE_METRICS_PT)

    print(f"[machine-eval] machine_mse       -> {metrics['machine_mse']:.6f}")
    print(f"[machine-eval] zero_mse          -> {metrics['zero_mse']:.6f}")
    print(f"[machine-eval] zero/machine      -> {metrics['zero_over_machine']:.4f}")
    print(f"[machine-eval] delta_cos         -> {metrics['delta_cos']:.4f}")
    print(f"[machine-eval] saved metrics     -> {MACHINE_METRICS_PT}")
    return metrics


def save_machine_debug_mp4(out_path=MACHINE_DEBUG_MP4, fps=4, max_frames=64):
    device = _device()
    ae_model = load_ae_model(device)
    model = _load_model(device)
    probe_data = torch.load(MACHINE_PROBE_PT, map_location="cpu")

    z_hist = probe_data["z_hist"][:max_frames].to(device)
    a_real = probe_data["a_real"][:max_frames].to(device)
    dz_true = probe_data["dz"][:max_frames].to(device)

    frames = []
    with torch.no_grad():
        z_t = z_hist[:, -1]
        dz_pred = model(z_hist, a_real)
        z_true_next = z_t + dz_true
        z_pred_next = z_t + dz_pred

        x_last = ae_model.decode(z_t).clamp(0, 1)
        x_true = ae_model.decode(z_true_next).clamp(0, 1)
        x_pred = ae_model.decode(z_pred_next).clamp(0, 1)

        for i in range(z_hist.shape[0]):
            mse = F.mse_loss(z_pred_next[i], z_true_next[i]).item()
            fig, axes = plt.subplots(1, 3, figsize=(6, 2.4))
            panels = [
                ("last input", x_last[i]),
                ("true next", x_true[i]),
                (f"machine pred\nmse={mse:.4f}", x_pred[i]),
            ]
            for ax, (title, img) in zip(axes, panels):
                ax.imshow(_to_img(img), cmap="gray", vmin=0, vmax=1)
                ax.set_title(title)
                ax.axis("off")
            plt.tight_layout()
            frames.append(_fig_to_rgb(fig))
            plt.close(fig)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"[machine-eval] saved debug mp4 -> {out_path}")
    return out_path


if __name__ == "__main__":
    exp = start_experiment("machine_forward_eval", MACHINE_FORWARD_CONFIG)
    metrics = evaluate_machine_forward(batch_size=MACHINE_FORWARD_CONFIG["batch_size"])
    save_machine_debug_mp4()
    finish_experiment(exp, metrics)
