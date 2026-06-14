from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..shared.loaders import load_ae_model, load_lstm_model
from ..shared.experiment import finish_experiment, start_experiment
from .models import FDMModel, IDMModel, overlay_frames
from ..shared.paths import (
    FDM_MODEL_PT,
    IDM_MODEL_PT,
    LOSSES_PT,
    METRICS_PT,
    SEQ_TEST_PT,
    VIDEO_DEBUG_MP4,
    VIDEO_LEARNER_RUN_DIR,
)


def _device():
    return torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )


def _lstm_predict_and_hidden(lstm_model, z_input):
    h_seq, _ = lstm_model.lstm(z_input)
    h_t = h_seq[:, -1, :]

    z_lstm_next = lstm_model(z_input)
    if isinstance(z_lstm_next, tuple):
        z_lstm_next = z_lstm_next[-1]

    return h_t, z_lstm_next


def _to_img(x):
    return x.detach().cpu().squeeze().clamp(0, 1).numpy()


def _fig_to_rgb(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[:, :, :3].copy()


def save_video_learner_loss_plot(losses_path=LOSSES_PT, out_path=None):
    losses = torch.load(losses_path, map_location="cpu")

    if out_path is None:
        out_path = Path(losses_path).parent / "video_learner_losses.png"

    xs = losses["epoch"]

    plt.figure(figsize=(10, 5))
    if "train_total" in losses:
        plt.plot(xs, losses["train_total"], lw=2, label="train")
    if "test_total" in losses:
        plt.plot(xs, losses["test_total"], lw=2, label="test")
    if "total" in losses:
        plt.plot(xs, losses["total"], lw=2, label="total")
    if "lstm_baseline" in losses:
        plt.plot(xs, losses["lstm_baseline"], lw=1.8, label="lstm baseline")

    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("Video Learner Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    print(f"[video-learner-eval] saved loss plot -> {out_path}")
    return out_path


def _load_video_learner_models(device, latent_a_dim=8):
    ae_model = load_ae_model(device)
    lstm_model = load_lstm_model(device)

    seq_test_data = torch.load(SEQ_TEST_PT, map_location="cpu")
    x0 = seq_test_data["x"][:1].to(device)

    with torch.no_grad():
        x_input = x0[:, :-1]
        x_next = x0[:, -1]

        x_overlay = overlay_frames(x_input, decay=0.9)
        _, m_t = ae_model(x_overlay)

        z_hist = torch.stack(
            [ae_model(x_input[:, t])[1] for t in range(x_input.shape[1])],
            dim=1,
        )

        z_t = z_hist[:, -1, :]
        _, z_true_next = ae_model(x_next)
        h_t, _ = _lstm_predict_and_hidden(lstm_model, z_hist)
        c_t = torch.cat([z_t, h_t, m_t], dim=-1)

    idm_model = IDMModel(c_dim=c_t.shape[-1], latent_a_dim=latent_a_dim).to(device)
    fdm_model = FDMModel(
        c_dim=c_t.shape[-1],
        z_dim=z_true_next.shape[-1],
        latent_a_dim=latent_a_dim,
    ).to(device)

    idm_model.load_state_dict(torch.load(IDM_MODEL_PT, map_location=device))
    fdm_model.load_state_dict(torch.load(FDM_MODEL_PT, map_location=device))

    ae_model.eval()
    lstm_model.eval()
    idm_model.eval()
    fdm_model.eval()

    return ae_model, lstm_model, idm_model, fdm_model


def _predict_next(ae_model, lstm_model, idm_model, fdm_model, x_input, overlay_decay):
    x_overlay = overlay_frames(x_input, decay=overlay_decay)
    _, m_t = ae_model(x_overlay)

    z_hist = torch.stack(
        [ae_model(x_input[:, t])[1] for t in range(x_input.shape[1])],
        dim=1,
    )

    z_t = z_hist[:, -1, :]
    h_t, z_lstm_next = _lstm_predict_and_hidden(lstm_model, z_hist)
    c_t = torch.cat([z_t, h_t, m_t], dim=-1)

    latent_a = idm_model(c_t)
    delta_z_pred = fdm_model(c_t, latent_a)
    z_video_next = z_t + delta_z_pred

    x_video_next = ae_model.decode(z_video_next).clamp(0, 1)
    x_lstm_next = ae_model.decode(z_lstm_next).clamp(0, 1)

    return {
        "x_overlay": x_overlay,
        "z_t": z_t,
        "z_lstm_next": z_lstm_next,
        "latent_a": latent_a,
        "delta_z_pred": delta_z_pred,
        "z_video_next": z_video_next,
        "x_video_next": x_video_next,
        "x_lstm_next": x_lstm_next,
    }


def evaluate_video_learner(
    run_dir=VIDEO_LEARNER_RUN_DIR,
    batch_size=32,
    overlay_decay=0.9,
    latent_a_dim=8,
):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = _device()
    ae_model, lstm_model, idm_model, fdm_model = _load_video_learner_models(
        device=device,
        latent_a_dim=latent_a_dim,
    )

    seq_test_data = torch.load(SEQ_TEST_PT, map_location="cpu")
    test_loader = DataLoader(
        TensorDataset(seq_test_data["x"]),
        batch_size=batch_size,
        shuffle=False,
    )

    sums = {
        "video_mse": 0.0,
        "lstm_mse": 0.0,
        "delta_mse": 0.0,
        "delta_cos": 0.0,
    }
    n = 0
    latent_chunks = []

    with torch.no_grad():
        for (x,) in test_loader:
            x = x.to(device)
            x_input = x[:, :-1]
            x_true_next = x[:, -1]

            pred = _predict_next(
                ae_model=ae_model,
                lstm_model=lstm_model,
                idm_model=idm_model,
                fdm_model=fdm_model,
                x_input=x_input,
                overlay_decay=overlay_decay,
            )
            _, z_true_next = ae_model(x_true_next)
            delta_z_true = z_true_next - pred["z_t"]

            video_mse = F.mse_loss(pred["z_video_next"], z_true_next)
            lstm_mse = F.mse_loss(pred["z_lstm_next"], z_true_next)
            delta_mse = F.mse_loss(pred["delta_z_pred"], delta_z_true)
            delta_cos = F.cosine_similarity(
                pred["delta_z_pred"],
                delta_z_true,
                dim=-1,
            ).mean()

            bs = x.shape[0]
            sums["video_mse"] += float(video_mse.item()) * bs
            sums["lstm_mse"] += float(lstm_mse.item()) * bs
            sums["delta_mse"] += float(delta_mse.item()) * bs
            sums["delta_cos"] += float(delta_cos.item()) * bs
            n += bs
            latent_chunks.append(pred["latent_a"].detach().cpu())

    latent_a = torch.cat(latent_chunks, dim=0)
    metrics = {k: v / max(n, 1) for k, v in sums.items()}
    metrics.update(
        {
            "n_test": n,
            "video_vs_lstm_mse_ratio": metrics["video_mse"] / max(metrics["lstm_mse"], 1e-12),
            "latent_a_mean": latent_a.mean(dim=0),
            "latent_a_std": latent_a.std(dim=0, unbiased=False),
            "latent_a_min": latent_a.min(dim=0).values,
            "latent_a_max": latent_a.max(dim=0).values,
        }
    )
    torch.save(metrics, METRICS_PT)

    print(f"[video-learner-eval] video_mse      -> {metrics['video_mse']:.6f}")
    print(f"[video-learner-eval] lstm_mse       -> {metrics['lstm_mse']:.6f}")
    print(f"[video-learner-eval] delta_mse      -> {metrics['delta_mse']:.6f}")
    print(f"[video-learner-eval] delta_cos      -> {metrics['delta_cos']:.4f}")
    print(f"[video-learner-eval] video/lstm mse -> {metrics['video_vs_lstm_mse_ratio']:.4f}")
    print(f"[video-learner-eval] saved metrics  -> {METRICS_PT}")

    return metrics


def save_latent_action_plot(metrics_path=METRICS_PT, out_path=None):
    metrics = torch.load(metrics_path, map_location="cpu")

    if out_path is None:
        out_path = Path(metrics_path).parent / "latent_action_stats.png"

    mean = metrics["latent_a_mean"].numpy()
    std = metrics["latent_a_std"].numpy()
    xs = np.arange(len(mean))

    plt.figure(figsize=(8, 4))
    plt.bar(xs, mean, yerr=std, capsize=4)
    plt.axhline(0, color="black", lw=1)
    plt.xlabel("latent action dim")
    plt.ylabel("mean +/- std")
    plt.title("Latent Action Stats")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    print(f"[video-learner-eval] saved latent plot -> {out_path}")
    return out_path


def save_video_learner_visual_check(
    run_dir=VIDEO_LEARNER_RUN_DIR,
    out_path=None,
    overlay_decay=0.9,
    latent_a_dim=8,
):
    run_dir = Path(run_dir)
    if out_path is None:
        out_path = run_dir / "video_learner_visual_check.png"

    device = _device()
    ae_model, lstm_model, idm_model, fdm_model = _load_video_learner_models(
        device=device,
        latent_a_dim=latent_a_dim,
    )

    seq_test_data = torch.load(SEQ_TEST_PT, map_location="cpu")
    x = seq_test_data["x"][:1].to(device)
    x_input = x[:, :-1]
    x_true_next = x[:, -1]

    with torch.no_grad():
        pred = _predict_next(
            ae_model=ae_model,
            lstm_model=lstm_model,
            idm_model=idm_model,
            fdm_model=fdm_model,
            x_input=x_input,
            overlay_decay=overlay_decay,
        )
        _, z_true_next = ae_model(x_true_next)
        delta_z_true = z_true_next - pred["z_t"]
        loss_video = F.mse_loss(pred["z_video_next"], z_true_next).item()
        loss_lstm = F.mse_loss(pred["z_lstm_next"], z_true_next).item()
        cos_delta = F.cosine_similarity(pred["delta_z_pred"], delta_z_true, dim=-1).item()

    seq_len = x_input.shape[1]
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle("Video Learner: True vs Predicted Next Frame", fontsize=16)
    gs = fig.add_gridspec(2, seq_len, height_ratios=[1.0, 1.25], hspace=0.35, wspace=0.08)
    top_axes = [fig.add_subplot(gs[0, t]) for t in range(seq_len)]
    bottom_gs = gs[1, :].subgridspec(1, 5, wspace=0.25)

    for t in range(seq_len):
        top_axes[t].imshow(_to_img(x_input[0, t]), cmap="gray", vmin=0, vmax=1)
        top_axes[t].set_title(str(t))
        top_axes[t].axis("off")

    panels = [
        ("overlay", pred["x_overlay"][0]),
        ("true next", x_true_next[0]),
        (f"video pred\nloss={loss_video:.4f}", pred["x_video_next"][0]),
        (f"lstm baseline\nloss={loss_lstm:.4f}", pred["x_lstm_next"][0]),
    ]
    for i, (title, img) in enumerate(panels):
        ax = fig.add_subplot(bottom_gs[0, i])
        ax.imshow(_to_img(img), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")

    ax_latent = fig.add_subplot(bottom_gs[0, 4])
    latent_a_np = pred["latent_a"][0].detach().cpu().numpy()
    ax_latent.bar([f"l{i}" for i in range(len(latent_a_np))], latent_a_np)
    ax_latent.set_title(f"latent action\ncos={cos_delta:.3f}")
    ax_latent.axhline(0, color="black", lw=1)
    ax_latent.set_ylim(-1.05, 1.05)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    print(f"[video-learner-eval] saved visual check -> {out_path}")
    return out_path


def save_video_debug_mp4(
    run_dir=VIDEO_LEARNER_RUN_DIR,
    out_path=VIDEO_DEBUG_MP4,
    overlay_decay=0.9,
    latent_a_dim=8,
    fps=4,
):
    device = _device()
    ae_model, lstm_model, idm_model, fdm_model = _load_video_learner_models(
        device=device,
        latent_a_dim=latent_a_dim,
    )

    seq_test_data = torch.load(SEQ_TEST_PT, map_location="cpu")
    x = seq_test_data["x"][:1].to(device)

    frames = []
    with torch.no_grad():
        for t in range(1, x.shape[1]):
            x_input = x[:, :t]
            x_true_next = x[:, t]
            pred = _predict_next(
                ae_model=ae_model,
                lstm_model=lstm_model,
                idm_model=idm_model,
                fdm_model=fdm_model,
                x_input=x_input,
                overlay_decay=overlay_decay,
            )

            fig, axes = plt.subplots(1, 4, figsize=(8, 2.4))
            panels = [
                ("last input", x_input[0, -1]),
                ("true next", x_true_next[0]),
                ("video pred", pred["x_video_next"][0]),
                ("lstm pred", pred["x_lstm_next"][0]),
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

    print(f"[video-learner-eval] saved debug mp4 -> {out_path}")
    return out_path


def run_latest_video_learner_eval():
    exp = start_experiment("video_learner_eval")
    run_dir = VIDEO_LEARNER_RUN_DIR

    save_video_learner_loss_plot(
        losses_path=run_dir / "losses.pt",
        out_path=run_dir / "video_learner_losses.png",
    )
    metrics = evaluate_video_learner(run_dir=run_dir)
    save_latent_action_plot(
        metrics_path=run_dir / "metrics.pt",
        out_path=run_dir / "latent_action_stats.png",
    )
    save_video_learner_visual_check(
        run_dir=run_dir,
        out_path=run_dir / "video_learner_visual_check.png",
    )
    save_video_debug_mp4(
        run_dir=run_dir,
        out_path=run_dir / "video_debug.mp4",
    )
    finish_experiment(exp, metrics)

    print(f"[video-learner-eval] completed eval -> {run_dir}")


if __name__ == "__main__":
    run_latest_video_learner_eval()
