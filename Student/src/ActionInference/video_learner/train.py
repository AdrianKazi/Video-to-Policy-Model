import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..shared.loaders import load_ae_model, load_lstm_model
from ..shared.experiment import finish_experiment, start_experiment
from ..shared.tensorboard import log_image_batch, log_step_features, log_tensor, make_writer
from .models import overlay_frames, IDMModel, FDMModel
from ..shared.paths import (
    SEQ_TRAIN_PT,
    IDM_MODEL_PT,
    FDM_MODEL_PT,
    LOSSES_PT,
    VIDEO_LEARNER_RUN_DIR,
)
from .config import VIDEO_LEARNER_CONFIG


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


def train_video_learner(
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 3e-4,
    overlay_decay: float = 0.9,
    latent_a_dim: int = 8,
    log_every: int = 1,
    early_stop_patience: int = 5,
    early_stop_min_delta: float = 1e-4,
):
    exp = start_experiment("video_learner_train", VIDEO_LEARNER_CONFIG)
    device = _device()

    ae_model = load_ae_model(device)
    lstm_model = load_lstm_model(device)

    ae_model.eval()
    lstm_model.eval()

    for p in ae_model.parameters():
        p.requires_grad = False

    for p in lstm_model.parameters():
        p.requires_grad = False

    seq_train_data = torch.load(SEQ_TRAIN_PT, map_location="cpu")
    n_total = seq_train_data["x"].shape[0]
    n_test = max(1, int(0.2 * n_total))
    n_train = n_total - n_test

    split_gen = torch.Generator().manual_seed(42)
    perm = torch.randperm(n_total, generator=split_gen)

    train_x = seq_train_data["x"][perm[:n_train]]
    test_x = seq_train_data["x"][perm[n_train:]]

    train_loader = DataLoader(
        TensorDataset(train_x),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        TensorDataset(test_x),
        batch_size=batch_size,
        shuffle=False,
    )

    # infer dimensions from one batch
    x0 = seq_train_data["x"][:batch_size].to(device)

    with torch.no_grad():
        x_hist = x0[:, :-1]
        x_next = x0[:, -1]

        x_overlay = overlay_frames(x_hist, decay=overlay_decay)
        _, m_t = ae_model(x_overlay)

        z_hist = torch.stack(
            [ae_model(x_hist[:, t])[1] for t in range(x_hist.shape[1])],
            dim=1,
        )

        z_t = z_hist[:, -1, :]
        _, z_true_next = ae_model(x_next)

        h_t, _ = _lstm_predict_and_hidden(lstm_model, z_hist)
        c_t = torch.cat([z_t, h_t, m_t], dim=-1)

    z_dim = z_true_next.shape[-1]
    c_dim = c_t.shape[-1]

    idm_model = IDMModel(
        c_dim=c_dim,
        latent_a_dim=latent_a_dim,
    ).to(device)

    fdm_model = FDMModel(
        c_dim=c_dim,
        z_dim=z_dim,
        latent_a_dim=latent_a_dim,
    ).to(device)

    opt = torch.optim.Adam(
        list(idm_model.parameters()) + list(fdm_model.parameters()),
        lr=lr,
    )

    losses = {
        "epoch": [],
        "train_total": [],
        "test_total": [],
        "z": [],
        "delta": [],
        "delta_cos": [],
        "lstm_baseline": [],
        "latent_a_mag": [],
        "latent_a_std": [],
    }

    best_test_loss = float("inf")
    best_idm_state = None
    best_fdm_state = None
    best_epoch = 0
    stale_epochs = 0
    writer = make_writer(VIDEO_LEARNER_RUN_DIR, "train")

    try:
        for epoch in range(1, epochs + 1):
            idm_model.train()
            fdm_model.train()

            sums = {
                "total": 0.0,
                "z": 0.0,
                "delta": 0.0,
                "delta_cos": 0.0,
                "lstm_baseline": 0.0,
                "latent_a_mag": 0.0,
                "latent_a_std": 0.0,
            }
            n = 0
            debug_batch = None

            for (x,) in train_loader:
                x = x.to(device)

                x_input = x[:, :-1]
                x_target = x[:, -1]

                with torch.no_grad():
                    x_overlay = overlay_frames(x_input, decay=overlay_decay)
                    _, m_t = ae_model(x_overlay)

                    z_input = torch.stack(
                        [ae_model(x_input[:, t])[1] for t in range(x_input.shape[1])],
                        dim=1,
                    )

                    z_t = z_input[:, -1, :]
                    _, z_true_next = ae_model(x_target)

                    h_t, z_lstm_next = _lstm_predict_and_hidden(lstm_model, z_input)

                    c_t = torch.cat([z_t, h_t, m_t], dim=-1)
                    loss_lstm = F.mse_loss(z_lstm_next, z_true_next)

                latent_a = idm_model(c_t)
                delta_z_pred = fdm_model(c_t, latent_a)
                delta_z_true = z_true_next - z_t
                z_pred_next = z_t + delta_z_pred

                loss_z = F.mse_loss(z_pred_next, z_true_next)
                loss_delta = F.mse_loss(delta_z_pred, delta_z_true)
                cos_delta = F.cosine_similarity(
                    delta_z_pred,
                    delta_z_true,
                    dim=-1,
                ).mean()

                loss = loss_z

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                bs = x.shape[0]

                sums["total"] += float(loss.item()) * bs
                sums["z"] += float(loss_z.item()) * bs
                sums["delta"] += float(loss_delta.item()) * bs
                sums["delta_cos"] += float(cos_delta.item()) * bs
                sums["lstm_baseline"] += float(loss_lstm.item()) * bs
                sums["latent_a_mag"] += float((latent_a ** 2).mean().item()) * bs
                sums["latent_a_std"] += float(
                    latent_a.std(dim=0, unbiased=False).mean().item()
                ) * bs

                debug_batch = {
                    "x_input": x_input.detach(),
                    "x_overlay": x_overlay.detach(),
                    "z_input": z_input.detach(),
                    "z_t": z_t.detach(),
                    "z_true_next": z_true_next.detach(),
                    "h_t": h_t.detach(),
                    "m_t": m_t.detach(),
                    "c_t": c_t.detach(),
                    "latent_a": latent_a.detach(),
                    "delta_z_pred": delta_z_pred.detach(),
                    "delta_z_true": delta_z_true.detach(),
                    "z_pred_next": z_pred_next.detach(),
                    "loss_z": loss_z.detach(),
                    "loss_delta": loss_delta.detach(),
                    "cos_delta": cos_delta.detach(),
                    "loss_lstm": loss_lstm.detach(),
                }

                n += bs

            means = {k: v / max(n, 1) for k, v in sums.items()}

            idm_model.eval()
            fdm_model.eval()
            test_sum = 0.0
            test_n = 0

            with torch.no_grad():
                for (x,) in test_loader:
                    x = x.to(device)

                    x_input = x[:, :-1]
                    x_target = x[:, -1]

                    x_overlay = overlay_frames(x_input, decay=overlay_decay)
                    _, m_t = ae_model(x_overlay)

                    z_input = torch.stack(
                        [ae_model(x_input[:, t])[1] for t in range(x_input.shape[1])],
                        dim=1,
                    )

                    z_t = z_input[:, -1, :]
                    _, z_true_next = ae_model(x_target)
                    h_t, _ = _lstm_predict_and_hidden(lstm_model, z_input)

                    c_t = torch.cat([z_t, h_t, m_t], dim=-1)

                    latent_a = idm_model(c_t)
                    delta_z_pred = fdm_model(c_t, latent_a)
                    z_pred_next = z_t + delta_z_pred

                    loss = F.mse_loss(z_pred_next, z_true_next)

                    bs = x.shape[0]
                    test_sum += float(loss.item()) * bs
                    test_n += bs

            test_total = test_sum / max(test_n, 1)

            losses["epoch"].append(epoch)
            losses["train_total"].append(means["total"])
            losses["test_total"].append(test_total)
            for k in ["z", "delta", "delta_cos", "lstm_baseline", "latent_a_mag", "latent_a_std"]:
                losses[k].append(means[k])

            for key, value in means.items():
                writer.add_scalar(f"loss/train/{key}", value, epoch)
            writer.add_scalar("loss/test/total", test_total, epoch)
            if debug_batch is not None:
                _log_video_learner_debug(writer, debug_batch, epoch)

            improved = test_total < (best_test_loss - early_stop_min_delta)
            if improved:
                best_test_loss = test_total
                best_epoch = epoch
                stale_epochs = 0
                best_idm_state = {
                    k: v.detach().cpu().clone()
                    for k, v in idm_model.state_dict().items()
                }
                best_fdm_state = {
                    k: v.detach().cpu().clone()
                    for k, v in fdm_model.state_dict().items()
                }
            else:
                stale_epochs += 1

            if epoch % log_every == 0:
                print(
                    f"[video-learner] epoch {epoch:03d}/{epochs} | "
                    f"train {means['total']:.6f} | test {test_total:.6f} | "
                    f"best_test {best_test_loss:.6f} @ {best_epoch:03d} | "
                    f"z {means['z']:.6f} | delta {means['delta']:.6f} | "
                    f"cos {means['delta_cos']:.4f} | lstm {means['lstm_baseline']:.6f} | "
                    f"latent std {means['latent_a_std']:.4f}"
                )

            if stale_epochs >= early_stop_patience:
                print(
                    f"[video-learner] early stop at epoch {epoch:03d}; "
                    f"best_test {best_test_loss:.6f} @ {best_epoch:03d}"
                )
                break
    finally:
        writer.flush()
        writer.close()

    if best_idm_state is not None:
        idm_model.load_state_dict(best_idm_state)
    if best_fdm_state is not None:
        fdm_model.load_state_dict(best_fdm_state)

    VIDEO_LEARNER_RUN_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(idm_model.state_dict(), IDM_MODEL_PT)
    torch.save(fdm_model.state_dict(), FDM_MODEL_PT)
    torch.save(losses, LOSSES_PT)

    print(f"[video-learner] saved IDM -> {IDM_MODEL_PT}")
    print(f"[video-learner] saved FDM -> {FDM_MODEL_PT}")
    print(f"[video-learner] saved losses -> {LOSSES_PT}")
    finish_experiment(
        exp,
        {
            "best_test": best_test_loss,
            "best_epoch": best_epoch,
            "last_train_total": losses["train_total"][-1] if losses["train_total"] else None,
            "last_test_total": losses["test_total"][-1] if losses["test_total"] else None,
            "delta_cos": losses["delta_cos"][-1] if losses["delta_cos"] else None,
        },
    )

    return idm_model, fdm_model, losses


def _log_video_learner_debug(writer, batch, step):
    log_step_features(writer, "video_learner_01_getlatent", ["AE reconstruction MSE", "decoded z_i_true frame preview"], step)
    log_step_features(writer, "video_learner_02_history", ["history length k", "temporal latent drift", "adjacent latent distance"], step)
    log_step_features(writer, "video_learner_03_lstmhidden", ["h_t activation distribution", "LSTM next-latent baseline MSE"], step)
    log_step_features(writer, "video_learner_04_overlay", ["overlay preview", "overlay intensity range", "motion visibility"], step)
    log_step_features(writer, "video_learner_05_overlaylatent", ["overlay reconstruction MSE", "m_t vs z_t_true distance"], step)
    log_step_features(writer, "video_learner_06_context", ["component norms", "concat dimension check"], step)
    log_step_features(writer, "video_learner_07_idm_latentaction", ["latent action mean / std / min / max", "saturation", "dead dimensions"], step)
    log_step_features(writer, "video_learner_08_fdm_input", ["c_t norm vs latent action norm", "concat dimension check"], step)
    log_step_features(writer, "video_learner_09_delta_research", ["delta-target MSE vs direct-next-latent MSE"], step)
    log_step_features(writer, "video_learner_10_fdm_delta", ["delta norm", "delta cosine", "delta MSE"], step)
    log_step_features(writer, "video_learner_11_build_nextlatent", ["decoded predicted next preview", "predicted next vs true next distance"], step)
    log_step_features(writer, "video_learner_12_loss", ["video MSE", "video/LSTM MSE ratio", "loss curve"], step)
    log_image_batch(writer, "video_learner_01_getlatent/sample", batch["x_input"], step)
    log_image_batch(writer, "video_learner_04_overlay/sample", batch["x_overlay"], step)
    log_tensor(writer, "video_learner_01_getlatent", batch["z_input"], step)
    log_tensor(writer, "video_learner_02_history", batch["z_input"], step)
    log_tensor(writer, "video_learner_03_lstmhidden", batch["h_t"], step)
    log_tensor(writer, "video_learner_05_overlaylatent", batch["m_t"], step)
    log_tensor(writer, "video_learner_06_context", batch["c_t"], step)
    log_tensor(writer, "video_learner_07_idm_latentaction", batch["latent_a"], step)
    log_tensor(writer, "video_learner_08_fdm_input", torch.cat([batch["c_t"], batch["latent_a"]], dim=-1), step)
    log_tensor(writer, "video_learner_10_fdm_delta", batch["delta_z_pred"], step)
    log_tensor(writer, "video_learner_10_true_delta", batch["delta_z_true"], step)
    log_tensor(writer, "video_learner_11_build_nextlatent", batch["z_pred_next"], step)
    writer.add_scalar("video_learner_10_fdm_delta/delta_cos", batch["cos_delta"].item(), step)
    writer.add_scalar("video_learner_10_fdm_delta/delta_mse", batch["loss_delta"].item(), step)
    writer.add_scalar("video_learner_12_loss/video_mse", batch["loss_z"].item(), step)
    writer.add_scalar("video_learner_12_loss/lstm_baseline_mse", batch["loss_lstm"].item(), step)


if __name__ == "__main__":
    train_video_learner(**VIDEO_LEARNER_CONFIG)
