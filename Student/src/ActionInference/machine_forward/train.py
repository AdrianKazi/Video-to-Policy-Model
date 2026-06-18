import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from ActionInference.machine_forward.config import MACHINE_FORWARD_CONFIG
from ActionInference.machine_forward.models import MachineForwardModel
from ActionInference.shared.experiment import finish_experiment, start_experiment
from ActionInference.shared.tensorboard import log_tensor, make_writer
from ActionInference.shared.paths import (
    MACHINE_FORWARD_RUN_DIR,
    MACHINE_LOSSES_PT,
    MACHINE_MODEL_PT,
    MACHINE_PROBE_PT,
)


def _device():
    return torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )


def train_machine_forward(
    z_dim=64,
    real_a_dim=2,
    hidden_dim=256,
    num_layers=1,
    hist_len=8,
    n_episodes=100,
    max_steps=300,
    epochs=30,
    batch_size=128,
    lr=3e-4,
    test_ratio=0.2,
    early_stop_patience=5,
    early_stop_min_delta=1e-5,
    min_delta_norm=None,
    max_stride=None,
):
    MACHINE_FORWARD_RUN_DIR.mkdir(parents=True, exist_ok=True)
    exp = start_experiment("machine_forward_train", MACHINE_FORWARD_CONFIG)

    device = _device()
    probe_data = torch.load(MACHINE_PROBE_PT, map_location="cpu")

    dataset = TensorDataset(
        probe_data["z_hist"].float(),
        probe_data["a_real"].float(),
        probe_data["dz"].float(),
    )

    n_test = max(1, int(len(dataset) * test_ratio))
    n_train = len(dataset) - n_test
    train_ds, test_ds = random_split(
        dataset,
        [n_train, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = MachineForwardModel(
        z_dim=z_dim,
        real_a_dim=real_a_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses = {
        "epoch": [],
        "train": [],
        "test": [],
        "zero_baseline": [],
    }
    best_test_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    writer = make_writer(MACHINE_FORWARD_RUN_DIR, "train")

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            train_sum = 0.0
            train_n = 0
            debug_batch = None

            for z_hist_b, a_b, dz_b in train_loader:
                z_hist_b = z_hist_b.to(device)
                a_b = a_b.to(device)
                dz_b = dz_b.to(device)

                dz_pred = model(z_hist_b, a_b)
                loss = F.mse_loss(dz_pred, dz_b)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                train_sum += float(loss.item()) * z_hist_b.shape[0]
                train_n += z_hist_b.shape[0]
                debug_batch = {
                    "z_hist": z_hist_b.detach(),
                    "a_real": a_b.detach(),
                    "dz_true": dz_b.detach(),
                    "dz_pred": dz_pred.detach(),
                    "loss": loss.detach(),
                }

            model.eval()
            test_sum = 0.0
            zero_sum = 0.0
            cos_sum = 0.0
            test_n = 0

            with torch.no_grad():
                for z_hist_b, a_b, dz_b in test_loader:
                    z_hist_b = z_hist_b.to(device)
                    a_b = a_b.to(device)
                    dz_b = dz_b.to(device)

                    dz_pred = model(z_hist_b, a_b)
                    loss = F.mse_loss(dz_pred, dz_b)
                    zero_loss = F.mse_loss(torch.zeros_like(dz_b), dz_b)
                    cos = F.cosine_similarity(dz_pred, dz_b, dim=-1).mean()

                    test_sum += float(loss.item()) * z_hist_b.shape[0]
                    zero_sum += float(zero_loss.item()) * z_hist_b.shape[0]
                    cos_sum += float(cos.item()) * z_hist_b.shape[0]
                    test_n += z_hist_b.shape[0]

            train_loss = train_sum / max(train_n, 1)
            test_loss = test_sum / max(test_n, 1)
            zero_loss = zero_sum / max(test_n, 1)
            delta_cos = cos_sum / max(test_n, 1)

            losses["epoch"].append(epoch)
            losses["train"].append(train_loss)
            losses["test"].append(test_loss)
            losses["zero_baseline"].append(zero_loss)

            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/test", test_loss, epoch)
            writer.add_scalar("loss/zero_baseline", zero_loss, epoch)
            writer.add_scalar("machine_forward/delta_cos", delta_cos, epoch)
            writer.add_scalar("machine_forward/zero_over_machine", zero_loss / max(test_loss, 1e-12), epoch)
            if debug_batch is not None:
                _log_machine_forward_debug(writer, model, debug_batch, epoch)

            improved = test_loss < (best_test_loss - early_stop_min_delta)
            if improved:
                best_test_loss = test_loss
                best_epoch = epoch
                stale_epochs = 0
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
            else:
                stale_epochs += 1

            print(
                f"[machine] epoch {epoch:03d}/{epochs} | "
                f"train {train_loss:.6f} | test {test_loss:.6f} | "
                f"zero {zero_loss:.6f} | best_test {best_test_loss:.6f} @ {best_epoch:03d}"
            )

            if stale_epochs >= early_stop_patience:
                print(
                    f"[machine] early stop at epoch {epoch:03d}; "
                    f"best_test {best_test_loss:.6f} @ {best_epoch:03d}"
                )
                break
    finally:
        writer.flush()
        writer.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), MACHINE_MODEL_PT)
    torch.save(losses, MACHINE_LOSSES_PT)

    plt.figure(figsize=(8, 4))
    plt.plot(losses["epoch"], losses["train"], lw=2, label="train")
    plt.plot(losses["epoch"], losses["test"], lw=2, label="test")
    plt.plot(losses["epoch"], losses["zero_baseline"], lw=1.8, label="zero baseline")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("Machine Forward Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(MACHINE_FORWARD_RUN_DIR / "machine_forward_losses.png", dpi=160)
    plt.close()

    print(f"[machine] saved model  -> {MACHINE_MODEL_PT}")
    print(f"[machine] saved losses -> {MACHINE_LOSSES_PT}")
    finish_experiment(
        exp,
        {
            "best_test": best_test_loss,
            "best_epoch": best_epoch,
            "last_train": losses["train"][-1] if losses["train"] else None,
            "machine_mse": losses["test"][-1] if losses["test"] else None,
            "zero_mse": losses["zero_baseline"][-1] if losses["zero_baseline"] else None,
            "zero_over_machine": (
                losses["zero_baseline"][-1] / max(losses["test"][-1], 1e-12)
                if losses["test"] else None
            ),
        },
    )
    return model, losses


def _log_machine_forward_debug(writer, model, batch, step):
    with torch.no_grad():
        _, (h_n, _) = model.history_encoder(batch["z_hist"])
        h_t_machine = h_n[-1]
        x_machine = torch.cat([h_t_machine, batch["a_real"]], dim=-1)
        z_t = batch["z_hist"][:, -1]
        z_next_true = z_t + batch["dz_true"]
        z_pred_next = z_t + batch["dz_pred"]

    writer.add_text(
        "MACHINE_FORWARD_READ_ME_FIRST",
        "\n".join(
            [
                "A_LOSS: MachineForward MSE and zero-delta baseline.",
                "B_TRUE_DELTA: latent movement produced by real env action over adaptive stride.",
                "C_PREDICTED_DELTA: MachineForward prediction for that real action.",
                "D_REAL_ACTION: sampled LunarLander action used for training.",
            ]
        ),
        0,
    )
    log_tensor(writer, "D_REAL_ACTION_USED_FOR_MACHINE_FORWARD_TRAINING", batch["a_real"], step)
    log_tensor(writer, "B_TRUE_DELTA_FROM_ENV_ADAPTIVE_STRIDE", batch["dz_true"], step)
    log_tensor(writer, "C_PREDICTED_DELTA_FROM_MACHINE_FORWARD", batch["dz_pred"], step)
    writer.add_scalar("machine_forward_01_sampleaction/real_action_std", batch["a_real"].std(dim=0, unbiased=False).mean().item(), step)
    writer.add_scalar("machine_forward_02_envstep/latent_transition_norm_after_action", batch["dz_true"].norm(dim=-1).mean().item(), step)
    writer.add_scalar("machine_forward_03_currentlatent/current_latent_norm", z_t.norm(dim=-1).mean().item(), step)
    writer.add_scalar("machine_forward_04_machinehistory/history_latent_drift", (batch["z_hist"][:, -1] - batch["z_hist"][:, 0]).norm(dim=-1).mean().item(), step)
    writer.add_scalar("machine_forward_05_nextlatent/current_to_next_latent_distance", batch["dz_true"].norm(dim=-1).mean().item(), step)
    writer.add_scalar("machine_forward_06_encodehistory/machine_hidden_norm", h_t_machine.norm(dim=-1).mean().item(), step)
    writer.add_scalar("machine_forward_07_machineinput/machine_input_norm", x_machine.norm(dim=-1).mean().item(), step)
    writer.add_scalar("machine_forward_08_true_delta/true_delta_norm", batch["dz_true"].norm(dim=-1).mean().item(), step)
    writer.add_scalar("machine_forward_09_pred_delta/predicted_delta_norm", batch["dz_pred"].norm(dim=-1).mean().item(), step)
    writer.add_scalar("machine_forward_10_build_nextlatent/predicted_next_vs_true_mse", F.mse_loss(z_pred_next, z_next_true).item(), step)
    writer.add_scalar("machine_forward_11_loss/train_batch_mse", batch["loss"].item(), step)
    writer.add_scalar("machine_forward_12_delta_research/delta_prediction_mse", F.mse_loss(batch["dz_pred"], batch["dz_true"]).item(), step)
    writer.add_scalar("A_LOSS/train_batch_mse", batch["loss"].item(), step)
    writer.add_scalar("B_TRUE_DELTA/norm_mean", batch["dz_true"].norm(dim=-1).mean().item(), step)
    writer.add_scalar("C_PREDICTED_DELTA/norm_mean", batch["dz_pred"].norm(dim=-1).mean().item(), step)
    writer.add_scalar(
        "C_PREDICTED_DELTA/cosine_pred_vs_true",
        F.cosine_similarity(batch["dz_pred"], batch["dz_true"], dim=-1).mean().item(),
        step,
    )


if __name__ == "__main__":
    train_machine_forward(**MACHINE_FORWARD_CONFIG)
