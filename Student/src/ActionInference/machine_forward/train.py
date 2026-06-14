import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from ActionInference.machine_forward.config import MACHINE_FORWARD_CONFIG
from ActionInference.machine_forward.models import MachineForwardModel
from ActionInference.shared.experiment import finish_experiment, start_experiment
from ActionInference.shared.tensorboard import log_step_features, log_tensor, make_writer
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
        z_pred_next = z_t + batch["dz_pred"]

    log_step_features(writer, "machine_forward_01_sampleaction", ["action mean / std / min / max", "action coverage"], step)
    log_step_features(writer, "machine_forward_02_envstep", ["frame transition preview", "episode length", "terminated / truncated rate"], step)
    log_step_features(writer, "machine_forward_03_currentlatent", ["AE reconstruction MSE", "decoded z_t_true preview"], step)
    log_step_features(writer, "machine_forward_04_machinehistory", ["machine history length", "latent drift"], step)
    log_step_features(writer, "machine_forward_05_nextlatent", ["decoded z_next_true preview", "z_next_true vs z_t_true distance"], step)
    log_step_features(writer, "machine_forward_06_encodehistory", ["h_t_machine activation distribution", "history encoder hidden norm"], step)
    log_step_features(writer, "machine_forward_07_machineinput", ["hidden norm vs action norm", "concat dimension check"], step)
    log_step_features(writer, "machine_forward_08_true_delta", ["true delta norm", "true delta distribution", "zero baseline MSE"], step)
    log_step_features(writer, "machine_forward_09_pred_delta", ["pred delta norm", "pred vs true delta cosine", "action sensitivity"], step)
    log_step_features(writer, "machine_forward_10_build_nextlatent", ["decoded predicted next frame", "predicted next vs true next distance"], step)
    log_step_features(writer, "machine_forward_11_loss", ["machine MSE", "zero baseline MSE", "zero/machine ratio"], step)
    log_step_features(writer, "machine_forward_12_delta_research", ["delta-target MSE vs direct-next-latent MSE"], step)
    log_tensor(writer, "machine_forward_01_sampleaction", batch["a_real"], step)
    log_tensor(writer, "machine_forward_03_currentlatent", z_t, step)
    log_tensor(writer, "machine_forward_04_machinehistory", batch["z_hist"], step)
    log_tensor(writer, "machine_forward_06_encodehistory", h_t_machine, step)
    log_tensor(writer, "machine_forward_07_machineinput", x_machine, step)
    log_tensor(writer, "machine_forward_08_true_delta", batch["dz_true"], step)
    log_tensor(writer, "machine_forward_09_pred_delta", batch["dz_pred"], step)
    log_tensor(writer, "machine_forward_10_build_nextlatent", z_pred_next, step)
    writer.add_scalar("machine_forward_09_pred_delta/action_sensitivity_proxy", batch["dz_pred"].std(dim=0, unbiased=False).mean().item(), step)
    writer.add_scalar("machine_forward_11_loss/train_batch_mse", batch["loss"].item(), step)


if __name__ == "__main__":
    train_machine_forward(**MACHINE_FORWARD_CONFIG)
