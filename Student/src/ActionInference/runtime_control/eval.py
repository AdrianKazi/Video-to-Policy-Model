import imageio.v2 as imageio
import matplotlib.pyplot as plt
import torch

from ActionInference.shared.experiment import finish_experiment, start_experiment
from ActionInference.shared.tensorboard import log_step_features, log_tensor, make_writer
from ActionInference.runtime_control.config import RUNTIME_CONTROL_CONFIG
from ActionInference.runtime_control.control import (
    load_runtime_models,
    rollout_lander_machine_forward,
)
from ActionInference.shared.paths import (
    RUNTIME_CONTROL_RUN_DIR,
    RUNTIME_DIAGNOSTICS_PNG,
    RUNTIME_ROLLOUT_MP4,
    RUNTIME_STATS_PT,
)


def _device():
    return torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )


def save_runtime_diagnostics(stats, out_path=RUNTIME_DIAGNOSTICS_PNG):
    actions = stats["actions"]
    latent_actions = stats["latent_actions"]
    search_losses = stats["search_losses"]
    steps = range(len(actions))

    fig, axs = plt.subplots(3, 1, figsize=(14, 10))

    axs[0].plot(steps, actions[:, 0], lw=1.5, label="a0")
    axs[0].plot(steps, actions[:, 1], lw=1.5, label="a1")
    axs[0].axhline(0, color="black", lw=1)
    axs[0].set_ylim(-1.05, 1.05)
    axs[0].set_title("Runtime Control Actions")
    axs[0].set_xlabel("step")
    axs[0].set_ylabel("action")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    for i in range(latent_actions.shape[1]):
        axs[1].plot(steps, latent_actions[:, i], lw=1.0, label=f"l{i}")
    axs[1].axhline(0, color="black", lw=1)
    axs[1].set_ylim(-1.05, 1.05)
    axs[1].set_title("IDM Latent Actions")
    axs[1].set_xlabel("step")
    axs[1].set_ylabel("latent")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(ncol=4, fontsize=8)

    axs[2].plot(steps, search_losses, color="tab:red", lw=1.8)
    axs[2].set_title("Machine Forward Action Search Loss")
    axs[2].set_xlabel("step")
    axs[2].set_ylabel("loss")
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    print(f"[runtime-control] saved diagnostics -> {out_path}")
    return out_path


def run_runtime_control_eval(**cfg):
    RUNTIME_CONTROL_RUN_DIR.mkdir(parents=True, exist_ok=True)
    exp = start_experiment("runtime_control_eval", cfg)
    writer = make_writer(RUNTIME_CONTROL_RUN_DIR, "eval")

    try:
        device = _device()
        ae_model, lstm_model, idm_model, fdm_model, machine_forward_model = load_runtime_models(
            device=device,
            latent_a_dim=cfg["latent_a_dim"],
        )

        stats = rollout_lander_machine_forward(
            ae_model=ae_model,
            lstm_model=lstm_model,
            idm_model=idm_model,
            fdm_model=fdm_model,
            machine_forward_model=machine_forward_model,
            max_steps=cfg["max_steps"],
            video_history_len=cfg["video_history_len"],
            machine_hist_len=cfg["machine_hist_len"],
            overlay_decay=cfg["overlay_decay"],
            n_action_candidates=cfg["n_action_candidates"],
            action_low=cfg["action_low"],
            action_high=cfg["action_high"],
            action_l2_penalty=cfg["action_l2_penalty"],
            delta_z_scale=cfg["delta_z_scale"],
            real_a_dim=cfg["real_a_dim"],
            device=device,
        )

        _log_runtime_tensorboard(writer, stats)
        imageio.mimsave(RUNTIME_ROLLOUT_MP4, stats["frames"], fps=cfg["fps"])
        torch.save(
            {
                "total_reward": stats["total_reward"],
                "actions": torch.from_numpy(stats["actions"]),
                "latent_actions": torch.from_numpy(stats["latent_actions"]),
                "desired_delta_norms": torch.from_numpy(stats["desired_delta_norms"]),
                "search_losses": torch.from_numpy(stats["search_losses"]),
                "rewards": torch.from_numpy(stats["rewards"]),
            },
            RUNTIME_STATS_PT,
        )
        save_runtime_diagnostics(stats)

        print(f"[runtime-control] saved rollout -> {RUNTIME_ROLLOUT_MP4}")
        print(f"[runtime-control] saved stats   -> {RUNTIME_STATS_PT}")
        print(f"[runtime-control] steps         -> {len(stats['frames'])}")
        print(f"[runtime-control] total_reward  -> {stats['total_reward']:.3f}")
        print(f"[runtime-control] action mean   -> {stats['actions'].mean(axis=0)}")
        print(f"[runtime-control] action std    -> {stats['actions'].std(axis=0)}")
        print(f"[runtime-control] search mean   -> {stats['search_losses'].mean():.6f}")
        finish_experiment(
            exp,
            {
                "total_reward": stats["total_reward"],
                "steps": len(stats["frames"]),
                "search_mean": float(stats["search_losses"].mean()),
                "action_mean_0": float(stats["actions"].mean(axis=0)[0]),
                "action_mean_1": float(stats["actions"].mean(axis=0)[1]),
                "action_std_0": float(stats["actions"].std(axis=0)[0]),
                "action_std_1": float(stats["actions"].std(axis=0)[1]),
            },
        )
        return stats
    finally:
        writer.flush()
        writer.close()


def _log_runtime_tensorboard(writer, stats):
    actions = torch.from_numpy(stats["actions"]).float()
    latent_actions = torch.from_numpy(stats["latent_actions"]).float()
    desired_delta_norms = torch.from_numpy(stats["desired_delta_norms"]).float()
    search_losses = torch.from_numpy(stats["search_losses"]).float()
    rewards = torch.from_numpy(stats["rewards"]).float()

    log_step_features(writer, "runtime_control_01_getlatent", ["AE reconstruction MSE", "decoded latent preview"], 0)
    log_step_features(writer, "runtime_control_02_video_history", ["video history length", "movement-window latent drift"], 0)
    log_step_features(writer, "runtime_control_03_machine_history", ["machine history length", "overlap with video history", "action-window latent drift"], 0)
    log_step_features(writer, "runtime_control_04_lstmhidden", ["h_t activation distribution", "hidden norm"], 0)
    log_step_features(writer, "runtime_control_05_overlay", ["overlay preview", "motion visibility", "overlay intensity range"], 0)
    log_step_features(writer, "runtime_control_06_overlaylatent", ["overlay reconstruction MSE", "m_t norm", "m_t vs z_t distance"], 0)
    log_step_features(writer, "runtime_control_07_context", ["component norms", "concat dimension check"], 0)
    log_step_features(writer, "runtime_control_08_idm_latentaction", ["latent action mean / std / min / max", "saturation", "dead dimensions"], 0)
    log_step_features(writer, "runtime_control_09_fdm_input", ["c_t norm vs latent action norm", "concat dimension check"], 0)
    log_step_features(writer, "runtime_control_10_desired_delta", ["desired delta norm", "desired delta distribution", "desired delta vs reachable delta range"], 0)
    log_step_features(writer, "runtime_control_11_machine_hidden", ["machine hidden activation distribution", "machine hidden norm"], 0)
    log_step_features(writer, "runtime_control_12_candidate_actions", ["candidate action coverage", "candidate action mean / std", "boundary coverage"], 0)
    log_step_features(writer, "runtime_control_13_machine_input", ["machine hidden norm vs candidate action norm", "batch dimension check"], 0)
    log_step_features(writer, "runtime_control_14_machine_candidate_delta", ["candidate delta spread", "action sensitivity", "reachable delta range"], 0)
    log_step_features(writer, "runtime_control_15_chooseaction", ["selected action distribution", "search loss", "best-vs-median candidate loss", "action L2 penalty contribution"], 0)
    log_step_features(writer, "runtime_control_16_applyaction", ["rollout reward", "episode length", "action trace", "rollout mp4"], 0)
    log_tensor(writer, "runtime_control_08_idm_latentaction", latent_actions, 0)
    log_tensor(writer, "runtime_control_10_desired_delta", desired_delta_norms, 0)
    log_tensor(writer, "runtime_control_14_machine_candidate_delta", search_losses, 0)
    log_tensor(writer, "runtime_control_15_chooseaction", actions, 0)
    log_tensor(writer, "runtime_control_16_applyaction", rewards, 0)

    for i in range(actions.shape[0]):
        writer.add_scalar("runtime_control_15_chooseaction/a0", actions[i, 0].item(), i)
        writer.add_scalar("runtime_control_15_chooseaction/a1", actions[i, 1].item(), i)
        writer.add_scalar("runtime_control_10_desired_delta/norm", desired_delta_norms[i].item(), i)
        writer.add_scalar("runtime_control_15_chooseaction/search_loss", search_losses[i].item(), i)
        writer.add_scalar("runtime_control_16_applyaction/reward", rewards[i].item(), i)


if __name__ == "__main__":
    run_runtime_control_eval(**RUNTIME_CONTROL_CONFIG)
