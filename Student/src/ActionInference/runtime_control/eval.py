import imageio.v2 as imageio
import matplotlib.pyplot as plt
import torch

from ActionInference.shared.experiment import finish_experiment, start_experiment
from ActionInference.shared.tensorboard import log_tensor, make_writer
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
    debug = stats.get("debug", {})

    chosen_machine_delta = debug.get("machine_candidate_delta")
    if chosen_machine_delta is not None:
        chosen_machine_delta_norms = torch.linalg.vector_norm(chosen_machine_delta.float(), dim=-1)
    else:
        chosen_machine_delta_norms = torch.empty(0)

    writer.add_text(
        "RUNTIME_CONTROL_READ_ME_FIRST",
        "\n".join(
            [
                "A_DESIRED_DELTA_NORM: what Video Learner asks the controller to do.",
                "B_CHOSEN_MACHINE_DELTA_NORM: what the selected real action can do according to Machine Forward.",
                "C_CHOSEN_REAL_ACTION_A0_A1: actual LunarLander action sent to env.",
                "D_SEARCH_LOSS: planner matching loss after action penalty.",
                "E_REWARD_PER_STEP: environment reward after applying chosen action.",
            ]
        ),
        0,
    )

    log_tensor(writer, "A_DESIRED_DELTA_VECTOR_FROM_VIDEO_LEARNER", debug.get("desired_delta"), 0)
    log_tensor(writer, "B_CHOSEN_MACHINE_DELTA_VECTOR_FROM_REAL_ACTION", chosen_machine_delta, 0)
    log_tensor(writer, "C_CHOSEN_REAL_ACTION_A0_A1_SENT_TO_ENV", actions, 0)
    log_tensor(writer, "C_IDM_LATENT_ACTION_NOT_REAL_ACTION", latent_actions, 0)
    log_tensor(writer, "D_SEARCH_LOSS_MATCH_DESIRED_VS_MACHINE_DELTA", search_losses, 0)
    log_tensor(writer, "E_REWARD_PER_STEP_AFTER_ACTION", rewards, 0)

    if debug:
        getlatent = debug.get("getlatent")
        video_history = debug.get("video_history")
        machine_history = debug.get("machine_history")
        lstmhidden = debug.get("lstmhidden")
        overlay = debug.get("overlay")
        overlaylatent = debug.get("overlaylatent")
        context = debug.get("context")
        fdm_input = debug.get("fdm_input")
        desired_delta = debug.get("desired_delta")
        machine_hidden = debug.get("machine_hidden")
        candidate_actions = debug.get("candidate_actions")
        machine_input = debug.get("machine_input")
        choose_latent_loss = debug.get("chooseaction_latent_loss")
        choose_action_penalty = debug.get("chooseaction_action_penalty")

    for i in range(actions.shape[0]):
        if debug:
            writer.add_scalar("runtime_control_01_getlatent/current_latent_norm", getlatent[i].norm().item(), i)
            writer.add_scalar("runtime_control_02_video_history/video_history_drift", (video_history[i, -1] - video_history[i, 0]).norm().item(), i)
            writer.add_scalar("runtime_control_03_machine_history/machine_history_drift", (machine_history[i, -1] - machine_history[i, 0]).norm().item(), i)
            writer.add_scalar("runtime_control_04_lstmhidden/lstm_hidden_norm", lstmhidden[i].norm().item(), i)
            writer.add_scalar("runtime_control_05_overlay/overlay_mean_intensity", overlay[i].mean().item(), i)
            writer.add_scalar("runtime_control_06_overlaylatent/overlay_latent_norm", overlaylatent[i].norm().item(), i)
            writer.add_scalar("runtime_control_07_context/context_norm", context[i].norm().item(), i)
            writer.add_scalar("runtime_control_08_idm_latentaction/latent_action_norm_not_real_action", latent_actions[i].norm().item(), i)
            writer.add_scalar("runtime_control_09_fdm_input/fdm_input_norm", fdm_input[i].norm().item(), i)
            writer.add_scalar("runtime_control_10_desired_delta/desired_delta_norm_from_video_learner", desired_delta[i].norm().item(), i)
            writer.add_scalar("runtime_control_11_machine_hidden/machine_hidden_norm", machine_hidden[i].norm().item(), i)
            writer.add_scalar("runtime_control_12_candidate_actions/candidate_action_std", candidate_actions[i].std(dim=0, unbiased=False).mean().item(), i)
            writer.add_scalar("runtime_control_13_machine_input/chosen_machine_input_norm", machine_input[i].norm().item(), i)
            if chosen_machine_delta_norms.numel():
                writer.add_scalar("runtime_control_14_machine_candidate_delta/chosen_machine_delta_norm", chosen_machine_delta_norms[i].item(), i)
            writer.add_scalar("runtime_control_15_chooseaction/latent_match_loss", choose_latent_loss[i].item(), i)
            writer.add_scalar("runtime_control_15_chooseaction/action_penalty", choose_action_penalty[i].item(), i)
            writer.add_scalar("runtime_control_16_applyaction/reward_after_chosen_action", rewards[i].item(), i)
        writer.add_scalar("A_DESIRED_DELTA_NORM_FROM_VIDEO_LEARNER", desired_delta_norms[i].item(), i)
        if chosen_machine_delta_norms.numel():
            writer.add_scalar("B_CHOSEN_MACHINE_DELTA_NORM_FROM_REAL_ACTION", chosen_machine_delta_norms[i].item(), i)
        writer.add_scalar("C_CHOSEN_REAL_ACTION/a0_main_engine", actions[i, 0].item(), i)
        writer.add_scalar("C_CHOSEN_REAL_ACTION/a1_side_engine", actions[i, 1].item(), i)
        writer.add_scalar("D_SEARCH_LOSS_MATCH_DESIRED_VS_MACHINE_DELTA", search_losses[i].item(), i)
        writer.add_scalar("E_REWARD_PER_STEP_AFTER_ACTION", rewards[i].item(), i)


if __name__ == "__main__":
    run_runtime_control_eval(**RUNTIME_CONTROL_CONFIG)
