import numpy as np
import torch
import gymnasium as gym
from PIL import Image

from ActionInference.machine_forward.config import MACHINE_FORWARD_CONFIG
from ActionInference.machine_forward.models import MachineForwardModel
from ActionInference.shared.loaders import load_ae_model, load_lstm_model
from ActionInference.shared.paths import (
    FDM_MODEL_PT,
    IDM_MODEL_PT,
    MACHINE_MODEL_PT,
)
from ActionInference.video_learner.models import FDMModel, IDMModel, overlay_frames


def _device():
    return torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )


def preprocess_rgb(frame):
    img = Image.fromarray(frame).convert("L").resize((84, 84))
    x = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(x)[None, None]


def _lstm_predict_and_hidden(lstm_model, z_input):
    h_seq, _ = lstm_model.lstm(z_input)
    h_t = h_seq[:, -1, :]

    z_lstm_next = lstm_model(z_input)
    if isinstance(z_lstm_next, tuple):
        z_lstm_next = z_lstm_next[-1]

    return h_t, z_lstm_next


def _pad_tail(items, length):
    if len(items) < length:
        return [items[0]] * (length - len(items)) + items
    return items[-length:]


def load_runtime_models(device=None, latent_a_dim=8):
    if device is None:
        device = _device()

    ae_model = load_ae_model(device)
    lstm_model = load_lstm_model(device)

    idm_model = IDMModel(c_dim=384, latent_a_dim=latent_a_dim).to(device)
    fdm_model = FDMModel(c_dim=384, z_dim=64, latent_a_dim=latent_a_dim).to(device)
    idm_model.load_state_dict(torch.load(IDM_MODEL_PT, map_location=device))
    fdm_model.load_state_dict(torch.load(FDM_MODEL_PT, map_location=device))

    machine_forward_model = MachineForwardModel(
        z_dim=MACHINE_FORWARD_CONFIG["z_dim"],
        real_a_dim=MACHINE_FORWARD_CONFIG["real_a_dim"],
        hidden_dim=MACHINE_FORWARD_CONFIG["hidden_dim"],
        num_layers=MACHINE_FORWARD_CONFIG["num_layers"],
    ).to(device)
    machine_forward_model.load_state_dict(torch.load(MACHINE_MODEL_PT, map_location=device))

    ae_model.eval()
    lstm_model.eval()
    idm_model.eval()
    fdm_model.eval()
    machine_forward_model.eval()

    return ae_model, lstm_model, idm_model, fdm_model, machine_forward_model


def desired_delta_from_video_context(
    ae_model,
    lstm_model,
    idm_model,
    fdm_model,
    x_hist,
    history_len=31,
    overlay_decay=0.9,
    delta_z_scale=1.0,
    return_debug=False,
):
    hist = _pad_tail(x_hist, history_len)
    x_seq = torch.stack(hist, dim=0)[None]

    x_overlay = overlay_frames(x_seq, decay=overlay_decay)
    _, m_t = ae_model(x_overlay)

    z_hist = torch.stack(
        [ae_model(x_seq[:, t])[1] for t in range(x_seq.shape[1])],
        dim=1,
    )
    z_t = z_hist[:, -1, :]
    h_t, _ = _lstm_predict_and_hidden(lstm_model, z_hist)
    c_t = torch.cat([z_t, h_t, m_t], dim=-1)

    latent_a = idm_model(c_t)
    x_fdm = torch.cat([c_t, latent_a], dim=-1)
    delta_z_desired = fdm_model(c_t, latent_a) * delta_z_scale

    if return_debug:
        return delta_z_desired, latent_a, {
            "x_seq": x_seq.detach(),
            "x_overlay": x_overlay.detach(),
            "z_hist": z_hist.detach(),
            "z_t": z_t.detach(),
            "h_t": h_t.detach(),
            "m_t": m_t.detach(),
            "c_t": c_t.detach(),
            "x_fdm": x_fdm.detach(),
        }

    return delta_z_desired, latent_a


def choose_action_by_machine_forward(
    machine_forward_model,
    z_hist,
    delta_z_desired,
    n_candidates=2048,
    action_low=-0.7,
    action_high=0.7,
    action_l2_penalty=0.05,
    real_a_dim=2,
    return_debug=False,
):
    machine_forward_model.eval()

    bsz = z_hist.shape[0]
    hist_len = z_hist.shape[1]
    z_dim = z_hist.shape[2]
    device = z_hist.device
    dtype = z_hist.dtype

    actions = torch.empty(
        bsz,
        n_candidates,
        real_a_dim,
        device=device,
        dtype=dtype,
    ).uniform_(action_low, action_high)

    z_hist_rep = z_hist[:, None, :, :].expand(bsz, n_candidates, hist_len, z_dim)
    dz_rep = delta_z_desired[:, None, :].expand(
        bsz,
        n_candidates,
        delta_z_desired.shape[-1],
    )

    z_hist_flat = z_hist_rep.reshape(bsz * n_candidates, hist_len, z_dim)
    a_flat = actions.reshape(bsz * n_candidates, real_a_dim)

    with torch.no_grad():
        dz_pred = machine_forward_model(z_hist_flat, a_flat)
        dz_pred = dz_pred.reshape(bsz, n_candidates, -1)

        latent_loss = ((dz_pred - dz_rep) ** 2).mean(dim=-1)
        action_penalty = (actions ** 2).mean(dim=-1)
        losses = latent_loss + action_l2_penalty * action_penalty
        best_idx = losses.argmin(dim=1)

    best_actions = actions[torch.arange(bsz, device=device), best_idx]
    best_losses = losses[torch.arange(bsz, device=device), best_idx]

    if return_debug:
        best_dz_pred = dz_pred[torch.arange(bsz, device=device), best_idx]
        best_latent_loss = latent_loss[torch.arange(bsz, device=device), best_idx]
        best_action_penalty = action_penalty[torch.arange(bsz, device=device), best_idx]
        return best_actions, best_losses, {
            "candidate_actions": actions.detach(),
            "best_dz_pred": best_dz_pred.detach(),
            "best_latent_loss": best_latent_loss.detach(),
            "best_action_penalty": best_action_penalty.detach(),
            "candidate_delta_norms": torch.linalg.vector_norm(dz_pred.detach(), dim=-1),
        }

    return best_actions, best_losses


def rollout_lander_machine_forward(
    ae_model,
    lstm_model,
    idm_model,
    fdm_model,
    machine_forward_model,
    max_steps=500,
    video_history_len=31,
    machine_hist_len=8,
    overlay_decay=0.9,
    n_action_candidates=8192,
    action_low=-0.7,
    action_high=0.7,
    action_l2_penalty=0.05,
    delta_z_scale=1.5,
    real_a_dim=2,
    device=None,
):
    if device is None:
        device = _device()

    env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
    env.reset()

    x_hist = []
    z_buffer = []
    frames = []
    actions = []
    latent_actions = []
    desired_delta_norms = []
    search_losses = []
    rewards = []
    debug = {
        "getlatent": [],
        "video_history": [],
        "machine_history": [],
        "lstmhidden": [],
        "overlay": [],
        "overlaylatent": [],
        "context": [],
        "fdm_input": [],
        "desired_delta": [],
        "machine_hidden": [],
        "candidate_actions": [],
        "machine_input": [],
        "machine_candidate_delta": [],
        "chooseaction_latent_loss": [],
        "chooseaction_action_penalty": [],
    }
    total_reward = 0.0

    for _ in range(max_steps):
        rgb = env.render()
        frames.append(rgb)

        x = preprocess_rgb(rgb).to(device)
        x_hist.append(x.squeeze(0))

        with torch.no_grad():
            _, z = ae_model(x)
        z_buffer.append(z.squeeze(0).detach().cpu())

        delta_z_desired, latent_a, video_debug = desired_delta_from_video_context(
            ae_model=ae_model,
            lstm_model=lstm_model,
            idm_model=idm_model,
            fdm_model=fdm_model,
            x_hist=x_hist,
            history_len=video_history_len,
            overlay_decay=overlay_decay,
            delta_z_scale=delta_z_scale,
            return_debug=True,
        )

        machine_hist = _pad_tail(z_buffer, machine_hist_len)
        z_hist_machine = torch.stack(machine_hist, dim=0)[None].to(device)

        action, search_loss, search_debug = choose_action_by_machine_forward(
            machine_forward_model=machine_forward_model,
            z_hist=z_hist_machine,
            delta_z_desired=delta_z_desired,
            n_candidates=n_action_candidates,
            action_low=action_low,
            action_high=action_high,
            action_l2_penalty=action_l2_penalty,
            real_a_dim=real_a_dim,
            return_debug=True,
        )

        with torch.no_grad():
            _, (h_n_machine, _) = machine_forward_model.history_encoder(z_hist_machine)
            h_t_machine = h_n_machine[-1]
            x_machine = torch.cat([h_t_machine, action], dim=-1)

        action_np = action[0].detach().cpu().numpy()
        action_np = np.clip(action_np, -1.0, 1.0).astype(np.float32)

        _, reward, terminated, truncated, _ = env.step(action_np)

        actions.append(action_np.copy())
        latent_actions.append(latent_a[0].detach().cpu().numpy().copy())
        desired_delta_norms.append(float(torch.linalg.vector_norm(delta_z_desired).item()))
        search_losses.append(float(search_loss.item()))
        rewards.append(float(reward))
        total_reward += float(reward)

        debug["getlatent"].append(z.detach().squeeze(0).cpu())
        debug["video_history"].append(video_debug["z_hist"].detach().squeeze(0).cpu())
        debug["machine_history"].append(z_hist_machine.detach().squeeze(0).cpu())
        debug["lstmhidden"].append(video_debug["h_t"].detach().squeeze(0).cpu())
        debug["overlay"].append(video_debug["x_overlay"].detach().squeeze(0).cpu())
        debug["overlaylatent"].append(video_debug["m_t"].detach().squeeze(0).cpu())
        debug["context"].append(video_debug["c_t"].detach().squeeze(0).cpu())
        debug["fdm_input"].append(video_debug["x_fdm"].detach().squeeze(0).cpu())
        debug["desired_delta"].append(delta_z_desired.detach().squeeze(0).cpu())
        debug["machine_hidden"].append(h_t_machine.detach().squeeze(0).cpu())
        debug["candidate_actions"].append(search_debug["candidate_actions"].detach().squeeze(0).cpu())
        debug["machine_input"].append(x_machine.detach().squeeze(0).cpu())
        debug["machine_candidate_delta"].append(search_debug["best_dz_pred"].detach().squeeze(0).cpu())
        debug["chooseaction_latent_loss"].append(search_debug["best_latent_loss"].detach().cpu())
        debug["chooseaction_action_penalty"].append(search_debug["best_action_penalty"].detach().cpu())

        if len(x_hist) > video_history_len:
            x_hist = x_hist[-video_history_len:]
        if len(z_buffer) > machine_hist_len:
            z_buffer = z_buffer[-machine_hist_len:]

        if terminated or truncated:
            break

    env.close()

    return {
        "frames": frames,
        "total_reward": total_reward,
        "actions": np.array(actions),
        "latent_actions": np.array(latent_actions),
        "desired_delta_norms": np.array(desired_delta_norms),
        "search_losses": np.array(search_losses),
        "rewards": np.array(rewards),
        "debug": {
            key: torch.stack(value, dim=0)
            for key, value in debug.items()
            if value
        },
    }
