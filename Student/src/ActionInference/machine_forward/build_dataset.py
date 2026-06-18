import numpy as np
import torch
import gymnasium as gym
from PIL import Image

from ActionInference.machine_forward.config import MACHINE_FORWARD_CONFIG
from ActionInference.shared.loaders import load_ae_model
from ActionInference.shared.paths import MACHINE_FORWARD_RUN_DIR, MACHINE_PROBE_PT


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


def sample_random_action(real_a_dim=2):
    return np.random.uniform(-1.0, 1.0, size=(real_a_dim,)).astype(np.float32)

def rollout_until_latent_motion(
    env,
    ae_model,
    z_t,
    a_real_np,
    *,
    min_delta_norm=0.15,
    max_stride=8,
    device=None,
):
    terminated = False
    truncated = False
    z_next = None
    rgb_next = None
    used_stride = 0

    for _ in range(max_stride):
        _, _, terminated, truncated, _ = env.step(a_real_np)
        used_stride += 1

        rgb_next = env.render()
        x_next = preprocess_rgb(rgb_next).to(device)

        with torch.no_grad():
            _, z_next_batch = ae_model(x_next)

        z_next = z_next_batch.squeeze(0)
        delta_norm = torch.linalg.vector_norm(z_next - z_t).item()

        if delta_norm >= min_delta_norm or terminated or truncated:
            break

    return z_next, rgb_next, used_stride, terminated, truncated

def build_machine_probe_dataset(
    n_episodes=100,
    max_steps=300,
    hist_len=8,
    real_a_dim=2,
    min_delta_norm=0.15,
    max_stride=8,
    device=None,
):
    if device is None:
        device = _device()

    MACHINE_FORWARD_RUN_DIR.mkdir(parents=True, exist_ok=True)

    ae_model = load_ae_model(device)
    ae_model.eval()

    env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")

    z_hist_list = []
    a_list = []
    dz_list = []
    stride_list = []

    for ep in range(n_episodes):
        env.reset()
        z_buffer = []

        rgb = env.render()
        x = preprocess_rgb(rgb).to(device)

        with torch.no_grad():
            _, z = ae_model(x)

        z_buffer.append(z.squeeze(0).cpu())

        for _ in range(max_steps):
            a_real_np = sample_random_action(real_a_dim)
            z_t = z_buffer[-1].to(device)
            z_next, _, used_stride, terminated, truncated = rollout_until_latent_motion(
                env,
                ae_model,
                z_t,
                a_real_np,
                min_delta_norm=min_delta_norm,
                max_stride=max_stride,
                device=device,
            )
            dz = z_next - z_t

            hist = z_buffer
            if len(hist) < hist_len:
                hist = [hist[0]] * (hist_len - len(hist)) + hist
            else:
                hist = hist[-hist_len:]

            z_hist_list.append(torch.stack(hist, dim=0).cpu())
            a_list.append(torch.from_numpy(a_real_np))
            dz_list.append(dz.cpu())
            stride_list.append(used_stride)

            z_buffer.append(z_next.squeeze(0).cpu())
            if len(z_buffer) > hist_len:
                z_buffer = z_buffer[-hist_len:]

            if terminated or truncated:
                break

        print(f"[machine-dataset] episode {ep + 1:03d}/{n_episodes} | samples {len(z_hist_list)}")

    env.close()

    probe_data = {
        "z_hist": torch.stack(z_hist_list).float(),
        "a_real": torch.stack(a_list).float(),
        "dz": torch.stack(dz_list).float(),
        "stride": torch.tensor(stride_list, dtype=torch.int32),
        "hist_len": hist_len,
        "real_a_dim": real_a_dim,
        "min_delta_norm": min_delta_norm,
        "max_stride": max_stride,
    }

    torch.save(probe_data, MACHINE_PROBE_PT)
    print(f"[machine-dataset] z_hist -> {probe_data['z_hist'].shape}")
    print(f"[machine-dataset] a_real -> {probe_data['a_real'].shape}")
    print(f"[machine-dataset] dz     -> {probe_data['dz'].shape}")
    print(f"[machine-dataset] stride -> {probe_data['stride'].float().mean().item():.2f} mean")
    print(f"[machine-dataset] saved  -> {MACHINE_PROBE_PT}")
    return MACHINE_PROBE_PT


if __name__ == "__main__":
    build_machine_probe_dataset(
        n_episodes=MACHINE_FORWARD_CONFIG["n_episodes"],
        max_steps=MACHINE_FORWARD_CONFIG["max_steps"],
        hist_len=MACHINE_FORWARD_CONFIG["hist_len"],
        real_a_dim=MACHINE_FORWARD_CONFIG["real_a_dim"],
        min_delta_norm=MACHINE_FORWARD_CONFIG["min_delta_norm"],
        max_stride=MACHINE_FORWARD_CONFIG["max_stride"],
    )
