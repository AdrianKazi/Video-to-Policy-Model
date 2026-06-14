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


def build_machine_probe_dataset(
    n_episodes=100,
    max_steps=300,
    hist_len=8,
    real_a_dim=2,
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
            _, _, terminated, truncated, _ = env.step(a_real_np)

            rgb_next = env.render()
            x_next = preprocess_rgb(rgb_next).to(device)

            with torch.no_grad():
                _, z_next = ae_model(x_next)

            z_t = z_buffer[-1].to(z_next.device)
            dz = z_next.squeeze(0) - z_t

            hist = z_buffer
            if len(hist) < hist_len:
                hist = [hist[0]] * (hist_len - len(hist)) + hist
            else:
                hist = hist[-hist_len:]

            z_hist_list.append(torch.stack(hist, dim=0).cpu())
            a_list.append(torch.from_numpy(a_real_np))
            dz_list.append(dz.cpu())

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
        "hist_len": hist_len,
        "real_a_dim": real_a_dim,
    }

    torch.save(probe_data, MACHINE_PROBE_PT)
    print(f"[machine-dataset] z_hist -> {probe_data['z_hist'].shape}")
    print(f"[machine-dataset] a_real -> {probe_data['a_real'].shape}")
    print(f"[machine-dataset] dz     -> {probe_data['dz'].shape}")
    print(f"[machine-dataset] saved  -> {MACHINE_PROBE_PT}")
    return MACHINE_PROBE_PT


if __name__ == "__main__":
    build_machine_probe_dataset(
        n_episodes=MACHINE_FORWARD_CONFIG["n_episodes"],
        max_steps=MACHINE_FORWARD_CONFIG["max_steps"],
        hist_len=MACHINE_FORWARD_CONFIG["hist_len"],
        real_a_dim=MACHINE_FORWARD_CONFIG["real_a_dim"],
    )
