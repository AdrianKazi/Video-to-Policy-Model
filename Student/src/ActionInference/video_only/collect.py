"""Record the Gymnasium scripted demonstrator as one MP4 per episode."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import heuristic


def record_episodes(output: Path, count: int, seed: int):
    paths = [output / f"episode-{seed + index:06d}.mp4" for index in range(count)]
    for path in paths:
        if path.exists():
            raise FileExistsError(f"{path} exists. Choose a new seed or output directory.")
    output.mkdir(parents=True, exist_ok=True)
    env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
    try:
        with TemporaryDirectory(dir=output) as temporary:
            for index, path in enumerate(paths):
                observation, _ = env.reset(seed=seed + index)
                first_frame = env.render()
                height, width = first_frame.shape[:2]
                temporary_path = Path(temporary) / path.name
                writer = cv2.VideoWriter(
                    str(temporary_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    env.metadata["render_fps"],
                    (width, height),
                )
                try:
                    if not writer.isOpened():
                        raise OSError(f"Could not start MP4 encoder for {path}")
                    writer.write(cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
                    while True:
                        action = heuristic(env.unwrapped, observation)
                        observation, _, terminated, truncated, _ = env.step(action)
                        writer.write(cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR))
                        if terminated or truncated:
                            break
                finally:
                    writer.release()
                if temporary_path.stat().st_size == 0:
                    raise OSError(f"Empty video for {path}")
                # Publish the completed file without overwriting another collection run.
                path.hardlink_to(temporary_path)
                temporary_path.unlink()
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error("Use at least one episode and a non-negative seed")
    record_episodes(args.output, args.episodes, args.seed)


if __name__ == "__main__":
    main()
