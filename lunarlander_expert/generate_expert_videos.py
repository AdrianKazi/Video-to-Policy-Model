import argparse
import csv
import os

import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import heuristic
from gymnasium.wrappers import RecordVideo


def generate_expert_videos(num_episodes: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=output_dir,
        episode_trigger=lambda episode_id: episode_id < num_episodes,
        name_prefix="lunar_lander_expert",
    )

    rewards_path = os.path.join(output_dir, "rewards.csv")

    with open(rewards_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "steps"])

        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0.0
            steps = 0

            while not done:
                action = heuristic(env.unwrapped, state)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                steps += 1

            writer.writerow([episode, episode_reward, steps])
            print(f"episode {episode:03d} | reward {episode_reward:9.3f} | steps {steps}")

    env.close()
    print(f"saved videos to: {output_dir}")
    print(f"saved rewards to: {rewards_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default="videos/expert_300")
    args = parser.parse_args()

    generate_expert_videos(
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
    )
