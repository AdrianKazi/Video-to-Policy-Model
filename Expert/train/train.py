# train/train.py

import os
import numpy as np
import torch

from env.env import make_env, get_env_dims
from agents.td3 import TD3_Agent
from config.config import *
from tracking.mlflow_utils import setup_mlflow, start_run
import mlflow


def train():

    os.makedirs(MODEL_DIR, exist_ok=True)

    env = make_env()
    state_dim, action_dim, max_action = get_env_dims(env)

    agent = TD3_Agent(state_dim, action_dim, max_action)

    setup_mlflow()

    reward_history = []

    with start_run("train"):

        for episode in range(MAX_EPISODES):

            state, _ = env.reset()
            ep_reward = 0

            for step in range(MAX_STEPS):

                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.store(state, action, reward, next_state, done)
                agent.train()

                state = next_state
                ep_reward += reward

                if done:
                    break

            reward_history.append(ep_reward)

            avg10 = np.mean(reward_history[-10:])

            print(f"Ep {episode} | Reward {ep_reward}")

            mlflow.log_metric("reward", ep_reward, step=episode)
            mlflow.log_metric("avg10", avg10, step=episode)

            if len(reward_history) >= 100 and np.mean(reward_history[-100:]) >= TARGET_LAST_100_MEAN_REWARD:
                print(f"Early stop at ep {episode} | avg100 = {np.mean(reward_history[-100:]):.2f}")
                break

        path = f"{MODEL_DIR}/actor.pth"
        torch.save(agent.actor.state_dict(), path)
        mlflow.log_artifact(path)

    env.close()