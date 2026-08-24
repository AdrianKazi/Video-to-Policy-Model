# test/test.py

import os
import torch
import numpy as np

from gymnasium.wrappers import RecordVideo

from env.env import make_env, get_env_dims
from agents.td3 import TD3_Agent
from config.config import *
from tracking.mlflow_utils import setup_mlflow, start_run
import mlflow


def test():

    VIDEO_PATH = "videos"
    MODEL_PATH = "models_saved/actor.pth"

    os.makedirs(VIDEO_PATH, exist_ok=True)

    env = make_env(render_mode="rgb_array")
    env = RecordVideo(env, VIDEO_PATH, episode_trigger=lambda e: True)

    state_dim, action_dim, max_action = get_env_dims(env)
    agent = TD3_Agent(state_dim, action_dim, max_action)

    agent.actor.load_state_dict(torch.load(MODEL_PATH))
    agent.actor.eval()

    setup_mlflow()

    with start_run("test"):

        for episode in range(NUM_TEST_EPISODES):

            state, _ = env.reset()
            ep_reward = 0
            done = False

            while not done:

                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = agent.actor(state_tensor).detach().cpu().numpy()[0]

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                state = next_state
                ep_reward += reward

            print(f"Test episode {episode} | reward: {ep_reward}")
            mlflow.log_metric("test_reward", ep_reward, step=episode)

    env.close()