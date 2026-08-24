# env/env.py

import gymnasium as gym


def make_env(render_mode=None):
    return gym.make("LunarLanderContinuous-v3", render_mode=render_mode)


def get_env_dims(env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    return state_dim, action_dim, max_action