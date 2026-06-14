# config/config.py

# Learning rates
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4

# Replay buffer
REPLAY_BUFFER_SIZE = int(1e6)
BATCH_SIZE = 128
MIN_MEMORY = int(1e4)

# RL params
GAMMA = 0.99
TAU = 0.005

# TD3 specific
POLICY_DELAY = 3
TARGET_POLICY_NOISE = 0.3
TARGET_POLICY_CLIP = 0.5

# Training
MAX_EPISODES = 1000
MAX_STEPS = int(1e3)
TARGET_LAST_100_MEAN_REWARD = 200

# Paths
MODEL_DIR = "./models_saved"
VIDEO_DIR = "./videos"
MLFLOW_URI = "file:./mlruns"
EXPERIMENT_NAME = "TARGET_POLICY_NOISE_"

# Test
NUM_TEST_EPISODES = 500