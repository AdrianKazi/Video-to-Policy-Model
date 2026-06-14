# DDPG and TD3 for LunarLander

## Introduction

Reinforcement Learning (RL) approach for continuous control tasks. DDPG provides a baseline; TD3 improves on it with twin critics, delayed policy updates, and target smoothing.

Environment: LunarLander-v2. Agent controls a spacecraft for soft, fuel-efficient landing.

## Report

[Read the full report](./RL_LunarLander_Report_AdrianSKazi.pdf)


## Train

```bash
python main.py --mode train
```

## Generate expert demonstrations

```bash
python main.py --mode test
```

Outputs videos to `videos/`.

## Outputs

After training:
- `models_saved/actor.pth`
- `mlruns/`

After testing:
- `videos/rl-video-episode-0.mp4`

## Config

All settings in `config/config.py`.