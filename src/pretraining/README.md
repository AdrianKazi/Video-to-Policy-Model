# Action Embedding Pretraining

Modular Lunar Lander pretraining pipeline for action embeddings.

Run the full pipeline from the project root:

```bash
.venv/bin/python -m src.pretraining.run_pretraining
```

This runs the full pipeline:

```text
1. Build control points from Lunar Lander expert videos
2. Build background optical-flow points per episode
3. Build normalized action embeddings and token files
4. Train the action embedding transformer
5. Save model, config, logs, and plots
```

Input videos are expected here:

```text
lunarlander_expert/videos/expert_300/*.mp4
```

Artifacts are written under:

```text
artifacts/pretrain/
  control_points/
  background_points/
  action_embeddings/
  tokens/
  models/
  plots/
  logs/
  reviews/
```

Main saved outputs:

```text
artifacts/pretrain/control_points/lunar_lander_control_points.csv
artifacts/pretrain/background_points/lunar_lander_background_points_episode_XXXX.csv
artifacts/pretrain/action_embeddings/lunar_lander_action_embeddings.pkl
artifacts/pretrain/action_embeddings/lunar_lander_action_embeddings_summary.json
artifacts/pretrain/tokens/lunar_lander_action_tokens_episode_XXXX.npz
artifacts/pretrain/models/action_embedding_transformer.pt
artifacts/pretrain/models/action_embedding_transformer_config.json
artifacts/pretrain/logs/action_embedding_transformer_history.json
artifacts/pretrain/plots/action_embedding_transformer_loss.png
artifacts/pretrain/plots/action_embedding_true_vs_pred.png
```

Individual stages can still be run directly:

```bash
.venv/bin/python -m src.pretraining.run_control_points
.venv/bin/python -m src.pretraining.run_background_points
.venv/bin/python -m src.pretraining.run_action_embeddings
.venv/bin/python -m src.pretraining.run_pretrain_transformer
```

Current config lives in:

```text
src/pretraining/config.py
```

Important config values:

```text
N_PRETRAIN_VIDEOS
MAX_FRAMES_PER_EPISODE
N_BACKGROUND_POINTS
BACKGROUND_GRID_STEP
BACKGROUND_DILATE_PX
```
