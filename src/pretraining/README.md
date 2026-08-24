# Action Embedding Pretraining

Modular version of `Action_inference_Pretrain.ipynb`.

Run from project root:

```bash
python -m src.pretraining.run_control_points
python -m src.pretraining.run_background_points
python -m src.pretraining.run_action_embeddings
python -m src.pretraining.run_pretrain_transformer
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

Current config lives in:

```text
src/pretraining/config.py
```

