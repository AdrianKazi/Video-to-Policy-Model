# Video-only latent policy

Can a policy learn actionable structure from passive LunarLander videos? This experiment isolates the video-learning stage and uses the continuous LunarLander setup from the repository. The learner reads one MP4 per episode. The supervised target is a future frame from the same video. The trained policy sees four recent grayscale frames by default and outputs an integer code.

For a history up to frame t, the dynamics network predicts K possible next frames. The model updates the candidate with the smallest motion-weighted error against frame t+1. Then the dynamics network is frozen. A policy learns to predict the winning code using only frame t and its history. This is an ILPO-style best-of-K baseline with a separate policy-training stage. K is a model hyperparameter.

The passive-video objective is invariant under permutation of the codes. Executing them in Box2D would require information about the simulator's actuator interface. These results measure video prediction, code utilization and latent imitation, with no claim about actuator semantics or landing performance.

## Run

From the repository's `Student` directory, use an environment with PyTorch, NumPy and OpenCV.

```bash
python -m pip install torch numpy opencv-python
export PYTHONPATH=src
python -m ActionInference.video_only.train \
  --videos ../Expert/videos \
  --output runs/video_only/expert_seed42
```

If you have no recordings, you can generate an initial dataset with Gymnasium's scripted demonstrator. Gymnasium and Box2D are used only by this recorder, which saves just MP4s. The same training command can read these videos by setting `--videos data/video_only`.

```bash
python -m pip install "gymnasium[box2d]>=1.0,<2"
python -m ActionInference.video_only.collect --output data/video_only --episodes 40
```

For a short run, add `--max-videos 6 --dynamics-epochs 1 --policy-epochs 1 --size 48 --batch-size 16` and use a new output directory. The trainer selects a device using the existing CPU, MPS and CUDA utility. For multiple initialization seeds on the same split, change `--seed` and keep `--split-seed` fixed.

## Read the results

The trainer splits complete videos before building contiguous windows. Use separate original episodes. Multiple edits or copies of the same episode could otherwise appear in both splits.

| Metric | What it measures |
| --- | --- |
| Oracle MSE | Lowest error among the candidates after observing the test target. This is a hindsight diagnostic. |
| Policy MSE | Error for the candidate selected by the policy from current and past frames. |
| Prior MSE | Expected error when choosing a code independently of the history, using frequencies estimated from training. |
| No-code MSE | Error of a separately trained single-output model with the same history and pixel objective. |
| Copy MSE | Error from repeating the last observed frame. |
| Code counts and agreement | Usage of oracle and policy codes, and policy agreement with the oracle alongside a most-frequent-training-code reference. |

Use the policy vs. no-code comparison as the primary check. An oracle advantage with a weak policy can arise when the best-of-K predictor can explain what happened after seeing the future but the policy cannot predict the same codes in advance. Check the code counts for collapse. Holding the code fixed across different states or correlating it with a visible feature would require further analysis to make claims about what it represents.

The free rollouts feed each model its own predictions and report motion-weighted error at steps 1, 5 and 10 by default. The one-step results average transitions within each episode, then average episodes equally. The generated `rollout.png` shows one held-out clip. Rows are increasing horizons, and columns are reference, latent policy, single-output predictor and copy.

A run saves `metrics.json`, `models.pt` and `rollout.png` in the output directory.

## References

- Edwards et al., [Imitating Latent Policies from Observation](https://proceedings.mlr.press/v97/edwards19a.html).
- Schmidt and Jiang, [Learning to Act without Actions](https://proceedings.iclr.cc/paper_files/paper/2024/hash/27985d21f0b751b933d675930aa25022-Abstract-Conference.html).
