# Action Inference

This branch is the action-embedding pretraining phase of the project. The goal is to infer action from video without starting from explicit action labels. Instead of treating action as a class name such as `left_engine`, `walk`, or `brake`, this project defines action as a structured relation between an object's control points and the background structure around it.

The core pipeline is:

```text
video frame -> object control points -> background points -> CP/background relation -> action embedding token -> transformer pretrain
```

## Core Idea

An action should be visible as a change in how an object moves relative to the background. A lander firing an engine, a car turning, or a person walking all create structured changes between object geometry and the surrounding scene. The project encodes that relation algebraically.

For one object in frame `t`, control points are the compact geometry of the segmentation blob. They include the centroid and selected boundary/control points, so the model sees more than just the object's center.

$$P_t=\begin{bmatrix}p_t^{(1)}\\p_t^{(2)}\\\vdots\\p_t^{(K)}\end{bmatrix}=\begin{bmatrix}x_{cp,t}^{(1)}&y_{cp,t}^{(1)}\\x_{cp,t}^{(2)}&y_{cp,t}^{(2)}\\\vdots&\vdots\\x_{cp,t}^{(K)}&y_{cp,t}^{(K)}\end{bmatrix}\in\mathbb{R}^{K\times2}$$

In the Lunar Lander experiment, `K = 6` control points are used: centroid, axis points, and contact-like points derived from the segmentation blob. The background is represented by optical-flow points. Each background point stores image position and frame-to-frame displacement.

$$B_t=\begin{bmatrix}b_t^{(1)}\\b_t^{(2)}\\\vdots\\b_t^{(N)}\end{bmatrix}=\begin{bmatrix}x_{bckg,t}^{(1)}&y_{bckg,t}^{(1)}&dx_t^{(1)}&dy_t^{(1)}\\x_{bckg,t}^{(2)}&y_{bckg,t}^{(2)}&dx_t^{(2)}&dy_t^{(2)}\\\vdots&\vdots&\vdots&\vdots\\x_{bckg,t}^{(N)}&y_{bckg,t}^{(N)}&dx_t^{(N)}&dy_t^{(N)}\end{bmatrix}\in\mathbb{R}^{N\times4}$$

where `N` is the fixed number of selected background points. In the current pretraining setup, `N = 100`. The scalar optical-flow speed is:

$$s_t^{(j)}=\sqrt{\left(dx_t^{(j)}\right)^2+\left(dy_t^{(j)}\right)^2}$$

The action embedding is built as an outer product between control-point coordinates and background-point coordinates. This creates one relation value for every control-point/background-point pair.

$$A_t^x=\underbrace{\begin{bmatrix}x_{cp,t}^{(1)}\\x_{cp,t}^{(2)}\\\vdots\\x_{cp,t}^{(K)}\end{bmatrix}}_{X_{cp,t}\in\mathbb{R}^{K\times1}}\underbrace{\begin{bmatrix}x_{bckg,t}^{(1)}\\x_{bckg,t}^{(2)}\\\vdots\\x_{bckg,t}^{(N)}\end{bmatrix}^{T}}_{X_{bckg,t}^{T}\in\mathbb{R}^{1\times N}}$$

$$A_t^y=\underbrace{\begin{bmatrix}y_{cp,t}^{(1)}\\y_{cp,t}^{(2)}\\\vdots\\y_{cp,t}^{(K)}\end{bmatrix}}_{Y_{cp,t}\in\mathbb{R}^{K\times1}}\underbrace{\begin{bmatrix}y_{bckg,t}^{(1)}\\y_{bckg,t}^{(2)}\\\vdots\\y_{bckg,t}^{(N)}\end{bmatrix}^{T}}_{Y_{bckg,t}^{T}\in\mathbb{R}^{1\times N}}$$

This is an outer product, not a dot product. It does not collapse the relation to one scalar. It produces two matrices:

$$A_t^x\in\mathbb{R}^{K\times N},\quad A_t^y\in\mathbb{R}^{K\times N}$$

Each entry stores one multiplicative relation:

$$A_t^x(k,j)=x_{cp,t}^{(k)}x_{bckg,t}^{(j)},\quad A_t^y(k,j)=y_{cp,t}^{(k)}y_{bckg,t}^{(j)}$$

The full action embedding stacks the x-relation and y-relation matrices:

$$A_t=\left(A_t^x,A_t^y\right)\in\mathbb{R}^{K\times N\times2}$$

Before the transformer sees the token, the action embedding is normalized and flattened:

$$z_t=\operatorname{flatten}\left(A_t^{norm}\right)\in\mathbb{R}^{D}$$

With `K = 6`, `N = 100`, and two relation channels, one flattened action token has:

$$D=K\cdot N\cdot2=6\cdot100\cdot2=1200$$

## Action Embedding Visuals

The first plot shows the raw ingredients: object control points in red and selected background points in blue. This is the spatial setup before the outer product.

![Action embedding inputs](docs/assets/action_embedding_inputs.png)

The next plot shows the actual action embedding matrices. `A_x` stores CP/background x-relations and `A_y` stores CP/background y-relations. These two matrices are stacked into one action token.

![Action embedding outer product](docs/assets/action_embedding_outer_product.png)

Across time, each frame becomes one action token. The transformer sees a sequence of these tokens exactly like a language model sees a sequence of token embeddings, except here the tokens are motion relations instead of words.

![Action embedding token sequence](docs/assets/action_embedding_token_sequence.png)

The temporal-change plot is the key sanity check. The action embedding does not vanish into a flat latent vector. It changes frame by frame, giving the model a measurable motion signal.

![Action embedding temporal change](docs/assets/action_embedding_temporal_change.png)

## Pretraining

Pretraining learns the temporal dynamics of action embeddings. The model is not yet learning the Lunar Lander action space. It is learning how CP/background relation tokens evolve over time.

The pretraining task is causal next-token prediction, but the tokens are action embeddings instead of word tokens.

$$z_{t+1}^{pred}=T_{\theta}\left(z_{t-L+1},z_{t-L+2},\ldots,z_t\right)$$

where `L` is the sequence length and `T_theta` is the causal transformer.

In words: given a short history of action embeddings, predict the next action embedding.

$$z_{t+1}^{true}=\operatorname{flatten}\left(A_{t+1}^{norm}\right)$$

The loss compares the predicted next CP/background relation token with the true next token computed directly from the next video frame.

$$\mathcal{L}_{pretrain}=\operatorname{MSE}\left(z_{t+1}^{pred},z_{t+1}^{true}\right)$$

This is the useful research result of the branch: the transformer can learn motion-token dynamics from the action embedding representation.

![Pretrain transformer loss](docs/assets/pretrain_transformer_loss.png)

The flattened true-vs-predicted embedding plot shows that the transformer is not only predicting a mean vector. It is learning part of the high-dimensional relation structure.

![Pretrain true vs predicted embedding](docs/assets/pretrain_true_vs_pred.png)

The spatial prediction view reconstructs an approximate CP/background view from the predicted token. It is not a perfect inverse, but it confirms that the predicted token stays close to the true next relation.

![Pretrain spatial prediction](docs/assets/pretrain_spatial_prediction.png)

## Why This Matters

The earlier LSTM + autoencoder path could predict future encoded frames and reconstruct decoded frames, but frame-level latent states did not contain enough explicit action information. The action often vanished into generic visual prediction.

The CP/background relation changes the target. The model no longer predicts pixels first. It predicts structured motion:

```text
object control points relative to selected background points
```

That makes the representation closer to what an action actually is: a controlled change in object-background relation.

## Fine-Tuning Status

The unresolved problem is action grounding: mapping a learned action embedding into an executable action interface.

For Lunar Lander, the action space is:

```text
0: noop
1: left_engine
2: main_engine
3: right_engine
```

The tested fine-tune direction was an Action Inference Net:

```text
predicted action embedding -> action logits
```

Because there are no direct expert action labels in the video-only setting, we tested a re-environment benchmark. For each candidate action, the environment is replayed, the candidate action is applied, the resulting frame is encoded back into CP/background space, and the candidate whose consequence is closest to the reference relation is treated as the benchmark action.

This produced signal, but it also exposed the main bottleneck: many candidate actions create very similar short-horizon CP/background deltas. The benchmark often has low margin, so the best action is weakly identified.

![AIN fine-tune loss and benchmark match](docs/assets/ain_reenv_loss_benchmark_match.png)

![AIN action distribution](docs/assets/ain_action_distribution.png)

![AIN confusion matrix](docs/assets/ain_confusion_matrix.png)

The candidate margin plot shows the problem directly: most candidate losses are extremely close. This means the action-grounding label is often low-confidence even when the pretraining representation is useful.

![AIN candidate loss margin](docs/assets/ain_candidate_loss_margin.png)

## Current Conclusion

The action embedding pretrain is promising. The action-space fine-tune is not solved yet.

What works:

- Lunar Lander control points are extracted from segmentation geometry.
- Background points are tracked and reduced to fixed `N = 100`.
- CP/background outer products produce a stable action token of dimension `1200`.
- A causal transformer learns to predict the next action embedding.
- The representation carries temporal motion signal.

What remains open:

- Mapping action embeddings to executable action spaces.
- Increasing action separation when candidate actions have very similar short-horizon effects.
- Adding stronger spatial structure instead of flattening too early.
- Testing pretrained visual or semantic models as a bridge from motion embedding to action concept.

## Next Research Direction

The next branch should focus on action-space grounding, not rebuilding object detection.

Practical next steps:

1. Keep the CP/background action embedding pretrain.
2. Add explicit spatial embeddings for control-point index, background-point index, relation channel, and time.
3. Test longer action-effect horizons and frame stride so action consequences are more separable.
4. Filter or downweight low-margin candidate actions.
5. Try a pretrained semantic bridge: map motion embeddings into concepts like `move_left`, `move_right`, `stabilize`, `fire_main_engine`, then map concepts into an agent-specific action space.
6. For environments with simulators, use teacher-forced re-env grounding instead of closed-loop rollout from an untrained policy.

## Repository Layout

```text
src/
  pretraining/      # modular action-embedding pretraining pipeline

notebooks/
  Action_Inference_Experiments_Object_and_Background_Segmentation_Model_Selection.ipynb
  Action_Inference_Experiments_LunarLander_Single_Episode_Action_Encoder_on_Transformer.ipynb
  Action_inference_Experiments_LunarLander_Pretraining.ipynb
  Action_Inference_Experiments_LunarLander_FineTune.ipynb

artifacts/
  raw_videos/      # source videos used by research and pretraining
  research/        # exploratory notebook outputs
  pretrain/        # generated pretraining datasets, plots, logs, and models

lunarlander_expert/
  # expert policy/video generation code for Lunar Lander
```

Large generated files are intentionally ignored by git. The repository should track code, notebooks, documentation, and lightweight plot assets, not multi-GB CSV/video artifacts.

## Pretraining Pipeline

Run the modular pretraining pipeline from the project root:

```bash
python -m src.pretraining.run_control_points
python -m src.pretraining.run_background_points
python -m src.pretraining.run_action_embeddings
python -m src.pretraining.run_pretrain_transformer
```

The pipeline writes outputs to:

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

## Hypothesis

An action is not just a label. An action is a structured transformation of an object relative to its environment. If that transformation can be encoded as a reusable action embedding, then pretraining can learn motion before any specific robot action space is attached.
