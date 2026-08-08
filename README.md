# Conclusion

The model works well as long as the LSTM and autoencoder remain the main components. The main problem is action inference. The model can accurately predict future latent states and decode them back into images, but we are not yet able to reliably infer actions from sequences of latent states or images. The project therefore requires further research to determine how to recover discrete actions directly from visual observations, in a way that would also be intuitive for a human observer.

---

# vid-to-pol

Video-to-policy research pipeline for inferring LunarLander controls from visual transitions.

## Start Here

Open the main notebook:

[vid-to-pol.deconstruction](Student/research/vid-to-pol.deconstruction.ipynb)

It is the compact, self-contained walkthrough of the video-to-policy pipeline. The sections below document the full project structure and command-line workflow.

## Project Summary

This model tries to infer control from video. It watches LunarLander frames, learns the latent movement implied by the video, learns how real environment actions move the same latent space, and then searches for a real action that best matches the video-implied movement.

This project explores whether real control actions can be recovered from video alone by learning a latent transition space.

The motivation is to test a mostly neural alternative to classical reinforcement learning. RL can be expensive, environment-specific, and harder to reuse outside the simulator or task it was trained on. A neural video-to-action pipeline could, in principle, make action inference more reusable by learning from visual transitions and an action space instead of repeatedly optimizing a policy through environment interaction.

At this stage, this project does not claim to solve that problem. I did not find a working fully neural replacement for the control loop here. The value of the project is the experimental evidence, diagnostics, and the failure analysis.

The pipeline is:

1. encode frames into latents with an autoencoder,
2. learn video-implied latent actions and desired latent deltas,
3. learn how real LunarLander actions move the same latent space,
4. search for the real action whose predicted latent delta best matches the video-implied desired delta.

The useful result is not a finished controller. The useful result is the research finding: the individual parts can be made to produce meaningful diagnostics, but the full architecture becomes too large to debug cleanly when started end-to-end.

The main lesson is that this kind of project should start from small controlled experiments:

- one short trajectory,
- one frame pair or one small frame window,
- one latent delta target,
- one action-search problem,
- then scale only after each piece is empirically understood.

Starting from a full multi-model architecture made the failure mode harder to isolate. The current bottleneck is the Runtime Control interface between video-implied latent deltas and real-action Machine Forward deltas, especially after introducing adaptive stride.

## Setup

Python 3.10.11 (pinned in `.venv`).

From `vid-to-pol/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r Expert/requirements.txt
pip install -r Student/requirements.txt
```

## Generate expert videos

Training runs for up to `MAX_EPISODES` episodes (default 1000), stopping early when the last 100-episode mean reward reaches `TARGET_LAST_100_MEAN_REWARD` (default 200). Both are configurable in `config/config.py`. Usually agent achieves 200 mean after ~600 epochs.

```bash
cd Expert
python main.py --mode train
```

Videos saved to `Expert/videos/`. Number of test episodes configurable via `NUM_TEST_EPISODES` in `config/config.py` (default 500).

```bash
python main.py --mode test
```

## Transform expert videos into frames

Reads `.mp4` files from `Expert/videos/` and writes grayscale 84×84 frames to `Student/data/frames/`.

```bash
cd ../Student
export PYTHONPATH=src
python src/videos_to_frames.py
```

## Preprocess Data, Train and Eval Autoencoder

All artefacts, model, dataset, plots will appear in `runs/autoencoder/autoencoder_[timestep]/`.

```bash
python -m Autoencoder dataset
python -m Autoencoder train
python -m Autoencoder eval
```

## Preprocess Data, Train and Eval Sequential (LSTM)

All artefacts, model, dataset, plots will appear in `runs/sequential/sequential_[timestep]/`.

```bash
python -m Sequential dataset
python -m Sequential train
python -m Sequential eval
```

## Action Inference

### Smoke Tests

Smoke test Video Learner forward pass

```bash
export PYTHONPATH=src
python - <<'PY'
from ActionInference.smoke import smoke_video_learner_forward

smoke_video_learner_forward()
PY
```

### Video Learner

Trains `IDMModel + FDMModel`. Parameters setup in `Student/src/ActionInference/video_learner/config.py`.

Learns:
- `IDMModel(c_t) -> latent_a`
- `FDMModel(c_t, latent_a) -> delta_z_pred`

```bash
export PYTHONPATH=src
python -m ActionInference.video_learner.train
python -m ActionInference.video_learner.eval
```

### Machine Forward

Trains `MachineForwardModel`. Parameters setup in `Student/src/ActionInference/machine_forward/config.py`.

Learns:
- `MachineForwardModel(H_t, a_real) -> delta_z_pred`

Builds probe data from LunarLander random real actions:
- `H_t = (z_{t-k+1}, ..., z_t)`
- `a_real = sampled LunarLander action`
- `delta_z = z_{t+s} - z_t`, where adaptive stride `s <= max_stride` advances the environment until latent motion reaches `min_delta_norm` or the episode ends

```bash
export PYTHONPATH=src
python -m ActionInference.machine_forward.build_dataset
python -m ActionInference.machine_forward.train
python -m ActionInference.machine_forward.eval
```

### Runtime Control

Runs planner/search using trained `IDMModel + FDMModel + MachineForwardModel`. Parameters setup in `Student/src/ActionInference/runtime_control/config.py`.

Uses:
- `IDMModel(c_t) -> latent_a`
- `FDMModel(c_t, latent_a) -> delta_z_desired`
- `MachineForwardModel(H_t, a_real) -> delta_z_pred`
- action search: `argmin_a ||delta_z_pred - delta_z_desired||^2`

```bash
export PYTHONPATH=src
python -m ActionInference.runtime_control.eval
```

### Experiment Tracking

Every ActionInference train/eval command writes experiment metadata to:

```txt
runs/action_inference/experiments/
```

Use one shared `ACTION_EXPERIMENT_ID` for the whole research run:

```bash
export ACTION_EXPERIMENT_ID=001_vanilla_baseline
export ACTION_EXPERIMENT_HYPOTHESIS="Baseline full pipeline run used to locate the first broken diagnostic step."
export ACTION_EXPERIMENT_SEED=42
```

Each stage writes:

```txt
<stage>_meta.json
<stage>_config.json
<stage>_metrics.json
git_diff.patch
result_summary.md
```

All runs are indexed in:

```txt
runs/action_inference/experiments/runs_index.csv
```

TensorBoard logs include the same run id:

```txt
runs/action_inference/video_learner/tensorboard/train/001_vanilla_baseline/
runs/action_inference/machine_forward/tensorboard/train/001_vanilla_baseline/
runs/action_inference/runtime_control/tensorboard/eval/001_vanilla_baseline/
```

| Section | Short explanation | Models trained | Output |
|---|-------|---|---|
| Autoencoder | Compresses frame into latent vector | $AE$ | $z_t = AE(o_t)$ |
| Sequential | Encodes latent history into hidden state | $LSTM$ | $h_t = LSTM(H_t)$ |
| Video Learner | Learns latent action and latent change from video context | $IDM$, $FDM$ | $\tilde{a}_t = IDM(c_t)$, $\Delta z_{t+1}^{pred} = FDM(c_t, \tilde{a}_t)$ |
| Machine Forward | Learns how true action changes latent state over adaptive stride | $MachineForwardModel$ | $\Delta z_{t+s}^{pred,machine} = MachineForward(H_t^{machine}, a_t^{true})$ |
| Runtime Control | Searches true action that matches desired latent change | none | $a_t^* = \arg\min_a \Vert \Delta z_{t+s}^{machine}(a) - \Delta z_{t+1}^{desired} \Vert^2 + \lambda \Vert a \Vert^2$ |

## Global Diagnostic Features

| Step Name | Features |
|---|---|
| `global_00_globalfeatures` | * shape<br>* dtype<br>* device<br>* mean / std / min / max<br>* nan / inf count<br>* norm<br>* histogram<br>* sample image / video when decodable |

## Video Learner

Plain words:

- watches video only
- invents a hidden latent action
- learns the latent movement from current frame to next frame

| Description | Step Name | Formula | Step Specific Diagnostic Features |
|---|---|---|---|
| Get latent of each true frame defined in history ($k$) | `video_learner_01_getlatent` | $z_i^{true} = AE(o_i^{true})$ | * AE reconstruction MSE<br>* decoded $z_i^{true}$ frame preview |
| Create history of true latents | `video_learner_02_history` | $H_t = (z_{t-k+1}^{true}, z_{t-k+2}^{true}, ..., z_t^{true})$ | * history length $k$<br>* temporal latent drift<br>* adjacent latent distance |
| Run history of true latents through LSTM and get hidden state | `video_learner_03_lstmhidden` | $h_t = LSTM(H_t)$ | * $h_t$ activation distribution<br>* LSTM next-latent baseline MSE |
| Overlay last $k$ true frames to mimic dynamics | `video_learner_04_overlay` | $M_t = Overlay(o_{t-k+1}^{true}, o_{t-k+2}^{true}, ..., o_t^{true})$ | * overlay preview<br>* overlay intensity range<br>* motion visibility |
| Encode overlay to get latent | `video_learner_05_overlaylatent` | $m_t = AE(M_t)$ | * overlay reconstruction MSE<br>* $m_t$ vs $z_t^{true}$ distance |
| Create representation of current $t$ from current true latent, hidden state of history, and latent of overlay | `video_learner_06_context` | $c_t = concat(z_t^{true}, h_t, m_t)$ | * component norms: $z_t^{true}$ / $h_t$ / $m_t$<br>* concat dimension check |
| Get predicted latent action based on neural net with input of current representation | `video_learner_07_idm_latentaction` | $\tilde{a}_t^{pred} = IDM(c_t)$ | * latent action mean / std / min / max<br>* saturation near -1 or 1<br>* dead dimensions |
| Concatenate current representation with predicted latent action | `video_learner_08_fdm_input` | $x_t^{fdm} = concat(c_t, \tilde{a}_t^{pred})$ | * $c_t$ norm vs $\tilde{a}_t^{pred}$ norm<br>* concat dimension check |
| Research note: delta target works better because it forces FDM to model change, not copy current state | `video_learner_09_delta_research` | $\Delta z_{t+1}^{pred} \text{ beats direct } z_{t+1}^{pred}$ | * delta-target MSE vs direct-next-latent MSE |
| Run concatenation of representation and predicted latent action through neural net and receive predicted latent change | `video_learner_10_fdm_delta` | $\Delta z_{t+1}^{pred} = FDM(x_t^{fdm})$ | * $\Delta z_{t+1}^{pred}$ norm<br>* $\Delta z_{t+1}^{pred}$ vs $\Delta z_{t+1}^{true}$ cosine<br>* delta MSE |
| Build predicted next latent from current true latent and predicted latent change | `video_learner_11_build_nextlatent` | $z_{t+1}^{pred} = z_t^{true} + \Delta z_{t+1}^{pred}$ | * $z_{t+1}^{pred}$ decoded preview<br>* $z_{t+1}^{pred}$ vs $z_{t+1}^{true}$ distance |
| Calculate MSE loss from predicted next latent and true next latent | `video_learner_12_loss` | $L_{video} = \Vert z_{t+1}^{pred} - z_{t+1}^{true} \Vert^2$ | * video MSE<br>* video/LSTM MSE ratio<br>* loss curve |

Diagnostic only:

$$ \Delta z_{t+1}^{true} = z_{t+1} - z_t $$

$$ L_{\Delta} = || \Delta z_{t+1}^{pred} - \Delta z_{t+1}^{true} ||^2 $$

Video Learner itself is not the current failure point.

It learns:

$$ IDM(c_t) \rightarrow \tilde{a}_t $$

$$ FDM(concat(c_t, \tilde{a}_t)) \rightarrow \Delta z_{t+1}^{pred} $$

and predicts the next latent state better than the LSTM baseline:

$$ MSE_{video} < MSE_{lstm} $$

So the model has learned a useful latent transition signal from video context.

The limitation is that:

$$ \tilde{a}_t $$

is still a latent action, not a real LunarLander action. The failure happens later, when Runtime Control tries to convert this desired latent transition into a real action using Machine Forward search.

## Machine Forward

Plain words:

- samples random real LunarLander actions
- applies them in the environment
- measures the latent change they caused
- learns what each real action does in latent space

| Description | Step Name | Formula | Step Specific Diagnostic Features |
|---|---|---|---|
| Select random action, e.g. `[main thrust, side thrust] = [0.8, -0.3]` | `machine_forward_01_sampleaction` | $a_t^{true} = sample_{probe}(\mathcal{A})$ | * action mean / std / min / max<br>* action coverage over $[-1, 1]^2$ |
| Feed the LunarLander environment with current true frame and random true action until enough latent motion or max stride | `machine_forward_02_envstep` | $o_{t+s}^{true} = Env^s(o_t^{true}, a_t^{true})$ | * frame transition preview<br>* episode length<br>* adaptive stride<br>* terminated / truncated rate |
| Create latent for current true frame | `machine_forward_03_currentlatent` | $z_t^{true} = AE(o_t^{true})$ | * AE reconstruction MSE<br>* decoded $z_t^{true}$ preview |
| Build history of true latents | `machine_forward_04_machinehistory` | $H_t^{machine} = (z_{t-k+1}^{true}, z_{t-k+2}^{true}, ..., z_t^{true})$ | * machine history length $k_m$<br>* latent drift over machine window |
| Create latent for reached true frame | `machine_forward_05_nextlatent` | $z_{t+s}^{true} = AE(o_{t+s}^{true})$ | * $z_{t+s}^{true}$ decoded preview<br>* $z_{t+s}^{true}$ vs $z_t^{true}$ distance |
| Get hidden state from LSTM run on history of true latents | `machine_forward_06_encodehistory` | $h_t^{machine} = EncodeHistory(H_t^{machine})$ | * $h_t^{machine}$ activation distribution<br>* history encoder hidden norm |
| Concatenate hidden state of true latent history and true action | `machine_forward_07_machineinput` | $x_t^{machine} = concat(h_t^{machine}, a_t^{true})$ | * $h_t^{machine}$ norm vs $a_t^{true}$ norm<br>* concat dimension check |
| Calculate true latent change caused by true action over adaptive stride | `machine_forward_08_true_delta` | $\Delta z_{t+s}^{true} = z_{t+s}^{true} - z_t^{true}$ | * true delta norm<br>* true delta distribution<br>* zero-delta baseline MSE |
| Get predicted latent change from MachineForward | `machine_forward_09_pred_delta` | $\Delta z_{t+s}^{pred,machine} = MachineForward(x_t^{machine})$ | * pred delta norm<br>* pred vs true delta cosine<br>* action sensitivity |
| Build predicted next latent from current true latent and predicted latent change | `machine_forward_10_build_nextlatent` | $z_{t+s}^{pred,machine} = z_t^{true} + \Delta z_{t+s}^{pred,machine}$ | * decoded predicted next frame<br>* predicted next vs true next latent distance |
| Calculate MSE loss between predicted next latent and true next latent | `machine_forward_11_loss` | $L_{machine} = \Vert z_{t+s}^{pred,machine} - z_{t+s}^{true} \Vert^2$ | * machine MSE<br>* zero baseline MSE<br>* zero/machine ratio<br>* loss curve |
| Research note: delta target works better than direct next-latent target | `machine_forward_12_delta_research` | $\Delta z_{t+1}^{pred,machine} \text{ beats } z_{t+1}^{pred,machine}$ | * delta-target MSE vs direct-next-latent MSE |


Machine Forward itself is not the current failure point.

It learns:

$$ MachineForward(concat(EncodeHistory(H_t^{machine}), a_t^{true})) \rightarrow \Delta z_{t+s}^{pred,machine} $$

and beats the zero baseline:

$$ MSE_{machine} < MSE_{zero} $$

So the model has learned that real LunarLander actions produce predictable movement in latent space over adaptive stride.

The failure happens after Machine Forward, in Runtime Control: the action search uses a stride-trained Machine Forward model while the desired delta still comes from the video-step FDM signal.

## Runtime Control

Plain words:

- takes the latent movement wanted by Video Learner
- tests many real actions with Machine Forward
- picks the real action whose predicted movement best matches the wanted movement

| Description | Step Name | Formula | Step Specific Diagnostic Features |
|---|---|---|---|
| Create latents for all frames in history | `runtime_control_01_getlatent` | $z_i = AE(o_i)$ | * AE reconstruction MSE<br>* decoded latent preview |
| Create longer history of latents for movement detection | `runtime_control_02_video_history` | $H_t = (z_{t-k_v+1}, z_{t-k_v+2}, ..., z_t)$ | * video history length $k_v$<br>* movement-window latent drift |
| Create shorter history of latents for action detection | `runtime_control_03_machine_history` | $H_t^{machine} = (z_{t-k_m+1}, z_{t-k_m+2}, ..., z_t)$ | * machine history length $k_m$<br>* overlap with $H_t$<br>* action-window latent drift |
| Get hidden state of movement history latents | `runtime_control_04_lstmhidden` | $h_t = LSTM(H_t)$ | * $h_t$ activation distribution<br>* hidden norm |
| Create overlay of frames | `runtime_control_05_overlay` | $M_t = Overlay(o_{t-k_v+1}, o_{t-k_v+2}, ..., o_t)$ | * overlay preview<br>* motion visibility<br>* overlay intensity range |
| Get latent of overlay of frames | `runtime_control_06_overlaylatent` | $m_t = AE(M_t)$ | * overlay reconstruction MSE<br>* $m_t$ norm<br>* $m_t$ vs $z_t$ distance |
| Create representation of current state from current latent, hidden state of movement history latents and latent of overlay | `runtime_control_07_context` | $c_t = concat(z_t, h_t, m_t)$ | * component norms: $z_t$ / $h_t$ / $m_t$<br>* concat dimension check |
| Generate action inferred by pretrained IDM from current representation | `runtime_control_08_idm_latentaction` | $\tilde{a}_t = IDM(c_t)$ | * latent action mean / std / min / max<br>* saturation<br>* dead dimensions |
| Concatenate current representation and predicted action | `runtime_control_09_fdm_input` | $x_t^{fdm} = concat(c_t, \tilde{a}_t)$ | * $c_t$ norm vs $\tilde{a}_t$ norm<br>* concat dimension check |
| Infer desired latent change by pretrained FDM on concatenated representation of current state and predicted latent action | `runtime_control_10_desired_delta` | $\Delta z_{t+1}^{desired} = \alpha FDM(x_t^{fdm})$ | * desired delta norm<br>* desired delta distribution<br>* desired delta vs machine reachable delta range |
| Get hidden state of machine history latents | `runtime_control_11_machine_hidden` | $h_t^{machine} = EncodeHistory(H_t^{machine})$ | * $h_t^{machine}$ activation distribution<br>* machine hidden norm |
| Sample many candidate real actions for search | `runtime_control_12_candidate_actions` | $a^{(j)} \sim Uniform([action\_low, action\_high]^2),\ j=1,\dots,N$ | * candidate action coverage<br>* candidate action mean / std<br>* boundary coverage |
| Concatenate machine hidden state with each candidate action | `runtime_control_13_machine_input` | $x_t^{machine}(a^{(j)}) = concat(h_t^{machine}, a^{(j)})$ | * $h_t^{machine}$ norm vs candidate action norm<br>* batch dimension check |
| Predict latent change for each candidate action | `runtime_control_14_machine_candidate_delta` | $\Delta z_{t+s}^{machine}(a^{(j)}) = MachineForward(x_t^{machine}(a^{(j)}))$ | * candidate delta spread<br>* action sensitivity<br>* reachable delta range |
| Choose candidate action whose predicted latent change is closest to desired latent change | `runtime_control_15_chooseaction` | $a_t^* = \arg\min_{a^{(j)}} \Vert \Delta z_{t+s}^{machine}(a^{(j)}) - \Delta z_{t+1}^{desired} \Vert^2 + \lambda \Vert a^{(j)} \Vert^2$ | * selected action distribution<br>* search loss<br>* best-vs-median candidate loss<br>* action L2 penalty contribution |
| Apply selected best action in LunarLander environment | `runtime_control_16_applyaction` | $o_{t+1} = Env(o_t, a_t^*)$ | * rollout reward<br>* episode length<br>* action trace<br>* rollout mp4 |

Runtime Control currently fails at the action-search layer.

Observed rollout:

$$ a_t^* \approx 0 $$

The lander receives almost no thrust and falls down.

Cause:

$$ L_{search} = || \Delta z_{t+s}^{machine}(a) - \Delta z_{t+1}^{desired} ||^2 + \lambda ||a||^2 $$

Machine Forward is trained on adaptive-stride deltas:

$$ \Delta z_{t+s}^{machine} $$

while Runtime Control compares it to the video-step desired delta:

$$ \Delta z_{t+1}^{desired} $$

Current code also applies:

$$ \alpha = 1.5 $$

through `delta_z_scale`, and keeps:

$$ \lambda = 0.05 $$

through `action_l2_penalty`.

So the current failure is a combination of scale mismatch and action penalty: the selected real actions collapse near zero even though Video Learner and Machine Forward are individually healthy.

Current TensorBoard caveat:

`runtime_control_14_machine_candidate_delta` currently logs `search_losses`, not real candidate deltas. Real candidate-delta logging still needs to be added before using that card as evidence.

Immediate fix to test:

$$ \Delta z^{desired}_{runtime} = \beta \Delta z_{t+1}^{desired} $$

with:

$$ \beta \approx \frac{1}{E[s]} $$

and a lower action penalty than:

$$ \lambda = 0.05 $$

while using the configured action range:

$$ a \in [action\_low, action\_high]^2 $$
