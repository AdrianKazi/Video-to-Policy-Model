from pathlib import Path

# Resolve Student root from current working directory.
_cwd = Path.cwd().resolve()
STUDENT = None

for p in [_cwd, *_cwd.parents]:
    if (p / "src").is_dir():
        STUDENT = p
        break

if STUDENT is None:
    raise RuntimeError("Could not find Student root")

STUDENT_ROOT = STUDENT
RUNS_ROOT = STUDENT_ROOT / "runs"
FRAMES_ROOT = STUDENT_ROOT / "data" / "frames"

RUNS_ROOT.mkdir(parents=True, exist_ok=True)
FRAMES_ROOT.mkdir(parents=True, exist_ok=True)

###################
### Autoencoder ###
###################

AE_DIR = RUNS_ROOT / "autoencoder"
AE_RUN_DIR = AE_DIR

AE_MODEL_PT = AE_RUN_DIR / "model.pth"
AE_TRAIN_PT = AE_RUN_DIR / "ae_train_dataset.pt"
AE_TEST_PT = AE_RUN_DIR / "ae_test_dataset.pt"

##################
### Sequential ###
##################

SEQ_DIR = RUNS_ROOT / "sequential"
SEQ_RUN_DIR = SEQ_DIR

SEQ_MODEL_PT = SEQ_RUN_DIR / "model.pth"
SEQ_TRAIN_PT = SEQ_RUN_DIR / "seq_train_dataset.pt"
SEQ_TEST_PT = SEQ_RUN_DIR / "seq_test_dataset.pt"

########################
### Action Inference ###
########################

    #####################
    ### Video Learner ###
    #####################

ACTION_INFERENCE_DIR = RUNS_ROOT / "action_inference"
EXPERIMENTS_DIR = ACTION_INFERENCE_DIR / "experiments"
VIDEO_LEARNER_DIR = ACTION_INFERENCE_DIR / "video_learner"
VIDEO_LEARNER_RUN_DIR = VIDEO_LEARNER_DIR

IDM_MODEL_PT = VIDEO_LEARNER_RUN_DIR / "idm_model.pth"
FDM_MODEL_PT = VIDEO_LEARNER_RUN_DIR / "fdm_model.pth"
LOSSES_PT = VIDEO_LEARNER_RUN_DIR / "losses.pt"
METRICS_PT = VIDEO_LEARNER_RUN_DIR / "metrics.pt"
VIDEO_DEBUG_MP4 = VIDEO_LEARNER_RUN_DIR / "video_debug.mp4"

    #######################
    ### Machine Forward ###
    #######################

MACHINE_FORWARD_DIR = ACTION_INFERENCE_DIR / "machine_forward"
MACHINE_FORWARD_RUN_DIR = MACHINE_FORWARD_DIR

MACHINE_PROBE_PT = MACHINE_FORWARD_RUN_DIR / "machine_probe_dataset.pt"
MACHINE_MODEL_PT = MACHINE_FORWARD_RUN_DIR / "machine_forward_model.pth"
MACHINE_LOSSES_PT = MACHINE_FORWARD_RUN_DIR / "losses.pt"
MACHINE_METRICS_PT = MACHINE_FORWARD_RUN_DIR / "metrics.pt"
MACHINE_DEBUG_MP4 = MACHINE_FORWARD_RUN_DIR / "machine_debug.mp4"

    #######################
    ### Runtime Control ###
    #######################

RUNTIME_CONTROL_DIR = ACTION_INFERENCE_DIR / "runtime_control"
RUNTIME_CONTROL_RUN_DIR = RUNTIME_CONTROL_DIR

RUNTIME_ROLLOUT_MP4 = RUNTIME_CONTROL_RUN_DIR / "runtime_lander_rollout.mp4"
RUNTIME_DIAGNOSTICS_PNG = RUNTIME_CONTROL_RUN_DIR / "runtime_diagnostics.png"
RUNTIME_STATS_PT = RUNTIME_CONTROL_RUN_DIR / "runtime_stats.pt"
