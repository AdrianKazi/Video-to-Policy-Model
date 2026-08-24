from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LUNAR_EXPERT_VIDEO_DIR = PROJECT_ROOT / "lunarlander_expert" / "videos" / "expert_300"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PRETRAIN_DIR = ARTIFACTS_DIR / "pretrain"

PLOTS_DIR = PRETRAIN_DIR / "plots"
MODELS_DIR = PRETRAIN_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
CONTROL_POINTS_DIR = PRETRAIN_DIR / "control_points"
BACKGROUND_POINTS_DIR = PRETRAIN_DIR / "background_points"
ACTION_EMBEDDINGS_DIR = PRETRAIN_DIR / "action_embeddings"
TOKENS_DIR = PRETRAIN_DIR / "tokens"
LOGS_DIR = PRETRAIN_DIR / "logs"
REVIEWS_DIR = PRETRAIN_DIR / "reviews"

N_PRETRAIN_VIDEOS = 20
MAX_FRAMES_PER_EPISODE = 80
N_BACKGROUND_POINTS = 100

START_FROM = "start"
NUM_TRAINING_STEPS = 100
CHECKPOINT_EVERY = 500

SEQ_LEN = 4
BATCH_SIZE = 8
EMBED_DIM = 256
N_HEADS = 4
N_BLOCKS = 4
DROPOUT = 0.1
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01

BACKGROUND_GRID_STEP = 32
BACKGROUND_DILATE_PX = 7

CONTROL_POINT_NAMES = [
    "centroid",
    "axis_front",
    "axis_back",
    "axis_left",
    "axis_right",
    "contact_low",
]


def ensure_dirs():
    for path in [
        PRETRAIN_DIR,
        PLOTS_DIR,
        MODELS_DIR,
        CHECKPOINTS_DIR,
        CONTROL_POINTS_DIR,
        BACKGROUND_POINTS_DIR,
        ACTION_EMBEDDINGS_DIR,
        TOKENS_DIR,
        LOGS_DIR,
        REVIEWS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def lunar_video_paths(n_videos=N_PRETRAIN_VIDEOS):
    return sorted(LUNAR_EXPERT_VIDEO_DIR.glob("*.mp4"))[:n_videos]
