import time

from src.pretraining.config import (
    ACTION_EMBEDDINGS_DIR,
    BACKGROUND_POINTS_DIR,
    CONTROL_POINTS_DIR,
    LOGS_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    TOKENS_DIR,
    ensure_dirs,
    lunar_video_paths,
)
from src.pretraining.run_action_embeddings import main as run_action_embeddings
from src.pretraining.run_background_points import main as run_background_points
from src.pretraining.run_control_points import main as run_control_points
from src.pretraining.run_pretrain_transformer import main as run_pretrain_transformer


def run_stage(name, fn):
    print(f"\n=== {name} ===")
    start = time.time()
    fn()
    elapsed = time.time() - start
    print(f"=== {name} done in {elapsed:.1f}s ===")


def main():
    ensure_dirs()

    videos = lunar_video_paths()
    if len(videos) == 0:
        raise FileNotFoundError(
            "No Lunar Lander expert videos found. "
            "Expected .mp4 files in lunarlander_expert/videos/expert_300."
        )

    print("pretraining videos:", len(videos))
    print("control points dir:", CONTROL_POINTS_DIR)
    print("background points dir:", BACKGROUND_POINTS_DIR)
    print("action embeddings dir:", ACTION_EMBEDDINGS_DIR)
    print("tokens dir:", TOKENS_DIR)
    print("models dir:", MODELS_DIR)
    print("logs dir:", LOGS_DIR)
    print("plots dir:", PLOTS_DIR)

    run_stage("control points", run_control_points)
    run_stage("background points", run_background_points)
    run_stage("action embeddings", run_action_embeddings)
    run_stage("transformer pretraining", run_pretrain_transformer)

    print("\npretraining complete")
    print("model:", MODELS_DIR / "action_embedding_transformer.pt")
    print("embeddings:", ACTION_EMBEDDINGS_DIR / "lunar_lander_action_embeddings.pkl")
    print("tokens:", TOKENS_DIR)
    print("plots:", PLOTS_DIR)


if __name__ == "__main__":
    main()
