import pickle
import json

import pandas as pd

from src.pretraining.action_embeddings import (
    build_action_embeddings,
    reduce_background_points,
    reduce_control_points,
    save_tokens_per_episode,
    select_fixed_background_points,
)
from src.pretraining.config import (
    ACTION_EMBEDDINGS_DIR,
    BACKGROUND_POINTS_DIR,
    CONTROL_POINTS_DIR,
    TOKENS_DIR,
    ensure_dirs,
)


def main():
    ensure_dirs()
    cp_path = CONTROL_POINTS_DIR / "lunar_lander_control_points.csv"
    background_paths = sorted(BACKGROUND_POINTS_DIR.glob("lunar_lander_background_points_episode_*.csv"))

    cp_df = pd.read_csv(cp_path)
    bckg_df = pd.concat([pd.read_csv(path) for path in background_paths], ignore_index=True)

    cp_reduced = reduce_control_points(cp_df)
    bckg_reduced = reduce_background_points(bckg_df)
    bckg_selected = select_fixed_background_points(bckg_reduced)
    embeddings = build_action_embeddings(cp_reduced, bckg_selected)

    embeddings_path = ACTION_EMBEDDINGS_DIR / "lunar_lander_action_embeddings.pkl"
    with embeddings_path.open("wb") as f:
        pickle.dump(embeddings, f)

    token_paths = save_tokens_per_episode(embeddings, TOKENS_DIR)
    first_episode_id = next(iter(embeddings))
    first_frame = next(iter(embeddings[first_episode_id]))
    first_track_id = next(iter(embeddings[first_episode_id][first_frame]))
    first_embedding = embeddings[first_episode_id][first_frame][first_track_id]

    summary = {
        "episodes": len(embeddings),
        "token_files": len(token_paths),
        "control_point_dim": int(first_embedding["control_point_dim"]),
        "background_dim": int(first_embedding["background_dim"]),
        "relation_channels": int(first_embedding["relation_channels"]),
        "flattened_dim": int(first_embedding["flattened_dim"]),
    }

    summary_path = ACTION_EMBEDDINGS_DIR / "lunar_lander_action_embeddings_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("saved:", embeddings_path)
    print("saved token files:", len(token_paths))
    print("saved summary:", summary_path)
    print("episodes:", len(embeddings))


if __name__ == "__main__":
    main()
