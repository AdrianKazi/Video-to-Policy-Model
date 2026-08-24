import numpy as np
import pandas as pd

from src.pretraining.config import CONTROL_POINT_NAMES, N_BACKGROUND_POINTS


def reduce_control_points(cp_df):
    reduced = cp_df[
        [
            "episode_id",
            "frame",
            "track_id",
            "point_name",
            "x",
            "y",
            "box_x1",
            "box_y1",
            "box_x2",
            "box_y2",
        ]
    ].copy()

    reduced["point_name"] = pd.Categorical(
        reduced["point_name"],
        categories=CONTROL_POINT_NAMES,
        ordered=True,
    )

    return reduced.sort_values(
        ["episode_id", "track_id", "frame", "point_name"]
    ).reset_index(drop=True)


def reduce_background_points(bckg_df):
    reduced = bckg_df[
        [
            "episode_id",
            "frame",
            "background_point_id",
            "x",
            "y",
            "dx",
            "dy",
            "speed",
            "valid",
            "prev_valid",
        ]
    ].copy()

    return reduced.sort_values(
        ["episode_id", "frame", "background_point_id"]
    ).reset_index(drop=True)


def select_fixed_background_points(bckg_df, n_points=N_BACKGROUND_POINTS):
    return (
        bckg_df.sort_values(["episode_id", "frame", "background_point_id"])
        .groupby(["episode_id", "frame"], group_keys=False)
        .apply(
            lambda frame_df: frame_df.sample(
                n=min(n_points, len(frame_df)),
                random_state=42,
            )
        )
        .sort_values(["episode_id", "frame", "background_point_id"])
        .reset_index(drop=True)
    )


def build_action_embeddings(cp_df, bckg_df):
    embeddings = {}

    for episode_id in sorted(cp_df["episode_id"].unique()):
        common_frames = sorted(
            set(cp_df.loc[cp_df["episode_id"] == episode_id, "frame"].unique())
            & set(bckg_df.loc[bckg_df["episode_id"] == episode_id, "frame"].unique())
        )

        episode_embeddings = {}

        for frame in common_frames:
            frame_bckg = bckg_df[
                (bckg_df["episode_id"] == episode_id)
                & (bckg_df["frame"] == frame)
            ].sort_values("background_point_id")

            X_bckg = frame_bckg[["x"]].to_numpy()
            Y_bckg = frame_bckg[["y"]].to_numpy()

            frame_embeddings = {}

            for track_id, frame_cp in cp_df[
                (cp_df["episode_id"] == episode_id)
                & (cp_df["frame"] == frame)
            ].groupby("track_id", sort=False):
                frame_cp = frame_cp.sort_values("point_name")

                X_cp = frame_cp[["x"]].to_numpy()
                Y_cp = frame_cp[["y"]].to_numpy()

                A_x = X_cp @ X_bckg.T
                A_y = Y_cp @ Y_bckg.T
                A = np.stack([A_x, A_y], axis=-1)

                A_min = A.min()
                A_max = A.max()
                A_norm = (A - A_min) / (A_max - A_min + 1e-8)

                frame_embeddings[track_id] = {
                    "A": A,
                    "A_norm": A_norm,
                    "token_flat": A.reshape(-1).astype(np.float32),
                    "token_norm_flat": A_norm.reshape(-1).astype(np.float32),
                    "control_point_dim": len(frame_cp),
                    "background_dim": len(frame_bckg),
                    "relation_channels": A.shape[-1],
                    "flattened_dim": A.reshape(-1).shape[0],
                    "A_min": A_min,
                    "A_max": A_max,
                }

            episode_embeddings[frame] = frame_embeddings

        embeddings[episode_id] = episode_embeddings

    return embeddings


def save_tokens_per_episode(embeddings, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    for episode_id, episode_embeddings in embeddings.items():
        rows = []

        for frame, frame_embeddings in episode_embeddings.items():
            for track_id, embedding in frame_embeddings.items():
                rows.append(
                    {
                        "frame": frame,
                        "track_id": track_id,
                        "token": embedding["token_norm_flat"],
                    }
                )

        if not rows:
            continue

        frames = np.asarray([row["frame"] for row in rows], dtype=np.int64)
        track_ids = np.asarray([row["track_id"] for row in rows], dtype=np.int64)
        tokens = np.stack([row["token"] for row in rows]).astype(np.float32)

        output_path = output_dir / f"lunar_lander_action_tokens_episode_{episode_id:04d}.npz"
        np.savez_compressed(output_path, frames=frames, track_ids=track_ids, tokens=tokens)
        output_paths.append(output_path)

    return output_paths

