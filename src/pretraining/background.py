import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.pretraining.config import (
    BACKGROUND_DILATE_PX,
    BACKGROUND_GRID_STEP,
    BACKGROUND_POINTS_DIR,
    MAX_FRAMES_PER_EPISODE,
)
from src.pretraining.control_points import lunar_lander_agent_records
from src.pretraining.video import read_video_even


def surface_grid_background_mask(frame, dilate_px=BACKGROUND_DILATE_PX):
    records = lunar_lander_agent_records(frame)
    active_mask = np.zeros(frame.shape[:2], dtype=bool)

    for record in records:
        active_mask |= record["mask"]

    if dilate_px > 0 and active_mask.any():
        kernel = np.ones((dilate_px, dilate_px), dtype=np.uint8)
        active_mask = cv2.dilate(
            active_mask.astype(np.uint8),
            kernel,
            iterations=1,
        ).astype(bool)

    return ~active_mask, active_mask, records


def make_surface_grid_points(mask, step=BACKGROUND_GRID_STEP, margin=12):
    height, width = mask.shape
    points = []

    for y in range(margin, height - margin, step):
        for x in range(margin, width - margin, step):
            if mask[y, x]:
                points.append((float(x), float(y)))

    return np.asarray(points, dtype=np.float32)


def add_missing_surface_grid_points(
    background_mask,
    points,
    valid,
    step=BACKGROUND_GRID_STEP,
    min_distance_ratio=0.72,
):
    candidate_points = make_surface_grid_points(background_mask, step=step)

    if len(candidate_points) == 0:
        return points.reshape(-1, 2), valid.astype(bool)

    points = points.reshape(-1, 2)
    valid = valid.astype(bool)
    existing_points = points[valid] if len(points) else np.empty((0, 2), dtype=np.float32)
    new_points = []
    min_dist_sq = float(step * min_distance_ratio) ** 2

    for candidate in candidate_points:
        if len(existing_points):
            nearest_existing = np.min(np.sum((existing_points - candidate) ** 2, axis=1))
            if nearest_existing < min_dist_sq:
                continue

        if new_points:
            added_points = np.asarray(new_points, dtype=np.float32)
            nearest_added = np.min(np.sum((added_points - candidate) ** 2, axis=1))
            if nearest_added < min_dist_sq:
                continue

        new_points.append(candidate)

    if not new_points:
        return points, valid

    new_points = np.asarray(new_points, dtype=np.float32)
    points = np.vstack([points, new_points]) if len(points) else new_points
    valid = np.concatenate([valid, np.ones(len(new_points), dtype=bool)])

    return points, valid


def track_surface_grid_points(frames_batch, source_indices, step=BACKGROUND_GRID_STEP):
    source_indices = np.asarray(source_indices, dtype=int)

    first_frame = frames_batch[0]
    background_mask, active_mask, records = surface_grid_background_mask(first_frame)
    points = make_surface_grid_points(background_mask, step=step)
    valid = np.ones(len(points), dtype=bool)

    tracks = [
        {
            "frame": int(source_indices[0]),
            "points": points.copy(),
            "valid": valid.copy(),
            "prev_valid": np.zeros(len(points), dtype=bool),
            "dx": np.zeros(len(points), dtype=np.float32),
            "dy": np.zeros(len(points), dtype=np.float32),
            "speed": np.zeros(len(points), dtype=np.float32),
        }
    ]

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
    prev_points = points.reshape(-1, 1, 2)
    prev_valid = valid.copy()

    for local_pos, frame in enumerate(frames_batch[1:], start=1):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        background_mask, _, _ = surface_grid_background_mask(frame)

        if len(prev_points) == 0:
            next_points = prev_points.copy()
            valid = np.zeros(0, dtype=bool)
        else:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                prev_points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )

            if next_points is None or status is None:
                next_points = prev_points.copy()
                valid = np.zeros(len(prev_points), dtype=bool)
            else:
                next_xy = next_points.reshape(-1, 2)
                height, width = background_mask.shape
                inside_frame = (
                    (next_xy[:, 0] >= 0)
                    & (next_xy[:, 0] < width)
                    & (next_xy[:, 1] >= 0)
                    & (next_xy[:, 1] < height)
                )

                on_background = np.zeros(len(next_xy), dtype=bool)
                rounded = np.floor(next_xy[inside_frame]).astype(int)
                rounded[:, 0] = np.clip(rounded[:, 0], 0, width - 1)
                rounded[:, 1] = np.clip(rounded[:, 1], 0, height - 1)
                on_background[inside_frame] = background_mask[rounded[:, 1], rounded[:, 0]]

                valid = (
                    prev_valid
                    & (status.reshape(-1) == 1)
                    & inside_frame
                    & on_background
                )

        tracked_xy = next_points.reshape(-1, 2)
        prev_xy = prev_points.reshape(-1, 2)
        dx = tracked_xy[:, 0] - prev_xy[:, 0] if len(tracked_xy) else np.zeros(0, dtype=np.float32)
        dy = tracked_xy[:, 1] - prev_xy[:, 1] if len(tracked_xy) else np.zeros(0, dtype=np.float32)
        speed = np.sqrt(dx ** 2 + dy ** 2)

        tracked_xy, valid = add_missing_surface_grid_points(
            background_mask,
            tracked_xy,
            valid,
            step=step,
        )

        added_count = len(tracked_xy) - len(dx)
        if added_count > 0:
            dx = np.concatenate([dx, np.zeros(added_count, dtype=np.float32)])
            dy = np.concatenate([dy, np.zeros(added_count, dtype=np.float32)])
            speed = np.concatenate([speed, np.zeros(added_count, dtype=np.float32)])
            prev_valid = np.concatenate([prev_valid, np.zeros(added_count, dtype=bool)])

        tracks.append(
            {
                "frame": int(source_indices[local_pos]),
                "points": tracked_xy.copy(),
                "valid": valid.copy(),
                "prev_valid": prev_valid.copy(),
                "dx": dx.astype(np.float32),
                "dy": dy.astype(np.float32),
                "speed": speed.astype(np.float32),
            }
        )

        prev_gray = gray
        prev_points = tracked_xy.reshape(-1, 1, 2)
        prev_valid = valid.copy()

    return tracks


def tracks_to_background_dataframe(tracks, episode_id, video_path):
    rows = []

    for track in tracks:
        points = track["points"]
        valid = track["valid"]
        prev_valid = track["prev_valid"]
        dx = track["dx"]
        dy = track["dy"]
        speed = track["speed"]

        for background_point_id, (x, y) in enumerate(points):
            rows.append(
                {
                    "episode_id": int(episode_id),
                    "video_path": str(video_path),
                    "frame": int(track["frame"]),
                    "background_point_id": int(background_point_id),
                    "x": float(x),
                    "y": float(y),
                    "dx": float(dx[background_point_id]),
                    "dy": float(dy[background_point_id]),
                    "speed": float(speed[background_point_id]),
                    "valid": bool(valid[background_point_id]),
                    "prev_valid": bool(prev_valid[background_point_id]),
                }
            )

    return pd.DataFrame(rows)


def build_background_points_per_episode(video_paths, max_frames=MAX_FRAMES_PER_EPISODE):
    output_paths = []

    for episode_id, video_path in enumerate(tqdm(video_paths, desc="background points")):
        frames, frame_indices = read_video_even(
            video_path,
            max_frames=max_frames,
            return_indices=True,
        )
        tracks = track_surface_grid_points(frames, frame_indices)
        episode_df = tracks_to_background_dataframe(tracks, episode_id, video_path)
        episode_df = episode_df.sort_values(
            ["episode_id", "frame", "background_point_id"]
        ).reset_index(drop=True)

        output_path = BACKGROUND_POINTS_DIR / f"lunar_lander_background_points_episode_{episode_id:04d}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        episode_df.to_csv(output_path, index=False)
        output_paths.append(output_path)

    return output_paths

