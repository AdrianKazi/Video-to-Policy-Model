import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.pretraining.config import CONTROL_POINT_NAMES, CONTROL_POINTS_DIR, MAX_FRAMES_PER_EPISODE
from src.pretraining.video import read_video_even


def mask_to_record(mask, object_id):
    ys, xs = np.nonzero(mask)

    if len(xs) == 0:
        return None

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1

    return {
        "object_id": object_id,
        "mask": mask,
        "box": (x1, y1, x2, y2),
        "area": int(mask.sum()),
        "centroid": (float(xs.mean()), float(ys.mean())),
    }


def largest_connected_component(mask, min_area=30):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    if num_labels <= 1:
        return np.zeros_like(mask, dtype=bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(np.argmax(areas)) + 1

    if areas[best_label - 1] < min_area:
        return np.zeros_like(mask, dtype=bool)

    return labels == best_label


def lunar_lander_agent_records(frame):
    rgb = frame.astype(np.int16)
    brightness = rgb.mean(axis=2)
    colorfulness = np.max(rgb, axis=2) - np.min(rgb, axis=2)

    height, width = frame.shape[:2]
    yy, xx = np.indices((height, width))

    mask = (
        ((brightness > 70) | (colorfulness > 55))
        & (yy < int(height * 0.70))
        & (xx > int(width * 0.20))
        & (xx < int(width * 0.80))
    )

    mask = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)

    mask = largest_connected_component(mask, min_area=25)
    record = mask_to_record(mask, object_id=0)

    if record is None:
        return []

    record.update(
        {
            "class_id": -101,
            "class_name": "lunar_lander_agent",
            "confidence": 1.0,
            "source": "env_specific_color_geometry",
        }
    )

    return [record]


def control_points_from_record(record):
    mask = record["mask"]
    ys, xs = np.nonzero(mask)

    if len(xs) == 0:
        return []

    x1, y1, x2, y2 = record["box"]
    centroid_x, centroid_y = record["centroid"]

    front_idx = np.argmin(xs)
    back_idx = np.argmax(xs)
    left_idx = np.argmax(ys)
    right_idx = np.argmin(ys)
    contact_idx = np.argmax(ys)

    points = [
        ("centroid", centroid_x, centroid_y),
        ("axis_front", float(xs[front_idx]), float(ys[front_idx])),
        ("axis_back", float(xs[back_idx]), float(ys[back_idx])),
        ("axis_left", float(xs[left_idx]), float(ys[left_idx])),
        ("axis_right", float(xs[right_idx]), float(ys[right_idx])),
        ("contact_low", float(xs[contact_idx]), float(ys[contact_idx])),
    ]

    rows = []

    for point_name, x, y in points:
        rows.append(
            {
                "point_name": point_name,
                "x": x,
                "y": y,
                "box_x1": x1,
                "box_y1": y1,
                "box_x2": x2,
                "box_y2": y2,
                "area": record["area"],
                "class_name": record["class_name"],
                "source": record["source"],
            }
        )

    return rows


def build_control_points(video_paths, max_frames=MAX_FRAMES_PER_EPISODE):
    rows = []

    for episode_id, video_path in enumerate(tqdm(video_paths, desc="control points")):
        frames, frame_indices = read_video_even(
            video_path,
            max_frames=max_frames,
            return_indices=True,
        )

        for local_frame_idx, frame in enumerate(frames):
            source_frame_idx = int(frame_indices[local_frame_idx])
            records = lunar_lander_agent_records(frame)

            for track_id, record in enumerate(records):
                for row in control_points_from_record(record):
                    row.update(
                        {
                            "episode_id": episode_id,
                            "video_path": str(video_path),
                            "frame": source_frame_idx,
                            "local_frame": local_frame_idx,
                            "track_id": track_id,
                        }
                    )
                    rows.append(row)

    df = pd.DataFrame(rows)
    df["point_name"] = pd.Categorical(
        df["point_name"],
        categories=CONTROL_POINT_NAMES,
        ordered=True,
    )

    return df.sort_values(
        ["episode_id", "track_id", "frame", "point_name"]
    ).reset_index(drop=True)


def save_control_points(df, path=None):
    path = path or (CONTROL_POINTS_DIR / "lunar_lander_control_points.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path

