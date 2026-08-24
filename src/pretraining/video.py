import cv2
import imageio.v2 as imageio
import numpy as np


def read_video(path, max_frames=None, stride=1, start_frame=0):
    cap = cv2.VideoCapture(str(path))

    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    frames_out = []
    frame_no = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_no % stride == 0:
            frames_out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if max_frames is not None and len(frames_out) >= max_frames:
                break

        frame_no += 1

    cap.release()

    if not frames_out:
        raise RuntimeError(f"No frames read from {path}")

    return np.stack(frames_out)


def sample_even_frame_indices(num_frames, max_frames):
    if num_frames <= 0:
        return np.asarray([], dtype=int)

    if max_frames is None or num_frames <= max_frames:
        return np.arange(num_frames, dtype=int)

    return np.unique(np.linspace(0, num_frames - 1, int(max_frames), dtype=int))


def read_video_even(path, max_frames=None, stride=1, return_indices=False):
    full_frames = read_video(path, max_frames=None, stride=stride)
    frame_indices = sample_even_frame_indices(len(full_frames), max_frames)
    frames = full_frames[frame_indices]

    if return_indices:
        return frames, frame_indices

    return frames


def save_video(frames, path, fps=30):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = frames.astype(np.uint8)
    imageio.mimsave(path, frames, fps=fps)
    return path

