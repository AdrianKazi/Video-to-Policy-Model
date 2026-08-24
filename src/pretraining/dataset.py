import numpy as np
import torch


def build_sequences_from_embeddings(embeddings, seq_len):
    sequences = []

    for episode_id in sorted(embeddings.keys()):
        available_track_ids = sorted(
            {
                track_id
                for frame_embeddings in embeddings[episode_id].values()
                for track_id in frame_embeddings.keys()
            }
        )

        for track_id in available_track_ids:
            track_frames = [
                frame
                for frame in sorted(embeddings[episode_id])
                if track_id in embeddings[episode_id][frame]
            ]

            if len(track_frames) <= seq_len:
                continue

            tokens = np.stack(
                [
                    embeddings[episode_id][frame][track_id]["token_norm_flat"]
                    for frame in track_frames
                ]
            ).astype(np.float32)

            sequences.append(
                {
                    "episode_id": episode_id,
                    "track_id": track_id,
                    "frames": track_frames,
                    "tokens": tokens,
                }
            )

    return sequences


def split_sequences(sequences, train_ratio=0.8):
    split_idx = int(len(sequences) * train_ratio)
    return sequences[:split_idx], sequences[split_idx:]


def get_data_batch(sequences, seq_len, batch_size):
    X_batch = []
    y_batch = []

    for _ in range(batch_size):
        seq = sequences[np.random.randint(0, len(sequences))]
        tokens = seq["tokens"]
        max_start = len(tokens) - seq_len - 1

        if max_start < 0:
            continue

        start = np.random.randint(0, max_start + 1)
        X_batch.append(tokens[start:start + seq_len])
        y_batch.append(tokens[start + 1:start + seq_len + 1])

    if len(X_batch) == 0:
        raise ValueError("No valid sequences for batch.")

    return (
        torch.tensor(np.stack(X_batch), dtype=torch.float32),
        torch.tensor(np.stack(y_batch), dtype=torch.float32),
    )

