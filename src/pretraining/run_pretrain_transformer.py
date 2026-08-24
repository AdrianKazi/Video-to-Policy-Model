import pickle

from src.pretraining.config import ACTION_EMBEDDINGS_DIR, LOGS_DIR, MODELS_DIR, ensure_dirs
from src.pretraining.dataset import build_sequences_from_embeddings, split_sequences
from src.pretraining.train import save_training_run, train_transformer


def main():
    ensure_dirs()
    embeddings_path = ACTION_EMBEDDINGS_DIR / "lunar_lander_action_embeddings.pkl"

    with embeddings_path.open("rb") as f:
        embeddings = pickle.load(f)

    seq_len = 4
    sequences = build_sequences_from_embeddings(embeddings, seq_len=seq_len)
    train_sequences, test_sequences = split_sequences(sequences)
    flattened_dim = sequences[0]["tokens"].shape[-1]

    print("sequences:", len(sequences))
    print("train_sequences:", len(train_sequences))
    print("test_sequences:", len(test_sequences))
    print("flattened_dim:", flattened_dim)

    model, history = train_transformer(
        train_sequences=train_sequences,
        test_sequences=test_sequences,
        flattened_dim=flattened_dim,
        seq_len=seq_len,
        num_samples=100,
    )

    save_training_run(
        model,
        history,
        MODELS_DIR / "action_embedding_transformer.pt",
        LOGS_DIR / "action_embedding_transformer_history.json",
    )


if __name__ == "__main__":
    main()

