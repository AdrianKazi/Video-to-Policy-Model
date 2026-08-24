import pickle

from src.pretraining.config import (
    ACTION_EMBEDDINGS_DIR,
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    CHECKPOINT_EVERY,
    DROPOUT,
    EMBED_DIM,
    LEARNING_RATE,
    LOGS_DIR,
    MODELS_DIR,
    N_BLOCKS,
    N_HEADS,
    NUM_TRAINING_STEPS,
    PLOTS_DIR,
    SEQ_LEN,
    START_FROM,
    WEIGHT_DECAY,
    ensure_dirs,
)
from src.pretraining.dataset import build_sequences_from_embeddings, get_data_batch, split_sequences
from src.pretraining.train import save_training_run, train_transformer


def main():
    ensure_dirs()
    embeddings_path = ACTION_EMBEDDINGS_DIR / "lunar_lander_action_embeddings.pkl"

    with embeddings_path.open("rb") as f:
        embeddings = pickle.load(f)

    sequences = build_sequences_from_embeddings(embeddings, seq_len=SEQ_LEN)
    train_sequences, test_sequences = split_sequences(sequences)
    flattened_dim = sequences[0]["tokens"].shape[-1]
    sample_token = sequences[0]["tokens"][0]

    config = {
        "seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "embed_dim": EMBED_DIM,
        "n_heads": N_HEADS,
        "n_blocks": N_BLOCKS,
        "dropout": DROPOUT,
        "num_training_steps": NUM_TRAINING_STEPS,
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "flattened_dim": int(flattened_dim),
        "token_shape": list(sample_token.shape),
        "num_sequences": len(sequences),
        "num_train_sequences": len(train_sequences),
        "num_test_sequences": len(test_sequences),
        "start_from": str(START_FROM),
        "checkpoint_every": CHECKPOINT_EVERY,
    }

    print("sequences:", len(sequences))
    print("train_sequences:", len(train_sequences))
    print("test_sequences:", len(test_sequences))
    print("flattened_dim:", flattened_dim)

    model, optimizer, history = train_transformer(
        train_sequences=train_sequences,
        test_sequences=test_sequences,
        flattened_dim=flattened_dim,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        n_blocks=N_BLOCKS,
        dropout=DROPOUT,
        num_samples=NUM_TRAINING_STEPS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        start_from=START_FROM,
        checkpoint_dir=CHECKPOINTS_DIR,
        checkpoint_every=CHECKPOINT_EVERY,
        config=config,
    )

    eval_batch = get_data_batch(test_sequences, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)

    save_training_run(
        model,
        optimizer,
        history,
        MODELS_DIR / "action_embedding_transformer.pt",
        LOGS_DIR / "action_embedding_transformer_history.json",
        config_path=MODELS_DIR / "action_embedding_transformer_config.json",
        loss_plot_path=PLOTS_DIR / "action_embedding_transformer_loss.png",
        eval_plot_path=PLOTS_DIR / "action_embedding_true_vs_pred.png",
        config=config,
        eval_batch=eval_batch,
    )

    print("saved model:", MODELS_DIR / "action_embedding_transformer.pt")
    print("saved checkpoints dir:", CHECKPOINTS_DIR)
    print("saved config:", MODELS_DIR / "action_embedding_transformer_config.json")
    print("saved history:", LOGS_DIR / "action_embedding_transformer_history.json")
    print("saved loss plot:", PLOTS_DIR / "action_embedding_transformer_loss.png")
    print("saved eval plot:", PLOTS_DIR / "action_embedding_true_vs_pred.png")


if __name__ == "__main__":
    main()
