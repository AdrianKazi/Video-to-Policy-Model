import pickle

from src.pretraining.config import ACTION_EMBEDDINGS_DIR, LOGS_DIR, MODELS_DIR, PLOTS_DIR, ensure_dirs
from src.pretraining.dataset import build_sequences_from_embeddings, get_data_batch, split_sequences
from src.pretraining.train import save_training_run, train_transformer


def main():
    ensure_dirs()
    embeddings_path = ACTION_EMBEDDINGS_DIR / "lunar_lander_action_embeddings.pkl"

    with embeddings_path.open("rb") as f:
        embeddings = pickle.load(f)

    seq_len = 4
    batch_size = 8
    embed_dim = 256
    n_heads = 4
    n_blocks = 4
    dropout = 0.1
    num_samples = 100
    lr = 0.001
    weight_decay = 0.01

    sequences = build_sequences_from_embeddings(embeddings, seq_len=seq_len)
    train_sequences, test_sequences = split_sequences(sequences)
    flattened_dim = sequences[0]["tokens"].shape[-1]
    sample_token = sequences[0]["tokens"][0]

    config = {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "embed_dim": embed_dim,
        "n_heads": n_heads,
        "n_blocks": n_blocks,
        "dropout": dropout,
        "num_samples": num_samples,
        "lr": lr,
        "weight_decay": weight_decay,
        "flattened_dim": int(flattened_dim),
        "token_shape": list(sample_token.shape),
        "num_sequences": len(sequences),
        "num_train_sequences": len(train_sequences),
        "num_test_sequences": len(test_sequences),
    }

    print("sequences:", len(sequences))
    print("train_sequences:", len(train_sequences))
    print("test_sequences:", len(test_sequences))
    print("flattened_dim:", flattened_dim)

    model, history = train_transformer(
        train_sequences=train_sequences,
        test_sequences=test_sequences,
        flattened_dim=flattened_dim,
        seq_len=seq_len,
        batch_size=batch_size,
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_blocks=n_blocks,
        dropout=dropout,
        num_samples=num_samples,
        lr=lr,
        weight_decay=weight_decay,
    )

    eval_batch = get_data_batch(test_sequences, seq_len=seq_len, batch_size=batch_size)

    save_training_run(
        model,
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
    print("saved config:", MODELS_DIR / "action_embedding_transformer_config.json")
    print("saved history:", LOGS_DIR / "action_embedding_transformer_history.json")
    print("saved loss plot:", PLOTS_DIR / "action_embedding_transformer_loss.png")
    print("saved eval plot:", PLOTS_DIR / "action_embedding_true_vs_pred.png")


if __name__ == "__main__":
    main()
