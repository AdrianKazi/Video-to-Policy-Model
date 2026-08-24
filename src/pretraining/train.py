import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/action_inference_matplotlib")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.pretraining.dataset import get_data_batch
from src.pretraining.model import ActionEmbeddingModel


def train_transformer(
    train_sequences,
    test_sequences,
    flattened_dim,
    seq_len=4,
    batch_size=8,
    embed_dim=256,
    n_heads=4,
    n_blocks=4,
    dropout=0.1,
    num_samples=100,
    lr=0.001,
    weight_decay=0.01,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionEmbeddingModel(
        flattened_dim=flattened_dim,
        seq_len=seq_len,
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_blocks=n_blocks,
        dropout=dropout,
    ).to(device)

    loss_function = nn.MSELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loss = []
    test_loss = []

    for step in range(num_samples):
        X, y = get_data_batch(train_sequences, seq_len=seq_len, batch_size=batch_size)
        model.zero_grad(set_to_none=True)
        pred = model(X.to(device))
        loss = loss_function(pred, y.to(device))
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

        with torch.no_grad():
            model.eval()
            X_test, y_test = get_data_batch(test_sequences, seq_len=seq_len, batch_size=batch_size)
            pred_test = model(X_test.to(device))
            thisloss = loss_function(pred_test, y_test.to(device))
            test_loss.append(thisloss.item())
            model.train()

        print(
            f"step={step:5d} "
            f"train_loss={train_loss[-1]:.6f} "
            f"test_loss={test_loss[-1]:.6f}"
        )

    return model, {"train_loss": train_loss, "test_loss": test_loss}


def save_training_run(
    model,
    history,
    model_path,
    history_path,
    config_path=None,
    loss_plot_path=None,
    eval_plot_path=None,
    config=None,
    eval_batch=None,
    device=None,
):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config or {},
    }

    torch.save(checkpoint, model_path)
    history_path.write_text(json.dumps(history, indent=2))

    if config_path is not None:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config or {}, indent=2))

    if loss_plot_path is not None:
        loss_plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 4))
        plt.plot(range(len(history["train_loss"])), history["train_loss"], label="train loss")
        plt.plot(range(len(history["test_loss"])), history["test_loss"], marker="o", label="test loss")
        plt.xlabel("training step")
        plt.ylabel("MSE loss")
        plt.title("Action Embedding Transformer Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(loss_plot_path, dpi=160, bbox_inches="tight")
        plt.close()

    if eval_plot_path is not None and eval_batch is not None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        X, y = eval_batch
        model.eval()

        with torch.no_grad():
            pred = model(X.to(device)).cpu()

        true_embedding = y[0, -1].numpy()
        pred_embedding = pred[0, -1].numpy()

        eval_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(true_embedding, label="true")
        ax[0].plot(pred_embedding, label="pred", alpha=0.75)
        ax[0].set_title("Flattened Action Embedding: True vs Pred")
        ax[0].set_xlabel("flattened embedding index")
        ax[0].set_ylabel("value")
        ax[0].legend()

        ax[1].scatter(true_embedding, pred_embedding, s=8, alpha=0.5)
        ax[1].set_title("Predicted vs True Embedding Values")
        ax[1].set_xlabel("true")
        ax[1].set_ylabel("pred")

        plt.tight_layout()
        plt.savefig(eval_plot_path, dpi=160, bbox_inches="tight")
        plt.close()
