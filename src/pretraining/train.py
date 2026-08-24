import json

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


def save_training_run(model, history, model_path, history_path):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    history_path.write_text(json.dumps(history, indent=2))

