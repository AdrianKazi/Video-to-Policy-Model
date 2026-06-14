import torch

from Autoencoder.network import AutoEncoder
from Sequential.network import LatentLSTM

from .paths import AE_MODEL_PT, SEQ_MODEL_PT


def load_ae_model(device, z_dim: int = 64):
    ae_model = AutoEncoder(z_dim=z_dim).to(device)
    ae_model.load_state_dict(torch.load(AE_MODEL_PT, map_location=device))
    ae_model.eval()

    print(f"[AE] loaded model -> {AE_MODEL_PT}")
    return ae_model


def load_lstm_model(
    device,
    z_dim: int = 64,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
):
    lstm_model = LatentLSTM(
        z_dim=z_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    lstm_model.load_state_dict(torch.load(SEQ_MODEL_PT, map_location=device))
    lstm_model.eval()

    print(f"[LSTM] loaded model -> {SEQ_MODEL_PT}")
    return lstm_model