import torch

from .loaders import load_ae_model, load_lstm_model
from .models import overlay_frames, IDMModel, ActionAdapter, FDMModel
from .paths import SEQ_TRAIN_PT


def smoke_video_learner_forward(batch_size: int = 32, decay: float = 0.9):
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    ae_model = load_ae_model(device)
    lstm_model = load_lstm_model(device)

    seq_train_data = torch.load(SEQ_TRAIN_PT, map_location="cpu")
    x = seq_train_data["x"][:batch_size].to(device)

    x_hist = x[:, :-1]
    x_next = x[:, -1]

    with torch.no_grad():
        x_overlay = overlay_frames(x_hist, decay=decay)
        _, m_t = ae_model(x_overlay)

        z_hist = torch.stack(
            [ae_model(x_hist[:, t])[1] for t in range(x_hist.shape[1])],
            dim=1,
        )

        z_t = z_hist[:, -1, :]
        _, z_true_next = ae_model(x_next)

        h_seq, _ = lstm_model.lstm(z_hist)
        h_t = h_seq[:, -1, :]

        c_t = torch.cat([z_t, h_t, m_t], dim=-1)

    z_dim = z_true_next.shape[-1]
    c_dim = c_t.shape[-1]
    latent_a_dim = 8
    real_a_dim = 2

    idm_model = IDMModel(c_dim=c_dim, latent_a_dim=latent_a_dim).to(device)
    adapter_model = ActionAdapter(latent_a_dim=latent_a_dim, real_a_dim=real_a_dim).to(device)
    fdm_model = FDMModel(c_dim=c_dim, z_dim=z_dim, latent_a_dim=latent_a_dim).to(device)

    latent_a = idm_model(c_t)
    a_pred = adapter_model(latent_a)
    delta_z_pred = fdm_model(c_t, latent_a)
    z_pred_next = z_t + delta_z_pred
    loss_z = ((z_pred_next - z_true_next) ** 2).mean()

    print("x:", x.shape)
    print("x_overlay:", x_overlay.shape)
    print("z_hist:", z_hist.shape)
    print("z_t:", z_t.shape)
    print("h_t:", h_t.shape)
    print("m_t:", m_t.shape)
    print("c_t:", c_t.shape)
    print("z_true_next:", z_true_next.shape)
    print("latent_a:", latent_a.shape)
    print("a_pred:", a_pred.shape)
    print("delta_z_pred:", delta_z_pred.shape)
    print("z_pred_next:", z_pred_next.shape)
    print("loss_z:", float(loss_z.item()))
