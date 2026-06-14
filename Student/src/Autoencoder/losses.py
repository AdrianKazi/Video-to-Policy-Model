import torch


def weighted_mse(x_hat: torch.Tensor, x: torch.Tensor, threshold: float = 0.1, high_weight: float = 20.0):
    mask = (x > threshold).float()
    weights = 1.0 + mask * high_weight
    loss = (x_hat - x) ** 2 * weights
    return loss.mean()
