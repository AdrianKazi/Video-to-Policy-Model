"""Save original vs reconstruction grid (shared by train end-of-run and eval)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_reconstruction_png(
    out_path: Path,
    x: torch.Tensor,
    x_hat: torch.Tensor,
    *,
    num_rows: int,
) -> None:
    """``x``, ``x_hat``: (T,1,84,84) on same device; writes a PNG file."""
    n = min(num_rows, x.shape[0])
    fig, axs = plt.subplots(n, 2, figsize=(6, 2 * n), squeeze=False)
    for i in range(n):
        axs[i, 0].imshow(x[i].squeeze().detach().cpu(), cmap="gray")
        axs[i, 0].axis("off")
        if i == 0:
            axs[i, 0].set_title("original")
        axs[i, 1].imshow(x_hat[i].squeeze().detach().cpu(), cmap="gray")
        axs[i, 1].axis("off")
        if i == 0:
            axs[i, 1].set_title("reconstruction")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
