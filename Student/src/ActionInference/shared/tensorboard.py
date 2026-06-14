from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


GLOBAL_FEATURES_TEXT = "\n".join(
    [
        "* shape",
        "* dtype",
        "* device",
        "* mean / std / min / max",
        "* nan / inf count",
        "* norm",
        "* histogram",
        "* sample image / video when decodable",
    ]
)


def make_writer(run_dir, name):
    log_dir = Path(run_dir) / "tensorboard" / name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_text("global_00_globalfeatures/features", GLOBAL_FEATURES_TEXT, 0)
    return writer


def log_tensor(writer, tag, value, step, histogram=True):
    if value is None:
        return

    if not torch.is_tensor(value):
        value = torch.as_tensor(value)

    x = value.detach()
    writer.add_text(f"{tag}/shape", str(tuple(x.shape)), step)
    writer.add_text(f"{tag}/dtype", str(x.dtype), step)
    writer.add_text(f"{tag}/device", str(x.device), step)

    if not torch.is_floating_point(x):
        x = x.float()

    x_cpu = x.detach().float().cpu()
    finite = torch.isfinite(x_cpu)
    nan_count = torch.isnan(x_cpu).sum().item()
    inf_count = torch.isinf(x_cpu).sum().item()

    writer.add_scalar(f"{tag}/nan_count", nan_count, step)
    writer.add_scalar(f"{tag}/inf_count", inf_count, step)

    if finite.any():
        xf = x_cpu[finite]
        writer.add_scalar(f"{tag}/mean", xf.mean().item(), step)
        writer.add_scalar(f"{tag}/std", xf.std(unbiased=False).item(), step)
        writer.add_scalar(f"{tag}/min", xf.min().item(), step)
        writer.add_scalar(f"{tag}/max", xf.max().item(), step)
        writer.add_scalar(f"{tag}/norm", torch.linalg.vector_norm(xf).item(), step)
        if histogram:
            writer.add_histogram(f"{tag}/histogram", xf, step)


def log_step_features(writer, tag, features, step):
    text = "\n".join([f"* {feature}" for feature in features])
    writer.add_text(f"{tag}/diagnostic_features", text, step)


def log_image_batch(writer, tag, value, step, max_images=4):
    if value is None or not torch.is_tensor(value):
        return

    x = value.detach().float().cpu().clamp(0, 1)
    if x.ndim == 5:
        x = x[:, -1]
    if x.ndim == 3:
        x = x[:, None]
    if x.ndim != 4:
        return

    writer.add_images(tag, x[:max_images], step)
