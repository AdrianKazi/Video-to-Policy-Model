"""Learn a latent policy and visual dynamics from MP4 episodes."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from Autoencoder.device_util import pick_device
from .data import VideoWindows, read_videos, split_episodes
from .model import LatentPolicy, VideoDynamics, prediction_errors


def video_tensor(frames, device):
    return frames.to(device=device, dtype=torch.float32).div_(255)


def fit_dynamics(model, baseline, loader, epochs, device):
    models = (model, baseline)
    optimizers = [torch.optim.AdamW(net.parameters(), lr=1e-3) for net in models]
    for epoch in range(epochs):
        totals = [0.0, 0.0]
        count = 0
        for history, target, _ in loader:
            history = video_tensor(history, device)
            target = video_tensor(target, device)
            for index, (net, optimizer) in enumerate(zip(models, optimizers)):
                optimizer.zero_grad(set_to_none=True)
                errors = prediction_errors(net(history), history[:, -1:], target)
                loss = errors.min(dim=1).values.mean()
                loss.backward()
                optimizer.step()
                totals[index] += loss.item() * history.size(0)
            count += history.size(0)
        print(
            f"dynamics {epoch + 1}/{epochs} | best code {totals[0] / count:.6f}"
            f" | no code {totals[1] / count:.6f}",
            flush=True,
        )


def fit_policy(policy, model, loader, epochs, codes, device):
    model.eval()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    for epoch in range(epochs):
        totals = torch.zeros(codes, dtype=torch.long)
        correct = 0
        for history, target, _ in loader:
            history = video_tensor(history, device)
            target = video_tensor(target, device)
            with torch.no_grad():
                labels = prediction_errors(model(history), history[:, -1:], target).argmin(dim=1)
            optimizer.zero_grad(set_to_none=True)
            logits = policy(history)
            F.cross_entropy(logits, labels).backward()
            optimizer.step()
            totals += torch.bincount(labels.cpu(), minlength=codes)
            correct += (logits.argmax(dim=1) == labels).sum().item()
        print(f"policy {epoch + 1}/{epochs} | train agreement {correct / totals.sum():.3f}", flush=True)
    return totals.float() / totals.sum()


@torch.inference_mode()
def evaluate(model, baseline, policy, loader, episodes, prior, device):
    metric_names = [
        "oracle_mse", "policy_mse", "prior_mse", "no_code_mse", "copy_mse",
        "policy_agreement", "majority_agreement",
    ]
    totals = np.zeros((len(episodes), len(metric_names)), dtype=np.float64)
    examples = np.zeros(len(episodes), dtype=np.int64)
    oracle_counts = torch.zeros_like(prior, dtype=torch.long, device="cpu")
    policy_counts = torch.zeros_like(oracle_counts)
    majority = prior.argmax().item()
    prior = prior.to(device)

    for history, target, episode in loader:
        history = video_tensor(history, device)
        target = video_tensor(target, device)
        current = history[:, -1:]
        errors = prediction_errors(model(history), current, target)
        indices = torch.arange(len(history), device=device)
        oracle_loss, labels = errors.min(dim=1)
        choices = policy(history).argmax(dim=1)
        values = torch.stack(
            (
                oracle_loss,
                errors[indices, choices],
                errors @ prior,
                prediction_errors(baseline(history), current, target)[:, 0],
                prediction_errors(current, current, target)[:, 0],
                choices == labels,
                labels == majority,
            ),
            dim=1,
        )
        np.add.at(totals, episode.numpy(), values.cpu().numpy())
        np.add.at(examples, episode.numpy(), 1)
        oracle_counts += torch.bincount(labels.cpu(), minlength=len(prior))
        policy_counts += torch.bincount(choices.cpu(), minlength=len(prior))

    by_episode = totals / examples[:, None]
    return {
        "episode_mean": dict(zip(metric_names, by_episode.mean(axis=0).tolist())),
        "by_episode": {
            episode.path.name: dict(zip(metric_names, values.tolist()))
            for episode, values in zip(episodes, by_episode)
        },
        "test_oracle_code_counts": oracle_counts.tolist(),
        "test_policy_code_counts": policy_counts.tolist(),
    }


@torch.inference_mode()
def evaluate_rollouts(model, baseline, policy, loader, episodes, steps, device, preview):
    horizons = sorted({1, min(5, steps), steps})
    totals = np.zeros((len(episodes), len(horizons), 3), dtype=np.float64)
    examples = np.zeros(len(episodes), dtype=np.int64)
    preview_rows = []
    for history, targets, episode in loader:
        history = video_tensor(history, device)
        targets = video_tensor(targets, device)
        policy_history = history
        baseline_history = history
        first_frame = history[:, -1:]
        indices = torch.arange(len(history), device=device)
        for step in range(1, steps + 1):
            choices = policy(policy_history).argmax(dim=1)
            prediction = model(policy_history)[indices, choices].unsqueeze(1)
            no_code = baseline(baseline_history)
            policy_history = torch.cat((policy_history[:, 1:], prediction), dim=1)
            baseline_history = torch.cat((baseline_history[:, 1:], no_code), dim=1)
            if step in horizons:
                target = targets[:, step - 1:step]
                current = first_frame if step == 1 else targets[:, step - 2:step - 1]
                candidates = torch.cat((prediction, no_code, first_frame), dim=1)
                errors = prediction_errors(candidates, current, target)
                np.add.at(totals[:, horizons.index(step)], episode.numpy(), errors.cpu().numpy())
                if examples.sum() == 0:
                    frames = torch.cat((target[:1], candidates[:1]), dim=1).squeeze(0)
                    row = frames.mul(255).round().to(torch.uint8).cpu().numpy()
                    preview_rows.append(np.concatenate(row, axis=1))
        np.add.at(examples, episode.numpy(), 1)
    image = np.concatenate(preview_rows, axis=0)
    if not cv2.imwrite(str(preview), image):
        raise OSError(f"Could not write {preview}")
    means = (totals / examples[:, None, None]).mean(axis=0)
    names = ("policy_mse", "no_code_mse", "copy_mse")
    return {str(step): dict(zip(names, values.tolist())) for step, values in zip(horizons, means)}


def run(args):
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    videos = read_videos(
        args.videos, args.size, args.history + args.rollout_steps, args.split_seed, args.max_videos
    )
    train, test = split_episodes(videos, args.split_seed)
    args.output.mkdir(parents=True, exist_ok=False)
    print(f"{device} | {len(train)} train episodes | {len(test)} test episodes", flush=True)

    train_windows = VideoWindows(train, args.history, horizon=1)
    train_loader = DataLoader(
        train_windows,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    test_windows = VideoWindows(test, args.history, horizon=1)
    test_loader = DataLoader(test_windows, batch_size=args.batch_size)
    rollout_windows = VideoWindows(test, args.history, args.rollout_steps, stride=args.rollout_steps)
    rollout_loader = DataLoader(rollout_windows, batch_size=args.batch_size)

    model = VideoDynamics(args.history, args.codes).to(device)
    baseline = VideoDynamics(args.history, codes=1).to(device)
    policy = LatentPolicy(args.history, args.codes).to(device)
    fit_dynamics(model, baseline, train_loader, args.dynamics_epochs, device)
    prior = fit_policy(policy, model, train_loader, args.policy_epochs, args.codes, device)
    model.eval()
    baseline.eval()
    policy.eval()

    one_step = evaluate(model, baseline, policy, test_loader, test, prior, device)
    rollouts = evaluate_rollouts(
        model, baseline, policy, rollout_loader, test, args.rollout_steps, device,
        args.output / "rollout.png",
    )
    settings = {**vars(args), "videos": str(args.videos), "output": str(args.output)}
    summary = {
        "settings": settings,
        "device": str(device),
        "train_episodes": [episode.path.name for episode in train],
        "test_episodes": [episode.path.name for episode in test],
        "train_code_prior": prior.tolist(),
        "one_step": one_step,
        "rollout_episode_mean": rollouts,
    }
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    torch.save(
        {
            "config": {"history": args.history, "codes": args.codes, "size": args.size},
            "dynamics": model.state_dict(),
            "policy": policy.state_dict(),
            "no_code": baseline.state_dict(),
        },
        args.output / "models.pt",
    )
    print(json.dumps({"one_step": one_step["episode_mean"], "rollouts": rollouts}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--codes", type=int, default=8)
    parser.add_argument("--size", type=int, default=84)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dynamics-epochs", type=int, default=8)
    parser.add_argument("--policy-epochs", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=10)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device")
    args = parser.parse_args()
    if (
        args.history < 2 or args.codes < 2 or args.size < 16 or args.size % 4 != 0
        or min(args.batch_size, args.dynamics_epochs, args.policy_epochs, args.rollout_steps) < 1
        or min(args.seed, args.split_seed) < 0 or (args.max_videos is not None and args.max_videos < 2)
    ):
        parser.error("Use at least 2 history frames and codes, a size >= 16 divisible by 4, "
                     "positive counts, and non-negative seeds")
    run(args)


if __name__ == "__main__":
    main()
