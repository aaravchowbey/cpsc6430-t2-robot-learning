"""Train the three simulation-only LIBERO behavior cloning policies."""

import argparse
import csv
from contextlib import ExitStack
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


VARIANTS = ("vision", "dense", "grid")
CAMERAS = ("agentview_rgb", "eye_in_hand_rgb")
IMAGE_SIZE = 48
GRID_SIZE = 4
WIDTH = 48


def read_manifest(path):
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def resize_view(image):
    return cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def load_episodes(data_dir, rows):
    images, actions, tasks = [], [], []
    with ExitStack() as stack:
        sources = {}
        for row in rows:
            filename = row["dataset"]
            if filename not in sources:
                sources[filename] = stack.enter_context(h5py.File(data_dir / filename, "r"))
            episode = sources[filename][f"data/{row['episode']}"]
            count = int(row["num_steps"])
            if episode["actions"].shape != (count, 7) or any(
                len(episode[f"obs/{key}"]) != count for key in CAMERAS
            ):
                raise ValueError(f"Manifest count or action shape mismatch: {filename}/{row['episode']}")
            # The collector records each image after applying the same-index action.
            views = [
                np.stack([resize_view(frame) for frame in episode[f"obs/{key}"][:-1]])
                for key in CAMERAS
            ]
            images.append(np.stack(views, axis=1).transpose(0, 1, 4, 2, 3))
            actions.append(episode["actions"][1:].astype(np.float32))
            tasks.append(np.full(count - 1, int(row["task_id"]), dtype=np.int64))
    return TensorDataset(*(torch.from_numpy(np.concatenate(parts)) for parts in (images, actions, tasks)))


def instruction_features(rows):
    """Encode the two fixed language goals with train-vocabulary bag-of-words counts."""
    goals = {int(row["task_id"]): row["task"].split("_") for row in rows}
    vocabulary = sorted({word for goal in goals.values() for word in goal})
    return torch.tensor(
        [[goals[task].count(word) for word in vocabulary] for task in sorted(goals)],
        dtype=torch.float32,
    )


class Policy(nn.Module):
    def __init__(self, variant, instructions):
        super().__init__()
        self.variant = variant
        self.register_buffer("instructions", instructions)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, WIDTH, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(GRID_SIZE),
        )
        axis = torch.linspace(0, 1, GRID_SIZE)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("dense_coordinates", torch.stack((x, y), dim=-1).reshape(1, 16, 2))
        self.coordinate_embedding = nn.Linear(2, WIDTH)
        self.camera_embedding = nn.Parameter(torch.randn(1, 2, 1, WIDTH) * 0.02)
        if variant != "vision":
            self.language = nn.Linear(instructions.shape[1], WIDTH)
        if variant == "grid":
            self.coordinate_predictor = nn.Sequential(
                nn.Linear(3 * WIDTH, WIDTH), nn.ReLU(), nn.Linear(WIDTH, 4),
            )
        self.queries = nn.Parameter(torch.randn(1, 4, WIDTH) * 0.1)
        self.attention = nn.MultiheadAttention(WIDTH, 4, batch_first=True)
        self.head = nn.Sequential(nn.Linear(5 * WIDTH, 128), nn.ReLU(), nn.Linear(128, 7), nn.Tanh())

    @property
    def visual_tokens(self):
        return 2 if self.variant == "grid" else 2 * GRID_SIZE**2

    def forward(self, images, task_ids):
        batch = images.shape[0]
        features = self.encoder(images.flatten(0, 1).float() / 255.0)
        language = features.new_zeros(batch, WIDTH)
        if self.variant != "vision":
            language = self.language(self.instructions[task_ids])
        if self.variant == "grid":
            pooled = features.mean(dim=(2, 3)).reshape(batch, 2 * WIDTH)
            coordinates = self.coordinate_predictor(torch.cat((pooled, language), dim=1))
            coordinates = coordinates.sigmoid().reshape(batch * 2, 1, 2)
            sampled = F.grid_sample(
                features, (2 * coordinates - 1).unsqueeze(2),
                mode="bilinear", align_corners=True,
            )
            tokens = sampled.flatten(2).transpose(1, 2)
        else:
            coordinates = self.dense_coordinates
            tokens = features.flatten(2).transpose(1, 2)
        tokens = tokens + self.coordinate_embedding(coordinates)
        tokens = tokens.reshape(batch, 2, -1, WIDTH) + self.camera_embedding
        tokens = tokens.flatten(1, 2)
        queries = self.queries + language.unsqueeze(1)
        context = self.attention(queries, tokens, tokens, need_weights=False)[0]
        return self.head(torch.cat((context.flatten(1), language), dim=1))


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total = 0.0
    with torch.set_grad_enabled(optimizer is not None):
        for images, actions, tasks in loader:
            images, actions, tasks = images.to(device), actions.to(device), tasks.to(device)
            loss = F.mse_loss(model(images, tasks), actions)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(images)
    return total / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=Path("splits"))
    parser.add_argument("--output", type=Path, default=Path("output/policies"))
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true", help="Overfit the same 32 training examples")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1:
        parser.error("Epochs, batch size and threads must be positive")
    torch.set_num_threads(args.threads)
    train_rows = read_manifest(args.splits / "train.csv")
    instructions = instruction_features(train_rows)
    if args.smoke:
        episode = load_episodes(args.data_dir, train_rows[:1])
        train = valid = TensorDataset(*(tensor[:32] for tensor in episode.tensors))
    else:
        train = load_episodes(args.data_dir, train_rows)
        valid = load_episodes(args.data_dir, read_manifest(args.splits / "validation.csv"))
    args.output.mkdir(parents=True, exist_ok=True)
    valid_loader = DataLoader(valid, batch_size=args.batch_size)
    print(f"device={args.device} | train_pairs={len(train)} | validation_pairs={len(valid)}", flush=True)
    for variant in VARIANTS if args.variant == "all" else (args.variant,):
        torch.manual_seed(args.seed)
        train_loader = DataLoader(
            train, batch_size=args.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        model = Policy(variant, instructions).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        best = float("inf")
        name = f"{variant}_smoke" if args.smoke else variant
        with (args.output / f"{name}_loss.csv").open("w", newline="", encoding="utf-8") as log:
            metrics = csv.writer(log)
            metrics.writerow(("epoch", "train_mse", "validation_mse"))
            for epoch in range(1, args.epochs + 1):
                train_loss = run_epoch(model, train_loader, args.device, optimizer)
                valid_loss = run_epoch(model, valid_loader, args.device)
                metrics.writerow((epoch, train_loss, valid_loss))
                log.flush()
                print(f"{variant} {epoch:03d} train={train_loss:.5f} val={valid_loss:.5f}", flush=True)
                if valid_loss < best:
                    best = valid_loss
                    path = args.output / f"{name}.pt"
                    temporary = path.with_suffix(".tmp")
                    torch.save({
                        "state_dict": model.state_dict(), "variant": variant, "epoch": epoch,
                        "validation_mse": best, "visual_tokens": model.visual_tokens,
                        "seed": args.seed, "learning_rate": args.learning_rate,
                        "batch_size": args.batch_size, "smoke": args.smoke,
                    }, temporary)
                    temporary.replace(path)


if __name__ == "__main__":
    main()
