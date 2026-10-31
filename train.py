"""Train on whole people; select the best epoch using validation loss only."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from common import FEATURE_VERSION
from dataset import load_data, split_people
from model import SignLSTM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/keypoints"))
    parser.add_argument("--output", type=Path, default=Path("runs/mvp"))
    parser.add_argument("--val-people", nargs="+")
    parser.add_argument("--test-people", nargs="+")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.hidden_size) < 1 or args.lr <= 0:
        parser.error("Training parameters must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new output directory to preserve previous runs")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest, x, y = load_data(args.data)
    groups, indices = split_people(manifest["records"], manifest["labels"], args.seed,
                                   args.val_people, args.test_people)
    print("People per split:", groups)
    args.output.mkdir(parents=True, exist_ok=True)
    split = {"people": groups, "files": {k: [manifest["records"][i]["path"] for i in v]
                                          for k, v in indices.items()}}
    (args.output / "split.json").write_text(json.dumps(split, indent=2))
    device = torch.device(args.device)
    model = SignLSTM(len(manifest["labels"]), args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    def loader(name, shuffle=False):
        idx = indices[name]
        return DataLoader(TensorDataset(torch.from_numpy(x[idx]), torch.from_numpy(y[idx]).long()),
                          batch_size=args.batch_size, shuffle=shuffle)
    train_loader, val_loader = loader("train", True), loader("val")
    best_loss = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(features), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(targets)
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                logits = model(features)
                val_loss += loss_fn(logits, targets).item() * len(targets)
                correct += (logits.argmax(1) == targets).sum().item()
        train_loss /= len(indices["train"])
        val_loss /= len(indices["val"])
        val_accuracy = correct / len(indices["val"])
        if not np.isfinite(train_loss + val_loss):
            raise RuntimeError("Non-finite loss; inspect data or reduce learning rate")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_accuracy": val_accuracy})
        print(f"Epoch {epoch:03d} train={train_loss:.4f} val={val_loss:.4f} val_acc={val_accuracy:.3f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"state_dict": model.state_dict(), "labels": manifest["labels"],
                        "hidden_size": args.hidden_size, "length": manifest["length"],
                        "seconds": manifest["seconds"], "feature_version": FEATURE_VERSION,
                        "split": split, "records": manifest["records"], "seed": args.seed,
                        "epoch": epoch}, args.output / "best.pt")
    (args.output / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Saved {args.output / 'best.pt'}; evaluate on untouched test people next.")


if __name__ == "__main__":
    main()
