"""Evaluate the saved test split, preserving label order in every metric."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from dataset import load_data
from model import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/keypoints"))
    parser.add_argument("--checkpoint", default="runs/mvp/best.pt")
    parser.add_argument("--output", type=Path, default=Path("runs/mvp/evaluation"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    model, checkpoint = load_checkpoint(args.checkpoint)
    manifest, x, y = load_data(args.data)
    for field in ("labels", "length", "seconds", "records"):
        if manifest[field] != checkpoint[field]:
            raise ValueError(f"Dataset {field} changed since training; use the original dataset")
    test_files = set(checkpoint["split"]["files"]["test"])
    indices = [i for i, row in enumerate(manifest["records"]) if row["path"] in test_files]
    predictions = []
    with torch.no_grad():
        for start in range(0, len(indices), args.batch_size):
            batch = torch.from_numpy(x[indices[start:start + args.batch_size]])
            predictions.extend(model(batch).argmax(1).tolist())
    labels = checkpoint["labels"]
    ids = list(range(len(labels)))
    actual = y[indices]
    matrix = confusion_matrix(actual, predictions, labels=ids)
    accuracy = float(accuracy_score(actual, predictions))
    report = classification_report(actual, predictions, labels=ids, target_names=labels,
                                   zero_division=0, output_dict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(
        {"accuracy": accuracy, "test_people": checkpoint["split"]["people"]["test"],
         "labels": labels, "confusion_matrix": matrix.tolist(), "report": report}, indent=2))
    np.savetxt(args.output / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d",
               header=",".join(labels), comments="")
    import matplotlib
    matplotlib.use("Agg")  # Works without a desktop/display server.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title(f"Unseen-person accuracy: {accuracy:.1%}")
    fig.tight_layout()
    fig.savefig(args.output / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Test accuracy: {accuracy:.2%} ({len(indices)} clips)")
    print(classification_report(actual, predictions, labels=ids, target_names=labels, zero_division=0))
    print(f"Saved metrics, CSV and confusion_matrix.png to {args.output}")


if __name__ == "__main__":
    main()
