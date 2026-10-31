"""Convert data/videos/PERSON/LABEL/*.avi into fixed-length landmark arrays."""
import argparse
import json
import hashlib
from pathlib import Path
import cv2
import numpy as np
from common import HandTracker, read_labels, resample, FEATURE_VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=Path("data/videos"))
    parser.add_argument("--output", type=Path, default=Path("data/keypoints"))
    parser.add_argument("--labels", default="labels.txt")
    parser.add_argument("--model", default="models/hand_landmarker.task")
    parser.add_argument("--length", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()
    if args.length < 2 or args.seconds <= 0:
        parser.error("Use --length >= 2 and --seconds > 0")
    labels = read_labels(args.labels)
    videos = sorted(p for p in args.videos.glob("*/*/*") if p.suffix.lower() in {".avi", ".mp4", ".mov"})
    if not videos:
        parser.error("No videos found under PERSON/LABEL/video")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be empty; use a new directory to avoid mixing feature versions")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for video in videos:
        person, label = video.parent.parent.name, video.parent.name
        if label not in labels:
            raise ValueError(f"Unknown label in {video}")
        cap = cv2.VideoCapture(str(video))
        tracker = None
        sequence = []
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
                raise ValueError(f"Cannot read video/FPS: {video}")
            tracker = HandTracker(args.model)  # Reset temporal tracking for every clip.
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                features, _ = tracker.extract(frame, len(sequence) * 1000 / fps)
                sequence.append(features)
        finally:
            cap.release()
            if tracker is not None:
                tracker.close()
        if not sequence:
            raise ValueError(f"Empty video: {video}")
        duration = len(sequence) / fps
        if abs(duration - args.seconds) > max(0.2, 2 / fps):
            raise ValueError(f"{video}: {duration:.2f}s, expected {args.seconds}s; record consistent durations")
        array = resample(sequence, args.length)
        if not np.any(array[:, [63, 127]]):
            raise ValueError(f"No hands detected in {video}; re-record or remove this clip")
        relative = Path(person) / label / (video.name + ".npy")
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, array)
        records.append({"path": relative.as_posix(), "person": person, "label": label,
                        "source": str(video.resolve()),
                        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
        print(f"Extracted {video}")
    # Training reads only this manifest, so stale/unlisted arrays cannot sneak in.
    manifest = {"labels": labels, "length": args.length, "seconds": args.seconds,
                "feature_version": FEATURE_VERSION, "records": records}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {len(records)} clips to {args.output}")


if __name__ == "__main__":
    main()
