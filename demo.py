"""Sliding-window webcam recognition. Q quits; C clears the current window."""
import argparse
from collections import deque
import time
import cv2
import numpy as np
import torch
from common import HandTracker, draw_hands
from model import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/mvp/best.pt")
    parser.add_argument("--model", default="models/hand_landmarker.task")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--stable", type=int, default=3, help="Consecutive matching predictions before display")
    parser.add_argument("--interval", type=float, default=0.15, help="Seconds between predictions")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or args.stable < 1 or args.interval <= 0:
        parser.error("Invalid threshold, stable count or interval")
    # Small CPU batches are faster with a single thread than a large thread pool.
    torch.set_num_threads(1)
    model, checkpoint = load_checkpoint(args.checkpoint)
    camera = cv2.VideoCapture(args.camera)
    tracker = None
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Cannot open webcam; check permissions and --camera")
    window = deque()
    last_prediction = 0.0
    candidate, count = None, 0
    word = "Warming up..."
    try:
        tracker = HandTracker(args.model)
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera stopped returning frames")
            frame = cv2.flip(frame, 1)
            now = time.monotonic()
            features, result = tracker.extract(frame, now * 1000)
            window.append((now, features))
            # Keep one frame before the cutoff, so a full-duration window is available.
            cutoff = now - checkpoint["seconds"]
            while len(window) > 1 and window[1][0] <= cutoff:
                window.popleft()
            hands_now = bool(features[63] or features[127])
            if not hands_now:
                window.clear()
                candidate, count = None, 0
                word = "No hands"
            elif now - window[0][0] < checkpoint["seconds"]:
                word = "Warming up..."
            elif now - last_prediction >= args.interval:
                last_prediction = now
                # Sample by wall-clock time, not frame count, to tolerate variable webcam FPS.
                times = np.array([t for t, _ in window])
                targets = np.linspace(cutoff, now, checkpoint["length"])
                right = np.searchsorted(times, targets).clip(0, len(times) - 1)
                left = (right - 1).clip(0)
                nearest = np.where(abs(times[left] - targets) < abs(times[right] - targets), left, right)
                sequence = np.stack([window[i][1] for i in nearest])
                with torch.no_grad():
                    probabilities = model(torch.from_numpy(sequence)[None]).softmax(1)[0]
                confidence, index = probabilities.max(0)
                label = checkpoint["labels"][index.item()]
                if confidence.item() < args.threshold:
                    candidate, count = None, 0
                    word = "Uncertain"
                else:
                    count = count + 1 if candidate == label else 1
                    candidate = label
                    word = f"{label} ({confidence.item():.0%})" if count >= args.stable else "Uncertain"
            draw_hands(frame, result)
            cv2.putText(frame, word, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, "Q quit | C clear", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Isolated sign recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                window.clear()
                candidate, count = None, 0
                word = "Warming up..."
    finally:
        if tracker is not None:
            tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
