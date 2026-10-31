"""Record one isolated sign at a time: SPACE starts a countdown; Q quits."""
import argparse
import time
import uuid
from pathlib import Path
import cv2
from common import read_labels, safe_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person", required=True, help="Stable signer ID across ALL recording sessions")
    parser.add_argument("--label", required=True)
    parser.add_argument("--labels", default="labels.txt")
    parser.add_argument("--output", type=Path, default=Path("data/videos"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()
    safe_name(args.person)
    if args.label not in read_labels(args.labels):
        parser.error("--label must be listed in labels.txt")
    if args.seconds <= 0 or args.fps <= 0:
        parser.error("--seconds and --fps must be positive")
    folder = args.output / args.person / args.label
    folder.mkdir(parents=True, exist_ok=True)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError("Cannot open camera; check permissions and --camera")
    writer = None
    path = None
    start = None
    frames = 0
    target_frames = max(1, round(args.seconds * args.fps))
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera stopped returning frames")
            # Store mirrored input in BOTH recording and demo.
            frame = cv2.flip(frame, 1)
            now = time.monotonic()
            message = f"{args.person} / {args.label}: SPACE record, Q quit"
            if start is not None:
                if now < start:
                    message = f"Get ready: {start - now:.1f}s"
                else:
                    if writer is None:
                        path = folder / f"{uuid.uuid4().hex}.avi"
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), args.fps, (w, h))
                        if not writer.isOpened():
                            raise RuntimeError("Video writer failed; check MJPG codec support")
                    if now >= start + frames / args.fps:
                        writer.write(frame)  # Never record the overlay text.
                        frames += 1
                    message = f"RECORD {frames}/{target_frames}: {args.label}"
                    if frames >= target_frames:
                        writer.release()
                        writer = None
                        if now - start > args.seconds + 0.3:
                            path.unlink(missing_ok=True)
                            print("Discarded slow capture; lower --fps and try again")
                        else:
                            print(f"Saved {path}")
                        path = None
                        start = None
                        frames = 0
            cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Record signs", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and start is None:
                start = time.monotonic() + 3.0
    finally:
        if writer is not None:
            writer.release()
        if path is not None:
            path.unlink(missing_ok=True)  # Discard an interrupted clip.
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
