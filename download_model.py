"""Download the official MediaPipe Hand Landmarker asset once."""
import argparse
from pathlib import Path
from urllib.request import urlopen

URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/hand_landmarker.task"))
    args = parser.parse_args()
    if args.output.exists():
        print(f"Already exists: {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".download")
    try:
        with urlopen(URL, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
