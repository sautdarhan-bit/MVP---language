"""Shared preprocessing. Keep training and webcam features identical."""
from pathlib import Path
import re
import numpy as np

FEATURES = 128  # 2 hands * (21 landmarks * 3 coordinates + presence flag)
FEATURE_VERSION = "image_xyz_handedness_v1"


def read_labels(path):
    labels = [s.strip() for s in Path(path).read_text().splitlines() if s.strip()]
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("labels.txt needs at least two unique labels")
    for label in labels:
        safe_name(label)
    return labels


def safe_name(name):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Use only letters, digits, underscores and hyphens for IDs/labels")
    return name


def resample(sequence, length):
    """Uniform temporal sampling: repeat short sequences, downsample long ones."""
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[1] != FEATURES or len(sequence) == 0:
        raise ValueError(f"Expected nonempty (frames, {FEATURES}) sequence")
    indices = np.rint(np.linspace(0, len(sequence) - 1, length)).astype(int)
    return sequence[indices]


class HandTracker:
    def __init__(self, model_path):
        import mediapipe as mp
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Missing {model_path}; run python download_model.py")
        self.mp = mp
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=2,
            )
        )
        self.last_ms = -1

    def extract(self, bgr, timestamp_ms):
        import cv2
        # Tasks VIDEO timestamps must increase strictly, even with unusual video FPS.
        timestamp_ms = max(int(timestamp_ms), self.last_ms + 1)
        self.last_ms = timestamp_ms
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self.detector.detect_for_video(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), timestamp_ms
        )
        features = np.zeros((2, 64), dtype=np.float32)
        scores = [-1.0, -1.0]
        for points, handedness in zip(result.hand_landmarks, result.handedness):
            category = handedness[0]
            slot = 0 if category.category_name == "Left" else 1
            # A mistaken duplicate handedness should not overwrite the better detection.
            if category.score <= scores[slot]:
                continue
            scores[slot] = category.score
            features[slot, :63] = np.array([(q.x, q.y, q.z) for q in points]).reshape(-1)
            features[slot, 63] = 1.0
        # Keep image-relative position: location and motion can distinguish signs.
        # Missing hands stay all-zero; the explicit flag distinguishes missing data.
        return features.reshape(-1), result

    def close(self):
        self.detector.close()


def draw_hands(frame, result):
    import cv2
    height, width = frame.shape[:2]
    for hand in result.hand_landmarks:
        for point in hand:
            cv2.circle(frame, (int(point.x * width), int(point.y * height)), 3, (0, 255, 0), -1)
