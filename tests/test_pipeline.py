"""Checks for person leakage, feature layout, and a synthetic train/evaluate run.

Synthetic landmarks verify plumbing, not sign-recognition quality.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import FEATURES, FEATURE_VERSION, HandTracker, resample
from dataset import split_people
from model import SignLSTM, load_checkpoint


class PipelineTests(unittest.TestCase):
    def test_resampling_preserves_endpoints(self):
        x = np.arange(3 * FEATURES, dtype=np.float32).reshape(3, FEATURES)
        result = resample(x, 30)
        self.assertEqual(result.shape, (30, FEATURES))
        np.testing.assert_array_equal(result[0], x[0])
        np.testing.assert_array_equal(result[-1], x[-1])
        np.testing.assert_array_equal(resample(x[:1], 30), np.repeat(x[:1], 30, axis=0))
        with self.assertRaises(ValueError):
            resample([], 30)

    def test_people_never_cross_splits(self):
        records = [{"person": person, "label": label}
                   for person in ("alice", "bob", "carol", "dan") for label in ("yes", "no")]
        groups, indices = split_people(records, ["yes", "no"], 42)
        self.assertFalse(set(groups["train"]) & set(groups["val"]))
        self.assertFalse(set(groups["train"]) & set(groups["test"]))
        self.assertFalse(set(groups["val"]) & set(groups["test"]))
        self.assertEqual(sorted(sum(indices.values(), [])), list(range(len(records))))
        self.assertEqual(groups, split_people(records, ["yes", "no"], 42)[0])
        with self.assertRaises(ValueError):
            split_people(records, ["yes", "no"], 42, ["alice"], ["alice"])
        with self.assertRaises(ValueError):
            split_people(records[:4], ["yes", "no"], 42)
        with self.assertRaises(ValueError):
            split_people(records[:-1], ["yes", "no"], 42, ["alice"], ["dan"])

    def test_hand_slots_and_monotonic_timestamps(self):
        # Test our adapter without needing a camera or downloaded model.
        tracker = object.__new__(HandTracker)
        tracker.last_ms = -1
        tracker.mp = SimpleNamespace(Image=lambda **kw: kw,
                                     ImageFormat=SimpleNamespace(SRGB=1))
        point = SimpleNamespace(x=0.2, y=0.3, z=-0.1)
        result = SimpleNamespace(hand_landmarks=[[point] * 21],
                                 handedness=[[SimpleNamespace(category_name="Right", score=0.9)]])
        timestamps = []
        def detect(image, timestamp):
            timestamps.append(timestamp)
            return result
        tracker.detector = SimpleNamespace(detect_for_video=detect)
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        features, _ = tracker.extract(frame, 10)
        tracker.extract(frame, 10)
        self.assertEqual(timestamps, [10, 11])
        self.assertEqual(features[63], 0)
        self.assertEqual(features[127], 1)
        np.testing.assert_array_equal(features[:64], 0)
        result.hand_landmarks, result.handedness = [], []
        missing, _ = tracker.extract(frame, 11)
        self.assertFalse(missing.any())

    def test_lstm_shape_and_gradients(self):
        model = SignLSTM(4, hidden_size=8)
        logits = model(torch.zeros(2, 30, FEATURES))
        self.assertEqual(tuple(logits.shape), (2, 4))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        self.assertIsNotNone(model.lstm.weight_ih_l0.grad)

    def test_synthetic_train_evaluate_and_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            data = folder / "data"
            data.mkdir()
            labels = ["hello", "thanks", "yes", "no"]
            records = []
            rng = np.random.default_rng(3)
            for person in ("alice", "bob", "carol"):
                for label_index, label in enumerate(labels):
                    for clip in range(2):
                        relative = f"{person}/{label}/{clip}.npy"
                        path = data / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        x = rng.normal(0, 0.01, (30, FEATURES)).astype(np.float32)
                        x[:, label_index] += 0.5
                        x[:, 63] = 1
                        np.save(path, x)
                        records.append({"path": relative, "person": person, "label": label,
                                        "source": relative,
                                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            (data / "manifest.json").write_text(json.dumps({
                "labels": labels, "length": 30, "seconds": 2.0,
                "feature_version": FEATURE_VERSION, "records": records}))
            run = folder / "run"
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       MPLCONFIGDIR=str(folder / "matplotlib"))
            def command(script, *args, success=True):
                result = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                                        env=env, capture_output=True, text=True, timeout=120)
                if success:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                return result
            command("train.py", "--data", data, "--output", run, "--epochs", 2,
                    "--hidden-size", 8, "--val-people", "bob", "--test-people", "carol")
            evaluation = folder / "evaluation"
            command("evaluate.py", "--data", data, "--checkpoint", run / "best.pt",
                    "--output", evaluation)
            metrics = json.loads((evaluation / "metrics.json").read_text())
            self.assertEqual(metrics["test_people"], ["carol"])
            matrix = np.array(metrics["confusion_matrix"])
            self.assertEqual(matrix.shape, (4, 4))
            self.assertEqual(matrix.sum(), 8)
            self.assertAlmostEqual(metrics["accuracy"], np.trace(matrix) / matrix.sum())
            self.assertTrue((evaluation / "confusion_matrix.png").is_file())
            automatic = folder / "automatic"
            command("train.py", "--data", data, "--output", automatic, "--epochs", 1,
                    "--hidden-size", 8)
            _, auto_checkpoint = load_checkpoint(automatic / "best.pt")
            self.assertEqual(set(sum(auto_checkpoint["split"]["people"].values(), [])),
                             {"alice", "bob", "carol"})
            # Changing extracted arrays after training must not silently change test results.
            changed = data / records[0]["path"]
            np.save(changed, np.zeros((30, FEATURES), dtype=np.float32))
            result = command("evaluate.py", "--data", data, "--checkpoint", run / "best.pt",
                             "--output", evaluation, success=False)
            self.assertIn("changed since extraction", result.stderr)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
