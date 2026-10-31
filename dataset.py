"""Manifest validation and whole-person splits (never random video splits)."""
import json
import hashlib
import numpy as np
from common import FEATURES, FEATURE_VERSION


def load_data(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["feature_version"] != FEATURE_VERSION:
        raise ValueError("Incompatible feature version")
    records = manifest["records"]
    if not records:
        raise ValueError("Manifest is empty")
    labels = manifest["labels"]
    arrays = []
    for row in records:
        path = root / row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Keypoints changed since extraction: {path}")
        array = np.load(path, allow_pickle=False)
        if array.shape != (manifest["length"], FEATURES) or not np.isfinite(array).all():
            raise ValueError(f"Invalid keypoints: {row['path']}")
        arrays.append(array)
    return manifest, np.stack(arrays), np.array([labels.index(r["label"]) for r in records])


def split_people(records, labels, seed, val_people=None, test_people=None):
    people = sorted({r["person"] for r in records})
    if len(people) < 3:
        raise ValueError("Need at least 3 distinct people: train, validation, test")
    if (val_people is None) != (test_people is None):
        raise ValueError("Specify BOTH --val-people and --test-people, or neither")
    if val_people is None:
        people = np.random.default_rng(seed).permutation(people).tolist()
        count = max(1, int(len(people) * 0.2))
        test_people, val_people = people[:count], people[count:2 * count]
    val_set, test_set = set(val_people), set(test_people)
    known = set(people)
    if not val_set or not test_set or val_set & test_set or not (val_set | test_set) <= known:
        raise ValueError("Validation/test IDs must be known, nonempty and disjoint")
    groups = {"train": sorted(known - val_set - test_set),
              "val": sorted(val_set), "test": sorted(test_set)}
    indices = {}
    for name, ids in groups.items():
        indices[name] = [i for i, row in enumerate(records) if row["person"] in ids]
        present = {records[i]["label"] for i in indices[name]}
        missing = set(labels) - present
        if missing:
            raise ValueError(f"{name} split missing labels {sorted(missing)}; collect more clips or choose other people")
    return groups, indices
