# Webcam isolated-sign recognition MVP

A small learning project: record short clips, extract MediaPipe hand landmarks,
train a PyTorch LSTM, evaluate on people it has never seen, and display predicted
words over an OpenCV webcam feed. The default vocabulary is `hello`, `thanks`,
`yes`, `no`; edit `labels.txt` before collecting data.

**You must record data and train first. No trained sign classifier is included.**
The downloaded MediaPipe model detects hands; it does not recognize these words.
This is isolated-sign classification, not continuous sign-language translation.
Use signs from one chosen sign language and consistent signing instructions.

## 1. Setup

Use Python 3.11 or 3.12 on a desktop with a webcam. Run commands from this folder.
A local graphical session is required for recording and the demo; extraction,
training and evaluation can run without a display.

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# CPU-only PyTorch (avoids large CUDA packages):
python -m pip install 'torch>=2.5,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python download_model.py
```

For GPU training, install the appropriate PyTorch build using its official
[installation selector](https://pytorch.org/get-started/locally/) before installing
requirements, then pass `--device cuda` to training. CPU is sufficient for an MVP.
On Linux, OpenCV may also need system packages `libgl1` and `libglib2.0-0`.
Use the GUI OpenCV distribution in requirements, not a headless build. MediaPipe
can depend on OpenCV contrib, so this project uses `opencv-contrib-python` rather
than installing competing OpenCV packages. If the camera fails, grant camera
permission, close other camera applications, or try `--camera 1`.

The downloader fetches Google's versioned `hand_landmarker.task` asset over HTTPS
into `models/`. Internet is needed for installation and this download only;
recording, landmarks, training and inference run locally.

## 2. Collect clips from multiple people

Use **at least three different people**, with every label represented in each
split. Five or more people, and roughly 20–50 varied clips per label per person,
are a more useful starting point. Three people are only enough to exercise the
pipeline; results will be noisy. Keep one stable person ID across all of that
person's sessions—changing their ID would leak identity across splits.

```bash
python record.py --person alice --label hello
python record.py --person alice --label thanks
python record.py --person alice --label yes
python record.py --person alice --label no
# Repeat all labels for bob, carol, and preferably more people.
```

For each clip:

1. Press **SPACE**; a three-second countdown gives you time to get ready.
2. Perform exactly one sign during the two-second recording, starting and ending
   in a consistent neutral position with hands visible. Do not concatenate signs.
3. Press SPACE again to collect another example. Press **Q** to quit.

Clips are mirrored, saved without overlays, and named uniquely:

```text
data/videos/
  alice/hello/<unique-id>.avi
  alice/thanks/<unique-id>.avi
  bob/hello/<unique-id>.avi
  ...
```

Defaults are two seconds at 15 FPS (30 frames), saved as MJPG AVI. Slow captures
are discarded with a message; reduce `--fps` if necessary. Vary distance,
background and lighting within each person's set; avoid assigning a different
background to every label. Keep both hands in frame. Imported clips must use the
same mirrored view and duration; the extractor does not mirror again.
Do not publish recordings without the participants' permission.

## 3. Extract landmarks

```bash
python extract.py
```

This writes `data/keypoints/PERSON/LABEL/CLIP.avi.npy` and `manifest.json`.
Each array is `[30, 128]`: two fixed Left/Right slots, each with 21 image-relative
`(x, y, z)` points plus a hand-present flag. A missing hand is all zeros. Hand
tracking resets between clips. Coordinates retain signing location and movement;
this simple baseline does not normalize away position/scale or use pose/face.

Clips are uniformly resampled to 30 steps; all-zero clips and incorrect durations
are rejected. Remove or re-record rejected clips. Extraction requires a new/empty
output folder to prevent mixing old and new preprocessing. If extraction fails
partway through, remove its partial output or choose a new `--output` directory.
The manifest records source paths, labels, person IDs and array hashes.

For a different clip duration, use the same setting when recording/extracting:

```bash
python record.py --person alice --label hello --seconds 3
python extract.py --seconds 3 --length 45 --output data/keypoints_3s
```

The trained checkpoint carries duration and sequence length into the demo. Keep
one duration across the dataset. No interpolation of missing hands is performed.

## 4. Train with a split by person

For a transparent three-person example:

```bash
python train.py --val-people bob --test-people carol --epochs 50
```

All other people go into training. Alternatively, let the seeded splitter choose
whole people (approximately 20% validation, 20% test, remainder training):

```bash
python train.py --output runs/automatic --epochs 50
```

The script rejects overlapping IDs, too few people, or a split missing a label.
It **never splits individual videos randomly**. `split.json` lists exact people
and files in each set. Check that these IDs truly represent different people.

The model is one LSTM layer (64 hidden units), then a linear classifier. Adam
minimizes cross-entropy with gradient clipping. Validation loss selects the best
epoch; test clips never participate in optimization or model selection.
No global dataset normalization is fitted, so there are no fitted preprocessing
statistics that could leak test information.

Outputs:

- `runs/mvp/best.pt`: weights, ordered labels, preprocessing settings, split and
  dataset metadata.
- `runs/mvp/split.json`: audit of the person split.
- `runs/mvp/history.json`: per-epoch training/validation losses and validation accuracy.

Use a new `--output` directory for each run. For custom keypoints use
`--data data/keypoints_3s`. `--seed 42` seeds the split and training RNGs; exact
numerical reproducibility across hardware/PyTorch versions is not guaranteed.

## 5. Evaluate once on held-out people

```bash
python evaluate.py --checkpoint runs/mvp/best.pt
```

Reports clip-level accuracy and per-class precision/recall/F1, and writes:

- `runs/mvp/evaluation/metrics.json`
- `runs/mvp/evaluation/confusion_matrix.csv`
- `runs/mvp/evaluation/confusion_matrix.png`

Matrix rows are true labels, columns predicted labels, in `labels.txt` order
stored at extraction. CSV has a column header; row labels use that same order.
All test clips are scored, without a confidence rejection threshold. The demo's
threshold therefore changes its behavior relative to these metrics.
Evaluation loads the checkpoint's original test files and refuses changed dataset
metadata or altered keypoint arrays. Keep the original extracted dataset.
Use validation people for tuning; repeated test-driven changes make the reported
score optimistic. An unseen-person test with just one signer is not a reliable
estimate for a broad population.

## 6. Run the webcam demo

```bash
python demo.py --checkpoint runs/mvp/best.pt --threshold 0.8
```

Perform one of the trained signs over about two seconds, using the same camera
framing as recording. The rolling window spans the checkpoint's clip duration.
The demo samples it by elapsed time, runs the LSTM roughly every 150 ms and shows
a word after three consecutive confident predictions. It draws hand points too.
**Q** quits and **C** clears the window. Taking both hands out of view clears the
window and prediction immediately; re-entry needs a fresh full window.

Expect at least the clip duration plus a short stability delay before a reliable
word appears. Low scores show `Uncertain`; no detected hands show `No hands`.
Softmax scores are not calibrated probabilities of correctness. An LSTM trained
only on these signs can still confidently mislabel idle motion or unknown signs.
Thresholding does not solve unknown-sign detection. For a stronger demo, add an
`idle` label, record diverse non-sign/transition clips for every person, re-extract
and retrain. It will appear as an ordinary predicted label. There is no automatic
sign-boundary detection or sentence assembly.

Hands alone miss facial expression, body position and linguistic context, so some
signs cannot be distinguished with this representation. This MVP is for learning
and controlled demonstrations, not accessibility-critical interpretation.

## Code map and checks

| File | Purpose |
| --- | --- |
| `record.py` | Countdown and webcam clip recording |
| `extract.py` | Video → fixed-length landmark arrays and manifest |
| `common.py` | Shared hand tracker, feature layout, sampling |
| `dataset.py` | Dataset validation and whole-person splits |
| `model.py` | LSTM and checkpoint loading |
| `train.py` | Training and validation-only checkpoint selection |
| `evaluate.py` | Held-out test metrics and confusion matrix |
| `demo.py` | Live webcam inference and confidence/stability display |
| `download_model.py` | Official MediaPipe hand-model download |

```bash
python -m unittest discover -s tests -v
```

Tests cover person isolation, split errors, temporal sampling, missing-hand slots,
strictly increasing tracker timestamps, LSTM gradients, and a synthetic training →
evaluation run including tamper detection. Synthetic tests verify pipeline
plumbing, not recognition accuracy. Actual accuracy requires your recorded dataset,
and camera/GUI behavior requires a local webcam check.

Implementation references: [MediaPipe Hand Landmarker Python guide](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python),
[PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html),
and [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).
