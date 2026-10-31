# Validation performed

- Five unittest tests passed, including synthetic training, validation-based
  checkpoint selection, held-out evaluation, confusion-matrix output, automatic
  and explicit person splits, and rejection of modified keypoint arrays.
- The official downloaded MediaPipe Hand Landmarker performed inference on
  blank frames successfully using the actual Tasks VIDEO API.
- OpenCV wrote and decoded a 30-frame MJPG AVI; extraction correctly rejected
  it because it contained no hands.
- All six command-line entrypoints accepted `--help`.

Test environment: Linux, Python 3.12, PyTorch 2.15.0+cpu, MediaPipe 0.10.35,
OpenCV contrib 4.14.0.94, NumPy 2.5.4, scikit-learn 1.9.1, matplotlib 3.11.2.

A physical webcam/GUI session and real signed clips were not available for
validation. No sign-recognition accuracy is claimed and no trained sign weights
are supplied. Synthetic tests check software plumbing, not recognition quality.
Follow the README to collect data from separate people, train, evaluate, and
check the webcam demo on your machine.
