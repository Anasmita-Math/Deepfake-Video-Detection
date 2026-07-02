# Deepfake Video Detection with ResNeXt101 + PyTorch

A face-level deepfake classifier that extracts faces from video frames and uses a fine-tuned **ResNeXt101 (32x8d)** CNN to classify each face as **REAL** or **FAKE**. Per-frame predictions are aggregated to produce a single video-level verdict with a confidence score.

Trained and evaluated on the **FaceForensics++ (FF++)** dataset.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Setup](#setup)
- [Usage](#usage)
  - [1. Face Extraction](#1-face-extraction)
  - [2. Data Preparation](#2-data-preparation)
  - [3. Model Architecture](#3-model-architecture)
  - [4. Training](#4-training)
  - [5. Evaluation](#5-evaluation)
  - [6. Video-Level Inference](#6-video-level-inference)
- [Results](#results)
- [Known Issues / Gaps](#known-issues--gaps)
- [Suggested Improvements](#suggested-improvements)
- [License](#license)

---

## Overview

This project tackles deepfake detection as an **image classification problem applied to video**:

1. Faces are detected and cropped from sampled frames of each video using OpenCV's Haar Cascade face detector.
2. Each cropped face is classified independently by a ResNeXt101-based binary classifier (REAL vs. FAKE).
3. Frame-level probabilities are averaged/aggregated to produce a final video-level label and confidence score.

The notebook (`deep-fake-video-5-resnext101.ipynb`) was originally built and run as a **Kaggle notebook** with GPU acceleration and the FaceForensics++ dataset mounted as a Kaggle input.

---

## How It Works

```
Video File
   │
   ▼
[Frame Sampling]  ── sample N evenly-spaced frames (default: 10)
   │
   ▼
[Face Detection]  ── Haar Cascade (frontal face), largest detected face kept
   │
   ▼
[Face Crop + Resize]  ── resized to 128x128
   │
   ▼
[Preprocessing]  ── ToTensor + ImageNet normalization
   │
   ▼
[ResNeXt101 32x8d]  ── pretrained backbone (frozen) + custom FC head
   │
   ▼
[Per-frame REAL/FAKE probabilities]
   │
   ▼
[Aggregation]  ── average across frames
   │
   ▼
Final Prediction (REAL / FAKE) + Confidence
```

---

## Project Structure

```
.
├── deep-fake-video-5-resnext101.ipynb   # Main notebook (training + evaluation + inference)
├── resnext101_deepfake_faces.pth        # Saved model weights (generated after training)
├── faces_tmp/                           # Temporary directory of extracted face crops (generated)
└── README.md
```

---

## Requirements

- Python 3.11 (as used in the original Kaggle environment)
- GPU strongly recommended (CUDA)

### Python packages

```
torch
torchvision
opencv-python
numpy
scikit-learn
tqdm
matplotlib
seaborn
```

Install with:

```bash
pip install torch torchvision opencv-python numpy scikit-learn tqdm matplotlib seaborn
```

> **Note:** OpenCV must be built with the `data` submodule available (`cv2.data.haarcascades`), which the standard `opencv-python` package provides.

---

## Dataset

The notebook uses the **FaceForensics++ (FF++)** dataset, expected in the following structure (as mounted in the original Kaggle environment):

```
/kaggle/input/faceforensics/FF++/
├── real/    # real videos (.mp4)
└── fake/    # fake / manipulated videos (.mp4)
```

To run this outside Kaggle, download FaceForensics++ (requires requesting access from the dataset authors) and update the paths:

```python
REAL_VID_DIR = "/path/to/FF++/real"
FAKE_VID_DIR = "/path/to/FF++/fake"
```

The notebook uses only the **first 200 videos** from each class (`os.listdir(vid_dir)[:200]`) to keep preprocessing time manageable.

---

## Setup

1. Clone/download this repository and place the notebook in your working directory.
2. Install the dependencies listed above.
3. Update `REAL_VID_DIR` and `FAKE_VID_DIR` to point to your local copy of FaceForensics++.
4. Launch Jupyter and run the notebook cells sequentially:

```bash
jupyter notebook deep-fake-video-5-resnext101.ipynb
```

---

## Usage

### 1. Face Extraction

`extract_faces_from_video()` samples `frame_count` (default 10) evenly-spaced frames from a video, runs Haar Cascade face detection, keeps the **largest** detected face per frame, and resizes it to `128x128`.

```python
faces = extract_faces_from_video(
    video_path="path/to/video.mp4",
    frame_count=10,
    output_size=(128, 128)
)
```

Videos where face detection fails to find faces in all 10 sampled frames are **skipped** from the training set.

### 2. Data Preparation

For each video in `real/` and `fake/`:
- Extracted faces are saved as individual `.jpg` files to `./faces_tmp/`, named `{label}_{video_filename}_{frame_idx}.jpg`.
- Labels: `0 = REAL`, `1 = FAKE`.
- The resulting list of face-image file paths and labels is split into train/validation sets (80/20, stratified) using `train_test_split`.

### 3. Model Architecture

- **Backbone:** `torchvision.models.resnext101_32x8d`, pretrained on ImageNet, with all backbone layers **frozen** (`requires_grad = False`).
- **Classifier head** (trainable):
  ```python
  nn.Sequential(
      nn.Linear(model.fc.in_features, 512),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(512, 2)
  )
  ```
- Output: 2 logits corresponding to `[REAL, FAKE]`.

### 4. Training

```python
train(model, train_loader, val_loader, epochs=40)
```

- **Loss:** `CrossEntropyLoss`
- **Optimizer:** `Adam`, `lr=1e-4`, applied only to the classifier head (`model.fc.parameters()`)
- **Batch size:** 32
- Only the final FC head is trained (transfer learning with a frozen backbone), which makes training fast even without fine-tuning the full network.

After training:

```python
torch.save(model.state_dict(), "resnext101_deepfake_faces.pth")
```

### 5. Evaluation

The notebook reloads the saved weights and evaluates on the held-out validation split (`X_val`, `y_val`), producing:
- Overall accuracy
- Per-class precision / recall / F1 (`classification_report`)
- Confusion matrix (visualized with `seaborn.heatmap`)

### 6. Video-Level Inference

The notebook calls a `predict_video()` function to classify a full video:

```python
label, confidence, details = predict_video(
    sample_path,
    model,
    transform,
    device,
    frame_count=10,
    return_details=True
)
```

It returns:
- `label`: `"REAL"` or `"FAKE"`
- `confidence`: aggregated confidence score
- `details`: dict containing per-frame `real_probs` and `fake_probs`

> ⚠️ **This function is referenced but not defined anywhere in the notebook** — see [Known Issues](#known-issues--gaps) below.

---

## Results

Results below are from the run captured in the notebook (200 real + 200 fake videos, 10 frames each, 40 epochs, backbone frozen):

### Training (final epochs)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|------:|-----------:|----------:|---------:|--------:|
| 32    | 0.132      | 94.82%    | 0.297    | **90.13%** |
| 35    | 0.124      | 95.52%    | 0.309    | 88.98%  |
| 40    | 0.129      | 94.61%    | 0.309    | 88.16%  |

### Held-out Test Evaluation (Cell 9)

**Test Accuracy: 88.16%**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| REAL  | 0.85      | 0.91   | 0.88     | 282     |
| FAKE  | 0.92      | 0.86   | 0.89     | 326     |
| **Accuracy** |    |        | **0.88** | 608     |

### Sample Video-Level Predictions

| Video | True Label | Predicted | Confidence |
|-------|-----------|-----------|------------|
| `01_02__outside_talking_still_laughing__YVGY8LOK.mp4` | FAKE | FAKE | 0.98 |
| `01__podium_speech_happy.mp4` | REAL | REAL | 0.97 |

Both sample predictions were highly confident and correct, with per-frame probabilities consistently aligned with the video-level label.

---

## Limitations

- **`predict_video()` is undefined.** Cells 10 and 11 call this function to run video-level inference, but it is never implemented in the notebook. To reproduce this part, you'll need to write it yourself — logically it should: (1) call `extract_faces_from_video`, (2) apply `transform` to each face and batch them, (3) run the model to get per-frame softmax probabilities, (4) average frame probabilities to get a final label/confidence, and (5) return `(label, confidence, details)` where `details` includes `real_probs` and `fake_probs` lists.
- **Small dataset subset.** Only the first 200 videos per class are used (`[:200]`), which is a small fraction of full FaceForensics++. Results may not generalize to the full dataset or to other deepfake generation methods.
- **Evaluation reuses the validation split as the "test" set.** Cell 9 explicitly reuses `X_val`/`y_val` rather than a held-out test split, so the reported 88.16% test accuracy is technically validation accuracy, not accuracy on unseen data.
- **Face detector is a Haar Cascade**, which is fast but less robust than modern deep-learning face detectors (e.g. MTCNN, RetinaFace), especially for occluded faces, extreme poses, or low-resolution video.
- **Frame-level, not identity/temporal aware.** Faces are classified independently per frame; the model does not use temporal information across frames (e.g. via an RNN/3D-CNN) and does not track a single identity across the video.
- **Class imbalance in confusion matrix.** Support differs between REAL (282) and FAKE (326) in the evaluation split.

---

## Future Work

- Fine-tune deeper layers of ResNeXt101 (currently fully frozen except the FC head) for better feature adaptation.
- Replace Haar Cascade with a deep learning-based face detector (MTCNN / RetinaFace / MediaPipe) for more robust face localization.
- Use a true held-out test set, separate from the validation set used for model selection.
- Incorporate temporal modeling (e.g., LSTM/GRU or 3D-CNN over frame sequences) instead of independent per-frame classification.
- Train on the full FaceForensics++ dataset (or additional datasets like DFDC, Celeb-DF) for better generalization.
- Add data augmentation (random crops, flips, compression artifacts) to improve robustness to real-world video degradation.

---

## License

No license specified. Add a `LICENSE` file if you intend to distribute this project. Note that use of the FaceForensics++ dataset is subject to its own [terms of use](https://github.com/ondyari/FaceForensics), which typically restrict use to research purposes and require signing a usage agreement.
