# dev-robosuite-pickplace-vision

A vision pipeline for a robotic pick-and-place task. Training data is
generated synthetically from a [robosuite](https://robosuite.ai/) PickPlace
simulation (Panda arm + Robotiq85 gripper, 4 objects: Bread/Can/Cereal/Milk),
annotated and hosted via [Roboflow](https://roboflow.com/).

Two independent pipelines are provided, each with its own entrypoint:

1. **Object detection** ([`train_yolo.py`](train_yolo.py)) — trains an
   [Ultralytics YOLO](https://docs.ultralytics.com/) model to detect objects
   (class + bounding box), evaluates it, and exports it to ONNX.
2. **Pose estimation** ([`train_pose.py`](train_pose.py)) — on top of the
   detected bounding boxes, trains a `PoseEstimator` model that predicts each
   object's camera-frame xyz position *and* 3D rotation, evaluates it, and
   exports it to ONNX.

## Table of contents

- [Highlights](#highlights)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Configuration](#configuration-configconfigyaml)
- [Object detection pipeline](#object-detection-pipeline-train_yolopy)
- [Pose estimation pipeline](#pose-estimation-pipeline-train_posepy)
- [Project structure](#project-structure)
- [Devices](#devices-deviceconfigurator)
- [Logging](#logging-setup_logger)

## Highlights

- End-to-end synthetic-data pipeline: robosuite → Roboflow annotation →
  YOLO detection → pose estimation, with no manual data collection.
- Every step (dataset generation, download, training, evaluation) can be
  toggled independently via one YAML config, so the pipeline can be resumed
  at any stage.
- Two-stream `PoseEstimator` (separate ResNet18 backbones for xyz and
  rotation) with a symmetry-aware geodesic rotation loss.
- Automatic ONNX export gated on evaluation thresholds, plus W&B tracking
  and live pose visualizations out of the box.

## Requirements

- Python >= 3.11, < 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management
- A [Roboflow](https://roboflow.com/) account + API key (for downloading the
  annotated dataset)
- A [Weights & Biases](https://wandb.ai/) account (for experiment tracking)

## Installation

This project uses [uv](https://docs.astral.sh/uv/). Pick the extra that
matches your hardware:

```bash
uv sync --extra cpu
```

```bash
uv sync --extra mps
```

```bash
uv sync --extra gpu
```

`cpu` and `mps` use the standard torch wheels (CPU / macOS with MPS
support), `gpu` installs torch with CUDA 12.4 and `onnxruntime-gpu`.

Then create a `.env` file with your Roboflow API key:

```
API_KEY=your-roboflow-api-key
```

## Quick start

1. Install dependencies (see above) and add your `.env` with `API_KEY`.
2. Check `config/config.yaml` — by default all steps are enabled, so a
   first run will generate a dataset, download the Roboflow annotations,
   train YOLO, and evaluate it.
3. Run the object detection pipeline:

   ```bash
   uv run --no-sync python train_yolo.py
   ```

4. Once a YOLO model has been trained and exported to ONNX (evaluation must
   clear `YOLO.EXPORT_ONNX_THRESHOLD`), run the pose estimation pipeline:

   ```bash
   uv run --no-sync python train_pose.py
   ```

## Configuration (`config/config.yaml`)

| Key | Description |
|---|---|
| `DATA_FOLDER` | Base folder where generated frames (`<DATA_FOLDER>/images`) and downloaded Roboflow datasets are stored. |
| `GENERATE_DATASET` | Toggles step 1 (dataset generation). |
| `DOWNLOAD_DATASET` | Toggles step 2 (dataset download from Roboflow). |
| `TRAIN_YOLO` | Toggles step 3 (YOLO training). |
| `EVAL_YOLO` | Toggles step 4 (YOLO evaluation). |
| `ENVIRONMENT.ROBOT_BASE_OFFSET` | World-frame XYZ offset applied to the Panda base in the pose pipeline's simulation (`PickPlaceWithRobotOffset`). |
| `ROBOFLOW.*` | Workspace/project/version/model format for the Roboflow download. |
| `YOLO.MODEL_NAME` | Base model (e.g. `yolov8n.pt`) training starts from. |
| `YOLO.EPOCHS` / `IMAGE_SIZE` / `BATCH_SIZE` | Training hyperparameters. |
| `YOLO.PATIENCE` | Number of epochs without improvement before early stopping kicks in. |
| `YOLO.PROJECT_NAME` / `RUN_NAME` | Naming for the Ultralytics run folder (train/eval). |
| `YOLO.EXPORT_ONNX_THRESHOLD` | Minimum mAP50-95 score on the test set required to export the model to ONNX. |
| `ULTRALYTICS.*` | Disables Ultralytics' built-in trackers (wandb, tensorboard, mlflow, comet, clearml) — we log to W&B ourselves. |
| `GENERATE_POSE_DATASET` | Toggles dataset generation for the pose pipeline (see `train_pose.py`). |
| `TRAIN_POSE_ESTIMATOR` / `EVAL_POSE_ESTIMATOR` | Toggle training and evaluation of `PoseEstimator`, respectively. |
| `POSE.POSE_DATASET_PATH` | Path to the generated `pose_dataset.pkl` (shared between train/eval). |
| `POSE.NUM_DATASET_IMAGES` | Number of frames the pose dataset generator builds. |
| `POSE.POS_IMAGE_SIZE` / `ROTATION_IMAGE_SIZE` | Input resolution of the full-scene image (xyz stream) and the object crop (rotation stream), respectively. |
| `POSE.DROPOUT` | Dropout probability in the class processor and rotation head of `PoseEstimator`. |
| `POSE.ROTATION_LOSS_WEIGHT` | Weight of the rotation loss relative to the xyz loss in the combined training loss. |
| `POSE.EARLY_STOPPING_PATIENCE` | Epochs to wait for val_loss to improve before stopping early. `0`/`null` disables it (always runs the full `POSE.EPOCHS`). |
| `POSE.EXPORT_POSITION_THRESHOLD_CM` / `EXPORT_ROTATION_THRESHOLD_DEG` | Maximum mean absolute xyz error (cm) and mean rotation error (degrees) — **every class** must meet **both** to export to ONNX. |

Each step is independently skipped when its flag is `false`. If a later step
needs the output of a skipped one (e.g. training without a downloaded
dataset), the pipeline logs a warning and skips that step too.

## Object detection pipeline (`train_yolo.py`)

Four steps, each toggled independently via `config/config.yaml`:

1. **Generate dataset** — synthetic camera frames from the robosuite
   simulation.
2. **Download dataset** — fetch the annotated dataset from Roboflow.
3. **Train** — train YOLO on the downloaded dataset, with W&B tracking.
4. **Evaluate** — evaluate the trained model on the test set and (above an
   mAP threshold) export it to ONNX.

### Step 1 — Generate dataset (`DatasetGenerator`)

[`src/data_preparation/dataset_generator.py`](src/data_preparation/dataset_generator.py)

Initializes a robosuite `PickPlace` environment (via
`robosuite_env.build_pickplace_env` — Panda robot, Robotiq85 gripper,
offscreen renderer, `agentview` camera) and generates synthetic training
frames:

- samples a random action per frame (`env.action_spec`),
- resets the environment and steps the action,
- processes the observation: flip, crop (`CropRegion`), and RGB→BGR
  conversion,
- saves the result as `frame_XXXXX.png` in `<DATA_FOLDER>/images`.

The index automatically continues after the last existing frame in the
output folder, so you can run the generator multiple times to extend the
dataset without overwriting existing frames. Resolution and crop area are
passed as `ImageSize`/`CropRegion` dataclasses
([`src/util/types.py`](src/util/types.py)). This step only produces raw,
unlabeled frames — labeling happens afterwards, by hand, in Roboflow (step 2
downloads the annotated dataset).

### Step 2 — Download dataset (`RoboflowDownloader`)

[`src/data_preparation/data_retriever.py`](src/data_preparation/data_retriever.py)

Fetches a specific dataset version from a Roboflow workspace/project
(`ROBOFLOW.*` in the config) in YOLO format. If the project folder already
exists locally, it is not downloaded again. After downloading, `data.yaml`
is rewritten so the `train`/`val`/`test` paths point to the local project
folder with absolute paths — necessary because Roboflow returns relative
paths that don't match where the project lives locally.

### Step 3 — Train (`YOLOTrainer`)

[`src/training/yolo_trainer.py`](src/training/yolo_trainer.py)

Loads the base model (`YOLO.MODEL_NAME`) and trains with Ultralytics on the
downloaded/labeled dataset (`data.yaml`), using `EPOCHS`, `IMAGE_SIZE`,
`BATCH_SIZE`, and `PATIENCE` (early stopping) from the config. Results are
written to `<YOLO.PROJECT_NAME>/<YOLO.RUN_NAME>-train`.

### Step 4 — Evaluate (`YOLOEvaluator`)

[`src/evaluation/yolo_evaluator.py`](src/evaluation/yolo_evaluator.py)

Loads the best checkpoint (`weights/best.pt`) from the training run and
evaluates it on the test split of `data.yaml`, reporting mAP50, mAP50-95,
precision, and recall. If the mAP50-95 score meets
`YOLO.EXPORT_ONNX_THRESHOLD`, the model is automatically exported to ONNX
(`imgsz` from `YOLO.IMAGE_SIZE`); otherwise the export is skipped. Results
are written to `<YOLO.PROJECT_NAME>/<YOLO.RUN_NAME>-test`.

### Experiment tracking (`YOLOWandBCallback`)

[`src/util/yolo_wandb_callback.py`](src/util/yolo_wandb_callback.py)

Starts a Weights & Biases run (when `TRAIN_YOLO` is enabled) and attaches
callbacks to the Ultralytics trainer to log to W&B ourselves, in addition to
Ultralytics' built-in trackers (disabled via `ULTRALYTICS.*`):

- train/validation/test metrics and losses, per epoch,
- learning rates,
- model info (parameters, FLOPs) at the first epoch,
- train/val/test plots as media (confusion matrix, PR curves, example
  batches),
- PR/F1/precision/recall curves (averaged and per class),
- the best model as a W&B artifact once training finishes.

## Pose estimation pipeline (`train_pose.py`)

Builds on the trained/exported YOLO model (`yolo_detector.onnx`) a second
model that predicts, for each detected object, the camera-frame `xyz`
position **and** 3D rotation. Three steps, each toggled independently via
`config/config.yaml`:

1. **Generate pose dataset** (`GENERATE_POSE_DATASET`) — live robosuite
   frames + YOLO detections + ground-truth object poses from `obs`, in the
   camera frame rather than the world frame (see `PoseDatasetGenerator`).
   Saved as `pose_dataset.pkl` (`{"frames", "samples", "class_names"}`).
2. **Train** (`TRAIN_POSE_ESTIMATOR`) — `PoseTrainer` trains `PoseEstimator`
   with W&B tracking.
3. **Evaluate** (`EVAL_POSE_ESTIMATOR`) — `PoseEvaluator` evaluates on the
   test split (xyz error in cm + rotation error, overall/macro-averaged and
   per class) and exports to ONNX only if every class meets both
   thresholds.

### Model (`PoseEstimator`)

[`src/models/pose_estimator.py`](src/models/pose_estimator.py)

A two-stream architecture, since xyz and rotation fundamentally need
different visual information:

- **xyz stream**: the full agentview frame + bbox features (`x1,y1,x2,y2,
  area,cx,cy`, normalized) + class one-hot → ResNet18 → xyz. A cropped
  object alone carries no scale/position information to regress absolute
  xyz; the full frame provides that context.
- **rotation stream**: a cropped object image (which preserves fine visual
  detail — which face/edge is visible — that a full-frame view would lose)
  → ResNet18, with class one-hot passed through its own small MLP branch
  (with dropout) → fused → 6D rotation (Zhou et al., 2019).
  `PoseEstimator.rot6d_to_matrix()` converts this to an orthonormal 3x3
  rotation matrix via Gram-Schmidt orthogonalization.

The two streams share nothing (separate ResNet18 backbones) — only the
training loop and checkpoint are shared.

### Training (`PoseTrainer`)

[`src/training/pose_trainer.py`](src/training/pose_trainer.py)

Combined loss: xyz MSE + `POSE.ROTATION_LOSS_WEIGHT` × a symmetry-aware
geodesic rotation loss (angular distance between the predicted rotation
matrix and the nearest symmetry-equivalent one — a generic
180°-about-the-local-z-axis candidate, applied to all classes, so that
visually indistinguishable rotations aren't unfairly penalized). Both
losses are macro-averaged per class (each class present in a batch
contributes equally to the gradient) rather than pooled across the whole
batch — a pooled loss lets visually easier classes dilute a harder one's
signal, letting it quietly stay under-optimized. Shared Adam optimizer +
`ReduceLROnPlateau` scheduler. Early stops after
`POSE.EARLY_STOPPING_PATIENCE` epochs without val_loss improvement (the
best checkpoint so far is kept).

### Evaluation (`PoseEvaluator`)

[`src/evaluation/pose_evaluator.py`](src/evaluation/pose_evaluator.py)

Evaluates on the test split: mean absolute xyz error (cm) and average
rotation error (degrees, symmetry-aware), both macro-averaged (equal
weight per class) plus a full per-class breakdown. xyz R^2 is also logged,
but only as a secondary reference — it's normalized by the *combined*
position variance across every class (spanning the whole table), so a few
cm of residual on one small object can round to ~1.0 there while still
being fatal for that object's own grasp tolerance. Exports to ONNX
(`image, bbox, class_onehot, crop` → `xyz, rot6d`) only if **every class**
meets **both** `POSE.EXPORT_POSITION_THRESHOLD_CM` (xyz error) **and**
`POSE.EXPORT_ROTATION_THRESHOLD_DEG` (rotation error) — a class-averaged
pass is not enough, since it can hide one weak class behind the others.

### Visualization (`PoseVisualizer`)

[`src/evaluation/pose_visualizer.py`](src/evaluation/pose_visualizer.py)

Renders live agentview scenes with ground-truth (white) vs. predicted
(orange) poses, for W&B media logging during training/evaluation. Builds
its own robosuite environment, independent of the training dataset.

## Project structure

```
dev-robosuite-pickplace-vision/
├── config/
│   ├── config.yaml          # Pipeline configuration (see above)
│   └── logging.yaml         # Logging configuration (Python logging.dictConfig)
├── src/
│   ├── data_preparation/
│   │   ├── robosuite_env.py           # Builds the shared robosuite PickPlace environment
│   │   ├── dataset_generator.py       # Generates raw (unlabeled) frames via robosuite
│   │   ├── data_retriever.py          # Downloads the labeled dataset from Roboflow
│   │   ├── pose_dataset_generator.py  # Generates labeled pose samples (xyz + rotation)
│   │   ├── pose_dataset.py            # torch Dataset over pose_dataset.pkl
│   │   └── pose_data_loader.py        # Builds train/val/test DataLoaders
│   ├── models/
│   │   └── pose_estimator.py      # PoseEstimator (xyz + 6D rotation, two-stream ResNet18)
│   ├── training/
│   │   ├── yolo_trainer.py        # YOLO training (Ultralytics)
│   │   └── pose_trainer.py        # PoseEstimator training (joint xyz + rotation loss)
│   ├── evaluation/
│   │   ├── yolo_evaluator.py      # YOLO evaluation + optional ONNX export
│   │   ├── onnx_detector.py       # Inference wrapper around yolo_detector.onnx
│   │   ├── pose_evaluator.py      # PoseEstimator evaluation + optional ONNX export
│   │   └── pose_visualizer.py     # Renders live scenes with predicted/ground-truth poses
│   └── util/
│       ├── types.py                  # Typed dataclasses for config/devices
│       ├── system_configurator.py    # Reads config.yaml into SystemConfig
│       ├── device_configurator.py    # Detects cuda/mps/cpu + onnx providers
│       ├── logging_configurator.py   # Sets up logging from logging.yaml
│       ├── yolo_wandb_callback.py    # W&B logging during YOLO training/evaluation
│       └── pose_wandb_callback.py    # W&B logging during pose training/evaluation
├── train_yolo.py              # Entrypoint of the object detection pipeline
├── train_pose.py              # Entrypoint of the pose estimation pipeline
├── pyproject.toml             # Dependencies + uv extras (cpu/mps/gpu)
└── uv.lock
```

## Devices (`DeviceConfigurator`)

[`src/util/device_configurator.py`](src/util/device_configurator.py)

Automatically detects the available compute device (`cuda` → `mps` →
otherwise `cpu`) for torch, and determines the matching `onnxruntime`
execution providers (`CUDAExecutionProvider`/`CoreMLExecutionProvider`, with
`CPUExecutionProvider` always as a fallback). Logged on every pipeline run.

## Logging (`setup_logger`)

[`src/util/logging_configurator.py`](src/util/logging_configurator.py)

Configures Python logging from `config/logging.yaml`
(`logging.config.dictConfig`): console output to stderr and a detailed file
log to `logs/main.log` (overwritten on each run). Falls back to a basic
configuration if the YAML file is missing or invalid.
