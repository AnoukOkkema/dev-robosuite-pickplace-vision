# dev-robosuite-pickplace-vision

A vision pipeline for a robot pick-and-place task. Training images come
from a [robosuite](https://robosuite.ai/) PickPlace simulation (Panda arm +
Robotiq85 gripper, 4 objects: Bread/Can/Cereal/Milk). The images are labeled and hosted on [Roboflow](https://roboflow.com/).

There are two separate pipelines, each with its own entrypoint:

1. **Object detection** ([`train_yolo.py`](train_yolo.py)): trains an
   [Ultralytics YOLO](https://docs.ultralytics.com/) model to find objects
   in an image (class + bounding box), tests it, and exports it to ONNX.
2. **Pose estimation** ([`train_pose.py`](train_pose.py)): takes those
   bounding boxes and trains a `PoseEstimator` model that predicts each
   object's position (`xyz`, camera-frame) and 3D rotation, tests it, and
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

- Full pipeline from synthetic data to trained model: robosuite →
  Roboflow labeling → YOLO detection → pose estimation. No manual data
  collection needed.
- Every step (dataset generation, download, training, evaluation) can be
  turned on/off in one YAML config file, so you can resume the pipeline at
  any stage.
- `PoseEstimator` uses two separate networks: one for position, one for
  rotation. It also uses a rotation loss that understands object symmetry.
- The model is only exported to ONNX if it passes set quality thresholds.
  Training also logs metrics and live pose visualizations to W&B.

## Requirements

- Python >= 3.11, < 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management
- A [Roboflow](https://roboflow.com/) account + API key (to download the
  labeled dataset)
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

`cpu` and `mps` install the standard torch build (CPU / macOS with MPS
support). `gpu` installs torch with CUDA 12.4 and `onnxruntime-gpu`.

Then create a `.env` file with your Roboflow API key:

```
API_KEY=your-roboflow-api-key
```

## Quick start

1. Install dependencies (see above) and add your `.env` with `API_KEY`.
2. Check `config/config.yaml`. By default, every step is turned on except
   dataset generation. So a first run will download the Roboflow dataset,
   train YOLO, and evaluate it.
3. Run the object detection pipeline:

   ```bash
   uv run --no-sync python train_yolo.py
   ```

4. Once a YOLO model is trained and exported to ONNX (it must pass
   `YOLO.EXPORT_ONNX_THRESHOLD` first), run the pose estimation pipeline:

   ```bash
   uv run --no-sync python train_pose.py
   ```

## Configuration (`config/config.yaml`)

| Key | Description |
|---|---|
| `DATA_FOLDER` | Base folder for generated frames (`<DATA_FOLDER>/images`) and downloaded Roboflow datasets. |
| `GENERATE_DATASET` | Turns step 1 (dataset generation) on/off. |
| `DOWNLOAD_DATASET` | Turns step 2 (dataset download from Roboflow) on/off. |
| `TRAIN_YOLO` | Turns step 3 (YOLO training) on/off. |
| `EVAL_YOLO` | Turns step 4 (YOLO evaluation) on/off. |
| `ENVIRONMENT.ROBOT_BASE_OFFSET` | XYZ offset applied to the Panda robot's base position in the pose pipeline's simulation (`PickPlaceWithRobotOffset`). |
| `ROBOFLOW.*` | Workspace, project, version, and model format used to download the dataset from Roboflow. |
| `YOLO.MODEL_NAME` | Base model (e.g. `yolov8n.pt`) that training starts from. |
| `YOLO.EPOCHS` / `IMAGE_SIZE` / `BATCH_SIZE` | Training settings. |
| `YOLO.PATIENCE` | How many epochs to wait, with no improvement, before stopping training early. |
| `YOLO.PROJECT_NAME` / `RUN_NAME` | Names the Ultralytics run folder (train/eval). |
| `YOLO.EXPORT_ONNX_THRESHOLD` | Minimum mAP50-95 score (on the test set) needed to export the model to ONNX. |
| `ULTRALYTICS.*` | Turns off Ultralytics' own trackers (wandb, tensorboard, mlflow, comet, clearml). We log to W&B ourselves instead. |
| `GENERATE_POSE_DATASET` | Turns on/off dataset generation for the pose pipeline (see `train_pose.py`). |
| `TRAIN_POSE_ESTIMATOR` / `EVAL_POSE_ESTIMATOR` | Turn training and evaluation of `PoseEstimator` on/off. |
| `POSE.POSE_DATASET_PATH` | Path to the generated `pose_dataset.pkl` file (used by both train and eval). |
| `POSE.NUM_DATASET_IMAGES` | Number of frames the pose dataset generator creates. |
| `POSE.POS_IMAGE_SIZE` / `ROTATION_IMAGE_SIZE` | Input image size for the position stream (full scene) and the rotation stream (object crop). |
| `POSE.DROPOUT` | Dropout rate used in the class processor and rotation head of `PoseEstimator`. |
| `POSE.ROTATION_LOSS_WEIGHT` | How much the rotation loss counts, relative to the position loss, in the combined training loss. |
| `POSE.EARLY_STOPPING_PATIENCE` | Epochs to wait for `val_loss` to improve before stopping early. `0`/`null` turns this off (always runs the full `POSE.EPOCHS`). |
| `POSE.EXPORT_POSITION_THRESHOLD_CM` / `EXPORT_ROTATION_THRESHOLD_DEG` | Max allowed position error (cm) and rotation error (degrees). **Every class** must pass **both** to export to ONNX (rotation isn't checked for classes in `ROTATION_SYMMETRIC_CLASSES`). |
| `POSE.ROTATION_SYMMETRIC_CLASSES` | Classes that look the same at any rotation (default: `["Can"]`). A can looks and grasps the same no matter how it's turned, so it has no real "correct" rotation to learn from. It's left out of the rotation loss (otherwise its noisy target would hurt every class, since they share one rotation network), the rotation metric, and the rotation part of the export check. Its position is still scored normally. |

Each step is skipped on its own when its flag is `false`. If a later step
needs the output of a skipped one (e.g. training without a downloaded
dataset), the pipeline logs a warning and skips that step too.

## Object detection pipeline (`train_yolo.py`)

Four steps, each turned on/off separately in `config/config.yaml`:

1. **Generate dataset**: create synthetic camera frames from the robosuite
   simulation.
2. **Download dataset**: get the labeled dataset from Roboflow.
3. **Train**: train YOLO on the downloaded dataset, with W&B tracking.
4. **Evaluate**: test the trained model, and export it to ONNX if it beats
   the mAP threshold.

### Step 1: Generate dataset (`DatasetGenerator`)

[`src/data_preparation/dataset_generator.py`](src/data_preparation/dataset_generator.py)

Starts a robosuite `PickPlace` environment (via
`robosuite_env.build_pickplace_env`, using a Panda robot, Robotiq85
gripper, offscreen renderer, and `agentview` camera) and generates training
frames, one at a time:

- pick a random action (`env.action_spec`),
- reset the environment and take that action,
- process the resulting frame: flip it, crop it (`CropRegion`), and convert
  it from RGB to BGR,
- save it as `frame_XXXXX.png` in `<DATA_FOLDER>/images`.

Frame numbering continues from the last existing frame in the output
folder, so you can run the generator again later to add more frames
without overwriting the old ones. Resolution and crop area are set via the
`ImageSize`/`CropRegion` dataclasses
([`src/util/types.py`](src/util/types.py)). This step only creates raw,
unlabeled frames. Labeling is done afterwards by hand, in Roboflow (step 2
then downloads that labeled dataset).

### Step 2: Download dataset (`RoboflowDownloader`)

[`src/data_preparation/data_retriever.py`](src/data_preparation/data_retriever.py)

Downloads a specific dataset version from a Roboflow workspace/project
(`ROBOFLOW.*` in the config), in YOLO format. The dataset is public on
[Roboflow Universe](https://universe.roboflow.com/anouk-okkema/robosuite-pickplace-object-detection-synthetic),
so any valid Roboflow API key can download it. If the dataset folder
already exists locally, it's not downloaded again. After downloading,
`data.yaml` is rewritten so the `train`/`val`/`test` paths point to the
local folder using absolute paths. This is needed because Roboflow's own
paths are relative and don't match where the project actually lives on
disk.

### Step 3: Train (`YOLOTrainer`)

[`src/training/yolo_trainer.py`](src/training/yolo_trainer.py)

Loads the base model (`YOLO.MODEL_NAME`) and trains it with Ultralytics on
the downloaded, labeled dataset (`data.yaml`), using `EPOCHS`,
`IMAGE_SIZE`, `BATCH_SIZE`, and `PATIENCE` (early stopping) from the
config. Results are saved to `<YOLO.PROJECT_NAME>/<YOLO.RUN_NAME>-train`.

### Step 4: Evaluate (`YOLOEvaluator`)

[`src/evaluation/yolo_evaluator.py`](src/evaluation/yolo_evaluator.py)

Loads the best checkpoint (`weights/best.pt`) from training and tests it on
the test split of `data.yaml`, reporting mAP50, mAP50-95, precision, and
recall. If the mAP50-95 score meets `YOLO.EXPORT_ONNX_THRESHOLD`, the model
is automatically exported to ONNX; otherwise the export step is skipped.
Results are saved to `<YOLO.PROJECT_NAME>/<YOLO.RUN_NAME>-test`.

### Experiment tracking (`YOLOWandBCallback`)

[`src/util/yolo_wandb_callback.py`](src/util/yolo_wandb_callback.py)

Starts a Weights & Biases run (when `TRAIN_YOLO` is on) and hooks into the
Ultralytics trainer to log to W&B ourselves, alongside Ultralytics' own
built-in trackers (which are turned off via `ULTRALYTICS.*`). It logs:

- train/validation/test metrics and losses, per epoch,
- learning rates,
- model info (parameters, FLOPs), logged once at the first epoch,
- train/val/test plots as images (confusion matrix, PR curves, example
  batches),
- PR/F1/precision/recall curves, both averaged and per class,
- the best model, saved as a W&B artifact once training finishes.

## Pose estimation pipeline (`train_pose.py`)

Builds a second model on top of the trained/exported YOLO model
(`yolo_detector.onnx`). For each object YOLO detects, this model predicts
its position (`xyz`, camera-frame) **and** its 3D rotation. Three steps,
each turned on/off separately in `config/config.yaml`:

1. **Generate pose dataset** (`GENERATE_POSE_DATASET`): combines live
   robosuite frames, YOLO detections, and the true object poses from `obs`
   (converted from world-frame to camera-frame, see
   `PoseDatasetGenerator`). Saved as `pose_dataset.pkl`
   (`{"frames", "samples", "class_names"}`).
2. **Train** (`TRAIN_POSE_ESTIMATOR`): `PoseTrainer` trains
   `PoseEstimator`, with W&B tracking.
3. **Evaluate** (`EVAL_POSE_ESTIMATOR`): `PoseEvaluator` tests it on the
   test split (position error in cm + rotation error, both overall and per
   class), and exports it to ONNX only if every class passes both
   thresholds.

### Model (`PoseEstimator`)

[`src/models/pose_estimator.py`](src/models/pose_estimator.py)

A two-stream design: position and rotation are predicted by two separate
ResNet18 networks, because each needs a different kind of input image.

- **Position stream**: predicts where the object is.
  - **Input**: the cropped agentview frame (the whole bin, with all four
    objects still visible), plus the object's bounding box
    (`x1,y1,x2,y2,area,cx,cy`, normalized) and its class (one-hot).
  - **Why the whole bin, not just the object?** Any crop gets resized to a
    fixed input size before it reaches the network. So a close, small
    object and a far, large object can end up looking the same size in the
    crop. The object on its own doesn't tell the model how big or far
    away it really is. The bin edges and the other objects around it give
    the model something to measure against, so it can work out real-world
    position. The bounding box is also passed in, so the model knows
    exactly where in that frame the object sits.
  - **Output**: `xyz` (3 numbers).
- **Rotation stream**: predicts how the object is turned.
  - **Input**: a tight crop of just the object, plus its class (passed
    through its own small MLP layer with dropout, then combined with the
    image features).
  - **Why a tight crop, not the whole bin?** In the full bin frame, the
    object is just one small part of the image among four objects. Once
    that frame is resized down to the network's fixed input size, the
    object itself only takes up a handful of pixels, too blurry to tell
    which face or edge is facing the camera. A tight crop spends all its
    pixels on the object alone, so it stays sharp enough to see that
    detail.
  - **Output**: a 6D rotation vector (Zhou et al., 2019).
    `PoseEstimator.rot6d_to_matrix()` turns this into a proper 3x3 rotation
    matrix (via Gram-Schmidt orthogonalization).

The two streams share nothing (separate ResNet18 networks). Only the
training loop and the checkpoint file are shared.

### Training (`PoseTrainer`)

[`src/training/pose_trainer.py`](src/training/pose_trainer.py)

Both streams train together, using one combined loss:

- **Loss**: `xyz MSE + POSE.ROTATION_LOSS_WEIGHT × rotation loss`.
  - For rotation, being off by 180° around the vertical axis counts as a
    smaller mistake than other errors. That's because many objects look
    almost the same when turned 180° that way, so the model shouldn't be
    punished hard for that specific mix-up.
  - Classes in `POSE.ROTATION_SYMMETRIC_CLASSES` (e.g. `Can`) skip the
    rotation loss completely (see the config table above for why).
- **Per-class averaging**: each class counts equally toward the loss,
  instead of every individual sample counting equally. This stops an easy
  or common class from dominating training while a harder class is quietly
  left behind.
- **Optimizer**: one shared Adam optimizer, with `ReduceLROnPlateau` to
  lower the learning rate once progress stalls.
- **Early stopping**: training stops after `POSE.EARLY_STOPPING_PATIENCE`
  epochs with no improvement. The best checkpoint is kept, not the last
  one.

### Evaluation (`PoseEvaluator`)

[`src/evaluation/pose_evaluator.py`](src/evaluation/pose_evaluator.py)

Tests the trained model, then checks if it's good enough to export:

- **Metrics**: for each class, and averaged across all classes:
  - mean position error, in cm
  - mean rotation error, in degrees (a 180° error around the vertical axis
    counts as small, same as in the loss)
  - xyz R² is also logged, but only as extra info. It's not used to
    decide anything. It compares each object's error to the spread of
    *all* objects' positions combined, so a small object can get a good R²
    score while still being off by more than its own tolerance allows.
- **Export check**: the model is exported to ONNX only if **every single
  class** passes both:
  - position error under `POSE.EXPORT_POSITION_THRESHOLD_CM`
  - rotation error under `POSE.EXPORT_ROTATION_THRESHOLD_DEG` (skipped for
    classes in `POSE.ROTATION_SYMMETRIC_CLASSES`)

  An average across classes isn't enough here: one weak class shouldn't
  be able to hide behind the others' good scores. `Can` skips the rotation
  check for the same reason it skips the rotation loss (see the config
  table above); this also matches the robot's controller, which already
  picks `Can` up the same way regardless of rotation. Its position (xyz)
  is unaffected and is still checked normally.

### Visualization (`PoseVisualizer`)

[`src/evaluation/pose_visualizer.py`](src/evaluation/pose_visualizer.py)

Renders live agentview scenes showing the true pose (white) next to the
predicted pose (orange), logged to W&B during training/evaluation. It uses
its own robosuite environment, separate from the training dataset.

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

Automatically finds the best available device (`cuda` → `mps` → otherwise
`cpu`) for torch, and picks the matching `onnxruntime` execution providers
(`CUDAExecutionProvider`/`CoreMLExecutionProvider`, with
`CPUExecutionProvider` always kept as a fallback). This is logged every
time the pipeline runs.

## Logging (`LoggingConfigurator`)

[`src/util/logging_configurator.py`](src/util/logging_configurator.py)

Sets up Python logging from `config/logging.yaml`
(`logging.config.dictConfig`), via `LoggingConfigurator.setup(log_filename)`.
Console output goes to stderr, and a detailed log file is written to
`logs/<log_filename>` (overwritten each run). Each entrypoint passes its
own filename, e.g. `train_yolo.py` uses `train_yolo.log` and
`train_pose.py` uses `train_pose.log`, so the two pipelines don't overwrite
each other's logs. If the YAML file is missing or invalid, it falls back
to a basic configuration.

`LoggingConfigurator.suppress_robosuite_warnings()` is called separately,
before robosuite is imported anywhere, to silence robosuite's own startup
warnings.
