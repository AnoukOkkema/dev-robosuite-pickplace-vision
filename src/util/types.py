from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class RoboflowConfig:
    """Roboflow project to download the labeled dataset from (config.yaml `ROBOFLOW`)."""

    workspace_name: str
    project_name: str
    # Local folder name for the downloaded dataset. This does not have to
    # match the Roboflow project's slug (see
    # RoboflowDownloader.resolve_local_folder_path).
    dataset_folder_name: str
    version_number: int
    model_format: str


@dataclass
class EnvironmentConfig:
    """Simulation settings that must match the real robot setup (config.yaml `ENVIRONMENT`).

    robot_base_offset must match the value used in the other repo that
    actually controls the robot. Here is why: PoseEstimator's xyz stream
    looks at the cropped agentview frame (see crop_region), and the robot
    arm is visible in that frame. So the training scenes need to show the
    arm in the same place it will be in production. If they don't match,
    the model partly learns to guess position from where the arm appears
    on screen, and it will be off by roughly the size of that mismatch.
    """

    robot_base_offset: Tuple[float, float, float]


@dataclass
class YoloConfig:
    """YOLO training/export settings (config.yaml `YOLO`)."""

    model_name: str
    epochs: int
    image_size: int
    batch_size: int
    # Epochs without improvement before early stopping kicks in.
    patience: int
    project_name: str
    run_name: str
    # Minimum test-set mAP50-95 required to export the trained model to ONNX.
    export_onnx_threshold: float
    export_onnx_name: str


@dataclass
class UltralyticsConfig:
    """
    On/off switches for Ultralytics' own built-in experiment trackers
    (config.yaml `ULTRALYTICS`). These are kept off, since this project
    logs to W&B itself instead, through YOLOWandBCallback and
    PoseWandBCallback.
    """

    wandb: bool
    tensorboard: bool
    mlflow: bool
    comet: bool
    clearml: bool


@dataclass
class PoseConfig:
    """Pose-estimation pipeline settings (config.yaml `POSE`)."""

    project_name: str
    run_name: str
    # Used by both dataset generation and train/eval. See
    # PoseDatasetGenerator.
    pose_dataset_path: str
    checkpoint_path: str
    onnx_export_path: str
    # Number of frames PoseDatasetGenerator builds the dataset from.
    num_dataset_images: int
    # Input resolution of the full-scene image (xyz stream).
    pos_image_size: int
    # Input resolution of the cropped object image (rotation stream).
    rotation_image_size: int
    # Dropout probability in the class-processor and rotation head.
    dropout: float
    batch_size: int
    epochs: int
    learning_rate: float
    early_stopping_patience: int
    val_split: float
    test_split: float
    # Weight of the rotation loss relative to the xyz loss in the
    # combined training loss.
    rotation_loss_weight: float
    # Maximum mean absolute xyz error (cm) *every* class must meet to
    # export the trained model to ONNX.
    export_position_threshold_cm: float
    # Maximum mean rotation error (degrees) that *every* class must meet
    # to export to ONNX. Both this and export_position_threshold_cm must
    # be met, per class. A good score averaged across classes is not
    # enough on its own.
    export_rotation_threshold_deg: float
    # Classes with full rotational symmetry around their vertical axis
    # (for example, a cylindrical can). Any yaw angle looks the same for
    # these classes, so their "correct" rotation is meaningless noise to
    # train on. They are left out of the rotation loss, the reported
    # average rotation metric, and the rotation half of the export check.
    # Their position is still checked normally.
    rotation_symmetric_classes: List[str] = field(default_factory=list)


@dataclass
class SystemConfig:
    """Top-level, typed view of config.yaml. See SystemConfigurator.load()."""

    data_folder: str
    generate_dataset: bool
    download_dataset: bool
    train_yolo: bool
    eval_yolo: bool
    generate_pose_dataset: bool
    train_pose_estimator: bool
    eval_pose_estimator: bool
    environment: EnvironmentConfig
    roboflow: RoboflowConfig
    yolo: YoloConfig
    ultralytics: UltralyticsConfig
    pose: PoseConfig


@dataclass
class DeviceConfig:
    """Compute device resolved by DeviceConfigurator, for torch and ONNX Runtime."""

    torch_device: str
    # Execution providers in priority order (see DeviceConfigurator._resolve_onnx_providers).
    onnx_providers: list[str] = field(default_factory=list)


@dataclass
class ImageSize:
    """Camera/image resolution in pixels."""

    height: int
    width: int


@dataclass
class CropRegion:
    """Pixel crop bounds applied to a captured frame: [y1:y2, x1:x2]."""

    y1: int
    y2: int
    x1: int
    x2: int


@dataclass
class Detection:
    """A single YOLO detection for one object in one frame."""

    box: Tuple[float, float, float, float]  # x1, y1, x2, y2 in original image space
    confidence: float
    class_id: int


@dataclass
class PoseLabel:
    """Camera-frame ground-truth pose of an object."""

    xyz: np.ndarray  # (3,) position in camera-frame coordinates
    rot_cam: np.ndarray  # (3, 3) rotation matrix, camera-frame


@dataclass
class PoseSample:
    """
    A single training sample for the pose model. It holds a detected
    object's bounding box (in its full agentview frame, see
    PoseDatasetGenerator's frames dict), plus its ground-truth pose in
    camera-frame coordinates. The model takes the full frame as input,
    not just a crop of the object, with the bounding box passed in as
    extra input. A tightly cropped and resized image of the object alone
    does not carry enough scale or position information to predict its
    absolute xyz position on its own.
    """

    frame_index: int  # index into PoseDatasetGenerator's saved frames dict
    class_name: str
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 in the full frame
    xyz_cam: np.ndarray  # (3,) ground-truth position, camera-frame
    rot_cam: np.ndarray  # (3, 3) ground-truth rotation matrix, camera-frame


@dataclass
class PoseDatasetGeneratorConfig:
    """Settings for PoseDatasetGenerator's detection and sampling pass."""

    save_path: str = "./data/pose_dataset/pose_dataset.pkl"
    # Minimum YOLO detection confidence for a box to be used as a sample.
    conf_threshold: float = 0.25
    # IoU threshold for YOLO's NMS.
    iou_threshold: float = 0.45
    # Number of frames captured and run through detection per ONNX
    # forward pass.
    batch_size: int = 16
    # Safety cap: give up after num_images * max_attempts_multiplier frames,
    # in case the detector rarely produces a clean one-per-class result.
    max_attempts_multiplier: int = 20
