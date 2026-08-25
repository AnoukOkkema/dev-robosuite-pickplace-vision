from pathlib import Path
from typing import Any, Mapping

import yaml

from src.util.types import (
    PoseConfig,
    RoboflowConfig,
    SystemConfig,
    UltralyticsConfig,
    YoloConfig,
)

class ConfigReader:
    """Read and parse YAML configuration from disk."""

    def __init__(self, config_path: Path):
        self._config_path = config_path

    def read(self) -> dict[str, Any]:
        if not self._config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found at '{self._config_path.resolve()}'"
            )

        try:
            with open(self._config_path, "r") as file:
                raw = yaml.safe_load(file)
        except yaml.YAMLError as error:
            raise ValueError(
                f"Invalid YAML syntax in '{self._config_path}': {error}"
            ) from error
        except OSError as error:
            raise RuntimeError(
                f"Failed to read config file '{self._config_path}': {error}"
            ) from error

        if not isinstance(raw, dict):
            raise TypeError(
                f"Invalid data in config structure: expected mapping at root, got {type(raw).__name__}"
            )
        return raw


class ConfigAssembler:
    """Assemble a typed SystemConfig from a raw YAML mapping."""

    def assemble(self, raw: Mapping[str, Any]) -> SystemConfig:
        try:
            return SystemConfig(
                data_folder=raw["DATA_FOLDER"],
                generate_dataset=raw["GENERATE_DATASET"],
                download_dataset=raw["DOWNLOAD_DATASET"],
                train_yolo=raw["TRAIN_YOLO"],
                eval_yolo=raw["EVAL_YOLO"],
                generate_pose_dataset=raw["GENERATE_POSE_DATASET"],
                train_pose_estimator=raw["TRAIN_POSE_ESTIMATOR"],
                eval_pose_estimator=raw["EVAL_POSE_ESTIMATOR"],
                roboflow=RoboflowConfig(
                    workspace_name=raw["ROBOFLOW"]["WORKSPACE_NAME"],
                    project_name=raw["ROBOFLOW"]["PROJECT_NAME"],
                    dataset_folder_name=raw["ROBOFLOW"]["DATASET_FOLDER_NAME"],
                    version_number=raw["ROBOFLOW"]["VERSION_NUMBER"],
                    model_format=raw["ROBOFLOW"]["MODEL_FORMAT"],
                ),
                yolo=YoloConfig(
                    model_name=raw["YOLO"]["MODEL_NAME"],
                    epochs=raw["YOLO"]["EPOCHS"],
                    image_size=raw["YOLO"]["IMAGE_SIZE"],
                    batch_size=raw["YOLO"]["BATCH_SIZE"],
                    patience=raw["YOLO"]["PATIENCE"],
                    project_name=raw["YOLO"]["PROJECT_NAME"],
                    run_name=raw["YOLO"]["RUN_NAME"],
                    export_onnx_threshold=raw["YOLO"]["EXPORT_ONNX_THRESHOLD"],
                    export_onnx_name=raw["YOLO"]["EXPORT_ONNX_NAME"],
                ),
                ultralytics=UltralyticsConfig(
                    wandb=raw["ULTRALYTICS"]["wandb"],
                    tensorboard=raw["ULTRALYTICS"]["tensorboard"],
                    mlflow=raw["ULTRALYTICS"]["mlflow"],
                    comet=raw["ULTRALYTICS"]["comet"],
                    clearml=raw["ULTRALYTICS"]["clearml"],
                ),
                pose=PoseConfig(
                    project_name=raw["POSE"]["PROJECT_NAME"],
                    run_name=raw["POSE"]["RUN_NAME"],
                    pose_dataset_path=raw["POSE"]["POSE_DATASET_PATH"],
                    checkpoint_path=raw["POSE"]["CHECKPOINT_PATH"],
                    onnx_export_path=raw["POSE"]["ONNX_EXPORT_PATH"],
                    num_dataset_images=raw["POSE"]["NUM_DATASET_IMAGES"],
                    pos_image_size=raw["POSE"]["POS_IMAGE_SIZE"],
                    rotation_image_size=raw["POSE"]["ROTATION_IMAGE_SIZE"],
                    dropout=raw["POSE"]["DROPOUT"],
                    batch_size=raw["POSE"]["BATCH_SIZE"],
                    epochs=raw["POSE"]["EPOCHS"],
                    learning_rate=raw["POSE"]["LEARNING_RATE"],
                    val_split=raw["POSE"]["VAL_SPLIT"],
                    test_split=raw["POSE"]["TEST_SPLIT"],
                    rotation_loss_weight=raw["POSE"]["ROTATION_LOSS_WEIGHT"],
                    export_onnx_threshold=raw["POSE"]["EXPORT_ONNX_THRESHOLD"],
                    export_rotation_threshold_deg=raw["POSE"]["EXPORT_ROTATION_THRESHOLD_DEG"],
                    early_stopping_patience=raw["POSE"]["EARLY_STOPPING_PATIENCE"],
                ),
            )
        except KeyError as error:
            raise KeyError(f"Missing required key in config file: {error}") from error
        except TypeError as error:
            raise TypeError(f"Invalid data in config structure: {error}") from error


class SystemConfigurator:
    """Loads and assembles the app's typed SystemConfig from config.yaml."""

    DEFAULT_CONFIG_PATH = Path("config", "config.yaml")

    @classmethod
    def load(cls, config_path: Path = DEFAULT_CONFIG_PATH) -> SystemConfig:
        """
        Reads and assembles the typed SystemConfig.

        Args:
            config_path (Path): Path to the YAML config file.

        Returns:
            SystemConfig: The assembled, typed configuration.
        """

        raw = ConfigReader(config_path).read()
        return ConfigAssembler().assemble(raw)
