from pathlib import Path
from typing import Optional

from ultralytics import YOLO


class YOLOTrainer:
    """
    Trains a YOLO detection model, using Ultralytics, on a labeled dataset.
    Early stopping and plots are handled by Ultralytics itself.
    """

    def __init__(
        self,
        model_name: str,
        data_yaml_path: str,
        epochs: int,
        image_size: int,
        batch_size: int,
        patience: int,
        device: str,
        project_name: str,
        run_name: str,
        logger,
    ) -> None:
        """
        Sets up the YOLOTrainer and loads the base model.

        Args:
            model_name (str): Base checkpoint to start training from
                (e.g. "yolov8n.pt").
            data_yaml_path (str): Path to the dataset's data.yaml.
            epochs (int): Maximum number of training epochs.
            image_size (int): Training image size.
            batch_size (int): Training batch size.
            patience (int): Epochs without improvement before Ultralytics
                stops training early.
            device (str): Torch device to train on.
            project_name (str): Ultralytics project (output) directory name.
            run_name (str): Run name. The actual run folder becomes
                "{run_name}-train" (see `train`).
            logger: Logger instance.

        Returns:
            None
        """

        self.model_name = model_name

        self.data_yaml_path = Path(data_yaml_path)

        self.epochs = epochs
        self.image_size = image_size
        self.batch_size = batch_size
        self.patience = patience

        self.device = device

        self.project_name = project_name
        self.run_name = run_name

        self.logger = logger

        self.model = YOLO(self.model_name)

        self.logger.info(
            (
                "YOLOTrainer initialized | "
                "model=%s | "
                "epochs=%d | "
                "imgsz=%d | "
                "batch=%d | "
                "patience=%d | "
                "device=%s"
            ),
            self.model_name,
            self.epochs,
            self.image_size,
            self.batch_size,
            self.patience,
            self.device,
        )

    def train(self, enabled: bool = True) -> Optional[Path]:
        """
        Trains the YOLO model.

        Args:
            enabled (bool): If False, training is skipped.

        Returns:
            Optional[Path]: Training output directory, or None if training
                was skipped.
        """

        if not enabled:
            self.logger.info("Training skipped.")
            return None

        self.logger.info("Starting YOLO training...")

        results = self.model.train(
            data=str(self.data_yaml_path),
            epochs=self.epochs,
            imgsz=self.image_size,
            batch=self.batch_size,
            patience=self.patience,
            device=self.device,
            project=str(self.project_name),
            name=f"{self.run_name}-train",
            exist_ok=True,
            plots=True,
        )

        save_dir = Path(results.save_dir)

        self.logger.info("Training completed successfully | save_dir=%s", save_dir)

        return save_dir
