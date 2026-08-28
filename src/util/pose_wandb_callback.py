from dataclasses import asdict
from typing import List, Optional

import numpy as np
import wandb

from src.util.types import SystemConfig


class PoseWandBCallback:
    """
    Weights & Biases callback shared by PoseTrainer and PoseEvaluator (the
    same PoseEstimator model, covering both its position and rotation
    streams).

    YOLOWandBCallback hooks into Ultralytics' own callback system. These
    trainers do not use that system: they run a plain PyTorch loop instead,
    so this callback is called directly, in this order:
    - log_model_info(), once, at the start of training ("Model/...").
    - log_epoch(), after each training epoch ("train/...", "val/...").
      It logs whatever metrics the trainer computed (xyz_r2 for position,
      rot_mean_angle_error_deg for orientation), so this one callback works
      for both models.
    - log_scene_media("train", labels_image, ...), each epoch. Logs
      ground-truth poses drawn on a live agentview scene
      ("media-train/labels").
    - log_scene_media("val", labels_image, pred_image, ...), each epoch.
      Logs ground-truth poses plus predicted poses
      ("media-val/labels", "media-val/pred").
    - log_test_results(), after the held-out test evaluation ("test/...").
    - log_scene_media("test", ...), once, same as for val ("media-test/...").

    W&B logs system metrics (GPU, CPU, memory) automatically for any
    active run, so no extra code is needed for that.
    """

    def __init__(self, logger) -> None:
        self.logger = logger

    @classmethod
    def setup(
        cls,
        project_name: str,
        run_name: str,
        config: SystemConfig,
        logger,
        enabled: bool = True,
    ) -> Optional["PoseWandBCallback"]:
        """
        Starts a W&B run.

        Args:
            project_name (str): W&B project name for this run (position
                and orientation each have their own project/run names in
                config.yaml).
            run_name (str): W&B run name.
            config (SystemConfig): The full run config, logged to W&B.
            logger: Logger instance.
            enabled (bool): If False, tracking is skipped.

        Returns:
            Optional[PoseWandBCallback]: The callback, or None if tracking
                was skipped.
        """

        if not enabled:
            logger.info("W&B tracking skipped.")
            return None

        wandb.init(
            project=project_name,
            name=run_name,
            config=asdict(config),
        )

        return cls(logger=logger)

    def log_model_info(self, model, image_size: int) -> None:
        """
        Logs a one-time model summary ("Model/...").

        Args:
            model: The model being trained.
            image_size (int): Input size the model was built for.

        Returns:
            None
        """

        num_parameters = sum(p.numel() for p in model.parameters())
        num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        wandb.log(
            {
                "Model/name": type(model).__name__,
                "Model/backbone": "resnet18",
                "Model/image_size": image_size,
                "Model/num_parameters": num_parameters,
                "Model/num_trainable_parameters": num_trainable_parameters,
            }
        )

        self.logger.info("Logged model info to W&B.")

    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict) -> None:
        """
        Logs train and validation metrics for one epoch. Logs whatever
        keys the caller computed, for example {"xyz_r2": ...} for
        position or {"rot_mean_angle_error_deg": ...} for orientation, so
        this works for both PositionTrainer and OrientationTrainer.

        Args:
            epoch (int): The epoch number, starting at 1.
            train_metrics (dict): Maps metric name to value.
            val_metrics (dict): Same shape as train_metrics.

        Returns:
            None
        """

        log_data = {f"train/{key}": value for key, value in train_metrics.items()}
        log_data.update({f"val/{key}": value for key, value in val_metrics.items()})

        wandb.log(log_data, step=epoch)

    def log_scene_media(
        self,
        prefix: str,
        labels_images: List[np.ndarray],
        pred_images: Optional[List[np.ndarray]] = None,
        step: Optional[int] = None,
    ) -> None:
        """
        Logs a gallery of full agentview scenes with ground-truth poses
        drawn on them ("media-{prefix}/labels"). Optionally also logs the
        same scenes with predicted poses drawn on them
        ("media-{prefix}/pred"). Logging several samples, instead of just
        one, means a single hard or mislabeled frame does not look like a
        systematic model failure.

        Args:
            prefix (str): "train", "val", or "test".
            labels_images (List[np.ndarray]): (H, W, 3) BGR scenes with
                the ground-truth poses drawn on them.
            pred_images (Optional[List[np.ndarray]]): The same scenes
                with the predicted poses drawn on them. Pass None or an
                empty list to skip this.
            step (Optional[int]): The step to log at, e.g. the epoch
                number. Leave this out for a one-off log, such as the
                test set.

        Returns:
            None
        """

        log_data = {
            f"media-{prefix}/labels": [wandb.Image(image[:, :, ::-1]) for image in labels_images],
        }

        if pred_images:
            log_data[f"media-{prefix}/pred"] = [wandb.Image(image[:, :, ::-1]) for image in pred_images]

        log_kwargs = {"step": step} if step is not None else {}
        wandb.log(log_data, **log_kwargs)

    def log_test_results(self, metrics: dict) -> None:
        """
        Logs metrics for the held-out test set.

        Args:
            metrics (dict): Maps metric name to value.

        Returns:
            None
        """

        wandb.log({f"test/{key}": value for key, value in metrics.items()})

        self.logger.info("Logged test metrics to W&B.")
