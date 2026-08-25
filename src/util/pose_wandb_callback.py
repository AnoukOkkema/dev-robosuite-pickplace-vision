from dataclasses import asdict
from typing import List, Optional

import numpy as np
import wandb

from src.util.types import SystemConfig


class PoseWandBCallback:
    """
    Weights & Biases callback shared by PositionTrainer/OrientationTrainer
    and PositionEvaluator/OrientationEvaluator.

    Unlike YOLOWandBCallback (which hooks into Ultralytics' callback
    system), these trainers run a plain PyTorch loop, so this callback is
    invoked directly:
    - log_model_info() once, at the start of training ("Model/...")
    - log_epoch() after each training epoch ("train/...", "val/...") --
      logs whatever metrics the trainer computed (xyz_r2 for position,
      rot_mean_angle_error_deg for orientation), so this one callback works
      for both models.
    - log_scene_media("train", labels_image, ...) each epoch -- ground-truth
      poses on a live agentview scene ("media-train/labels")
    - log_scene_media("val", labels_image, pred_image, ...) each epoch --
      ground-truth + predicted poses ("media-val/labels", "media-val/pred")
    - log_test_results() after the held-out test evaluation ("test/...")
    - log_scene_media("test", ...) once, same as val ("media-test/...")

    System metrics (GPU/CPU/memory) are logged automatically by W&B for any
    active run -- no extra code needed for that.
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
            project_name (str): W&B project name for this run (position or
                orientation have their own project/run names in config.yaml).
            run_name (str): W&B run name.
            config (SystemConfig): Full run config, logged to W&B.
            logger: Logger instance.
            enabled (bool): If False, tracking is skipped.

        Returns:
            Optional[PoseWandBCallback]: The callback, or None if skipped.
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
        Logs train/val metrics for one epoch. Logs whatever keys the caller
        computed (e.g. {"xyz_r2": ...} for position, {"rot_mean_angle_error_deg": ...}
        for orientation), so this works for both PositionTrainer and
        OrientationTrainer.

        Args:
            epoch (int): 1-indexed epoch number.
            train_metrics (dict): Metric name -> value.
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
        drawn ("media-{prefix}/labels"), and optionally the same scenes with
        predicted poses drawn ("media-{prefix}/pred"). Multiple samples
        (rather than one) so a single hard/mislabeled frame doesn't look
        like a systematic model failure.

        Args:
            prefix (str): "train", "val" or "test".
            labels_images (List[np.ndarray]): (H, W, 3) BGR scenes with
                ground-truth poses overlaid.
            pred_images (Optional[List[np.ndarray]]): Same scenes with
                predicted poses overlaid, or None/empty to skip.
            step (Optional[int]): Step to log at (e.g. epoch). Omit for a
                one-off log (e.g. test set).

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
        Logs held-out test set metrics.

        Args:
            metrics (dict): Metric name -> value.

        Returns:
            None
        """

        wandb.log({f"test/{key}": value for key, value in metrics.items()})

        self.logger.info("Logged test metrics to W&B.")
