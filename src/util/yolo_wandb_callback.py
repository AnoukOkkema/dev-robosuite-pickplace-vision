from dataclasses import asdict
from pathlib import Path
from typing import Optional

import wandb
import numpy as np

from ultralytics.utils.torch_utils import (
    model_info_for_loggers
)
from src.training.yolo_trainer import YOLOTrainer
from src.util.types import SystemConfig


class YOLOWandBCallback:
    """
    Custom Weights & Biases callback for Ultralytics YOLO.

    Logs:
    - train metrics
    - validation metrics
    - test metrics
    - learning rates
    - train/val/test media
    - model information
    - PR / F1 / Precision / Recall curves
    - best model artifact
    """

    def __init__(
        self,
        logger
    ) -> None:
        """
        Initializes the callback.

        Args:
            logger: Logger instance.

        Returns:
            None
        """

        self.logger = logger

        self.processed_plots = {}

    @classmethod
    def setup(
        cls,
        trainer: YOLOTrainer,
        config: SystemConfig,
        logger,
        enabled: bool = True,
    ) -> Optional["YOLOWandBCallback"]:
        """
        Starts a W&B run and attaches a YOLOWandBCallback to the trainer.

        Args:
            trainer (YOLOTrainer): Trainer whose model callbacks are wired up.
            config (SystemConfig): Full run config, logged to W&B.
            logger: Logger instance.
            enabled (bool): If False, tracking is skipped.

        Returns:
            Optional[YOLOWandBCallback]: The attached callback,
                or None if tracking was skipped.
        """

        if not enabled:
            logger.info("W&B tracking skipped.")
            return None

        wandb.init(
            project=config.yolo.project_name,
            name=config.yolo.run_name,
            config=asdict(config),
        )

        callback = cls(
            logger=logger
        )

        trainer.model.add_callback(
            "on_train_epoch_end",
            callback.on_train_epoch_end
        )

        trainer.model.add_callback(
            "on_fit_epoch_end",
            callback.on_fit_epoch_end
        )

        trainer.model.add_callback(
            "on_train_end",
            callback.on_train_end
        )

        return callback

    def _log_plots(
        self,
        plots: dict,
        step: int,
        prefix: str
    ) -> None:
        """
        Logs media plots to W&B.

        Args:
            plots (dict): Plot dictionary.
            step (int): Current epoch.
            prefix (str): Namespace prefix.

        Returns:
            None
        """

        for plot_path, params in plots.items():

            timestamp = params["timestamp"]

            if self.processed_plots.get(plot_path) != timestamp:

                wandb.log(
                    {
                        f"media-{prefix}/{Path(plot_path).stem}":
                            wandb.Image(str(plot_path))
                    },
                    step=step
                )

                self.processed_plots[plot_path] = timestamp

    def _log_curves(
        self,
        trainer,
        step: int
    ) -> None:
        """
        Logs PR/F1/P/R curves to W&B.

        Args:
            trainer: Ultralytics trainer instance.
            step (int): Current epoch.

        Returns:
            None
        """

        if not hasattr(
            trainer.validator.metrics,
            "curves_results"
        ):
            return

        class_names = list(
            trainer.validator.metrics.names.values()
        )

        for curve_name, curve_values in zip(
            trainer.validator.metrics.curves,
            trainer.validator.metrics.curves_results
        ):

            x, y, x_title, y_title = curve_values

            data = []

            # ===== MEAN CURVE =====
            mean_curve = np.mean(
                y,
                axis=0
            )

            for xi, yi in zip(x, mean_curve):

                data.append([
                    float(xi),
                    float(yi),
                    "mean"
                ])

            # ===== PER CLASS CURVES =====
            for class_index, class_name in enumerate(
                class_names
            ):

                class_curve = y[class_index]

                for xi, yi in zip(
                    x,
                    class_curve
                ):

                    data.append([
                        float(xi),
                        float(yi),
                        class_name
                    ])

            table = wandb.Table(
                data=data,
                columns=[
                    x_title,
                    y_title,
                    "class"
                ]
            )

            wandb.log(
                {
                    f"curves/{curve_name}":
                        wandb.plot_table(
                            "wandb/area-under-curve/v0",
                            table,
                            {
                                "x": x_title,
                                "y": y_title,
                                "class": "class"
                            },
                            {
                                "title": curve_name,
                                "x-axis-title": x_title,
                                "y-axis-title": y_title,
                            }
                        )
                },
                step=step
            )

    def on_train_epoch_end(
        self,
        trainer
    ) -> None:
        """
        Logs training metrics after each epoch.

        Args:
            trainer: Ultralytics trainer instance.

        Returns:
            None
        """

        epoch = trainer.epoch + 1

        # ===== TRAIN LOSSES =====
        train_metrics = trainer.label_loss_items(
            trainer.tloss,
            prefix="train"
        )

        wandb.log(
            train_metrics,
            step=epoch
        )

        # ===== LEARNING RATES =====
        wandb.log(
            trainer.lr,
            step=epoch
        )

        # ===== MODEL INFO =====
        if epoch == 1:

            wandb.log(
                model_info_for_loggers(
                    trainer
                ),
                step=epoch
            )

        # ===== TRAIN MEDIA =====
        self._log_plots(
            trainer.plots,
            step=epoch,
            prefix="train"
        )

        self.logger.info(
            "Logged training metrics | epoch=%d",
            epoch
        )

    def on_fit_epoch_end(
        self,
        trainer
    ) -> None:
        """
        Logs validation metrics after each epoch.

        Args:
            trainer: Ultralytics trainer instance.

        Returns:
            None
        """

        epoch = trainer.epoch + 1

        validation_metrics = {}

        for key, value in trainer.metrics.items():

            # ===== VALIDATION LOSSES =====
            if key.startswith("val/"):

                validation_metrics[key] = value

            # ===== VALIDATION PERFORMANCE =====
            elif key.startswith("metrics/"):

                clean_key = (
                    key
                    .replace("metrics/", "")
                    .replace("(B)", "")
                )

                validation_metrics[
                    f"val/{clean_key}"
                ] = value

        wandb.log(
            validation_metrics,
            step=epoch
        )

        # ===== VALIDATION MEDIA =====
        self._log_plots(
            trainer.validator.plots,
            step=epoch,
            prefix="val"
        )

        # ===== VALIDATION CURVES =====
        self._log_curves(
            trainer,
            step=epoch
        )

        self.logger.info(
            "Logged validation metrics | epoch=%d",
            epoch
        )

    def on_train_end(
        self,
        trainer
    ) -> None:
        """
        Logs final trained model artifact.

        Args:
            trainer: Ultralytics trainer instance.

        Returns:
            None
        """

        artifact = wandb.Artifact(
            name="best-model",
            type="model"
        )

        artifact.add_file(
            str(trainer.best)
        )

        wandb.log_artifact(
            artifact
        )

        self.logger.info(
            "Logged best model artifact."
        )

    def log_test_results(
        self,
        metrics: dict,
        save_dir: str
    ) -> None:
        """
        Logs test metrics and test media.

        Args:
            metrics (dict): Test metrics.
            save_dir (str): Directory containing test plots.

        Returns:
            None
        """

        # ===== TEST METRICS =====
        test_metrics = {}

        for key, value in metrics.items():

            clean_key = (
                key
                .replace("metrics/", "")
                .replace("(B)", "")
            )

            test_metrics[
                f"test/{clean_key}"
            ] = value

        wandb.log(
            test_metrics
        )

        # ===== TEST MEDIA =====
        save_dir = Path(save_dir)

        for image_path in save_dir.rglob("*"):

            if image_path.suffix.lower() in [
                ".png",
                ".jpg",
                ".jpeg"
            ]:

                wandb.log(
                    {
                        f"media-test/{image_path.stem}":
                            wandb.Image(
                                str(image_path)
                            )
                    }
                )

        self.logger.info(
            "Logged test metrics and media."
        )