import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from src.models.pose_estimator import PoseEstimator


class PoseEvaluator:
    """
    Handles PoseEstimator (xyz + rotation, single model) evaluation and
    ONNX export.

    Rotation error accounts for the object's local 180-degree symmetry
    (see `symmetry_local_rotations`): for each sample, the reported error
    is the angle to whichever symmetry-equivalent target rotation is
    closest to the prediction, matching what the model was actually
    trained to minimize (PoseTrainer uses the same candidates). ONNX
    export only happens if BOTH the xyz R^2 score and mean rotation error
    meet their configured thresholds.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str,
        image_size: int,
        rotation_image_size: int,
        num_classes: int,
        logger,
        wandb_callback=None,
        pose_visualizer=None,
    ) -> None:
        """
        Initializes the PoseEvaluator.

        Args:
            checkpoint_path (str): Path to the trained PoseEstimator
                checkpoint (.pt state dict).
            device (str): Torch device to run evaluation on.
            image_size (int): Input resolution of the full-scene image
                (xyz stream), used for the ONNX dummy input.
            rotation_image_size (int): Input resolution of the cropped
                object image (rotation stream), used for the ONNX dummy
                input.
            num_classes (int): Number of object classes the model was
                trained on.
            logger: Logger instance.
            wandb_callback: Optional PoseWandBCallback to log test
                metrics/media to. If None, evaluation still runs but
                nothing is logged to W&B.
            pose_visualizer: Optional PoseVisualizer used to capture
                labeled/predicted scene media for W&B, if wandb_callback
                is also given.

        Returns:
            None
        """

        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.image_size = image_size
        self.rotation_image_size = rotation_image_size
        self.num_classes = num_classes
        self.logger = logger
        self.wandb_callback = wandb_callback
        self.pose_visualizer = pose_visualizer

        # Same generic 180-degree local-symmetry candidate as PoseTrainer --
        # reported errors should reflect what the model was actually trained
        # to minimize.
        self.symmetry_local_rotations = [
            torch.eye(3, device=device),
            torch.tensor(
                [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], device=device
            ),
        ]

        self.logger.info(
            "PoseEvaluator initialized | checkpoint=%s",
            self.checkpoint_path,
        )

    def _load_model(self) -> PoseEstimator:
        """Loads the PoseEstimator checkpoint in eval mode on `self.device`."""

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.checkpoint_path}"
            )

        model = PoseEstimator(num_classes=self.num_classes, pretrained=False)
        model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        return model

    def _r2_score(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """R^2 score, computed over all elements (flattened): 1 - SS_res / SS_tot."""

        residual_sum_squares = torch.sum((targets - preds) ** 2)
        total_sum_squares = torch.sum((targets - targets.mean()) ** 2)

        if total_sum_squares.item() == 0.0:
            return float("nan")

        return (1.0 - residual_sum_squares / total_sum_squares).item()

    def _mae_cm(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """Mean absolute xyz error in centimetres, over all elements (flattened)."""

        return (torch.abs(targets - preds).mean() * 100.0).item()

    def _angle_rad(self, rot_pred: torch.Tensor, rot_target: torch.Tensor) -> torch.Tensor:
        """Per-sample geodesic angle (radians) between two batches of rotation matrices."""

        relative_rotation = torch.matmul(rot_pred.transpose(1, 2), rot_target)
        trace = relative_rotation.diagonal(dim1=1, dim2=2).sum(-1)
        cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)

        return torch.acos(cos_angle)

    def _per_sample_angle_deg(self, rot_pred: torch.Tensor, rot_target: torch.Tensor) -> torch.Tensor:
        """
        Per-sample geodesic rotation error, in degrees, to the *closest*
        symmetry-equivalent target rotation (see symmetry_local_rotations).
        Returns a (N,) tensor -- callers reduce with .mean() (overall) or
        a boolean mask + .mean() (per class).
        """

        candidate_angles = torch.stack(
            [
                self._angle_rad(rot_pred, torch.matmul(rot_target, symmetry))
                for symmetry in self.symmetry_local_rotations
            ],
            dim=0,
        )

        return torch.rad2deg(candidate_angles.min(dim=0).values)

    def _run_test_set(self, model: PoseEstimator, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Runs inference over the full test set and computes overall +
        per-class metrics (xyz R^2, mean rotation angle error).

        Args:
            model (PoseEstimator): Loaded model in eval mode.
            test_loader (DataLoader): Test set loader.

        Returns:
            Dict[str, Any]: `{"xyz_r2", "xyz_mae_cm_macro",
                "rot_mean_angle_error_deg_macro", "rot_mean_angle_error_deg",
                "per_class"}`, where `per_class` maps each class name to its
                own `{"xyz_r2", "xyz_mae_cm", "rot_mean_angle_error_deg",
                "num_samples"}`. The `_macro` scores average the per-class
                scores with equal weight per class -- the metrics that
                actually gate ONNX export (see `evaluate`), since a pooled
                score can hide one weak class behind the others.
        """

        all_xyz_pred, all_xyz_target = [], []
        all_rot_pred, all_rot_target = [], []
        all_class_indices = []

        with torch.no_grad():
            for image, bbox, class_onehot, crop, xyz, rot6d_target in test_loader:
                image = image.to(self.device)
                bbox = bbox.to(self.device)
                class_onehot = class_onehot.to(self.device)
                crop = crop.to(self.device)
                xyz = xyz.to(self.device)
                rot6d_target = rot6d_target.to(self.device)

                xyz_pred, rot6d_pred = model(image, bbox, class_onehot, crop)
                rot_pred_matrix = PoseEstimator.rot6d_to_matrix(rot6d_pred)
                rot_target_matrix = PoseEstimator.rot6d_to_matrix(rot6d_target)

                all_xyz_pred.append(xyz_pred)
                all_xyz_target.append(xyz)
                all_rot_pred.append(rot_pred_matrix)
                all_rot_target.append(rot_target_matrix)
                all_class_indices.append(class_onehot.argmax(dim=1))

        xyz_pred_all = torch.cat(all_xyz_pred, dim=0)
        xyz_target_all = torch.cat(all_xyz_target, dim=0)
        rot_pred_all = torch.cat(all_rot_pred, dim=0)
        rot_target_all = torch.cat(all_rot_target, dim=0)
        class_indices_all = torch.cat(all_class_indices, dim=0)

        per_sample_error_deg = self._per_sample_angle_deg(rot_pred_all, rot_target_all)

        metrics = {
            "xyz_r2": self._r2_score(xyz_pred_all, xyz_target_all),
            "rot_mean_angle_error_deg": per_sample_error_deg.mean().item(),
        }

        class_names = test_loader.dataset.dataset.class_names
        per_class = {}

        for class_index, class_name in enumerate(class_names):
            mask = class_indices_all == class_index

            if mask.sum().item() == 0:
                continue

            per_class[class_name] = {
                "xyz_r2": self._r2_score(xyz_pred_all[mask], xyz_target_all[mask]),
                "xyz_mae_cm": self._mae_cm(xyz_pred_all[mask], xyz_target_all[mask]),
                "rot_mean_angle_error_deg": per_sample_error_deg[mask].mean().item(),
                "num_samples": int(mask.sum().item()),
            }

        # Macro-average (equal weight per class) instead of a pooled mean:
        # gates should reflect the worst class, not be washed out by easier
        # ones (see xyz_r2 docstring).
        metrics["xyz_mae_cm_macro"] = sum(
            class_metrics["xyz_mae_cm"] for class_metrics in per_class.values()
        ) / len(per_class)
        metrics["rot_mean_angle_error_deg_macro"] = sum(
            class_metrics["rot_mean_angle_error_deg"] for class_metrics in per_class.values()
        ) / len(per_class)

        metrics["per_class"] = per_class

        return metrics

    def _export_onnx(self, model: PoseEstimator, onnx_path: Path) -> None:
        """
        Exports `model` to ONNX using dummy inputs of the right shapes,
        with dynamic batch axes on every input/output.

        Args:
            model (PoseEstimator): Loaded model to export.
            onnx_path (Path): Destination path for the .onnx file.

        Returns:
            None
        """

        os.makedirs(onnx_path.parent, exist_ok=True)

        dummy_image = torch.randn(1, 3, self.image_size, self.image_size, device=self.device)
        dummy_bbox = torch.rand(1, PoseEstimator.BBOX_FEATURES, device=self.device)
        dummy_class_onehot = torch.zeros(1, model.num_classes, device=self.device)
        dummy_class_onehot[0, 0] = 1.0
        dummy_crop = torch.randn(1, 3, self.rotation_image_size, self.rotation_image_size, device=self.device)

        torch.onnx.export(
            model,
            (dummy_image, dummy_bbox, dummy_class_onehot, dummy_crop),
            str(onnx_path),
            input_names=["image", "bbox", "class_onehot", "crop"],
            output_names=["xyz", "rot6d"],
            dynamic_axes={
                "image": {0: "batch"},
                "bbox": {0: "batch"},
                "class_onehot": {0: "batch"},
                "crop": {0: "batch"},
                "xyz": {0: "batch"},
                "rot6d": {0: "batch"},
            },
            opset_version=17,
        )

        self.logger.info(
            "ONNX export completed | path=%s",
            onnx_path
        )

    def evaluate(
        self,
        test_loader: DataLoader,
        enabled: bool = True,
        export_position_threshold_cm: float = 1.0,
        export_rotation_threshold_deg: float = 15.0,
        export_rotation_exempt_classes: tuple = (),
        onnx_export_path: str = "./runs/pose_estimator/pose_estimator.onnx",
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates the PoseEstimator model on a held-out test set, and
        exports it to ONNX only if EVERY class's mean absolute xyz error
        (cm) meets its threshold, AND every class *not* in
        export_rotation_exempt_classes also meets the rotation threshold --
        a class-averaged score is not enough, since it can hide one weak
        class behind the others (see xyz_r2's docstring).

        Args:
            test_loader (DataLoader): Test set loader.
            enabled (bool): If False, evaluation is skipped.
            export_position_threshold_cm (float): Maximum mean absolute
                xyz error (cm) every class must meet to export to ONNX.
            export_rotation_threshold_deg (float): Maximum mean rotation
                angular error (degrees) every non-exempt class must meet
                to export to ONNX.
            export_rotation_exempt_classes (tuple): Class names skipped by
                the rotation check -- classes with full rotational
                symmetry (e.g. a cylindrical can) report meaningless
                rotation "error" since any yaw is equally valid.
            onnx_export_path (str): Path to write the ONNX export to.

        Returns:
            Optional[Dict[str, Any]]: Evaluation metrics + onnx_path,
                or None if evaluation was skipped.
        """

        if not enabled:
            self.logger.info("Evaluation skipped.")
            return None

        model = self._load_model()

        self.logger.info("Starting PoseEstimator evaluation...")

        metrics = self._run_test_set(model, test_loader)

        self.logger.info(
            "Evaluation completed | xyz_mae_cm_macro=%.2f | "
            "rot_mean_angle_error_deg_macro=%.2f | xyz_r2=%.4f",
            metrics["xyz_mae_cm_macro"],
            metrics["rot_mean_angle_error_deg_macro"],
            metrics["xyz_r2"],
        )

        for class_name, class_metrics in metrics["per_class"].items():
            self.logger.info(
                "  per class | %-8s | n=%-4d | xyz_mae_cm=%.2f | "
                "rot_mean_angle_error_deg=%.2f | xyz_r2=%.4f",
                class_name,
                class_metrics["num_samples"],
                class_metrics["xyz_mae_cm"],
                class_metrics["rot_mean_angle_error_deg"],
                class_metrics["xyz_r2"],
            )

        if self.wandb_callback is not None:
            # Combine overall + per-class metrics into a single log_test_results()
            # call: log_test_results() doesn't pass a wandb step, so multiple
            # separate calls would each land on their own auto-incremented
            # step instead of together as one "test evaluation" data point.
            wandb_metrics = {
                key: value for key, value in metrics.items() if key != "per_class"
            }

            for class_name, class_metrics in metrics["per_class"].items():
                wandb_metrics.update({
                    f"per_class/{class_name}/{key}": value
                    for key, value in class_metrics.items()
                })

            self.wandb_callback.log_test_results(wandb_metrics)

            if self.pose_visualizer is not None:
                test_labels_images, test_pred_images = self.pose_visualizer.capture_media(model=model)
                self.wandb_callback.log_scene_media("test", test_labels_images, test_pred_images)

        failing_classes = [
            class_name
            for class_name, class_metrics in metrics["per_class"].items()
            if class_metrics["xyz_mae_cm"] > export_position_threshold_cm
            or (
                class_name not in export_rotation_exempt_classes
                and class_metrics["rot_mean_angle_error_deg"] > export_rotation_threshold_deg
            )
        ]
        export_ok = not failing_classes

        onnx_path = None

        if export_ok:
            self.logger.info(
                "Exporting PoseEstimator model to ONNX | every class meets "
                "xyz_mae_cm <= %.2f | every non-exempt class (exempt=%s) meets "
                "rot_mean_angle_error_deg <= %.2f",
                export_position_threshold_cm,
                list(export_rotation_exempt_classes),
                export_rotation_threshold_deg,
            )

            onnx_path = Path(onnx_export_path)
            self._export_onnx(model, onnx_path)

        else:
            self.logger.info(
                "ONNX export skipped | needs xyz_mae_cm <= %.2f and "
                "rot_mean_angle_error_deg <= %.2f | failing classes: %s",
                export_position_threshold_cm,
                export_rotation_threshold_deg,
                ", ".join(
                    f"{class_name} (xyz_mae_cm={metrics['per_class'][class_name]['xyz_mae_cm']:.2f}, "
                    f"rot_mean_angle_error_deg="
                    f"{metrics['per_class'][class_name]['rot_mean_angle_error_deg']:.2f})"
                    for class_name in failing_classes
                ),
            )

        return {
            "metrics": metrics,
            "onnx_path": str(onnx_path) if onnx_path else None,
        }
