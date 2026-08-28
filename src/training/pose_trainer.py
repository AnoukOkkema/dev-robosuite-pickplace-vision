import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.pose_estimator import PoseEstimator


class PoseTrainer:
    """
    Trains the PoseEstimator model (xyz position + rotation). Both streams
    train together with one combined loss: xyz MSE loss plus a weighted,
    symmetry-aware rotation loss. They share one optimizer and one learning
    rate scheduler.
    """

    def __init__(
        self,
        model: PoseEstimator,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
        learning_rate: float,
        epochs: int,
        checkpoint_path: str,
        rotation_loss_weight: float,
        logger,
        early_stopping_patience: Optional[int] = None,
        rotation_symmetric_classes: list = (),
        wandb_callback=None,
        pose_visualizer=None,
    ) -> None:
        """
        Sets up the PoseTrainer, its optimizer, and its learning rate
        scheduler.

        Args:
            model (PoseEstimator): Model to train.
            train_loader (DataLoader): Training set loader.
            val_loader (DataLoader): Validation set loader.
            device (str): Torch device to train on.
            learning_rate (float): Adam initial learning rate.
            epochs (int): Maximum number of training epochs.
            checkpoint_path (str): Where to save the best checkpoint
                (overwritten each time val_loss improves).
            rotation_loss_weight (float): Weight of the rotation loss
                relative to the xyz loss in the combined training loss.
            logger: Logger instance.
            early_stopping_patience (Optional[int]): Epochs to wait for
                val_loss to improve before stopping early. None or 0
                disables this, so training always runs the full `epochs`.
            rotation_symmetric_classes (list): Class names that look the
                same at any rotation around their vertical axis (e.g. a
                cylindrical can). Any yaw is equally valid for these
                classes, so their rotation target is meaningless noise.
                They are left out of the rotation loss and out of the
                reported macro rotation metric. All classes share one
                rotation backbone, so without this, the noisy targets from
                these classes would pollute the gradients for every other
                class too. Their xyz position is still trained and scored
                normally.
            wandb_callback: Optional PoseWandBCallback for per-epoch and
                per-model logging. If None, training still runs, but
                nothing is logged to W&B.
            pose_visualizer: Optional PoseVisualizer used to capture
                labeled and predicted scene images each epoch, if
                wandb_callback is also given.

        Returns:
            None
        """

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = model.num_classes
        self.class_names = train_loader.dataset.dataset.class_names
        self.rotation_symmetric_classes = set(rotation_symmetric_classes)

        self.epochs = epochs
        self.checkpoint_path = Path(checkpoint_path)
        self.rotation_loss_weight = rotation_loss_weight
        # Epochs to wait for val_loss to improve before stopping early.
        # None or 0 disables this, so training always runs the full self.epochs.
        self.early_stopping_patience = early_stopping_patience
        self.wandb_callback = wandb_callback
        self.pose_visualizer = pose_visualizer

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        # Without this decay, val error tends to oscillate late in training
        # instead of settling down.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # A generic 180-degree local-symmetry candidate, applied to all
        # classes. Many objects (e.g. a box) look almost the same when
        # turned 180 degrees around their own vertical axis. Without this,
        # the model would be penalized for "wrong" rotations that actually
        # look correct, which pollutes both the gradient and the reported
        # error.
        self.symmetry_local_rotations = [
            torch.eye(3, device=device),
            torch.tensor(
                [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], device=device
            ),
        ]

        self.logger = logger

        self.logger.info(
            "PoseTrainer initialized | epochs=%d | device=%s",
            self.epochs,
            self.device,
        )

    def _r2_score(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """R^2 score, computed over all elements (flattened), as 1 - SS_res / SS_tot."""

        residual_sum_squares = torch.sum((targets - preds) ** 2)
        total_sum_squares = torch.sum((targets - targets.mean()) ** 2)

        if total_sum_squares.item() == 0.0:
            return float("nan")

        return (1.0 - residual_sum_squares / total_sum_squares).item()

    def _macro_mse_by_class(
        self, pred: torch.Tensor, target: torch.Tensor, class_indices: torch.Tensor
    ) -> torch.Tensor:
        """MSE averaged per class, with each class counted equally (a class
        that is missing from this batch is just skipped), instead of one
        MSE pooled over the whole batch.

        A pooled loss lets whichever classes are visually easier dominate
        the gradient at every step. Averaging per class instead makes sure
        each class that is present pulls equally, so a harder class (e.g. a
        small, round object) cannot be quietly left behind by easier ones.
        """

        per_class_losses = []

        for class_index in range(self.num_classes):
            mask = class_indices == class_index

            if mask.sum() == 0:
                continue

            per_class_losses.append(F.mse_loss(pred[mask], target[mask]))

        return torch.stack(per_class_losses).mean()

    def _rotation_scored_mask(self, class_indices: torch.Tensor) -> torch.Tensor:
        """Boolean mask selecting samples whose class is not rotation-symmetric.

        This keeps rotation_symmetric_classes (e.g. Can) out of the pooled
        rot_mean_angle_error_deg reference metric too. The loss and the
        macro metric already exclude these classes, but without this mask,
        their meaningless rotation "error" would still distort that number
        every epoch.
        """

        symmetric_indices = [
            index
            for index, name in enumerate(self.class_names)
            if name in self.rotation_symmetric_classes
        ]

        if not symmetric_indices:
            return torch.ones_like(class_indices, dtype=torch.bool)

        symmetric_indices_tensor = torch.tensor(
            symmetric_indices, device=class_indices.device
        )
        return ~torch.isin(class_indices, symmetric_indices_tensor)

    def _per_class_metrics(
        self,
        xyz_pred: torch.Tensor,
        xyz_target: torch.Tensor,
        rot_pred: torch.Tensor,
        rot_target: torch.Tensor,
        class_indices: torch.Tensor,
    ) -> dict:
        """Per-class mean absolute xyz error, in cm, and mean rotation error,
        in degrees."""

        per_sample_error_deg = torch.rad2deg(
            self._symmetry_min_angle_rad(rot_pred, rot_target)
        )
        per_class = {}

        for class_index, class_name in enumerate(self.class_names):
            mask = class_indices == class_index

            if mask.sum().item() == 0:
                continue

            per_class[class_name] = {
                "xyz_mae_cm": (
                    torch.abs(xyz_target[mask] - xyz_pred[mask]).mean() * 100.0
                ).item(),
                "rot_mean_angle_error_deg": per_sample_error_deg[mask].mean().item(),
            }

        return per_class

    def _per_sample_angle_rad(
        self, rot_pred: torch.Tensor, rot_target: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """Per-sample geodesic angle, in radians.

        Computed as arccos((trace(R_pred^T @ R_target) - 1) / 2).
        """

        relative_rotation = torch.matmul(rot_pred.transpose(1, 2), rot_target)
        trace = relative_rotation.diagonal(dim1=1, dim2=2).sum(-1)
        cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0 + eps, 1.0 - eps)

        return torch.acos(cos_angle)

    def _symmetry_min_angle_rad(
        self, rot_pred: torch.Tensor, rot_target: torch.Tensor
    ) -> torch.Tensor:
        """Per-sample geodesic angle to the closest symmetry-equivalent
        target rotation."""

        candidate_angles = torch.stack(
            [
                self._per_sample_angle_rad(rot_pred, torch.matmul(rot_target, symmetry))
                for symmetry in self.symmetry_local_rotations
            ],
            dim=0,
        )

        return candidate_angles.min(dim=0).values

    def _rotation_loss(
        self,
        rot6d_pred: torch.Tensor,
        rot_cam_target: torch.Tensor,
        class_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Symmetry-aware geodesic rotation loss (in radians), averaged per class.

        See `_macro_mse_by_class` for why: a pooled mean would let easier
        classes dilute a harder class's gradient signal.
        """

        rot_pred = PoseEstimator.rot6d_to_matrix(rot6d_pred)
        per_sample_angle = self._symmetry_min_angle_rad(rot_pred, rot_cam_target)

        per_class_losses = []

        for class_index in range(self.num_classes):
            if self.class_names[class_index] in self.rotation_symmetric_classes:
                continue

            mask = class_indices == class_index

            if mask.sum() == 0:
                continue

            per_class_losses.append(per_sample_angle[mask].mean())

        if not per_class_losses:
            return rot6d_pred.new_zeros(())

        return torch.stack(per_class_losses).mean()

    def _mean_angular_error_deg(
        self, rot_pred: torch.Tensor, rot_target: torch.Tensor
    ) -> float:
        """Mean symmetry-aware geodesic rotation error over the batch (in degrees)."""

        return (
            torch.rad2deg(self._symmetry_min_angle_rad(rot_pred, rot_target))
            .mean()
            .item()
        )

    def _run_epoch(self, loader: DataLoader, is_training: bool) -> dict:
        """
        Runs one full pass over `loader`. If `is_training` is True, it
        updates the model weights; otherwise it runs a no-grad evaluation
        pass. Returns the aggregate loss and metric values for the epoch.

        Args:
            loader (DataLoader): Data to iterate over (train or val).
            is_training (bool): If True, backpropagates and steps the
                optimizer. If False, runs in eval mode with gradients
                disabled.

        Returns:
            dict: `{"xyz_loss", "rot_loss", "xyz_r2", "rot_mean_angle_error_deg",
                "xyz_mae_cm_macro", "rot_mean_angle_error_deg_macro", "per_class"}`,
                averaged or aggregated over the whole loader. The `_macro`
                scores, and the per-class breakdown, weight each class
                equally (see `_macro_mse_by_class`). `xyz_r2` and the pooled
                `rot_mean_angle_error_deg` are kept only as secondary
                reference metrics.
        """

        self.model.train(is_training)

        total_xyz_loss = 0.0
        total_rot_loss = 0.0
        num_batches = 0

        all_xyz_pred, all_xyz_target = [], []
        all_rot_pred, all_rot_target = [], []
        all_class_indices = []

        for image, bbox, class_onehot, crop, xyz, rot6d_target in loader:
            image = image.to(self.device)
            bbox = bbox.to(self.device)
            class_onehot = class_onehot.to(self.device)
            crop = crop.to(self.device)
            xyz = xyz.to(self.device)
            rot6d_target = rot6d_target.to(self.device)
            class_indices = class_onehot.argmax(dim=1)

            rot_cam_target = PoseEstimator.rot6d_to_matrix(rot6d_target)

            with torch.set_grad_enabled(is_training):
                xyz_pred, rot6d_pred = self.model(image, bbox, class_onehot, crop)

                xyz_loss = self._macro_mse_by_class(xyz_pred, xyz, class_indices)
                rot_loss = self._rotation_loss(
                    rot6d_pred, rot_cam_target, class_indices
                )
                loss = xyz_loss + self.rotation_loss_weight * rot_loss

                if is_training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            total_xyz_loss += xyz_loss.item()
            total_rot_loss += rot_loss.item()
            num_batches += 1

            all_xyz_pred.append(xyz_pred.detach())
            all_xyz_target.append(xyz.detach())
            all_rot_pred.append(PoseEstimator.rot6d_to_matrix(rot6d_pred).detach())
            all_rot_target.append(rot_cam_target.detach())
            all_class_indices.append(class_indices.detach())

        xyz_pred_all = torch.cat(all_xyz_pred, dim=0)
        xyz_target_all = torch.cat(all_xyz_target, dim=0)
        rot_pred_all = torch.cat(all_rot_pred, dim=0)
        rot_target_all = torch.cat(all_rot_target, dim=0)
        class_indices_all = torch.cat(all_class_indices, dim=0)

        per_class = self._per_class_metrics(
            xyz_pred_all,
            xyz_target_all,
            rot_pred_all,
            rot_target_all,
            class_indices_all,
        )
        # xyz still matters for every class. Rotation only matters for
        # classes where it's meaningful (see rotation_symmetric_classes).
        rot_scored_classes = {
            name: c
            for name, c in per_class.items()
            if name not in self.rotation_symmetric_classes
        }
        rotation_scored_mask = self._rotation_scored_mask(class_indices_all)

        return {
            "xyz_loss": total_xyz_loss / num_batches,
            "rot_loss": total_rot_loss / num_batches,
            "xyz_r2": self._r2_score(xyz_pred_all, xyz_target_all),
            "rot_mean_angle_error_deg": self._mean_angular_error_deg(
                rot_pred_all[rotation_scored_mask], rot_target_all[rotation_scored_mask]
            ),
            "xyz_mae_cm_macro": sum(c["xyz_mae_cm"] for c in per_class.values())
            / len(per_class),
            "rot_mean_angle_error_deg_macro": (
                sum(c["rot_mean_angle_error_deg"] for c in rot_scored_classes.values())
                / len(rot_scored_classes)
            ),
            "per_class": per_class,
        }

    def train(self, enabled: bool = True) -> Optional[Path]:
        """
        Trains the PoseEstimator model.

        Args:
            enabled (bool): If False, training is skipped.

        Returns:
            Optional[Path]: Path to the best checkpoint, or None if
                training was skipped.
        """

        if not enabled:
            self.logger.info("Training skipped.")
            return None

        self.logger.info("Starting PoseEstimator training...")

        os.makedirs(self.checkpoint_path.parent, exist_ok=True)

        if self.wandb_callback is not None:
            image_size = self.train_loader.dataset.dataset.image_size
            self.wandb_callback.log_model_info(self.model, image_size)

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            train_metrics = self._run_epoch(self.train_loader, is_training=True)
            val_metrics = self._run_epoch(self.val_loader, is_training=False)

            # The per-class breakdown (val only, logged below) is not sent
            # to wandb's per-epoch scalar log.
            train_metrics.pop("per_class")
            val_per_class = val_metrics.pop("per_class")

            val_loss = (
                val_metrics["xyz_loss"]
                + self.rotation_loss_weight * val_metrics["rot_loss"]
            )
            self.scheduler.step(val_loss)

            self.logger.info(
                "epoch=%d/%d | train_xyz_mae_cm_macro=%.2f "
                "train_rot_err_deg_macro=%.2f | val_xyz_mae_cm_macro=%.2f "
                "val_rot_err_deg_macro=%.2f | lr=%.2e",
                epoch + 1,
                self.epochs,
                train_metrics["xyz_mae_cm_macro"],
                train_metrics["rot_mean_angle_error_deg_macro"],
                val_metrics["xyz_mae_cm_macro"],
                val_metrics["rot_mean_angle_error_deg_macro"],
                self.optimizer.param_groups[0]["lr"],
            )

            for class_name, class_metrics in val_per_class.items():
                # Rotation is meaningless for rotation-symmetric classes
                # (see rotation_symmetric_classes). Showing a real-looking
                # number for it would just invite mistaking noise for a
                # regression.
                rotation_display = (
                    "n/a (rotation-symmetric)"
                    if class_name in self.rotation_symmetric_classes
                    else f"{class_metrics['rot_mean_angle_error_deg']:.2f}"
                )
                self.logger.debug(
                    "  val per class | %-8s | xyz_mae_cm=%.2f | "
                    "rot_mean_angle_error_deg=%s",
                    class_name,
                    class_metrics["xyz_mae_cm"],
                    rotation_display,
                )

            if self.wandb_callback is not None:
                self.wandb_callback.log_epoch(epoch + 1, train_metrics, val_metrics)

                if self.pose_visualizer is not None:
                    train_labels_images, _ = self.pose_visualizer.capture_media(
                        model=None
                    )
                    self.wandb_callback.log_scene_media(
                        "train", train_labels_images, step=epoch + 1
                    )

                    val_labels_images, val_pred_images = (
                        self.pose_visualizer.capture_media(model=self.model)
                    )
                    self.wandb_callback.log_scene_media(
                        "val", val_labels_images, val_pred_images, step=epoch + 1
                    )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)

                self.logger.info(
                    "New best checkpoint saved | val_xyz_mae_cm_macro=%.2f "
                    "val_rot_err_deg_macro=%.2f -> %s",
                    val_metrics["xyz_mae_cm_macro"],
                    val_metrics["rot_mean_angle_error_deg_macro"],
                    self.checkpoint_path,
                )
            else:
                epochs_without_improvement += 1

            if (
                self.early_stopping_patience
                and epochs_without_improvement >= self.early_stopping_patience
            ):
                self.logger.info(
                    "Early stopping at epoch=%d/%d | no val_loss improvement "
                    "for %d epochs (best_val_loss=%.6f)",
                    epoch + 1,
                    self.epochs,
                    epochs_without_improvement,
                    best_val_loss,
                )
                break

        self.logger.info(
            "Training completed successfully | checkpoint=%s",
            self.checkpoint_path,
        )

        return self.checkpoint_path
