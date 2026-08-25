import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.pose_estimator import PoseEstimator


class PoseTrainer:
    """
    Handles PoseEstimator (xyz + rotation) training: joint xyz MSE loss +
    weighted symmetry-aware geodesic rotation loss, shared optimizer/scheduler.
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
        wandb_callback=None,
        pose_visualizer=None,
    ) -> None:
        """
        Initializes the PoseTrainer, optimizer and LR scheduler.

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
                val_loss to improve before stopping early. None/0 disables
                it (always runs the full `epochs`).
            wandb_callback: Optional PoseWandBCallback for per-epoch/model
                logging. If None, training still runs but nothing is
                logged to W&B.
            pose_visualizer: Optional PoseVisualizer used to capture
                labeled/predicted scene media each epoch, if
                wandb_callback is also given.

        Returns:
            None
        """

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.epochs = epochs
        self.checkpoint_path = Path(checkpoint_path)
        self.rotation_loss_weight = rotation_loss_weight
        # Epochs to wait for val_loss to improve before stopping early --
        # None/0 disables it (always runs the full self.epochs).
        self.early_stopping_patience = early_stopping_patience
        self.wandb_callback = wandb_callback
        self.pose_visualizer = pose_visualizer

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        # Without decay, val error tends to oscillate late in training instead
        # of settling.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Generic 180-degree local-symmetry candidate, applied to all classes:
        # without it, symmetric objects (e.g. a box that looks the same
        # rotated 180 degrees about its own up-axis) get penalized for
        # "wrong" rotations that are visually indistinguishable from correct
        # ones, which pollutes the gradient and the reported error.
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
        """R^2 score, computed over all elements (flattened): 1 - SS_res / SS_tot."""

        residual_sum_squares = torch.sum((targets - preds) ** 2)
        total_sum_squares = torch.sum((targets - targets.mean()) ** 2)

        if total_sum_squares.item() == 0.0:
            return float("nan")

        return (1.0 - residual_sum_squares / total_sum_squares).item()

    def _per_sample_angle_rad(self, rot_pred: torch.Tensor, rot_target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Per-sample geodesic angle (radians): arccos((trace(R_pred^T @ R_target) - 1) / 2)."""

        relative_rotation = torch.matmul(rot_pred.transpose(1, 2), rot_target)
        trace = relative_rotation.diagonal(dim1=1, dim2=2).sum(-1)
        cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0 + eps, 1.0 - eps)

        return torch.acos(cos_angle)

    def _symmetry_min_angle_rad(self, rot_pred: torch.Tensor, rot_target: torch.Tensor) -> torch.Tensor:
        """Per-sample geodesic angle to the closest symmetry-equivalent target rotation."""

        candidate_angles = torch.stack(
            [
                self._per_sample_angle_rad(rot_pred, torch.matmul(rot_target, symmetry))
                for symmetry in self.symmetry_local_rotations
            ],
            dim=0,
        )

        return candidate_angles.min(dim=0).values

    def _rotation_loss(self, rot6d_pred: torch.Tensor, rot_cam_target: torch.Tensor) -> torch.Tensor:
        """Symmetry-aware geodesic rotation loss (radians)."""

        rot_pred = PoseEstimator.rot6d_to_matrix(rot6d_pred)

        return self._symmetry_min_angle_rad(rot_pred, rot_cam_target).mean()

    def _mean_angular_error_deg(self, rot_pred: torch.Tensor, rot_target: torch.Tensor) -> float:
        """Mean symmetry-aware geodesic rotation error over the batch, in degrees."""

        return torch.rad2deg(self._symmetry_min_angle_rad(rot_pred, rot_target)).mean().item()

    def _run_epoch(self, loader: DataLoader, is_training: bool) -> dict:
        """
        Runs one full pass over `loader`, updating model weights if
        `is_training` (else a no-grad evaluation pass), and returns
        aggregate loss/metric values for the epoch.

        Args:
            loader (DataLoader): Data to iterate over (train or val).
            is_training (bool): If True, backprops and steps the optimizer;
                if False, runs in eval mode with gradients disabled.

        Returns:
            dict: `{"xyz_loss", "rot_loss", "xyz_r2", "rot_mean_angle_error_deg"}`,
                averaged/aggregated over the whole loader.
        """

        self.model.train(is_training)

        total_xyz_loss = 0.0
        total_rot_loss = 0.0
        num_batches = 0

        all_xyz_pred, all_xyz_target = [], []
        all_rot_pred, all_rot_target = [], []

        for image, bbox, class_onehot, crop, xyz, rot6d_target in loader:
            image = image.to(self.device)
            bbox = bbox.to(self.device)
            class_onehot = class_onehot.to(self.device)
            crop = crop.to(self.device)
            xyz = xyz.to(self.device)
            rot6d_target = rot6d_target.to(self.device)

            rot_cam_target = PoseEstimator.rot6d_to_matrix(rot6d_target)

            with torch.set_grad_enabled(is_training):
                xyz_pred, rot6d_pred = self.model(image, bbox, class_onehot, crop)

                xyz_loss = F.mse_loss(xyz_pred, xyz)
                rot_loss = self._rotation_loss(rot6d_pred, rot_cam_target)
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

        xyz_pred_all = torch.cat(all_xyz_pred, dim=0)
        xyz_target_all = torch.cat(all_xyz_target, dim=0)
        rot_pred_all = torch.cat(all_rot_pred, dim=0)
        rot_target_all = torch.cat(all_rot_target, dim=0)

        return {
            "xyz_loss": total_xyz_loss / num_batches,
            "rot_loss": total_rot_loss / num_batches,
            "xyz_r2": self._r2_score(xyz_pred_all, xyz_target_all),
            "rot_mean_angle_error_deg": self._mean_angular_error_deg(rot_pred_all, rot_target_all),
        }

    def train(self, enabled: bool = True) -> Optional[Path]:
        """
        Trains the PoseEstimator model.

        Args:
            enabled (bool): If False, training is skipped.

        Returns:
            Optional[Path]: Path to the best checkpoint,
                or None if training was skipped.
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

            val_loss = val_metrics["xyz_loss"] + self.rotation_loss_weight * val_metrics["rot_loss"]
            self.scheduler.step(val_loss)

            self.logger.info(
                "epoch=%d/%d | train_xyz_r2=%.4f train_rot_err_deg=%.2f | "
                "val_xyz_r2=%.4f val_rot_err_deg=%.2f | lr=%.2e",
                epoch + 1,
                self.epochs,
                train_metrics["xyz_r2"],
                train_metrics["rot_mean_angle_error_deg"],
                val_metrics["xyz_r2"],
                val_metrics["rot_mean_angle_error_deg"],
                self.optimizer.param_groups[0]["lr"],
            )

            if self.wandb_callback is not None:
                self.wandb_callback.log_epoch(epoch + 1, train_metrics, val_metrics)

                if self.pose_visualizer is not None:
                    train_labels_images, _ = self.pose_visualizer.capture_media(model=None)
                    self.wandb_callback.log_scene_media("train", train_labels_images, step=epoch + 1)

                    val_labels_images, val_pred_images = self.pose_visualizer.capture_media(model=self.model)
                    self.wandb_callback.log_scene_media(
                        "val", val_labels_images, val_pred_images, step=epoch + 1
                    )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)

                self.logger.info(
                    "New best checkpoint saved | val_xyz_r2=%.4f val_rot_err_deg=%.2f -> %s",
                    val_metrics["xyz_r2"],
                    val_metrics["rot_mean_angle_error_deg"],
                    self.checkpoint_path,
                )
            else:
                epochs_without_improvement += 1

            if self.early_stopping_patience and epochs_without_improvement >= self.early_stopping_patience:
                self.logger.info(
                    "Early stopping at epoch=%d/%d | no val_loss improvement for %d epochs (best_val_loss=%.6f)",
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
