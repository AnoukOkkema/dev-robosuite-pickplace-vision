import gc
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
import robosuite as suite
import torch
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.base import MujocoEnv
from robosuite.utils.transform_utils import quat2mat

from src.data_preparation.pose_dataset_generator import PoseDatasetGenerator
from src.environments.pickplace_with_robot_offset import PickPlaceWithRobotOffset
from src.evaluation.onnx_detector import OnnxDetector
from src.models.pose_estimator import PoseEstimator
from src.util.types import CropRegion, Detection, ImageSize, PoseLabel


class PoseVisualizer:
    """
    Renders live agentview scenes with the predicted xyz and rotation drawn
    on top, for W&B media logging.

    Builds its own PickPlace environment (mirroring PoseDatasetGenerator),
    separate from the training dataset, so it can be used mid-training to
    check the model on fresh frames.
    """

    AXIS_LENGTH = 0.05
    AXIS_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: x=red, y=green, z=blue
    GT_COLOR = (255, 255, 255)  # white
    PRED_COLOR = (0, 165, 255)  # orange

    def __init__(
        self,
        detector: OnnxDetector,
        image_size: ImageSize,
        crop_region: CropRegion,
        pose_image_size: int,
        rotation_image_size: int,
        device: str,
        robot_base_offset: tuple = (0.0, 0.0, 0.0),
        logger=None,
    ) -> None:
        """
        Initializes the PoseVisualizer and its own PickPlace environment.

        Args:
            detector (OnnxDetector): YOLO ONNX detector used to find
                objects in each captured frame.
            image_size (ImageSize): Camera resolution.
            crop_region (CropRegion): Crop applied to each captured frame.
            pose_image_size (int): Input resolution of the full-scene
                image (xyz stream), for model inference.
            rotation_image_size (int): Input resolution of the cropped
                object image (rotation stream), for model inference.
            device (str): Torch device to run model inference on.
            robot_base_offset (tuple): World-frame XYZ offset applied to
                the Panda base. Must match the offset used by
                PoseDatasetGenerator, and the offset used by the consuming
                control repo itself (see PickPlaceWithRobotOffset for why
                this matters).
            logger: Logger instance. Defaults to a module logger.

        Returns:
            None
        """

        self.detector = detector
        self.image_size = image_size
        self.crop_region = crop_region
        self.pose_image_size = pose_image_size
        self.rotation_image_size = rotation_image_size
        self.device = device
        self.robot_base_offset = robot_base_offset
        self.logger = logger or logging.getLogger(__name__)

        self.env = self._init_env()
        self.cam_xpos, self.cam_xmat, self.fovy = self._resolve_camera_params()

        self.logger.info("PoseVisualizer initialized.")

    # ------------------------------------------------------------------
    # Environment setup
    # ------------------------------------------------------------------

    def _init_env(self) -> MujocoEnv:
        """Builds the PickPlace environment (same setup as PoseDatasetGenerator)."""

        controller_config = load_composite_controller_config(controller="BASIC")

        return suite.make(
            PickPlaceWithRobotOffset.__name__,
            robots="Panda",
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names="agentview",
            camera_heights=self.image_size.height,
            camera_widths=self.image_size.width,
            gripper_types="Robotiq85Gripper",
            robot_base_offset=self.robot_base_offset,
        )

    def _resolve_camera_params(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Resolves the (static) agentview camera's world-frame position,
        rotation, and vertical FOV, for `_project`.

        Returns:
            Tuple[np.ndarray, np.ndarray, float]: (cam_xpos, cam_xmat, fovy).
        """

        self.env.reset()

        cam_id = self.env.sim.model.camera_name2id("agentview")
        cam_xpos = self.env.sim.data.cam_xpos[cam_id].copy()
        cam_xmat = self.env.sim.data.cam_xmat[cam_id].copy().reshape(3, 3)
        fovy = self.env.sim.model.cam_fovy[cam_id]

        return cam_xpos, cam_xmat, fovy

    # ------------------------------------------------------------------
    # Projection and drawing
    # ------------------------------------------------------------------

    def _project(self, xyz_cam: np.ndarray) -> Optional[Tuple[float, float]]:
        """Pinhole-projects a camera-frame point to a pixel in the
        crop_region-cropped image."""

        x, y, z = xyz_cam
        if -z <= 1e-6:
            return None  # behind the camera

        f = 0.5 * self.image_size.height / np.tan(np.deg2rad(self.fovy) / 2)
        cx, cy = self.image_size.width / 2, self.image_size.height / 2

        u = cx + f * (x / -z) - self.crop_region.x1
        v = cy - f * (y / -z) - self.crop_region.y1

        return u, v

    def _draw_pose(
        self,
        image: np.ndarray,
        xyz_cam: np.ndarray,
        rot_cam: np.ndarray,
        color: Tuple[int, int, int],
    ) -> None:
        """
        Draws a pose as a colored dot at its origin plus three short
        RGB-colored axis lines (x=red, y=green, z=blue) for orientation.
        Draws nothing if the origin projects behind the camera.

        Args:
            image (np.ndarray): Image to draw on, modified in place.
            xyz_cam (np.ndarray): Origin position, camera-frame.
            rot_cam (np.ndarray): Rotation matrix, camera-frame.
            color (Tuple[int, int, int]): BGR color for the origin dot.

        Returns:
            None
        """

        origin = self._project(xyz_cam)
        if origin is None:
            return

        ox, oy = origin
        cv2.circle(image, (int(ox), int(oy)), 5, color, -1)

        for axis_index, axis_color in enumerate(self.AXIS_COLORS):
            axis_point = xyz_cam + rot_cam[:, axis_index] * self.AXIS_LENGTH
            axis_pixel = self._project(axis_point)

            if axis_pixel is None:
                continue

            ax, ay = axis_pixel
            cv2.line(
                image,
                (int(ox), int(oy)),
                (int(ax), int(ay)),
                axis_color,
                2,
                cv2.LINE_AA,
            )

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dedupe_highest_confidence(detections: List[Detection]) -> List[Detection]:
        """Keeps only the highest-confidence detection per class.

        Unlike PoseDatasetGenerator (which rejects the whole frame on any
        duplicate or count mismatch, since training data must be exact),
        this is just a debug visualization. A best-effort per-class pick
        keeps the wandb sanity images readable, instead of drawing every
        overlapping box and axes that NMS left standing on one object.
        """

        best_by_class = {}

        for detection in detections:
            current_best = best_by_class.get(detection.class_id)
            if current_best is None or detection.confidence > current_best.confidence:
                best_by_class[detection.class_id] = detection

        return list(best_by_class.values())

    def _preprocess_image(self, image_bgr: np.ndarray, image_size: int) -> torch.Tensor:
        """Resizes and normalizes a BGR image into a (1, 3, image_size,
        image_size) model input tensor."""

        resized = cv2.resize(image_bgr, (image_size, image_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))
        return torch.from_numpy(chw).unsqueeze(0)

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def _capture_single(
        self, model: Optional[PoseEstimator] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Resets the env (this randomizes object placement, since PickPlace
        only re-randomizes on reset(), not on step()) and renders:
        - labels_image: agentview scene with ground-truth object poses drawn.
        - pred_image: same scene with the model's predicted xyz and
          rotation drawn (None if no model was given).

        robosuite's reset() rebuilds the whole MuJoCo model from XML. This
        leaks native memory if it's called many times without the old
        model/sim being garbage-collected. See capture_media(), which
        forces a gc pass after each batch to keep this in check.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: (labels_image, pred_image).
        """

        self.env.reset()
        obs, _, _, _ = self.env.step(np.random.uniform(*self.env.action_spec))
        image = PoseDatasetGenerator.capture_agentview_image(obs, self.crop_region)

        object_names = PoseDatasetGenerator.get_object_names(obs)

        gt_labels = {}
        for name in object_names:
            world_xpos = obs[f"{name}_pos"]
            world_xquat_xyzw = obs[f"{name}_quat"]
            world_xmat = quat2mat(world_xquat_xyzw)

            xyz_cam, rot_cam = PoseDatasetGenerator.world_to_camera_frame(
                world_xpos, world_xmat, self.cam_xpos, self.cam_xmat
            )

            gt_labels[name] = PoseLabel(xyz=xyz_cam, rot_cam=rot_cam)

        labels_image = image.copy()
        for label in gt_labels.values():
            self._draw_pose(labels_image, label.xyz, label.rot_cam, self.GT_COLOR)

        if model is None:
            return labels_image, None

        detections, _, _ = self.detector.predict(
            image, conf_threshold=0.25, iou_threshold=0.45
        )
        detections = self._dedupe_highest_confidence(detections)

        pred_image = image.copy()
        was_training = model.training
        model.eval()

        frame_height, frame_width = image.shape[:2]
        image_blob = self._preprocess_image(image, self.pose_image_size).to(self.device)

        with torch.no_grad():
            for detection in detections:
                x1, y1, x2, y2 = detection.box

                class_onehot = torch.zeros(
                    1, model.num_classes, dtype=torch.float32, device=self.device
                )
                class_onehot[0, detection.class_id] = 1.0

                x1i, y1i = max(int(round(x1)), 0), max(int(round(y1)), 0)
                x2i, y2i = min(int(round(x2)), frame_width), min(
                    int(round(y2)), frame_height
                )
                cv2.rectangle(pred_image, (x1i, y1i), (x2i, y2i), (0, 255, 0), 1)

                object_crop = image[y1i:y2i, x1i:x2i]
                if object_crop.size == 0:
                    continue

                x1n, y1n, x2n, y2n = (
                    x1 / frame_width,
                    y1 / frame_height,
                    x2 / frame_width,
                    y2 / frame_height,
                )
                area = (x2n - x1n) * (y2n - y1n)
                cx, cy = (x1n + x2n) / 2, (y1n + y2n) / 2

                bbox_norm = torch.tensor(
                    [[x1n, y1n, x2n, y2n, area, cx, cy]],
                    dtype=torch.float32,
                    device=self.device,
                )

                crop_blob = self._preprocess_image(
                    object_crop, self.rotation_image_size
                ).to(self.device)

                xyz_pred, rot6d_pred = model(
                    image_blob, bbox_norm, class_onehot, crop_blob
                )
                xyz_pred = xyz_pred[0].cpu().numpy()
                rot_pred = PoseEstimator.rot6d_to_matrix(rot6d_pred)[0].cpu().numpy()

                self._draw_pose(pred_image, xyz_pred, rot_pred, self.PRED_COLOR)

        model.train(was_training)

        return labels_image, pred_image

    def capture_media(
        self,
        model: Optional[PoseEstimator] = None,
        num_samples: int = 4,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Captures `num_samples` independent live frames (each with its own
        env.reset(), so object placement genuinely differs between
        samples) and renders ground-truth and predicted poses for each.

        Forces a gc pass after the batch. See `_capture_single` for why
        this is necessary (repeated env.reset() calls cause memory
        growth).

        Args:
            model (Optional[PoseEstimator]): If given, also renders predictions.
            num_samples (int): Number of frames to capture.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: (labels_images, pred_images).
                pred_images is empty if no model was given.
        """

        labels_images = []
        pred_images = []

        for _ in range(num_samples):
            labels_image, pred_image = self._capture_single(model)
            labels_images.append(labels_image)

            if pred_image is not None:
                pred_images.append(pred_image)

        gc.collect()

        return labels_images, pred_images

    def close(self) -> None:
        self.env.close()
