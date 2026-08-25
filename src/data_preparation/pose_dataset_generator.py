import logging
import os
from dataclasses import asdict
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.base import MujocoEnv
from robosuite.utils.transform_utils import quat2mat
from tqdm.auto import tqdm

from src.evaluation.onnx_detector import OnnxDetector
from src.util.types import (
    CropRegion,
    ImageSize,
    PoseDatasetGeneratorConfig,
    PoseLabel,
    PoseSample,
)


class PoseDatasetGenerator:
    """
    Generates a dataset for the xyz/orientation pose model.

    Builds its own PickPlace environment (mirrors DatasetGenerator), then for
    each frame:
    - retrieves the camera-frame ground-truth pose of all objects from obs
      (via world_to_camera_frame);
    - runs the YOLO ONNX model on the agentview frame;
    - requires exactly one detection per class before using the frame --
      incomplete/duplicate detections are skipped;
    - collects a PoseSample for each matched object.

    All samples are pickled as a single pandas DataFrame to config.save_path.
    """

    def __init__(
        self,
        detector: OnnxDetector,
        image_size: ImageSize,
        crop_region: CropRegion,
        config: Optional[PoseDatasetGeneratorConfig] = None,
        logger=None,
    ) -> None:
        """
        Initializes the PoseDatasetGenerator and its own PickPlace
        environment.

        Args:
            detector (OnnxDetector): YOLO ONNX detector used to find
                objects in each captured frame.
            image_size (ImageSize): Camera resolution.
            crop_region (CropRegion): Crop applied to each captured frame,
                identical to DatasetGenerator's.
            config (Optional[PoseDatasetGeneratorConfig]): Generation
                settings. Defaults to PoseDatasetGeneratorConfig().
            logger: Logger instance. Defaults to a module logger.

        Returns:
            None
        """

        self.detector = detector
        self.class_names = detector.class_names
        self.image_size = image_size
        self.crop_region = crop_region
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or PoseDatasetGeneratorConfig()

        self.env = self._init_env()
        self.cam_xpos, self.cam_xmat = self._resolve_camera_extrinsics()

        self.logger.info(
            "PoseDatasetGenerator initialized | classes=%s | save_path=%s",
            self.class_names,
            self.config.save_path,
        )

    def _init_env(self) -> MujocoEnv:
        """
        Initializes the robosuite environment.

        Returns:
            MujocoEnv: Initialized robosuite environment.
        """

        controller_config = load_composite_controller_config(controller="BASIC")

        return suite.make(
            "PickPlace",
            robots="Panda",
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names="agentview",
            camera_heights=self.image_size.height,
            camera_widths=self.image_size.width,
            gripper_types="Robotiq85Gripper",
        )

    def _resolve_camera_extrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resolves the (static) agentview camera position/rotation, world-frame.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (cam_xpos, cam_xmat).
        """

        self.env.reset()

        cam_id = self.env.sim.model.camera_name2id("agentview")
        cam_xpos = self.env.sim.data.cam_xpos[cam_id].copy()
        cam_xmat = self.env.sim.data.cam_xmat[cam_id].copy().reshape(3, 3)

        return cam_xpos, cam_xmat

    @staticmethod
    def capture_agentview_image(obs: dict, crop_region: Optional[CropRegion] = None) -> np.ndarray:
        """
        Processes obs["agentview_image"] identically to DatasetGenerator: flip
        vertically, crop, RGB -> BGR.

        Args:
            obs (dict): Environment observation dictionary.
            crop_region (Optional[CropRegion]): Crop coordinates (y1, y2, x1, x2).

        Returns:
            np.ndarray: Processed image (H, W, 3) in BGR.
        """

        image = np.flipud(obs["agentview_image"])

        if crop_region is not None:
            image = image[
                crop_region.y1:crop_region.y2,
                crop_region.x1:crop_region.x2
            ]

        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    @staticmethod
    def world_to_camera_frame(
        world_xpos: np.ndarray,
        world_xmat: np.ndarray,
        cam_xpos: np.ndarray,
        cam_xmat: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts a world-frame pose into a camera-frame pose.

        Args:
            world_xpos (np.ndarray): Object position, world-frame.
            world_xmat (np.ndarray): Object rotation matrix, world-frame.
            cam_xpos (np.ndarray): Camera position, world-frame.
            cam_xmat (np.ndarray): Camera rotation matrix, world-frame.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (xyz_cam, rot_cam).
        """

        xyz_cam = cam_xmat.T @ (world_xpos - cam_xpos)
        rot_cam = cam_xmat.T @ world_xmat

        return xyz_cam, rot_cam

    @staticmethod
    def match_label_for_class_name(class_name: str, labels: dict) -> Optional[PoseLabel]:
        """
        Finds the pose-label that belongs to a YOLO class name.

        robosuite object names are not necessarily identical to the Roboflow
        class names (e.g. object name "Milk_main" vs class "Milk"), so we
        match case-insensitively on substring instead of exact equality.

        Args:
            class_name (str): YOLO class name, e.g. "Milk".
            labels (dict): {object_name: PoseLabel}.

        Returns:
            Optional[PoseLabel]: The matching label, or None if not found.
        """

        class_name_lower = class_name.lower()

        for object_name, label in labels.items():
            if class_name_lower in object_name.lower():
                return label

        return None

    def _sample_action(self) -> np.ndarray:
        """Samples a random action from the environment's action space."""

        low, high = self.env.action_spec
        return np.random.uniform(low, high)

    def _capture(self):
        """Resets the env (re-randomizing object placement) and captures one frame."""

        self.env.reset()
        obs, _, _, _ = self.env.step(self._sample_action())
        image = self.capture_agentview_image(obs, self.crop_region)
        return obs, image

    def _capture_batch(self, batch_size: int) -> Tuple[List[dict], List[np.ndarray]]:
        """Calls `_capture` `batch_size` times, for batched detector inference."""

        batch_obs = []
        batch_images = []

        for _ in range(batch_size):
            obs, image = self._capture()
            batch_obs.append(obs)
            batch_images.append(image)

        return batch_obs, batch_images

    @staticmethod
    def get_object_names(obs) -> List[str]:
        """
        Finds the loose per-object pose keys in obs (e.g. "Milk"), excluding
        relative keys like "Milk_to_robot0_eef_pos/_quat" and "robot0_eef_pos/_quat".
        """

        return sorted({
            key.rsplit("_", 1)[0]
            for key in obs.keys()
            if (
                key.endswith("_pos")
                and f"{key.rsplit('_', 1)[0]}_quat" in obs
                and "_to_" not in key
                and not key.startswith("robot")
            )
        })

    def _world_labels_camera_frame(self, obs, object_names) -> dict:
        """
        Builds the camera-frame ground-truth pose for every object in
        `object_names`, from their world-frame position/quaternion in `obs`.

        Args:
            obs (dict): Environment observation dictionary.
            object_names (List[str]): Object names, as returned by `get_object_names`.

        Returns:
            dict: {object_name: PoseLabel}.
        """

        labels = {}
        for name in object_names:
            world_xpos = obs[f"{name}_pos"]
            world_xquat_xyzw = obs[f"{name}_quat"]
            world_xmat = quat2mat(world_xquat_xyzw)

            xyz_cam, rot_cam = self.world_to_camera_frame(
                world_xpos, world_xmat, self.cam_xpos, self.cam_xmat
            )

            labels[name] = PoseLabel(xyz=xyz_cam, rot_cam=rot_cam)

        return labels

    def _match_detections(self, detections, world_labels) -> Optional[List[tuple]]:
        """
        Matches each detection to its pose-label. Returns None if not every
        class occurs exactly once, or a detection has no match -- the frame
        is then skipped entirely (see `generate`) rather than used with
        partial/ambiguous labels.

        Args:
            detections (List[Detection]): This frame's YOLO detections.
            world_labels (dict): {object_name: PoseLabel}, from `_world_labels_camera_frame`.

        Returns:
            Optional[List[tuple]]: List of (detection, class_name, label)
                triples, or None if the frame isn't usable.
        """

        num_classes = len(self.class_names)

        if len(detections) != num_classes:
            return None

        detected_class_names = [self.class_names[d.class_id] for d in detections]
        if len(set(detected_class_names)) != num_classes:
            return None

        samples = []
        for detection, class_name in zip(detections, detected_class_names):
            label = self.match_label_for_class_name(class_name, world_labels)
            if label is None:
                return None

            samples.append((detection, class_name, label))

        return samples

    def generate(self, num_images: int, enabled: bool = True) -> pd.DataFrame:
        """
        Generates the pose dataset. Keeps sampling frames until exactly
        `num_images` frames with a complete (one-per-class) detection have
        been saved -- frames that get skipped (missed/duplicate/extra
        detections) do not count towards `num_images`.

        Stores the full agentview frame once per frame_index (deduplicated,
        since up to `num_classes` samples share the same frame) plus a
        samples table referencing it by frame_index -- see PoseSample.
        Pickled as {"frames": {frame_index: np.ndarray}, "samples": DataFrame}.

        Args:
            num_images (int): Number of usable frames to collect.
            enabled (bool): If False, generation is skipped.

        Returns:
            pd.DataFrame: The generated samples (empty if skipped).
        """

        if not enabled:
            self.env.close()
            self.logger.info("PoseDatasetGenerator: generation skipped.")
            return pd.DataFrame()

        self.logger.info(
            "Starting pose dataset generation | num_images=%d | batch_size=%d",
            num_images,
            self.config.batch_size,
        )

        frames: dict = {}
        samples: List[PoseSample] = []
        saved = 0
        skipped = 0
        frame_index = 0

        max_attempts = num_images * self.config.max_attempts_multiplier

        progress_bar = tqdm(total=num_images, desc="PoseDatasetGenerator")

        while saved < num_images and frame_index < max_attempts:
            batch_size = min(self.config.batch_size, max_attempts - frame_index)
            batch_obs, batch_images = self._capture_batch(batch_size)

            batch_detections = self.detector.predict_batch(
                batch_images,
                conf_threshold=self.config.conf_threshold,
                iou_threshold=self.config.iou_threshold,
            )

            for obs, image, detections in zip(batch_obs, batch_images, batch_detections):
                object_names = self.get_object_names(obs)
                world_labels = self._world_labels_camera_frame(obs, object_names)

                matched = self._match_detections(detections, world_labels)
                frame_index += 1

                if matched is None:
                    skipped += 1
                    progress_bar.set_postfix(skipped=skipped)
                    continue

                frames[frame_index] = image

                for detection, class_name, label in matched:
                    samples.append(
                        PoseSample(
                            frame_index=frame_index,
                            class_name=class_name,
                            bbox=detection.box,
                            xyz_cam=label.xyz,
                            rot_cam=label.rot_cam,
                        )
                    )

                saved += 1
                progress_bar.update(1)
                progress_bar.set_postfix(skipped=skipped)

                if saved >= num_images:
                    break

        progress_bar.close()

        if saved < num_images:
            self.logger.warning(
                "Stopped after %d attempts, only found %d/%d usable frames.",
                frame_index,
                saved,
                num_images,
            )

        df = pd.DataFrame([asdict(sample) for sample in samples])

        os.makedirs(os.path.dirname(self.config.save_path), exist_ok=True)
        pd.to_pickle(
            {"frames": frames, "samples": df, "class_names": self.class_names},
            self.config.save_path,
        )

        self.env.close()

        self.logger.info(
            "Pose dataset generation completed | saved=%d | path=%s",
            saved,
            self.config.save_path,
        )

        return df
