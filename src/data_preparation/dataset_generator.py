import logging
import os
from typing import Optional

import cv2
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.base import MujocoEnv
from tqdm import tqdm

from src.util.types import CropRegion, ImageSize


class DatasetGenerator:
    """
    Generates synthetic images using the robosuite PickPlace environment.

    This class does the following:
    - Starts a robosuite environment
    - Picks random actions
    - Captures images from the environment
    - Processes and crops the images
    - Saves the images to disk

    Image numbering picks up automatically from the last existing image
    in the output directory.
    """

    def __init__(
        self,
        save_dir: str = "./data/images",
        num_images: int = 100,
        image_size: ImageSize = ImageSize(height=1080, width=1920),
        crop_region: CropRegion = CropRegion(y1=370, y2=748, x1=400, x2=975),
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initializes the DatasetGenerator.

        Args:
            save_dir (str):
                Directory where the generated images are saved.

            num_images (int):
                Number of images to generate.

            image_size (ImageSize):
                Camera resolution (height, width).

            crop_region (CropRegion):
                Crop coordinates (y1, y2, x1, x2).

            logger (Optional[logging.Logger]):
                Logger instance.

        Returns:
            None
        """

        self.save_dir = save_dir

        self.num_images = num_images

        self.image_size = image_size

        self.crop_region = crop_region

        self.logger = logger

        self.env = None

    def _init_env(self) -> MujocoEnv:
        """
        Initializes the robosuite environment.

        Returns:
            MujocoEnv:
                Initialized robosuite environment.
        """

        controller_config = load_composite_controller_config(controller="BASIC")

        env = suite.make(
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

        return env

    def _warmup(self) -> None:
        """
        Discards the first render after environment creation.

        The very first offscreen render after `suite.make(...)` is
        sometimes stale (empty/incomplete scene), because the render
        context is still warming up. Doing one throwaway reset+step here
        avoids that frame ending up as `frame_00000.png`.

        Returns:
            None
        """

        self.env.reset()

        action = self._sample_action()

        self.env.step(action)

    def _sample_action(self) -> np.ndarray:
        """
        Samples a random action from the environment.

        Returns:
            np.ndarray:
                Random action vector.
        """

        low, high = self.env.action_spec

        return np.random.uniform(low, high)

    def _process_image(self, obs: dict) -> np.ndarray:
        """
        Processes the observation image.

        This function:
        - extracts the image
        - flips it vertically
        - crops the image
        - converts RGB to BGR

        Args:
            obs (dict):
                Environment observation dictionary.

        Returns:
            np.ndarray:
                Processed image.
        """

        image = np.flipud(obs["agentview_image"])

        image = image[
            self.crop_region.y1 : self.crop_region.y2,
            self.crop_region.x1 : self.crop_region.x2,
        ]

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image

    def _get_start_index(self) -> int:
        """
        Retrieves the next image index based on
        already existing images.

        Returns:
            int:
                Next available image index.
        """

        existing_images = sorted(
            [
                file
                for file in os.listdir(self.save_dir)
                if (file.startswith("frame_") and file.endswith(".png"))
            ]
        )

        if not existing_images:

            return 0

        last_image = existing_images[-1]

        last_index = int(last_image.replace("frame_", "").replace(".png", ""))

        return last_index + 1

    def generate(self, enabled: bool = True) -> None:
        """
        Generates the synthetic dataset.

        Args:
            enabled (bool):
                If False, generation is skipped.

        Returns:
            None
        """

        if not enabled:

            self.logger.info("Dataset generation skipped.")

            return

        os.makedirs(self.save_dir, exist_ok=True)

        self.env = self._init_env()

        self._warmup()

        start_index = self._get_start_index()

        self.logger.info(
            ("Starting dataset generation | " "num_images=%d | " "start_index=%d"),
            self.num_images,
            start_index,
        )

        pbar = tqdm(total=self.num_images)

        for step in range(start_index, start_index + self.num_images):

            obs = self.env.reset()

            action = self._sample_action()

            obs, _, _, _ = self.env.step(action)

            image = self._process_image(obs)

            filepath = os.path.join(self.save_dir, f"frame_{step:05d}.png")

            cv2.imwrite(filepath, image)

            pbar.update(1)

            pbar.set_description((f"Processed: " f"{pbar.n}/{pbar.total}"))

        pbar.close()

        self.logger.info(
            ("Dataset generation completed | " "generated=%d images"), self.num_images
        )

        self.env.close()
