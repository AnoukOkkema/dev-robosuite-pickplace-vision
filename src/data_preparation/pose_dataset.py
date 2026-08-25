import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PoseDataset(Dataset):
    """
    Wraps a pose_dataset.pkl (produced by PoseDatasetGenerator) as a torch
    Dataset. Each item is an (image, bbox, class_onehot, crop, xyz, rot6d)
    tuple:
    - image: resized/normalized RGB tensor of the *full* agentview frame
      (3, image_size, image_size) -- gives spatial/perspective context for
      xyz. A tight object crop alone carries no scale/position information
      to regress absolute xyz from.
    - bbox: the object's bbox in the frame, normalized to [0, 1] by the
      frame's width/height, plus derived area and center (7,) -- tells the
      model which object in the frame to predict the pose for. width/height
      are left out: PoseEstimator's heads sit behind a hidden layer now, so
      linear combinations of x1..y2 add nothing the network can't already
      compute itself. area (non-linear, a depth cue -- bigger in frame =
      closer to camera) and center (directly tied to lateral x/y via the
      camera projection) are kept as explicit, physically-grounded hints.
    - class_onehot: one-hot class vector (num_classes,) -- an explicit
      class signal (canonical object sizes/shapes differ, which helps
      depth/xyz).
    - crop: resized/normalized RGB tensor of the *tight object crop*
      (3, rotation_image_size, rotation_image_size) -- rotation depends on
      fine visual detail (which face/edges/orientation is visible), which
      is largely lost once the object is a small part of the downscaled
      full frame. Used only for the rotation head.
    - xyz: camera-frame position (3,)
    - rot6d: first two columns of the camera-frame rotation matrix,
      flattened (6,) -- the 6D rotation representation (Zhou et al., 2019)
    """

    def __init__(self, pickle_path: str, image_size: int = 224, rotation_image_size: int = 128) -> None:
        data = pd.read_pickle(pickle_path)
        self.frames = data["frames"]
        self.df = data["samples"]
        self.class_names = data["class_names"]
        self.class_to_index = {name: index for index, name in enumerate(self.class_names)}
        self.image_size = image_size
        self.rotation_image_size = rotation_image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        frame = self.frames[row["frame_index"]]
        frame_height, frame_width = frame.shape[:2]

        image = cv2.resize(frame, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        x1, y1, x2, y2 = row["bbox"]
        x1n, y1n, x2n, y2n = x1 / frame_width, y1 / frame_height, x2 / frame_width, y2 / frame_height
        area = (x2n - x1n) * (y2n - y1n)
        cx, cy = (x1n + x2n) / 2, (y1n + y2n) / 2

        bbox_norm = torch.tensor(
            [x1n, y1n, x2n, y2n, area, cx, cy],
            dtype=torch.float32,
        )

        class_onehot = torch.zeros(len(self.class_names), dtype=torch.float32)
        class_onehot[self.class_to_index[row["class_name"]]] = 1.0

        x1i, y1i = max(int(round(x1)), 0), max(int(round(y1)), 0)
        x2i, y2i = min(int(round(x2)), frame_width), min(int(round(y2)), frame_height)
        object_crop = frame[y1i:y2i, x1i:x2i]

        if object_crop.size == 0:
            object_crop = frame

        crop = cv2.resize(object_crop, (self.rotation_image_size, self.rotation_image_size))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        crop = torch.from_numpy(crop).permute(2, 0, 1)

        xyz = torch.from_numpy(np.asarray(row["xyz_cam"], dtype=np.float32))

        rot_cam = np.asarray(row["rot_cam"], dtype=np.float32)
        # .T before reshape: rot_cam[:, :2] is (3, 2) and .reshape(-1) alone flattens
        # row-major (interleaving both columns' entries per row), NOT the column0-then
        # -column1 layout rot6d_to_matrix() expects. Transposing first gives (2, 3) --
        # row0 = full column0, row1 = full column1 -- so the flatten is [col0, col1].
        rot6d = torch.from_numpy(rot_cam[:, :2].T.reshape(-1))

        return image, bbox_norm, class_onehot, crop, xyz, rot6d
