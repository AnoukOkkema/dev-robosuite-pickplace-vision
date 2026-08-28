import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PoseDataset(Dataset):
    """
    Wraps a pose_dataset.pkl (produced by PoseDatasetGenerator) as a torch
    Dataset. Each item is a tuple: (image, bbox, class_onehot, crop, xyz,
    rot6d).

    - image: a resized and normalized RGB tensor of the *full* agentview
      frame (3, image_size, image_size). This gives the model spatial and
      perspective context for predicting xyz. A tight crop of just the
      object has no scale or position information on its own, so it can't
      be used to work out the object's absolute xyz position.
    - bbox: the object's bounding box in the frame, normalized to [0, 1] by
      the frame's width and height, plus the derived area and center (7
      values in total). This tells the model which object in the frame to
      predict the pose for. Width and height are left out on purpose:
      PoseEstimator's heads now sit behind a hidden layer, so a linear
      combination of x1..y2 adds nothing the network can't already work
      out itself. Area and center are kept because they carry useful
      signal on their own: area is a non-linear depth cue (a bigger box in
      the frame means the object is closer to the camera), and center is
      directly tied to the object's lateral x/y position through the
      camera projection.
    - class_onehot: a one-hot class vector (num_classes,). This gives the
      model an explicit class signal, since different object classes have
      different canonical sizes and shapes, which helps with depth/xyz.
    - crop: a resized and normalized RGB tensor of the *tight object crop*
      (3, rotation_image_size, rotation_image_size). Rotation depends on
      fine visual detail, such as which face or edge is facing the camera.
      That detail is mostly lost once the object is only a small part of
      the downscaled full frame. This crop is used only for the rotation
      head.
    - xyz: camera-frame position (3,)
    - rot6d: the first two columns of the camera-frame rotation matrix,
      flattened (6,). This is the 6D rotation representation from Zhou et
      al., 2019.
    """

    def __init__(
        self, pickle_path: str, image_size: int = 224, rotation_image_size: int = 128
    ) -> None:
        data = pd.read_pickle(pickle_path)
        self.frames = data["frames"]
        self.df = data["samples"]
        self.class_names = data["class_names"]
        self.class_to_index = {
            name: index for index, name in enumerate(self.class_names)
        }
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
        x1n, y1n, x2n, y2n = (
            x1 / frame_width,
            y1 / frame_height,
            x2 / frame_width,
            y2 / frame_height,
        )
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

        crop = cv2.resize(
            object_crop, (self.rotation_image_size, self.rotation_image_size)
        )
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        crop = torch.from_numpy(crop).permute(2, 0, 1)

        xyz = torch.from_numpy(np.asarray(row["xyz_cam"], dtype=np.float32))

        rot_cam = np.asarray(row["rot_cam"], dtype=np.float32)
        # .T before reshape: rot_cam[:, :2] has shape (3, 2). Calling
        # .reshape(-1) on it directly would flatten it row by row, mixing
        # entries from both columns together. That is not the order
        # rot6d_to_matrix() expects, which is all of column 0 first, then
        # all of column 1. Transposing first gives shape (2, 3), where row
        # 0 is column 0 and row 1 is column 1, so flattening that gives
        # [col0, col1] in the correct order.
        rot6d = torch.from_numpy(rot_cam[:, :2].T.reshape(-1))

        return image, bbox_norm, class_onehot, crop, xyz, rot6d
