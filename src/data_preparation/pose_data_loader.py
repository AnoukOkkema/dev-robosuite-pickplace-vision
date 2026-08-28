import logging

import numpy as np
from torch.utils.data import DataLoader, Subset

from src.data_preparation.pose_dataset import PoseDataset


class PoseDataLoader:
    """
    Builds train/val/test DataLoaders from a pose_dataset.pkl.

    The data is split by whole captured frame, not by individual sample, so
    that the held-out sets are genuinely unseen scenes. See `_split` for
    details.
    """

    def __init__(
        self,
        pose_dataset_path: str,
        image_size: int,
        rotation_image_size: int,
        batch_size: int,
        val_split: float,
        test_split: float,
        logger=None,
    ) -> None:
        """
        Initializes the PoseDataLoader and builds the train/val/test splits.

        Args:
            pose_dataset_path (str): Path to the pickled pose_dataset.pkl
                (as written by PoseDatasetGenerator).
            image_size (int): Input resolution of the full-scene image
                (xyz stream).
            rotation_image_size (int): Input resolution of the cropped
                object image (rotation stream).
            batch_size (int): Batch size for all three DataLoaders.
            val_split (float): Fraction of frames assigned to validation.
            test_split (float): Fraction of frames assigned to test (the
                remainder goes to train).
            logger: Logger instance. Defaults to a module logger.

        Returns:
            None
        """

        self.dataset = PoseDataset(
            pose_dataset_path,
            image_size=image_size,
            rotation_image_size=rotation_image_size,
        )
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.logger = logger or logging.getLogger(__name__)

        train_dataset, val_dataset, test_dataset = self._split()

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        self.logger.info(
            "PoseDataLoader initialized | total=%d | train=%d | val=%d | test=%d",
            len(self.dataset),
            len(train_dataset),
            len(val_dataset),
            len(test_dataset),
        )

    def _split(self, seed: int = 42):
        """
        Splits by frame_index, not by raw sample index. Each captured frame
        produces one sample per visible object (for example 4 samples:
        Bread, Can, Cereal, Milk), and all of them share the exact same
        full-frame image that the xyz head is conditioned on.

        A plain per-sample random_split would let different objects from
        the SAME frame land in different splits. That would let the model
        partly memorize a frame's background or robot pose during
        training, and then "predict" a held-out object in that same,
        already-seen frame during validation. This would inflate the
        val/test metrics without the model actually generalizing to unseen
        scenes.

        Splitting by whole frames keeps every sample from a given frame on
        the same side of the split, so val/test are genuinely unseen
        scenes.

        Args:
            seed (int): Seed for shuffling frame indices before splitting,
                so the split is reproducible across runs.

        Returns:
            Tuple[Subset, Subset, Subset]: (train, val, test) subsets.
        """

        frame_indices = self.dataset.df["frame_index"].unique()
        frame_indices = np.random.default_rng(seed).permutation(frame_indices)

        val_frame_count = int(len(frame_indices) * self.val_split)
        test_frame_count = int(len(frame_indices) * self.test_split)

        val_frames = set(frame_indices[:val_frame_count])
        test_frames = set(
            frame_indices[val_frame_count : val_frame_count + test_frame_count]
        )
        train_frames = set(frame_indices[val_frame_count + test_frame_count :])

        frame_index_column = self.dataset.df["frame_index"]
        train_indices = frame_index_column[
            frame_index_column.isin(train_frames)
        ].index.tolist()
        val_indices = frame_index_column[
            frame_index_column.isin(val_frames)
        ].index.tolist()
        test_indices = frame_index_column[
            frame_index_column.isin(test_frames)
        ].index.tolist()

        return (
            Subset(self.dataset, train_indices),
            Subset(self.dataset, val_indices),
            Subset(self.dataset, test_indices),
        )
