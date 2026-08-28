import logging
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import onnxruntime as ort
import yaml

from src.util.types import Detection


class OnnxDetector:
    """
    Loads a YOLO ONNX model and runs inference on one image at a time,
    using onnxruntime. It repeats the same preprocessing steps used to
    build the training dataset: "Auto-Orient" plus a "Fit (black edges)"
    resize to a square canvas.
    """

    LETTERBOX_PAD_COLOR = (0, 0, 0)  # Black padding, matching Roboflow's "Fit" resize.

    def __init__(
        self,
        model_path: str,
        data_yaml_path: str,
        image_size: int = 640,
        intra_op_threads: int = 4,
        inter_op_threads: int = 4,
        logger=None,
    ) -> None:
        """
        Initializes the OnnxDetector and loads the ONNX Runtime session.

        Args:
            model_path (str): Path to the exported YOLO ONNX model.
            data_yaml_path (str): Path to the dataset's data.yaml file.
                Only used to read the ordered list of class names.
            image_size (int): Size of the square letterbox canvas that
                the model expects as input.
            intra_op_threads (int): ONNX Runtime intra-op thread count.
            inter_op_threads (int): ONNX Runtime inter-op thread count.
            logger: Logger instance. Defaults to a module logger.

        Returns:
            None
        """

        self.logger = logger or logging.getLogger(__name__)

        self.class_names = self._load_class_names(data_yaml_path)
        self.image_size = image_size

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = max(1, int(intra_op_threads))
        session_options.inter_op_num_threads = max(1, int(inter_op_threads))

        self.session = ort.InferenceSession(
            model_path,
            providers=providers,
            sess_options=session_options,
        )

        self.input_name = self.session.get_inputs()[0].name

        self.logger.info(
            "OnnxDetector initialized | model=%s | providers=%s | active=%s",
            model_path,
            providers,
            self.session.get_providers(),
        )

    def _letterbox(self, image: np.ndarray):
        """
        Resizes `image` to fit inside a square `image_size` canvas while
        keeping its aspect ratio. Pads the rest with black, matching
        Roboflow's "Fit (black edges)" preprocessing used to build the
        training dataset.

        Args:
            image (np.ndarray): Source image (H, W, 3) in BGR.

        Returns:
            Tuple[np.ndarray, float, int, int]:
                (letterboxed canvas, scale factor, left pad, top pad).
                `_postprocess` needs the scale and pads to map detections
                back to the original image coordinates.
        """

        height, width = image.shape[:2]

        scale = min(self.image_size / height, self.image_size / width)

        new_height = round(height * scale)
        new_width = round(width * scale)

        resized = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )

        canvas = np.full(
            (self.image_size, self.image_size, 3),
            self.LETTERBOX_PAD_COLOR,
            dtype=np.uint8,
        )

        pad_top = (self.image_size - new_height) // 2
        pad_left = (self.image_size - new_width) // 2

        canvas[pad_top : pad_top + new_height, pad_left : pad_left + new_width] = (
            resized
        )

        return canvas, scale, pad_left, pad_top

    def _to_blob(self, letterboxed_bgr: np.ndarray) -> np.ndarray:
        """Converts a letterboxed BGR image to a normalized NCHW float32
        blob for the model input."""

        rgb = cv2.cvtColor(letterboxed_bgr, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    def _postprocess(
        self,
        output,
        scale,
        pad_left,
        pad_top,
        original_size,
        conf_threshold,
        iou_threshold,
    ) -> List[Detection]:
        """
        Converts the raw model output into filtered Detections, after
        non-max suppression (NMS), in original-image pixel coordinates.

        Args:
            output: Raw model output for one image, shape (4 + nc, num_anchors).
            scale (float): Letterbox scale factor from `_letterbox`.
            pad_left (int): Letterbox left padding from `_letterbox`.
            pad_top (int): Letterbox top padding from `_letterbox`.
            original_size (Tuple[int, int]): (height, width) of the source image.
            conf_threshold (float): Minimum class confidence to keep a box.
            iou_threshold (float): IoU threshold used for NMS.

        Returns:
            List[Detection]: Detections in original-image pixel coordinates.
        """

        predictions = output.T  # (num_anchors, 4 + nc)

        boxes_cxcywh = predictions[:, :4]
        class_scores = predictions[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        keep_mask = confidences >= conf_threshold

        boxes_cxcywh = boxes_cxcywh[keep_mask]
        confidences = confidences[keep_mask]
        class_ids = class_ids[keep_mask]

        if len(boxes_cxcywh) == 0:
            return []

        boxes_xywh = np.column_stack(
            [
                boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2,
                boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2,
                boxes_cxcywh[:, 2],
                boxes_cxcywh[:, 3],
            ]
        )

        keep_indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            confidences.tolist(),
            conf_threshold,
            iou_threshold,
        )

        original_height, original_width = original_size

        detections = []

        for index in np.array(keep_indices).flatten():
            x, y, w, h = boxes_xywh[index]

            x1 = (x - pad_left) / scale
            y1 = (y - pad_top) / scale
            x2 = (x + w - pad_left) / scale
            y2 = (y + h - pad_top) / scale

            detections.append(
                Detection(
                    box=(
                        float(np.clip(x1, 0, original_width)),
                        float(np.clip(y1, 0, original_height)),
                        float(np.clip(x2, 0, original_width)),
                        float(np.clip(y2, 0, original_height)),
                    ),
                    confidence=float(confidences[index]),
                    class_id=int(class_ids[index]),
                )
            )

        return detections

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        """
        Runs the full pipeline on a single image: preprocess, then run
        inference, then postprocess the output.

        Args:
            image (np.ndarray): Source image (H, W, 3) in BGR.
            conf_threshold (float): Minimum class confidence to keep a box.
            iou_threshold (float): IoU threshold used for NMS.

        Returns:
            Tuple[List[Detection], float, float]:
                Detections, preprocessing time (s), inference time (s).
        """

        preprocess_start = time.perf_counter()

        letterboxed, scale, pad_left, pad_top = self._letterbox(image)
        blob = self._to_blob(letterboxed)

        preprocess_time = time.perf_counter() - preprocess_start

        inference_start = time.perf_counter()

        out = self.session.run(None, {self.input_name: blob})[0][0]

        inference_time = time.perf_counter() - inference_start

        detections = self._postprocess(
            output=out,
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            original_size=image.shape[:2],
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

        return detections, preprocess_time, inference_time

    def predict_batch(
        self,
        images: List[np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[List[Detection]]:
        """
        Runs the full pipeline (preprocess, inference, postprocess) on a
        batch of images in a single ONNX forward pass. This is much faster
        than calling predict() in a loop, which matters when generating
        large datasets.

        Args:
            images (List[np.ndarray]): Source images (H, W, 3) in BGR.
            conf_threshold (float): Minimum class confidence to keep a box.
            iou_threshold (float): IoU threshold used for NMS.

        Returns:
            List[List[Detection]]: Detections per image, same order as `images`.
        """

        if not images:
            return []

        scales = []
        pads = []
        blobs = []

        for image in images:
            letterboxed, scale, pad_left, pad_top = self._letterbox(image)
            scales.append(scale)
            pads.append((pad_left, pad_top))
            blobs.append(self._to_blob(letterboxed))

        batch_blob = np.concatenate(blobs, axis=0)

        outputs = self.session.run(None, {self.input_name: batch_blob})[0]

        return [
            self._postprocess(
                output=outputs[index],
                scale=scales[index],
                pad_left=pads[index][0],
                pad_top=pads[index][1],
                original_size=images[index].shape[:2],
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
            for index in range(len(images))
        ]

    @staticmethod
    def _load_class_names(data_yaml_path: Path) -> List[str]:
        with open(data_yaml_path, "r") as file:
            data = yaml.safe_load(file)

        return data["names"]
