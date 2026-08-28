import os
from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO


class YOLOEvaluator:
    """
    Evaluates a trained YOLO checkpoint (weights/best.pt from a training
    run) on the held-out test split, and exports it to ONNX if the
    resulting mAP50-95 meets the configured threshold.
    """

    def __init__(
        self,
        train_dir: str,
        data_yaml_path: str,
        device: str,
        image_size: int,
        project_name: str,
        run_name: str,
        export_onnx_name: str,
        export_onnx_threshold: float,
        logger,
    ) -> None:
        """
        Initializes the evaluator.

        Args:
            train_dir (str): Training directory.
            data_yaml_path (str): Path to data.yaml.
            device (str): Device used for evaluation.
            image_size (int): Image size used for ONNX export.
            project_name (str): Project name.
            run_name (str): Run name.
            export_onnx_name (str): Filename for the exported ONNX model.
            export_onnx_threshold (float): Minimum mAP50-95
                score required to export the model to ONNX.
            logger: Logger instance.

        Returns:
            None
        """

        self.train_dir = Path(train_dir)

        self.data_yaml_path = Path(data_yaml_path)

        self.device = device
        self.image_size = image_size

        self.project_name = project_name
        self.run_name = run_name
        self.export_onnx_name = export_onnx_name
        self.export_onnx_threshold = export_onnx_threshold

        self.logger = logger

        self.model_path = self.train_dir / "weights" / "best.pt"

        self.logger.info("YOLOEvaluator initialized | checkpoint=%s", self.model_path)

    def evaluate(self, enabled: bool = True) -> Optional[Dict[str, Any]]:
        """
        Evaluates the YOLO model.

        Args:
            enabled (bool): If False, evaluation is skipped.

        Returns:
            Optional[Dict[str, Any]]: Evaluation metrics,
                or None if evaluation was skipped.
        """

        if not enabled:
            self.logger.info("Evaluation skipped.")
            return None

        if not self.model_path.exists():

            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        model = YOLO(str(self.model_path))

        # Remove default Ultralytics callbacks
        for event in model.callbacks:
            model.callbacks[event] = []

        self.logger.info("Starting YOLO evaluation...")

        metrics = model.val(
            data=str(self.data_yaml_path),
            split="test",
            device=self.device,
            save_json=True,
            plots=True,
            project=str(self.project_name),
            name=f"{self.run_name}-test",
            exist_ok=True,
        )

        results = metrics.results_dict

        self.logger.info(
            (
                "Evaluation completed | "
                "mAP50=%.4f | "
                "mAP50-95=%.4f | "
                "precision=%.4f | "
                "recall=%.4f"
            ),
            results.get("metrics/mAP50(B)", 0.0),
            results.get("metrics/mAP50-95(B)", 0.0),
            results.get("metrics/precision(B)", 0.0),
            results.get("metrics/recall(B)", 0.0),
        )

        score = results.get("metrics/mAP50-95(B)", 0.0)

        onnx_path = None

        if score >= self.export_onnx_threshold:

            self.logger.info(
                "Exporting YOLO model to ONNX | " "mAP50-95=%.4f >= threshold=%.4f",
                score,
                self.export_onnx_threshold,
            )

            exported_path = Path(
                model.export(
                    format="onnx",
                    device=self.device,
                    imgsz=self.image_size,
                    simplify=True,
                    dynamic=True,
                )
            )

            onnx_path = str(exported_path.with_name(self.export_onnx_name))
            os.replace(exported_path, onnx_path)

            self.logger.info("ONNX export completed | path=%s", onnx_path)

        else:
            self.logger.info(
                "ONNX export skipped | " "mAP50-95=%.4f < threshold=%.4f",
                score,
                self.export_onnx_threshold,
            )

        return {
            "results": results,
            "save_dir": str(metrics.save_dir),
            "onnx_path": onnx_path,
        }
