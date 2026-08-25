from src.util.logging_configurator import LoggingConfigurator

# Must run before robosuite is imported (below, transitively via
# DatasetGenerator) -- robosuite emits its startup warnings at import time.
LoggingConfigurator.suppress_robosuite_warnings()

from dataclasses import asdict
import wandb
import os

from src.util.system_configurator import SystemConfigurator
from src.util.device_configurator import DeviceConfigurator
from src.util.types import CropRegion, ImageSize
from src.data_preparation.dataset_generator import DatasetGenerator
from src.data_preparation.data_retriever import RoboflowDownloader
from src.training.yolo_trainer import YOLOTrainer
from src.evaluation.yolo_evaluator import YOLOEvaluator
from src.util.yolo_wandb_callback import YOLOWandBCallback

from ultralytics import settings


def main() -> None:
    config = SystemConfigurator.load()
    logger = LoggingConfigurator.setup("train_yolo.log")
    settings.update(asdict(config.ultralytics))

    logger.info(
        "YOLO pipeline started | generate_dataset=%s | download_dataset=%s | "
        "train_yolo=%s | eval_yolo=%s",
        config.generate_dataset,
        config.download_dataset,
        config.train_yolo,
        config.eval_yolo,
    )

    device_config = DeviceConfigurator(logger=logger).resolve()
    device = device_config.torch_device

    # ===== DATASET GENERATION (LABEL DATASET IN ROBOFLOW) =====
    generator = DatasetGenerator(
        save_dir=f"{config.data_folder}/images",
        num_images=10000,
        image_size=ImageSize(height=1080, width=1920),
        crop_region=CropRegion(y1=350, y2=748, x1=400, x2=975),
        logger=logger
    )

    generator.generate(
        enabled=config.generate_dataset
    )

    # ===== DOWNLOAD =====
    downloader = RoboflowDownloader(
        api_key=os.getenv("API_KEY"),
        workspace_name=config.roboflow.workspace_name,
        project_name=config.roboflow.project_name,
        dataset_folder_name=config.roboflow.dataset_folder_name,
        version_number=config.roboflow.version_number,
        data_path=config.data_folder,
        model=config.roboflow.model_format,
        logger=logger
    )

    project_folder_path = downloader.download_dataset(
        config.download_dataset
    )

    data_yaml_path = (
        f"{project_folder_path}/data.yaml"
        if project_folder_path
        else ""
    )

    if not data_yaml_path:
        logger.warning(
            "No data.yaml resolved (dataset download was skipped or failed) -- "
            "training/evaluation will fail if they are enabled."
        )

    # ===== TRAINER =====
    trainer = YOLOTrainer(
        model_name=config.yolo.model_name,
        data_yaml_path=data_yaml_path,
        epochs=config.yolo.epochs,
        image_size=config.yolo.image_size,
        batch_size=config.yolo.batch_size,
        patience=config.yolo.patience,
        device=device,
        project_name=config.yolo.project_name,
        run_name=config.yolo.run_name,
        logger=logger
    )

    callback = YOLOWandBCallback.setup(
        trainer=trainer,
        config=config,
        logger=logger,
        enabled=config.train_yolo,
    )

    train_dir = trainer.train(
        config.train_yolo
    )

    # ===== EVALUATION =====
    evaluator = YOLOEvaluator(
        train_dir=str(train_dir) if train_dir else "",
        data_yaml_path=data_yaml_path,
        device=device,
        image_size=config.yolo.image_size,
        project_name=config.yolo.project_name,
        run_name=config.yolo.run_name,
        export_onnx_name=config.yolo.export_onnx_name,
        export_onnx_threshold=config.yolo.export_onnx_threshold,
        logger=logger
    )

    evaluation_results = evaluator.evaluate(
        enabled=config.eval_yolo
    )

    if evaluation_results is not None and callback is not None:

        callback.log_test_results(
            metrics=evaluation_results["results"],
            save_dir=evaluation_results["save_dir"]
        )

    if wandb.run is not None:

        wandb.finish()

    logger.info(
        "Pipeline completed successfully."
    )


if __name__ == "__main__":
    main()