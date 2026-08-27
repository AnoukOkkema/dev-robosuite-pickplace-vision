from src.util.logging_configurator import LoggingConfigurator

# Must run before robosuite is imported (below, transitively via
# PoseDatasetGenerator/PoseVisualizer) -- robosuite emits its startup
# warnings at import time.
LoggingConfigurator.suppress_robosuite_warnings()

import wandb

from src.data_preparation.data_retriever import RoboflowDownloader
from src.data_preparation.pose_data_loader import PoseDataLoader
from src.data_preparation.pose_dataset_generator import PoseDatasetGenerator
from src.evaluation.onnx_detector import OnnxDetector
from src.evaluation.pose_evaluator import PoseEvaluator
from src.evaluation.pose_visualizer import PoseVisualizer
from src.models.pose_estimator import PoseEstimator
from src.training.pose_trainer import PoseTrainer
from src.util.device_configurator import DeviceConfigurator
from src.util.pose_wandb_callback import PoseWandBCallback
from src.util.system_configurator import SystemConfigurator
from src.util.types import CropRegion, ImageSize, PoseDatasetGeneratorConfig


def main() -> None:
    config = SystemConfigurator.load()
    logger = LoggingConfigurator.setup("train_pose.log")

    logger.info(
        "Pose pipeline started | generate_pose_dataset=%s | "
        "train_pose_estimator=%s | eval_pose_estimator=%s",
        config.generate_pose_dataset,
        config.train_pose_estimator,
        config.eval_pose_estimator,
    )

    device_config = DeviceConfigurator(logger=logger).resolve()
    device = device_config.torch_device

    # ===== DATASET GENERATION =====
    yolo_model_path = (
        f"./runs/detect/{config.yolo.project_name}/"
        f"{config.yolo.run_name}-train/weights/{config.yolo.export_onnx_name}"
    )

    project_folder_path = RoboflowDownloader.resolve_local_folder_path(
        data_path=config.data_folder,
        dataset_folder_name=config.roboflow.dataset_folder_name,
        version_number=config.roboflow.version_number,
    )
    data_yaml_path = f"{project_folder_path}/data.yaml"

    detector = OnnxDetector(
        model_path=yolo_model_path,
        data_yaml_path=data_yaml_path,
        image_size=640,
        logger=logger,
    )

    pose_dataset_generator = PoseDatasetGenerator(
        detector=detector,
        image_size=ImageSize(height=1080, width=1920),
        crop_region=CropRegion(y1=350, y2=748, x1=400, x2=975),
        config=PoseDatasetGeneratorConfig(save_path=config.pose.pose_dataset_path),
        robot_base_offset=config.environment.robot_base_offset,
        logger=logger,
    )

    pose_dataset_generator.generate(
        num_images=config.pose.num_dataset_images,
        enabled=config.generate_pose_dataset,
    )

    # ===== DATASET =====
    pose_data_loader = PoseDataLoader(
        pose_dataset_path=config.pose.pose_dataset_path,
        image_size=config.pose.pos_image_size,
        rotation_image_size=config.pose.rotation_image_size,
        batch_size=config.pose.batch_size,
        val_split=config.pose.val_split,
        test_split=config.pose.test_split,
        logger=logger,
    )

    # ===== TRAINER =====
    wandb_callback = PoseWandBCallback.setup(
        project_name=config.pose.project_name,
        run_name=config.pose.run_name,
        config=config,
        logger=logger,
        enabled=config.train_pose_estimator,
    )

    pose_visualizer = None
    if wandb_callback is not None:
        pose_visualizer = PoseVisualizer(
            detector=detector,
            image_size=ImageSize(height=1080, width=1920),
            crop_region=CropRegion(y1=350, y2=748, x1=400, x2=975),
            pose_image_size=config.pose.pos_image_size,
            rotation_image_size=config.pose.rotation_image_size,
            device=device,
            robot_base_offset=config.environment.robot_base_offset,
            logger=logger,
        )

    model = PoseEstimator(
        num_classes=len(detector.class_names),
        pretrained=True,
        rotation_dropout_prob=config.pose.dropout,
    )

    trainer = PoseTrainer(
        model=model,
        train_loader=pose_data_loader.train_loader,
        val_loader=pose_data_loader.val_loader,
        device=device,
        learning_rate=config.pose.learning_rate,
        epochs=config.pose.epochs,
        checkpoint_path=config.pose.checkpoint_path,
        rotation_loss_weight=config.pose.rotation_loss_weight,
        logger=logger,
        early_stopping_patience=config.pose.early_stopping_patience,
        rotation_symmetric_classes=config.pose.rotation_symmetric_classes,
        wandb_callback=wandb_callback,
        pose_visualizer=pose_visualizer,
    )

    trainer.train(
        enabled=config.train_pose_estimator
    )

    # ===== EVALUATION =====
    evaluator = PoseEvaluator(
        checkpoint_path=config.pose.checkpoint_path,
        device=device,
        image_size=config.pose.pos_image_size,
        rotation_image_size=config.pose.rotation_image_size,
        num_classes=len(detector.class_names),
        logger=logger,
        rotation_symmetric_classes=config.pose.rotation_symmetric_classes,
        wandb_callback=wandb_callback,
        pose_visualizer=pose_visualizer,
    )

    evaluator.evaluate(
        test_loader=pose_data_loader.test_loader,
        enabled=config.eval_pose_estimator,
        export_position_threshold_cm=config.pose.export_position_threshold_cm,
        export_rotation_threshold_deg=config.pose.export_rotation_threshold_deg,
        onnx_export_path=config.pose.onnx_export_path,
    )

    if pose_visualizer is not None:
        pose_visualizer.close()

    if wandb.run is not None:
        wandb.finish()

    logger.info(
        "Pipeline completed successfully."
    )


if __name__ == "__main__":
    main()
