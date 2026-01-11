#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

from config import Config
from data import create_data_loaders
from models import ResNet50, ResNetConfig
from training import PlantTrainer

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def __train_model_inner(
    config: Config,
    logger: logging.Logger,
):
    if MLFLOW_AVAILABLE:
        mlflow.log_params(config.into_dict())  # ty:ignore[possibly-missing-attribute]

    logging.info("Starting plant classification training")
    logger.info(
        f"Learning rate from config: {config.training_config.learning_rate}",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Создание data loaders
    try:
        train_loader, val_loader, class_names = create_data_loaders(
            config.data_config,
            config.training_config,
            config.transform_config,
            logger,
        )
        logging.info(f"Loaded {len(train_loader.dataset)} training images")  # ty:ignore[invalid-argument-type]
        logging.info(f"Loaded {len(val_loader.dataset)} validation images")  # ty:ignore[invalid-argument-type]
        logging.info(f"Classes: {class_names}")
    except Exception as e:
        logging.exception(f"Error creating data loaders: {e}")
        return

    # Сид:
    if config.training_config.seed:
        torch.manual_seed(config.training_config.seed)

    # Создание модели
    try:
        model_config = ResNetConfig(num_classes=config.model_config.num_classes)
        model = ResNet50(model_config)
        model.to(device)
        logging.info(
            f"Initialized {config.model_config.name} model "
            f"with {config.model_config.num_classes} classes",
        )
    except Exception as e:
        logging.exception(f"Error creating model: {e}")
        return

    # Создание тренера и запуск обучения
    try:
        trainer = PlantTrainer(
            model,
            train_loader,
            val_loader,
            device,
            config,
            class_names,
        )
        trainer.train()
        logging.info("Training completed successfully!")

        # Вывод итогов
        summary = trainer.get_training_summary()
        logging.info(f" Training summary: {summary}")

        # Простейший артефакт в виде таблицы с метриками
        metrics_df = pd.DataFrame(
            {
                "metric": ["accuracy", "f1"],
                "value": [summary["final_val_accuracy"], summary["best_val_f1"]],
            }
        )
        metrics_df.to_csv("metrics.csv", index=False)

        if MLFLOW_AVAILABLE:
            mlflow.log_metric("f1-score", summary["best_val_f1"])  # ty:ignore[possibly-missing-attribute]
            mlflow.log_metric("accuracy", summary["final_val_accuracy"])  # ty:ignore[possibly-missing-attribute]

            mlflow.pytorch.log_model(model, "model")
            mlflow.log_artifact("metrics.csv")  # ty:ignore[possibly-missing-attribute]

    except Exception:
        logging.exception(" Error during training")
        raise

    if MLFLOW_AVAILABLE:
        logger.info(f"Run ID: {mlflow.active_run().info.run_id}")  # ty:ignore[possibly-missing-attribute]


def train_model(
    config: Config,
    logger: logging.Logger,
):
    if MLFLOW_AVAILABLE:
        mlflow.mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("plant_classification")
        mlflow.pytorch.autolog()

    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name="ResNet50"):  # ty:ignore[possibly-missing-attribute]
            __train_model_inner(config, logger)
    else:
        __train_model_inner(config, logger)


def main():
    parser = argparse.ArgumentParser(description="Train plant classification model")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configuration file",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config)
    log_dir = config.output_config.log_dir

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Start training with config: {args.config.resolve()}")

    train_model(config, logger)


if __name__ == "__main__":
    main()
