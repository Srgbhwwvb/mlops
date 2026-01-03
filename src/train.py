#!/usr/bin/env python3
import argparse
import logging
import os

from utils import load_config
from data import create_data_loaders
from models import ResNet50, ResNetConfig
from training import PlantTrainer


def setup_logging(log_dir: str, level: str = "INFO"):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "training.log")),
            logging.StreamHandler(),
        ],
    )


def train_model(config_path: str, verbose: bool = False):
    """Main training function."""
    print(f" Starting training with config: {config_path}")

    # Проверка существования конфига
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    # Загрузка конфигурации
    try:
        config = load_config(config_path)
        print("Config loaded successfully")

        # Логируем ключевые параметры для отладки
        print(
            f"Learning rate from config: {config['training']['learning_rate']} (type: {type(config['training']['learning_rate'])})"
        )

    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # Настройка логирования
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(config["output"]["log_dir"], log_level)

    logging.info("Starting plant classification training")

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Создание data loaders
    try:
        train_loader, val_loader, class_names = create_data_loaders(config)
        logging.info(f"Loaded {len(train_loader.dataset)} training images")
        logging.info(f"Loaded {len(val_loader.dataset)} validation images")
        logging.info(f"Classes: {class_names}")
    except Exception as e:
        logging.error(f"Error creating data loaders: {e}")
        return

    # Создание модели
    try:
        model_config = ResNetConfig(num_classes=config["model"]["num_classes"])
        model = ResNet50(model_config)
        model.to(device)
        logging.info(
            f"Initialized {config['model']['name']} model with {config['model']['num_classes']} classes"
        )
    except Exception as e:
        logging.error(f"Error creating model: {e}")
        return

    # Создание тренера и запуск обучения
    try:
        trainer = PlantTrainer(
            model, train_loader, val_loader, device, config, class_names
        )
        trainer.train()
        logging.info("Training completed successfully!")

        # Вывод итогов
        summary = trainer.get_training_summary()
        if summary:
            logging.info(f" Training summary: {summary}")

    except Exception as e:
        logging.error(f" Error during training: {e}")
        import traceback

        logging.error(traceback.format_exc())
        return


def main():
    parser = argparse.ArgumentParser(description="Train plant classification model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    train_model(args.config, args.verbose)


if __name__ == "__main__":
    main()
