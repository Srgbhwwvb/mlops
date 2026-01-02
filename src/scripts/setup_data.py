#!/usr/bin/env python3
import argparse
import os
import sys

from ..utils.config import load_config
from ..data.dataset import create_data_loaders

import torch


def setup_data(config_path: str):
    """TODO"""

    config_path = os.path.abspath(config_path)
    print(f"  >> {config_path}")
    print(f"Starting training with config: {config_path}")

    # Проверка существования конфига
    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found: {config_path}")

    # Загрузка конфигурации
    try:
        config = load_config(config_path)
        print("Config loaded successfully")

        # Логируем ключевые параметры для отладки
        print(
            f"Learning rate from config: {config['training']['learning_rate']} (type: {type(config['training']['learning_rate'])})"
        )
        
    except Exception as e:
        raise ValueError(f"Error loading config: {e}")
    
    from src.data.preprocessing import create_train_transforms
    from src.data.dataset import PlantDataset
    train_transform = create_train_transforms(config)
    
    # Load dataset
    full_dataset = PlantDataset(
        config["data"]["train_path"], transform=train_transform, is_train=True
    )

    torch.save(full_dataset, 'data/prepared_data')


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )

    args = parser.parse_args()

    setup_data(args.config)

    
if __name__ == "__main__":
    main()
