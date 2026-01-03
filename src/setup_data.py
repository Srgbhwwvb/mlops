#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

import torch

from config import Config
from data import PlantDataset, create_train_transforms


def setup_data(config: Config, logger: logging.Logger) -> None:
    """Read raw dataset and preprocess into the prepared one."""
    train_transform = create_train_transforms(config.transform_config)

    # Load dataset
    full_dataset = PlantDataset(
        logger,
        config.data_config.train_path,
        transform=train_transform,
        is_train=True,
    )

    torch.save(full_dataset, "data/prepared_data")
    full_dataset.print_basic_stats()


def main() -> None:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configuration file",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    logger.info(f"Setup data with config: {args.config.resolve()}")
    setup_data(config, logger)


if __name__ == "__main__":
    main()
