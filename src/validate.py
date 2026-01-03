#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from config import Config
from data import create_data_loaders
from models import ResNet50
from training import calculate_classification_metrics


def validate_model(config: Config, model_path: Path, logger: logging.Logger) -> None:
    """Validate a trained model using existing components."""
    logger.info(f"Model: {model_path}")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create data loaders (reusing existing function)
    _, val_loader, class_names = create_data_loaders(
        config.data_config,
        config.training_config,
        config.transform_config,
        logger,
    )
    logger.info(f"Loaded {len(val_loader.dataset)} validation images")  # ty:ignore[invalid-argument-type]
    logger.info(f"Classes: {class_names}")

    # Load model
    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")

    try:
        model = ResNet50.from_pretrained(model_path)
        model.to(device)
        model.eval()
        logger.info(f"Loaded model from {model_path}")
    except Exception:
        logger.exception("Error during loading model")
        raise

    # Validation loop
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Calculate metrics using existing function
    metrics = calculate_classification_metrics(all_predictions, all_targets)

    logger.info("Validation results:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
    logger.info(f"Total samples: {len(all_targets)}")

    # Detailed classification report
    logger.info(
        classification_report(
            all_targets,
            all_predictions,
            target_names=class_names,
            digits=4,
        )
    )

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)  # ty:ignore[invalid-argument-type]
    logger.info(f"\nConfusion Matrix:\n{cm_df}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate plant classification model")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to trained model directory",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    logger.info(f"Starting validation with config: {args.config}")
    validate_model(config, args.model_path, logger)


if __name__ == "__main__":
    main()
