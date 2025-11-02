#!/usr/bin/env python3
import argparse
import os
import sys
import torch
import logging
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Добавляем корень проекта в Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.utils.config import load_config
from src.data.dataset import create_data_loaders
from src.models.resnet import ResNet50
from src.training.metrics import calculate_classification_metrics


def validate_model(config_path, model_path, detailed=False):
    """Validate a trained model using existing components"""
    print(f"Starting validation with config: {config_path}")
    print(f"Model: {model_path}")

    # Load configuration
    config = load_config(config_path)

    # Setup logging
    log_level = "DEBUG" if detailed else "INFO"
    logging.basicConfig(level=log_level, format="%(message)s")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data loaders (reusing existing function)
    _, val_loader, class_names = create_data_loaders(config)
    print(f"Loaded {len(val_loader.dataset)} validation images")
    print(f"Classes: {class_names}")

    # Load model
    if not os.path.exists(model_path):
        print(f"Model path does not exist: {model_path}")
        return

    try:
        model = ResNet50.from_pretrained(model_path)
        model.to(device)
        model.eval()
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

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

    print("\n" + "=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
    print(f"Total samples: {len(all_targets)}")

    if detailed:
        # Detailed classification report
        print("\nDetailed Classification Report:")
        print(
            classification_report(
                all_targets, all_predictions, target_names=class_names, digits=4
            )
        )

        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        print("\nConfusion Matrix:")
        print(cm_df)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate plant classification model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model directory"
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Show detailed metrics per class"
    )

    args = parser.parse_args()

    validate_model(args.config, args.model_path, args.detailed)
