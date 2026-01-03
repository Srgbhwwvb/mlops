#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from api import PlantPredictor
from config import Config
from data import PlantDataset, create_val_transforms
from models import ResNet50


def predict_single_image(
    model_path: Path,
    image_path: Path,
    config: Config,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Predict plant class for a single image using PlantPredictor"""
    # Load model and class names
    model = ResNet50.from_pretrained(model_path)
    class_names = PlantDataset.LABELS

    # Create predictor instance
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = PlantPredictor(model, class_names, device)

    # Create transforms
    transform = create_val_transforms(config.transform_config)

    # Load and preprocess image
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Use existing predictor
        predictions = predictor.predict(input_tensor)
        return predictions[0]  # Single image in batch
    except Exception:
        logger.exception(f"Error processing {image_path}")
        raise


def predict_batch(
    model_path: Path,
    folder_path: Path,
    config: Config,
    logger: logging.Logger,
) -> dict[Path, dict[str, Any]]:
    """Predict plant classes for all images in a folder"""
    # Load model and setup
    model = ResNet50.from_pretrained(model_path)
    class_names = PlantDataset.LABELS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = PlantPredictor(model, class_names, device)
    transform = create_val_transforms(config.transform_config)

    # Find all images
    image_paths = list(folder_path.glob("*.png"))

    results: dict[Path, dict[str, Any]] = {}
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0)
            prediction = predictor.predict(input_tensor)[0]

            results[image_path] = prediction
        except Exception:
            logger.exception(f"Error processing {image_path}")
            raise

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict plant class using PlantPredictor",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model directory",
    )
    parser.add_argument("--image", type=Path, help="Path to single image")
    parser.add_argument("--folder", type=Path, help="Path to folder with images")
    parser.add_argument(
        "--config",
        type=Path,
        default="configs/train_config.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    if args.image:
        result = predict_single_image(args.model_path, args.image, config, logger)

        print("\nTop 3 predictions:")
        sorted_probs = sorted(
            result["probabilities"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        for i, (class_name, prob) in enumerate(sorted_probs):
            print(f"  {i + 1}. {class_name}: {prob:.4f}")
        print(f"Confidence: {result['confidence']:.4f}")

    elif args.folder:
        result = predict_batch(args.model_path, args.folder, config, logger)

        for png, res in result.items():
            print(f"{png}: {res['class_name']} (index: {res['class_index']})")

    else:
        print("Please specify either --image or --folder")


if __name__ == "__main__":
    main()
