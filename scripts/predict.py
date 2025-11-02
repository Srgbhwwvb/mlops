#!/usr/bin/env python3
import argparse
import os
import sys
import glob
from PIL import Image
import torchvision.transforms as transforms

# Добавляем корень проекта в Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.models.resnet import ResNet50
from src.data.dataset import PlantDataset
from src.api.predictor import PlantPredictor
from src.utils.config import load_config
from src.data.preprocessing import create_val_transforms


def predict_single_image(
    model_path, image_path, config_path="configs/train_config.yaml"
):
    """Predict plant class for a single image using PlantPredictor"""
    # Load model and class names
    model = ResNet50.from_pretrained(model_path)
    class_names = PlantDataset.LABELS

    # Load config for transforms
    config = load_config(config_path)

    # Create predictor instance
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = PlantPredictor(model, class_names, device)

    # Create transforms
    transform = create_val_transforms(config)

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Use existing predictor
    predictions = predictor.predict(input_tensor)
    result = predictions[0]  # Single image in batch

    # Display results
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Predicted class: {result['class_name']} (index: {result['class_index']})")
    print(f"Confidence: {result['confidence']:.4f}")

    # Show top-3 predictions
    print("\nTop-3 predictions:")
    sorted_probs = sorted(
        result["probabilities"].items(), key=lambda x: x[1], reverse=True
    )[:3]
    for i, (class_name, prob) in enumerate(sorted_probs):
        print(f"  {i+1}. {class_name}: {prob:.4f}")

    return result


def predict_batch(model_path, folder_path, config_path="configs/train_config.yaml"):
    """Predict plant classes for all images in a folder"""
    # Load model and setup
    model = ResNet50.from_pretrained(model_path)
    class_names = PlantDataset.LABELS
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = PlantPredictor(model, class_names, device)
    transform = create_val_transforms(config)

    # Find all images
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    image_paths = []
    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, extension)))

    print(f"Found {len(image_paths)} images in {folder_path}")

    results = []
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0)
            prediction = predictor.predict(input_tensor)[0]

            results.append(
                {"image": os.path.basename(image_path), "prediction": prediction}
            )

            print(
                f"\n{os.path.basename(image_path)}: {prediction['class_name']} "
                f"(confidence: {prediction['confidence']:.4f})"
            )

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    return results


if __name__ == "__main__":
    import torch

    parser = argparse.ArgumentParser(
        description="Predict plant class using PlantPredictor"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model directory"
    )
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--folder", type=str, help="Path to folder with images")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()

    if args.image:
        predict_single_image(args.model_path, args.image, args.config)
    elif args.folder:
        predict_batch(args.model_path, args.folder, args.config)
    else:
        print("Please specify either --image or --folder")
