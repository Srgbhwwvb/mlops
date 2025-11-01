import argparse
import logging
import os
import sys
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config
from src.data.dataset import create_data_loaders
from src.models.resnet import ResNet50, ResNetConfig
from src.training.trainer import PlantTrainer

def setup_logging(log_dir: str, level: str = "INFO"):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

def main(config_path: str, verbose: bool = False):
    """Main training function."""
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    log_level = "DEBUG" if verbose else config.get('logging', {}).get('level', 'INFO')
    setup_logging(config['output']['log_dir'], log_level)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, class_names = create_data_loaders(config)
    logging.info(f"Loaded {len(train_loader.dataset)} training images")
    logging.info(f"Loaded {len(val_loader.dataset)} validation images")
    
    # Create model
    model_config = ResNetConfig(num_classes=config['model']['num_classes'])
    model = ResNet50(model_config)
    model.to(device)
    
    logging.info(f"Initialized {config['model']['name']} model with {config['model']['num_classes']} classes")
    
    # Create trainer and start training
    trainer = PlantTrainer(model, train_loader, val_loader, device, config)
    trainer.train()
    
    logging.info("Training completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train plant classification model")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    main(args.config, args.verbose)