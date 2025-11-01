import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from typing import Optional, Callable, List, Tuple
import torch

from typing import *


class PlantDataset(Dataset):
    """Dataset for plant seedlings classification."""
    
    LABELS = [
        'Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 
        'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize', 
        'Scentless Mayweed', 'Shepherds Purse', 
        'Small-flowered Cranesbill', 'Sugar beet'
    ]

    def __init__(self, img_dir: str, transform: Optional[Callable] = None, 
                 is_train: bool = True):
        self.img_paths = glob.glob(img_dir)
        self.transform = transform
        self.is_train = is_train
        
        if not self.img_paths:
            raise ValueError(f"No images found in {img_dir}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        if self.is_train:
            label_name = img_path.split(os.sep)[-2]
            label = self.LABELS.index(label_name)
            return image, label
        else:
            image_name = os.path.basename(img_path)
            return image_name, image


def create_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Create train and validation data loaders."""
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((config['transforms']['image_size'], 
                          config['transforms']['image_size'])),
        transforms.RandomHorizontalFlip(
            config['transforms']['train']['RandomHorizontalFlip']
        ),
        transforms.RandomRotation(
            config['transforms']['train']['RandomRotation']
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['transforms']['mean'],
            std=config['transforms']['std']
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config['transforms']['image_size'], 
                          config['transforms']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['transforms']['mean'],
            std=config['transforms']['std']
        )
    ])

    # Load dataset
    full_dataset = PlantDataset(
        config['data']['train_path'], 
        transform=train_transform,
        is_train=True
    )

    # Split dataset
    train_size = int((1 - config['data']['val_size']) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size]
    )
    
    # Apply val transform to validation set
    val_dataset.dataset.transform = val_transform

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 4)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 4)
    )
    
    return train_loader, val_loader, full_dataset.LABELS