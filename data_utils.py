"""
Data utilities for loading and preprocessing IC images
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from typing import Tuple, Optional, Dict, List
import numpy as np


class ICDataset(Dataset):
    """
    Dataset class for IC images (Genuine vs Counterfeit)
    """
    
    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None, 
                 class_to_idx: Optional[Dict[str, int]] = None):
        """
        Initialize dataset
        
        Args:
            data_dir: Directory containing class subdirectories
            transform: Image transformations to apply
            class_to_idx: Optional mapping from class names to indices
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Get all image paths and labels
        self.image_paths = []
        self.labels = []
        
        # If class_to_idx not provided, create it
        if class_to_idx is None:
            if os.path.exists(data_dir):
                classes = sorted([d for d in os.listdir(data_dir) 
                                if os.path.isdir(os.path.join(data_dir, d))])
                self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            else:
                # Default mapping for binary classification
                self.class_to_idx = {'genuine': 0, 'counterfeit': 1}
        else:
            self.class_to_idx = class_to_idx
        
        # Load data if directory exists
        if os.path.exists(data_dir):
            self._load_data()
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
    
    def _load_data(self):
        """Load image paths and labels from directory structure"""
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(self.data_dir, class_name)
            
            if not os.path.exists(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)
    
    def __len__(self) -> int:
        """Return the number of samples"""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image tensor, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(image_size: Tuple[int, int], augmentation: bool = True, 
                  mean: List[float] = None, std: List[float] = None) -> Dict[str, transforms.Compose]:
    """
    Get image transformations for training and validation
    
    Args:
        image_size: Target image size (height, width)
        augmentation: Whether to apply data augmentation
        mean: Normalization mean values
        std: Normalization std values
        
    Returns:
        Dictionary with 'train' and 'val' transforms
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    # Training transforms with augmentation
    if augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0] + 32, image_size[1] + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    # Validation/test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return {
        'train': train_transform,
        'val': val_transform
    }


def create_dataloaders(data_config: Dict, train_config: Dict) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing
    
    Args:
        data_config: Data configuration dictionary
        train_config: Training configuration dictionary
        
    Returns:
        Dictionary with 'train', 'val', and 'test' DataLoaders
    """
    # Get transforms
    transforms_dict = get_transforms(
        image_size=data_config['image_size'],
        augmentation=data_config['augmentation'],
        mean=data_config['mean'],
        std=data_config['std']
    )
    
    # Create datasets
    train_dataset = ICDataset(
        data_dir=data_config.get('train_dir', 'data/train'),
        transform=transforms_dict['train']
    )
    
    val_dataset = ICDataset(
        data_dir=data_config.get('val_dir', 'data/val'),
        transform=transforms_dict['val'],
        class_to_idx=train_dataset.class_to_idx
    )
    
    test_dataset = ICDataset(
        data_dir=data_config.get('test_dir', 'data/test'),
        transform=transforms_dict['val'],
        class_to_idx=train_dataset.class_to_idx
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def preprocess_image(image_path: str, image_size: Tuple[int, int], 
                    mean: List[float] = None, std: List[float] = None) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocess a single image for inference
    
    Args:
        image_path: Path to image file
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Tuple of (preprocessed tensor, original image as numpy array)
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    original = np.array(image)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return tensor, original
