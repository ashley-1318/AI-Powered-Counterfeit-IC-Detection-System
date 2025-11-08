"""
Deep Learning Model for Counterfeit IC Detection
Implements transfer learning with ResNet and EfficientNet architectures
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Any


class CounterfeitICDetector(nn.Module):
    """
    Transfer learning based model for IC counterfeit detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model
        
        Args:
            config: Model configuration dictionary
        """
        super(CounterfeitICDetector, self).__init__()
        
        self.architecture = config.get('architecture', 'resnet50')
        self.num_classes = config.get('num_classes', 2)
        self.pretrained = config.get('pretrained', True)
        self.freeze_backbone = config.get('freeze_backbone', False)
        
        # Load pretrained model
        self.backbone = self._load_backbone()
        
        # Freeze backbone if specified
        if self.freeze_backbone:
            self._freeze_backbone()
        
        # Get the number of input features for the classifier
        num_features = self._get_num_features()
        
        # Replace the classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
        
    def _load_backbone(self) -> nn.Module:
        """Load the pretrained backbone model"""
        weights = 'DEFAULT' if self.pretrained else None
        
        if self.architecture == 'resnet50':
            model = models.resnet50(weights=weights)
            # Remove the final fully connected layer
            model = nn.Sequential(*list(model.children())[:-1])
        elif self.architecture == 'resnet101':
            model = models.resnet101(weights=weights)
            model = nn.Sequential(*list(model.children())[:-1])
        elif self.architecture == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=weights)
            model = nn.Sequential(*list(model.children())[:-1])
        elif self.architecture == 'efficientnet_b4':
            model = models.efficientnet_b4(weights=weights)
            model = nn.Sequential(*list(model.children())[:-1])
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
        
        return model
    
    def _freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def _get_num_features(self) -> int:
        """Get the number of features from the backbone"""
        if 'resnet' in self.architecture:
            return 2048 if '101' in self.architecture or '50' in self.architecture else 512
        elif 'efficientnet_b0' in self.architecture:
            return 1280
        elif 'efficientnet_b4' in self.architecture:
            return 1792
        else:
            return 2048
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Flatten features
        features = features.view(features.size(0), -1)
        
        # Classification
        output = self.classifier(features)
        
        return output
    
    def get_activation_layer(self) -> nn.Module:
        """
        Get the target layer for Grad-CAM
        
        Returns:
            Target layer for activation extraction
        """
        if 'resnet' in self.architecture:
            # For ResNet, return the last convolutional layer
            return self.backbone[-2]  # layer4
        elif 'efficientnet' in self.architecture:
            # For EfficientNet, return the last convolutional block
            return self.backbone[0].features[-1]
        else:
            return self.backbone[-1]


def create_model(config: Dict[str, Any]) -> CounterfeitICDetector:
    """
    Factory function to create a model
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized model
    """
    model = CounterfeitICDetector(config)
    return model


def load_model(checkpoint_path: str, config: Dict[str, Any], device: str = 'cpu') -> CounterfeitICDetector:
    """
    Load a trained model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Model configuration dictionary
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    model = create_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model
