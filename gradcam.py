"""
Grad-CAM implementation for explainable AI
Visualizes which parts of the image the model focuses on for classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Provides visual explanations for model predictions
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initialize Grad-CAM
        
        Args:
            model: The neural network model
            target_layer: The target layer for activation extraction
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Class Activation Map
        
        Args:
            input_tensor: Input image tensor of shape (1, 3, H, W)
            target_class: Target class index. If None, uses predicted class
            
        Returns:
            CAM as numpy array of shape (H, W)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # If target class not specified, use predicted class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Calculate weights using global average pooling
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # Compute weighted combination of activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def overlay_heatmap(self, image: np.ndarray, cam: np.ndarray, 
                       alpha: float = 0.4, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Overlay CAM heatmap on original image
        
        Args:
            image: Original image as numpy array (H, W, 3) in RGB format
            cam: Class Activation Map (H, W)
            alpha: Transparency factor for overlay
            colormap: OpenCV colormap to use
            
        Returns:
            Overlayed image as numpy array
        """
        # Resize CAM to match image size
        cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
        
        # Convert CAM to heatmap
        cam_uint8 = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(cam_uint8, colormap)
        
        # Convert heatmap from BGR to RGB
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on original image
        overlayed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlayed
    
    def __call__(self, input_tensor: torch.Tensor, image: np.ndarray, 
                target_class: Optional[int] = None, alpha: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate CAM and overlay on image
        
        Args:
            input_tensor: Input tensor (1, 3, H, W)
            image: Original image as numpy array (H, W, 3) in RGB
            target_class: Target class index
            alpha: Transparency for overlay
            
        Returns:
            Tuple of (CAM, overlayed_image)
        """
        cam = self.generate_cam(input_tensor, target_class)
        overlayed = self.overlay_heatmap(image, cam, alpha)
        
        return cam, overlayed


def create_gradcam(model: nn.Module, architecture: str = 'resnet50') -> GradCAM:
    """
    Factory function to create GradCAM instance
    
    Args:
        model: The neural network model
        architecture: Model architecture name
        
    Returns:
        GradCAM instance
    """
    # Get the appropriate target layer based on architecture
    if 'resnet' in architecture:
        # For ResNet, use the last layer before global pooling
        target_layer = model.backbone[-2][-1]  # Last block of layer4
    elif 'efficientnet' in architecture:
        # For EfficientNet, use the last convolutional block
        target_layer = model.backbone[0].features[-1]
    else:
        # Default to last layer of backbone
        target_layer = model.backbone[-1]
    
    return GradCAM(model, target_layer)
