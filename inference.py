"""
Inference utilities for Counterfeit IC Detection
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional
import os

from model import load_model, create_model
from data_utils import preprocess_image
from gradcam import create_gradcam
from config import MODEL_CONFIG, DATA_CONFIG, CLASS_LABELS


class ICDetector:
    """
    Inference class for IC counterfeit detection
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = None):
        """
        Initialize detector
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path, MODEL_CONFIG, self.device)
            print(f"Loaded model from {model_path}")
        else:
            # Create untrained model for demo purposes
            self.model = create_model(MODEL_CONFIG)
            self.model.to(self.device)
            self.model.eval()
            print("Created new model (not trained)")
        
        # Create Grad-CAM
        self.gradcam = create_gradcam(self.model, MODEL_CONFIG['architecture'])
        
        # Class labels
        self.class_labels = CLASS_LABELS
    
    def predict(self, image_path: str) -> Dict:
        """
        Predict whether an IC is genuine or counterfeit
        
        Args:
            image_path: Path to IC image
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        tensor, original_image = preprocess_image(
            image_path,
            DATA_CONFIG['image_size'],
            DATA_CONFIG['mean'],
            DATA_CONFIG['std']
        )
        tensor = tensor.to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = predicted.item()
            confidence_score = confidence.item()
        
        # Get class probabilities
        probs = probabilities[0].cpu().numpy()
        
        result = {
            'predicted_class': predicted_class,
            'predicted_label': self.class_labels.get(predicted_class, 'Unknown'),
            'confidence': confidence_score,
            'probabilities': {
                self.class_labels.get(i, f'Class {i}'): float(probs[i])
                for i in range(len(probs))
            }
        }
        
        return result
    
    def predict_with_explanation(self, image_path: str, alpha: float = 0.4) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """
        Predict with Grad-CAM visualization
        
        Args:
            image_path: Path to IC image
            alpha: Transparency for heatmap overlay
            
        Returns:
            Tuple of (prediction dict, CAM, overlayed image)
        """
        # Get prediction
        result = self.predict(image_path)
        
        # Preprocess image for Grad-CAM
        tensor, original_image = preprocess_image(
            image_path,
            DATA_CONFIG['image_size'],
            DATA_CONFIG['mean'],
            DATA_CONFIG['std']
        )
        tensor = tensor.to(self.device)
        
        # Resize original image to match model input size
        original_pil = Image.open(image_path).convert('RGB')
        original_resized = original_pil.resize(DATA_CONFIG['image_size'][::-1])  # (W, H)
        original_array = np.array(original_resized)
        
        # Generate Grad-CAM
        cam, overlayed = self.gradcam(
            tensor,
            original_array,
            target_class=result['predicted_class'],
            alpha=alpha
        )
        
        return result, cam, overlayed
    
    def batch_predict(self, image_paths: list) -> list:
        """
        Batch prediction for multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image_path in image_paths:
            result = self.predict(image_path)
            result['image_path'] = image_path
            results.append(result)
        
        return results


def predict_single_image(image_path: str, model_path: str) -> Dict:
    """
    Convenience function for single image prediction
    
    Args:
        image_path: Path to image
        model_path: Path to trained model
        
    Returns:
        Prediction dictionary
    """
    detector = ICDetector(model_path)
    result = detector.predict(image_path)
    return result


def main():
    """Demo inference function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path> [model_path]")
        return
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'models/best_model.pth'
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Create detector
    detector = ICDetector(model_path)
    
    # Make prediction
    print(f"\nAnalyzing: {image_path}")
    print("-" * 50)
    
    result = detector.predict(image_path)
    
    print(f"Prediction: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nClass Probabilities:")
    for label, prob in result['probabilities'].items():
        print(f"  {label}: {prob:.2%}")


if __name__ == '__main__':
    main()
