"""
Utility script to evaluate a trained model on test set
"""

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

from model import load_model
from data_utils import create_dataloaders
from config import MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG, PATHS, CLASS_LABELS


def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate model on test set
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to evaluate on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    results = {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs)
    }
    
    return results


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    return fig


def main():
    """Main evaluation function"""
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    dataloaders = create_dataloaders(DATA_CONFIG, TRAIN_CONFIG)
    
    if len(dataloaders['test'].dataset) == 0:
        print("\nWarning: No test data found!")
        print(f"Please add your test data to:")
        print(f"  - {PATHS['test_dir']}/genuine/")
        print(f"  - {PATHS['test_dir']}/counterfeit/")
        return
    
    # Load model
    model_path = os.path.join(PATHS['models_dir'], 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train a model first using train.py")
        return
    
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, MODEL_CONFIG, device)
    
    # Evaluate
    print(f"\nEvaluating on {len(dataloaders['test'].dataset)} test samples...")
    results = evaluate_model(model, dataloaders['test'], device)
    
    # Print classification report
    class_names = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS.keys())]
    print("\nClassification Report:")
    print("=" * 60)
    print(classification_report(results['labels'], results['predictions'], 
                                target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(results['labels'], results['predictions'])
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot and save confusion matrix
    results_dir = PATHS.get('results_dir', 'results')
    os.makedirs(results_dir, exist_ok=True)
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, cm_path)
    
    # Calculate per-class accuracy
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        class_mask = results['labels'] == i
        class_acc = (results['predictions'][class_mask] == i).sum() / class_mask.sum()
        print(f"  {class_name}: {class_acc:.2%}")
    
    # Overall accuracy
    overall_acc = (results['predictions'] == results['labels']).sum() / len(results['labels'])
    print(f"\nOverall Accuracy: {overall_acc:.2%}")


if __name__ == '__main__':
    main()
