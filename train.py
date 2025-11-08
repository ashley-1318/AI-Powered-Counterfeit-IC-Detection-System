"""
Training script for Counterfeit IC Detection model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
import os
import json
from tqdm import tqdm
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from model import create_model
from data_utils import create_dataloaders
from config import MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG, PATHS, CLASS_LABELS


class Trainer:
    """
    Trainer class for model training and evaluation
    """
    
    def __init__(self, model: nn.Module, train_loader, val_loader, 
                 config: Dict, device: str = 'cuda'):
        """
        Initialize trainer
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = self._get_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._get_scheduler()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.best_val_acc = 0.0
        self.early_stopping_counter = 0
    
    def _get_optimizer(self):
        """Get optimizer based on config"""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def _get_scheduler(self):
        """Get learning rate scheduler based on config"""
        scheduler_name = self.config.get('scheduler', 'step').lower()
        
        if scheduler_name == 'step':
            step_size = self.config.get('step_size', 10)
            gamma = self.config.get('gamma', 0.1)
            return StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == 'cosine':
            return CosineAnnealingLR(self.optimizer, T_max=self.config.get('num_epochs', 50))
        elif scheduler_name == 'plateau':
            return ReduceLROnPlateau(self.optimizer, mode='max', patience=5, factor=0.1)
        else:
            return StepLR(self.optimizer, step_size=10, gamma=0.1)
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float, Dict]:
        """
        Validate the model
        
        Returns:
            Tuple of (average loss, accuracy, metrics dict)
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        # Calculate additional metrics
        metrics = {
            'accuracy': epoch_acc,
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
        }
        
        return epoch_loss, epoch_acc, metrics
    
    def train(self, num_epochs: int, save_dir: str):
        """
        Train the model
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, metrics = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print statistics
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
            
            # Update learning rate
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_acc)
            else:
                self.scheduler.step()
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.early_stopping_counter = 0
                
                checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'metrics': metrics
                }, checkpoint_path)
                print(f"Saved best model with validation accuracy: {val_acc:.4f}")
            else:
                self.early_stopping_counter += 1
            
            # Early stopping
            patience = self.config.get('early_stopping_patience', 10)
            if self.early_stopping_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Save training history
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")


def main():
    """Main training function"""
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    dataloaders = create_dataloaders(DATA_CONFIG, TRAIN_CONFIG)
    
    # Check if data exists
    if len(dataloaders['train'].dataset) == 0:
        print("\nWarning: No training data found!")
        print(f"Please add your data to:")
        print(f"  - {PATHS['train_dir']}/genuine/")
        print(f"  - {PATHS['train_dir']}/counterfeit/")
        print(f"  - {PATHS['val_dir']}/genuine/")
        print(f"  - {PATHS['val_dir']}/counterfeit/")
        return
    
    # Create model
    print("Creating model...")
    model = create_model(MODEL_CONFIG)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        config=TRAIN_CONFIG,
        device=device
    )
    
    # Train
    trainer.train(
        num_epochs=TRAIN_CONFIG['num_epochs'],
        save_dir=PATHS['models_dir']
    )


if __name__ == '__main__':
    main()
