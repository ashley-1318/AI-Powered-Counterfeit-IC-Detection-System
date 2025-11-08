"""
Configuration file for the Counterfeit IC Detection System
"""

# Model Configuration
MODEL_CONFIG = {
    'architecture': 'resnet50',  # Options: resnet50, resnet101, efficientnet_b0, efficientnet_b4
    'num_classes': 2,  # Binary classification: genuine vs counterfeit
    'pretrained': True,
    'freeze_backbone': False,
}

# Training Configuration
TRAIN_CONFIG = {
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'optimizer': 'adam',  # Options: adam, sgd, adamw
    'scheduler': 'step',  # Options: step, cosine, plateau
    'step_size': 10,
    'gamma': 0.1,
    'early_stopping_patience': 10,
    'save_best_only': True,
}

# Data Configuration
DATA_CONFIG = {
    'image_size': (224, 224),
    'mean': [0.485, 0.456, 0.406],  # ImageNet normalization
    'std': [0.229, 0.224, 0.225],
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
    'augmentation': True,
}

# Grad-CAM Configuration
GRADCAM_CONFIG = {
    'target_layer': 'layer4',  # For ResNet models
    'colormap': 'jet',
    'alpha': 0.4,  # Overlay transparency
}

# Paths
PATHS = {
    'data_dir': 'data',
    'train_dir': 'data/train',
    'val_dir': 'data/val',
    'test_dir': 'data/test',
    'models_dir': 'models',
    'checkpoints_dir': 'checkpoints',
    'results_dir': 'results',
}

# Class Labels
CLASS_LABELS = {
    0: 'Genuine',
    1: 'Counterfeit'
}

# Streamlit Configuration
STREAMLIT_CONFIG = {
    'page_title': 'IC Counterfeit Detection',
    'page_icon': '🔍',
    'layout': 'wide',
    'max_upload_size': 10,  # MB
}
