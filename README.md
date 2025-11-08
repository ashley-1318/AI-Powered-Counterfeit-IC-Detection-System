# AI-Powered Counterfeit IC Detection System

A deep learning-based system that identifies counterfeit integrated circuits (ICs) from images using transfer learning and explainable AI (Grad-CAM). Built with PyTorch and Streamlit for real-time image analysis, visualization, and model interpretation.

## Features

- 🧠 **Deep Learning with Transfer Learning**: Uses pre-trained ResNet/EfficientNet architectures for high accuracy
- 🔍 **Explainable AI**: Grad-CAM visualization shows what the model focuses on
- 🚀 **Real-time Analysis**: Fast inference with GPU acceleration support
- 🖥️ **Interactive Web Interface**: User-friendly Streamlit application
- 📊 **Comprehensive Metrics**: Detailed performance evaluation and visualization
- ⚙️ **Configurable**: Easy-to-modify configuration for different use cases

## System Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Input Image   │─────▶│  Transfer        │─────▶│  Classification │
│   (IC Photo)    │      │  Learning Model  │      │  (Genuine/Fake) │
└─────────────────┘      │  (ResNet/Effnet) │      └─────────────────┘
                         └──────────────────┘              │
                                  │                         │
                                  │                         ▼
                         ┌────────▼──────────┐     ┌──────────────┐
                         │   Feature Maps    │     │   Grad-CAM   │
                         │   Extraction      │────▶│ Visualization│
                         └───────────────────┘     └──────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ashley-1318/AI-Powered-Counterfeit-IC-Detection-System.git
cd AI-Powered-Counterfeit-IC-Detection-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

Organize your IC images in the following structure:

```
data/
├── train/
│   ├── genuine/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── counterfeit/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── val/
│   ├── genuine/
│   └── counterfeit/
└── test/
    ├── genuine/
    └── counterfeit/
```

## Usage

### 1. Configuration

Modify `config.py` to adjust model and training parameters:

```python
MODEL_CONFIG = {
    'architecture': 'resnet50',  # resnet50, resnet101, efficientnet_b0, etc.
    'num_classes': 2,
    'pretrained': True,
}

TRAIN_CONFIG = {
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 0.001,
    # ... other parameters
}
```

### 2. Train the Model

```bash
python train.py
```

Training outputs:
- Model checkpoints in `models/best_model.pth`
- Training history in `models/training_history.json`

### 3. Evaluate the Model

```bash
python evaluate.py
```

Generates:
- Classification report
- Confusion matrix
- Per-class accuracy metrics

### 4. Run Inference

#### Command Line:
```bash
python inference.py path/to/image.jpg models/best_model.pth
```

#### Streamlit Web App:
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## Web Application

The Streamlit web interface provides:

- **Image Upload**: Drag-and-drop or browse to upload IC images
- **Real-time Prediction**: Instant classification with confidence scores
- **Grad-CAM Visualization**: Visual explanation of model decisions
- **Probability Distribution**: Bar charts showing class probabilities
- **Adjustable Settings**: Configure Grad-CAM overlay transparency

### Screenshot

*(Upload an IC image to see real-time analysis)*

## Project Structure

```
AI-Powered-Counterfeit-IC-Detection-System/
├── app.py                  # Streamlit web application
├── config.py              # Configuration parameters
├── model.py               # Neural network model definition
├── gradcam.py            # Grad-CAM implementation
├── data_utils.py         # Data loading and preprocessing
├── train.py              # Training script
├── inference.py          # Inference utilities
├── evaluate.py           # Model evaluation script
├── requirements.txt      # Python dependencies
├── README.md            # Documentation
└── .gitignore          # Git ignore file
```

## Model Architecture

The system uses transfer learning with the following architectures:

- **ResNet-50/101**: Deep residual networks with skip connections
- **EfficientNet-B0/B4**: Efficient neural networks with compound scaling

Key components:
1. **Backbone**: Pre-trained on ImageNet for feature extraction
2. **Classifier**: Custom fully-connected layers for binary classification
3. **Grad-CAM**: Targets last convolutional layer for visualization

## Technical Details

### Transfer Learning
- Uses pre-trained weights from ImageNet
- Fine-tunes all layers or freezes backbone (configurable)
- Adds custom classifier head for IC classification

### Grad-CAM (Gradient-weighted Class Activation Mapping)
- Generates visual explanations for predictions
- Highlights important regions in the input image
- Uses gradients of target class w.r.t. feature maps

### Data Augmentation
- Random crops and flips
- Rotation and color jittering
- ImageNet normalization

## Performance

The model achieves high accuracy on IC counterfeit detection:

- **Training**: ~50 epochs with early stopping
- **Optimization**: Adam optimizer with learning rate scheduling
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Validation**: K-fold cross-validation recommended for small datasets

## Customization

### Adding New Architectures

Edit `model.py` to add support for new architectures:

```python
elif self.architecture == 'your_model':
    model = models.your_model(pretrained=self.pretrained)
    # Configure as needed
```

### Adjusting Hyperparameters

Modify `config.py`:

```python
TRAIN_CONFIG = {
    'learning_rate': 0.0001,  # Lower for fine-tuning
    'batch_size': 16,         # Reduce for memory constraints
    'num_epochs': 100,        # Increase for better convergence
}
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config
- Use smaller model architecture
- Enable gradient checkpointing

### Low Accuracy
- Increase training epochs
- Add more training data
- Use data augmentation
- Try different architectures

### Model Not Found
- Ensure model is trained: `python train.py`
- Check model path in configuration
- Verify checkpoint file exists

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the deep learning framework
- Streamlit for the web application framework
- Original Grad-CAM paper: [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{ic_counterfeit_detection,
  title={AI-Powered Counterfeit IC Detection System},
  author={Your Name},
  year={2025},
  url={https://github.com/ashley-1318/AI-Powered-Counterfeit-IC-Detection-System}
}
```

## Contact

For questions or feedback, please open an issue on GitHub.
