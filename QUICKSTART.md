# Quick Start Guide

This guide will help you get started with the AI-Powered Counterfeit IC Detection System.

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up project directories:**
   ```bash
   python setup.py
   ```

## Preparing Your Dataset

1. **Collect IC images:**
   - Gather images of genuine ICs
   - Gather images of counterfeit ICs

2. **Organize the images:**
   ```
   data/
   ├── train/          # 80% of data
   │   ├── genuine/
   │   └── counterfeit/
   ├── val/            # 10% of data
   │   ├── genuine/
   │   └── counterfeit/
   └── test/           # 10% of data
       ├── genuine/
       └── counterfeit/
   ```

3. **Image requirements:**
   - Format: JPG, PNG, BMP, or TIFF
   - Recommended size: 224x224 or higher
   - Clear, focused images of IC chips
   - Balanced classes (similar numbers of genuine/counterfeit)

## Training a Model

1. **Configure training (optional):**
   Edit `config.py` to adjust model architecture, hyperparameters, etc.

2. **Start training:**
   ```bash
   python train.py
   ```

3. **Monitor training:**
   - Training progress is displayed in the console
   - Best model is saved to `models/best_model.pth`
   - Training history is saved to `models/training_history.json`

4. **Training will stop when:**
   - Maximum epochs reached
   - Early stopping triggered (no improvement for N epochs)

## Making Predictions

### Command Line Interface

```bash
python inference.py path/to/ic_image.jpg models/best_model.pth
```

### Web Interface (Recommended)

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- Drag-and-drop image upload
- Real-time classification
- Confidence scores
- Grad-CAM visualization (shows what the model focuses on)
- Probability distribution charts

## Evaluating Your Model

After training, evaluate on the test set:

```bash
python evaluate.py
```

This generates:
- Classification report (precision, recall, F1-score)
- Confusion matrix
- Per-class accuracy
- Visualization saved to `results/confusion_matrix.png`

## Configuration Options

Edit `config.py` to customize:

### Model Architecture
```python
MODEL_CONFIG = {
    'architecture': 'resnet50',  # Options: resnet50, resnet101, efficientnet_b0, efficientnet_b4
    'num_classes': 2,
    'pretrained': True,
}
```

### Training Parameters
```python
TRAIN_CONFIG = {
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'optimizer': 'adam',  # adam, sgd, adamw
    # ... other parameters
}
```

### Data Augmentation
```python
DATA_CONFIG = {
    'image_size': (224, 224),
    'augmentation': True,  # Enable/disable data augmentation
    # ... other parameters
}
```

## Tips for Better Results

1. **More Data = Better Performance**
   - Aim for at least 100-200 images per class
   - More diverse images improve generalization

2. **Balanced Dataset**
   - Keep similar numbers of genuine and counterfeit samples
   - Prevents bias toward one class

3. **Data Quality**
   - Use clear, well-lit images
   - Consistent image quality across dataset
   - Remove blurry or corrupted images

4. **Hyperparameter Tuning**
   - Try different learning rates (0.0001 - 0.01)
   - Adjust batch size based on available memory
   - Experiment with different architectures

5. **Data Augmentation**
   - Helps prevent overfitting
   - Can enable with `augmentation: True` in config

6. **Monitor Validation Metrics**
   - If training accuracy >> validation accuracy → overfitting
   - If both are low → underfitting or need more training

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Use smaller model (e.g., resnet50 instead of resnet101)
- Close other applications

### Model Not Converging
- Check data is correctly organized
- Try lower learning rate
- Increase number of epochs
- Check for data quality issues

### Low Accuracy
- Need more training data
- Increase training epochs
- Try different model architecture
- Enable data augmentation
- Check class balance

### Grad-CAM Not Showing
- Ensure model is properly trained
- Check target layer configuration
- Verify image preprocessing

## Example Workflow

```bash
# 1. Setup
pip install -r requirements.txt
python setup.py

# 2. Add your data to data/train, data/val, data/test directories

# 3. Train
python train.py

# 4. Evaluate
python evaluate.py

# 5. Run web app
streamlit run app.py
```

## Next Steps

- Collect and prepare your IC image dataset
- Train your first model
- Experiment with different architectures and hyperparameters
- Deploy the Streamlit app for end-users

## Support

For issues or questions:
1. Check this Quick Start Guide
2. Review the main README.md
3. Open an issue on GitHub

Happy detecting! 🔍
