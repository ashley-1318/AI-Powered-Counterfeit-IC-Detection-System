#!/usr/bin/env python
"""
System Test Script
Tests all major components of the IC Counterfeit Detection System
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import io


def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    try:
        import torch
        import torchvision
        import streamlit
        import cv2
        import numpy
        import matplotlib
        import sklearn
        import pandas
        import seaborn
        print("  ✓ All dependencies available")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_custom_modules():
    """Test that all custom modules can be imported"""
    print("\nTesting custom modules...")
    try:
        import config
        import model
        import gradcam
        import data_utils
        import inference
        print("  ✓ All custom modules available")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_model_creation():
    """Test model creation and basic operations"""
    print("\nTesting model creation...")
    try:
        from model import create_model
        from config import MODEL_CONFIG
        
        # Create model
        model = create_model(MODEL_CONFIG)
        print(f"  ✓ Model created: {MODEL_CONFIG['architecture']}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  ✓ Total parameters: {total_params:,}")
        print(f"  ✓ Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == (2, 2), f"Expected shape (2, 2), got {output.shape}"
        print(f"  ✓ Forward pass successful")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_data_utilities():
    """Test data loading and preprocessing utilities"""
    print("\nTesting data utilities...")
    try:
        from data_utils import get_transforms, ICDataset
        from config import DATA_CONFIG
        
        # Test transforms
        transforms = get_transforms(
            DATA_CONFIG['image_size'],
            augmentation=True,
            mean=DATA_CONFIG['mean'],
            std=DATA_CONFIG['std']
        )
        print("  ✓ Data transforms created")
        
        # Test dataset creation (will be empty without data)
        dataset = ICDataset('data/train')
        print(f"  ✓ Dataset created (contains {len(dataset)} samples)")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_gradcam():
    """Test Grad-CAM functionality"""
    print("\nTesting Grad-CAM...")
    try:
        from model import create_model
        from gradcam import create_gradcam
        from config import MODEL_CONFIG
        
        # Create model and Grad-CAM
        model = create_model(MODEL_CONFIG)
        model.eval()
        gradcam = create_gradcam(model, MODEL_CONFIG['architecture'])
        print("  ✓ Grad-CAM created")
        
        # Create dummy image
        dummy_input = torch.randn(1, 3, 224, 224)
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Generate CAM
        cam, overlayed = gradcam(dummy_input, dummy_image, target_class=0)
        
        # CAM is the raw feature map (7x7 for ResNet50), overlay is resized to match input
        assert len(cam.shape) == 2, f"Expected 2D CAM, got shape {cam.shape}"
        assert overlayed.shape == (224, 224, 3), f"Expected overlay shape (224, 224, 3), got {overlayed.shape}"
        print("  ✓ Grad-CAM visualization generated")
        print(f"    - Raw CAM shape: {cam.shape} (feature map)")
        print(f"    - Overlay shape: {overlayed.shape} (resized to input)")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_inference():
    """Test inference functionality"""
    print("\nTesting inference...")
    try:
        from inference import ICDetector
        
        # Create detector (will use untrained model)
        detector = ICDetector()
        print("  ✓ IC Detector created")
        
        # Create a temporary test image
        test_image_path = '/tmp/test_ic.jpg'
        img = Image.new('RGB', (224, 224), color='white')
        img.save(test_image_path)
        
        # Test prediction
        result = detector.predict(test_image_path)
        
        assert 'predicted_class' in result
        assert 'predicted_label' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        print("  ✓ Prediction successful")
        print(f"    - Predicted: {result['predicted_label']}")
        print(f"    - Confidence: {result['confidence']:.2%}")
        
        # Test with Grad-CAM
        result, cam, overlayed = detector.predict_with_explanation(test_image_path)
        print("  ✓ Prediction with Grad-CAM successful")
        
        # Cleanup
        os.remove(test_image_path)
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_configuration():
    """Test configuration loading"""
    print("\nTesting configuration...")
    try:
        from config import MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG, CLASS_LABELS
        
        print(f"  ✓ Model config: {MODEL_CONFIG['architecture']}, {MODEL_CONFIG['num_classes']} classes")
        print(f"  ✓ Training config: {TRAIN_CONFIG['num_epochs']} epochs, batch size {TRAIN_CONFIG['batch_size']}")
        print(f"  ✓ Data config: Image size {DATA_CONFIG['image_size']}")
        print(f"  ✓ Classes: {CLASS_LABELS}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_file_structure():
    """Test that all required files and directories exist"""
    print("\nTesting file structure...")
    
    required_files = [
        'app.py', 'config.py', 'model.py', 'gradcam.py',
        'data_utils.py', 'train.py', 'inference.py', 'evaluate.py',
        'setup.py', 'requirements.txt', 'README.md', 'QUICKSTART.md',
        '.gitignore'
    ]
    
    required_dirs = [
        'data/train/genuine', 'data/train/counterfeit',
        'data/val/genuine', 'data/val/counterfeit',
        'data/test/genuine', 'data/test/counterfeit',
        'models', 'checkpoints', 'results'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    missing_dirs = [d for d in required_dirs if not os.path.isdir(d)]
    
    if missing_files:
        print(f"  ✗ Missing files: {missing_files}")
        return False
    else:
        print(f"  ✓ All {len(required_files)} required files present")
    
    if missing_dirs:
        print(f"  ✗ Missing directories: {missing_dirs}")
        return False
    else:
        print(f"  ✓ All {len(required_dirs)} required directories present")
    
    return True


def main():
    """Run all tests"""
    print("="*70)
    print("AI-POWERED COUNTERFEIT IC DETECTION SYSTEM - COMPREHENSIVE TESTS")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("Custom Modules", test_custom_modules),
        ("Configuration", test_configuration),
        ("File Structure", test_file_structure),
        ("Model Creation", test_model_creation),
        ("Data Utilities", test_data_utilities),
        ("Grad-CAM", test_gradcam),
        ("Inference", test_inference),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test failed with exception: {e}")
            results.append((name, False))
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:>8} - {name}")
    
    print("="*70)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED - System is ready for use!")
        print("\nNext steps:")
        print("1. Add your IC image dataset to data/train, data/val, data/test")
        print("2. Train the model: python train.py")
        print("3. Launch the web app: streamlit run app.py")
        return 0
    else:
        print("\n✗ Some tests failed - please review the errors above")
        return 1


if __name__ == '__main__':
    sys.exit(main())
