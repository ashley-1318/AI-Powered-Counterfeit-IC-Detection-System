"""
Setup script to create directory structure for the project
Run this script to initialize the project folders
"""

import os
from config import PATHS


def create_directory_structure():
    """Create necessary directories for the project"""
    
    directories = [
        PATHS['data_dir'],
        PATHS['train_dir'],
        os.path.join(PATHS['train_dir'], 'genuine'),
        os.path.join(PATHS['train_dir'], 'counterfeit'),
        PATHS['val_dir'],
        os.path.join(PATHS['val_dir'], 'genuine'),
        os.path.join(PATHS['val_dir'], 'counterfeit'),
        PATHS['test_dir'],
        os.path.join(PATHS['test_dir'], 'genuine'),
        os.path.join(PATHS['test_dir'], 'counterfeit'),
        PATHS['models_dir'],
        PATHS['checkpoints_dir'],
        PATHS['results_dir'],
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    # Create README files in data directories
    readme_content = """# Dataset Directory

Place your IC images in this directory following the structure:

- genuine/: Images of genuine (authentic) integrated circuits
- counterfeit/: Images of counterfeit (fake) integrated circuits

## Image Requirements:

- Format: JPG, PNG, BMP, or TIFF
- Resolution: 224x224 or higher (will be resized automatically)
- Clear, well-lit images of IC chips
- Focused on the IC markings and surface features

## Tips for Dataset Collection:

1. Ensure balanced classes (similar number of genuine and counterfeit samples)
2. Include various IC types and manufacturers
3. Capture different lighting conditions and angles
4. Maintain consistent image quality across the dataset
5. Split data: ~80% train, ~10% validation, ~10% test

## Example:
```
train/
├── genuine/
│   ├── genuine_ic_001.jpg
│   ├── genuine_ic_002.jpg
│   └── ...
└── counterfeit/
    ├── fake_ic_001.jpg
    ├── fake_ic_002.jpg
    └── ...
```
"""
    
    # Write README to data directories
    for split in ['train', 'val', 'test']:
        readme_path = os.path.join(PATHS['data_dir'], split, 'README.md')
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"✓ Created README: {readme_path}")
    
    print("\n" + "="*60)
    print("Directory structure created successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Add your IC images to the data directories")
    print("2. Configure model parameters in config.py")
    print("3. Run training: python train.py")
    print("4. Launch web app: streamlit run app.py")


if __name__ == '__main__':
    create_directory_structure()
