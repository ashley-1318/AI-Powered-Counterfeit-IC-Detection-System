import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import numpy as np
from pathlib import Path

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
CLASS_NAMES = ['fake', 'genuine']

# Paths
DATA_DIR = Path("data/counterfeit_detection/organized")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Data preparation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    DATA_DIR / 'train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=CLASS_NAMES,
    shuffle=True,
    seed=42
)

validation_generator = val_test_datagen.flow_from_directory(
    DATA_DIR / 'val',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=CLASS_NAMES,
    shuffle=False,
    seed=42
)

# Create model
def create_model():
    try:
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        print("✅ EfficientNetB0 loaded with standard input shape.")
    except ValueError as e:
        print(f"⚠️ Initial model loading failed: {e}")
        print("Attempting fallback: Explicitly defining input tensor...")
        input_tensor = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_tensor=input_tensor
        )
        print("✅ Fallback successful. EfficientNetB0 loaded with explicit input tensor.")
    
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

# Create and compile model
model = create_model()
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

# Train model
print("Training model...")
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_DIR / 'counterfeit_ic_detector.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ],
    verbose=1
)

print("Training completed. Model saved to:", MODEL_DIR / 'counterfeit_ic_detector.h5')