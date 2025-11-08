"""
CircuitCheck Camera Integration Module

This module handles camera operations for capturing component images,
including support for USB cameras, webcams, and microscope cameras.

Features:
- Auto-detection of available cameras
- Image capture with adjustable parameters
- Image preprocessing and enhancement
- Integration with analysis pipeline

Author: CircuitCheck Team
Version: 1.0
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraController:
    def __init__(self, camera_id=0):
        """Initialize camera controller"""
        self.camera_id = camera_id
        self.camera = None
        self.is_connected = False
        self.frame_width = 1280
        self.frame_height = 720
        self.capture_settings = {
            'brightness': 0.5,
            'contrast': 0.5,
            'saturation': 0.5,
            'exposure': -6,  # Auto exposure
            'focus': 0,      # Auto focus
        }
        
    def connect(self):
        """Connect to the camera"""
        try:
            self.camera = cv2.VideoCapture(self.camera_id)
            
            if not self.camera.isOpened():
                raise Exception(f"Cannot open camera {self.camera_id}")
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Apply capture settings
            self._apply_settings()
            
            self.is_connected = True
            logger.info(f"Camera {self.camera_id} connected successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to camera: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the camera"""
        if self.camera:
            self.camera.release()
            self.is_connected = False
            logger.info("Camera disconnected")
    
    def _apply_settings(self):
        """Apply camera settings"""
        if not self.camera:
            return
            
        # Set camera properties if supported
        try:
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, self.capture_settings['brightness'])
            self.camera.set(cv2.CAP_PROP_CONTRAST, self.capture_settings['contrast'])
            self.camera.set(cv2.CAP_PROP_SATURATION, self.capture_settings['saturation'])
            self.camera.set(cv2.CAP_PROP_EXPOSURE, self.capture_settings['exposure'])
        except Exception as e:
            logger.warning(f"Could not apply all camera settings: {e}")
    
    def capture_image(self, filename=None, preprocessing=True):
        """
        Capture an image from the camera
        
        Args:
            filename: Optional filename to save the image
            preprocessing: Whether to apply image preprocessing
            
        Returns:
            numpy.ndarray: Captured image
        """
        if not self.is_connected:
            logger.error("Camera not connected")
            return None
            
        try:
            # Capture frame
            if self.camera is None:
                logger.error("Camera object is None")
                return None
                
            ret, frame = self.camera.read()
            
            if not ret:
                logger.error("Failed to capture frame")
                return None
            
            # Apply preprocessing if requested
            if preprocessing:
                frame = self._preprocess_image(frame)
            
            # Save image if filename provided
            if filename:
                success = cv2.imwrite(filename, frame)
                if success:
                    logger.info(f"Image saved to {filename}")
                else:
                    logger.error(f"Failed to save image to {filename}")
            
            return frame
            
        except Exception as e:
            logger.error(f"Error capturing image: {e}")
            return None
    
    def _preprocess_image(self, image):
        """
        Apply preprocessing to improve image quality
        
        Args:
            image: Input image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        try:
            # Convert to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                # OpenCV uses BGR, convert to RGB for consistency
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply noise reduction
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
            
            # Enhance contrast using CLAHE
            if len(denoised.shape) == 3:
                # Convert to LAB color space
                lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                
                # Merge channels and convert back
                lab = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                # Grayscale image
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(denoised)
            
            # Sharpening filter
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Blend original and sharpened (subtle sharpening)
            result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
            
            return result
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image
    
    def get_camera_info(self):
        """Get camera information"""
        if not self.is_connected or self.camera is None:
            return None
            
        try:
            info = {
                'camera_id': self.camera_id,
                'width': int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.camera.get(cv2.CAP_PROP_FPS),
                'brightness': self.camera.get(cv2.CAP_PROP_BRIGHTNESS),
                'contrast': self.camera.get(cv2.CAP_PROP_CONTRAST),
                'exposure': self.camera.get(cv2.CAP_PROP_EXPOSURE),
            }
            return info
        except Exception as e:
            logger.error(f"Error getting camera info: {e}")
            return None
    
    def update_settings(self, settings):
        """Update camera settings"""
        self.capture_settings.update(settings)
        if self.is_connected:
            self._apply_settings()

def detect_cameras():
    """
    Detect available cameras
    
    Returns:
        list: List of available camera IDs
    """
    available_cameras = []
    
    # Test camera IDs 0-5
    for camera_id in range(6):
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            available_cameras.append(camera_id)
            cap.release()
    
    logger.info(f"Detected cameras: {available_cameras}")
    return available_cameras

def create_capture_session(output_dir="captures"):
    """
    Create a new capture session
    
    Args:
        output_dir: Directory to save captured images
        
    Returns:
        tuple: (session_id, session_dir)
    """
    # Create session ID based on timestamp
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(output_dir, f"session_{session_id}")
    
    # Create directory
    os.makedirs(session_dir, exist_ok=True)
    
    # Create session metadata
    metadata = {
        'session_id': session_id,
        'created_at': datetime.now().isoformat(),
        'output_dir': session_dir
    }
    
    metadata_file = os.path.join(session_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Created capture session: {session_id}")
    return session_id, session_dir

# Example usage and testing functions
def test_camera_capture():
    """Test camera capture functionality"""
    print("Testing camera capture...")
    
    # Detect available cameras
    cameras = detect_cameras()
    if not cameras:
        print("No cameras detected!")
        return
    
    # Use first available camera
    camera_controller = CameraController(cameras[0])
    
    if not camera_controller.connect():
        print("Failed to connect to camera!")
        return
    
    # Get camera info
    info = camera_controller.get_camera_info()
    print(f"Camera info: {info}")
    
    # Create capture session
    session_id, session_dir = create_capture_session()
    
    # Capture a test image
    filename = os.path.join(session_dir, "test_capture.jpg")
    image = camera_controller.capture_image(filename)
    
    if image is not None:
        print(f"Test capture successful: {filename}")
        print(f"Image shape: {image.shape}")
    else:
        print("Test capture failed!")
    
    # Disconnect
    camera_controller.disconnect()

if __name__ == "__main__":
    test_camera_capture()