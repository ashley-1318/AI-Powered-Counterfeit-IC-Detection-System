"""
Streamlit Web Application for IC Counterfeit Detection
Real-time image analysis with Grad-CAM visualization
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import os
import matplotlib.pyplot as plt
import seaborn as sns

from inference import ICDetector
from config import STREAMLIT_CONFIG, MODEL_CONFIG, PATHS


# Page configuration
st.set_page_config(
    page_title=STREAMLIT_CONFIG['page_title'],
    page_icon=STREAMLIT_CONFIG['page_icon'],
    layout=STREAMLIT_CONFIG['layout']
)


@st.cache_resource
def load_detector(model_path: str):
    """Load the IC detector model (cached)"""
    if os.path.exists(model_path):
        return ICDetector(model_path)
    else:
        st.warning("No trained model found. Using untrained model for demo.")
        return ICDetector()


def plot_probabilities(probabilities: dict):
    """Plot class probabilities as a bar chart"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    
    colors = ['#2ecc71' if p == max(probs) else '#3498db' for p in probs]
    bars = ax.bar(classes, probs, color=colors, alpha=0.7)
    
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Class Probabilities', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.2%}',
                ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    return fig


def display_gradcam(original_image: np.ndarray, cam: np.ndarray, overlayed: np.ndarray):
    """Display original image, CAM heatmap, and overlay"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # CAM heatmap
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlayed)
    axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🔍 AI-Powered Counterfeit IC Detection System")
    st.markdown("""
    This system uses deep learning and explainable AI (Grad-CAM) to identify counterfeit 
    integrated circuits from images. Upload an IC image to get started!
    """)
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Model selection
    model_path = st.sidebar.text_input(
        "Model Path",
        value=os.path.join(PATHS['models_dir'], 'best_model.pth'),
        help="Path to the trained model checkpoint"
    )
    
    # Grad-CAM settings
    st.sidebar.subheader("Grad-CAM Settings")
    show_gradcam = st.sidebar.checkbox("Show Grad-CAM Visualization", value=True)
    alpha = st.sidebar.slider("Overlay Transparency", 0.0, 1.0, 0.4, 0.05)
    
    # Model info
    st.sidebar.subheader("Model Information")
    st.sidebar.write(f"**Architecture:** {MODEL_CONFIG['architecture']}")
    st.sidebar.write(f"**Classes:** {MODEL_CONFIG['num_classes']}")
    st.sidebar.write(f"**Device:** {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Load detector
    try:
        detector = load_detector(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure the model file exists or train a model first.")
        return
    
    # Main content
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload IC Image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image of an integrated circuit"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_column_width=True)
        
        # Save temporary file
        temp_path = "/tmp/temp_ic_image.jpg"
        image.save(temp_path)
        
        # Make prediction
        with st.spinner("Analyzing image..."):
            try:
                if show_gradcam:
                    result, cam, overlayed = detector.predict_with_explanation(temp_path, alpha)
                else:
                    result = detector.predict(temp_path)
                
                # Display results
                with col2:
                    st.subheader("Analysis Results")
                    
                    # Prediction
                    prediction = result['predicted_label']
                    confidence = result['confidence']
                    
                    # Color-coded result
                    if prediction == 'Genuine':
                        st.success(f"✅ **{prediction}**")
                        st.metric("Confidence", f"{confidence:.2%}")
                    else:
                        st.error(f"⚠️ **{prediction}**")
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Additional info
                    if confidence < 0.6:
                        st.warning("⚠️ Low confidence prediction. Results may be uncertain.")
                    elif confidence > 0.9:
                        st.info("✓ High confidence prediction.")
                
                # Probability distribution
                st.markdown("---")
                st.subheader("Probability Distribution")
                prob_fig = plot_probabilities(result['probabilities'])
                st.pyplot(prob_fig)
                plt.close()
                
                # Grad-CAM visualization
                if show_gradcam:
                    st.markdown("---")
                    st.subheader("Explainable AI Visualization (Grad-CAM)")
                    st.markdown("""
                    The heatmap shows which parts of the image the model focused on when making its prediction.
                    **Warmer colors** (red/yellow) indicate regions of **higher importance**.
                    """)
                    
                    # Resize original image for display
                    from data_utils import preprocess_image
                    from config import DATA_CONFIG
                    _, original_array = preprocess_image(
                        temp_path,
                        DATA_CONFIG['image_size'],
                        DATA_CONFIG['mean'],
                        DATA_CONFIG['std']
                    )
                    original_resized = image.resize(DATA_CONFIG['image_size'][::-1])
                    original_array = np.array(original_resized)
                    
                    gradcam_fig = display_gradcam(original_array, cam, overlayed)
                    st.pyplot(gradcam_fig)
                    plt.close()
                
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.info("Please ensure the image is valid and try again.")
    
    # Instructions
    else:
        st.info("👆 Please upload an IC image to start the analysis.")
        
        st.markdown("---")
        st.subheader("How to Use")
        st.markdown("""
        1. **Upload** an image of an integrated circuit (IC)
        2. The system will **analyze** the image using deep learning
        3. View the **prediction** (Genuine or Counterfeit) with confidence score
        4. Examine the **Grad-CAM visualization** to understand what the model focuses on
        5. Review the **probability distribution** for all classes
        """)
        
        st.subheader("About the Technology")
        st.markdown("""
        - **Transfer Learning**: Uses pre-trained models (ResNet/EfficientNet) for better accuracy
        - **Grad-CAM**: Provides visual explanations for model predictions
        - **Real-time Analysis**: Fast inference with GPU acceleration (if available)
        - **Binary Classification**: Distinguishes between genuine and counterfeit ICs
        """)


if __name__ == '__main__':
    main()
