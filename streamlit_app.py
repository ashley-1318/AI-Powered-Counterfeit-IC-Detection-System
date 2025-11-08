import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import json  # Added missing import
# Use keras with a TensorFlow fallback to avoid attribute errors in some environments
try:
    import tensorflow as tf
    from tensorflow import keras
except Exception:
    import keras
    tf = None  # ensure tf is defined when TensorFlow import fails

class CounterfeitICDetector:
    """Detector wrapping a binary EfficientNet-based model."""
    def __init__(self, model_path: Path):
        self.model = keras.models.load_model(model_path)
        self.img_size = 224  # EfficientNetB0 input size
        self.class_names = ['Fake', 'Genuine']

    def preprocess_image(self, image: Image.Image):
        image = image.convert("RGB").resize((self.img_size, self.img_size))
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def predict_single_image(self, image: Image.Image):
        processed_image = self.preprocess_image(image)
        prediction_prob = float(self.model.predict(processed_image, verbose=0)[0][0])
        predicted_class_idx = int(prediction_prob > 0.5)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = prediction_prob if predicted_class_idx == 1 else (1 - prediction_prob)
        return {
            'prediction': predicted_class,
            'confidence': float(confidence),
            'raw_score': prediction_prob,
            'is_genuine': predicted_class == 'Genuine'
        }

    def get_prediction_explanation(self, prediction_result):
        pred = prediction_result['prediction']
        conf = prediction_result['confidence']
        if pred == 'Genuine':
            if conf > 0.9:
                explanation = f"This component appears to be GENUINE with high confidence ({conf:.1%}). All visual indicators suggest authenticity."
            elif conf > 0.7:
                explanation = f"This component is likely GENUINE ({conf:.1%} confidence). Most visual indicators suggest authenticity."
            else:
                explanation = f"This component appears to be GENUINE but with lower confidence ({conf:.1%}). Consider additional verification."
        else:
            if conf > 0.9:
                explanation = f"This component appears to be COUNTERFEIT with high confidence ({conf:.1%}). Strong visual indicators of counterfeiting detected."
            elif conf > 0.7:
                explanation = f"This component is likely COUNTERFEIT ({conf:.1%} confidence). Several visual indicators of counterfeiting detected."
            else:
                explanation = f"This component appears to be COUNTERFEIT but with lower confidence ({conf:.1%}). Further inspection recommended."
        return explanation

# Streamlit app configuration
st.set_page_config(
    page_title="CircuitCheck - Counterfeit IC Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App header
st.title("🔍 CircuitCheck - AI-Powered Counterfeit IC Detection")
st.markdown("""
### Detect Counterfeit Electronic Components with Deep Learning

Upload an image of an IC or connector to analyze its authenticity using our trained EfficientNetB0 model.
""")

# Sidebar with model information
st.sidebar.header("📊 Model Information")

# Load model info if available
model_info_path = Path("models/model_info.json")
if model_info_path.exists():
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    st.sidebar.metric("Test Accuracy", f"{model_info['test_accuracy']:.1%}")
    st.sidebar.metric("F1 Score", f"{model_info['f1_score']:.3f}")
    st.sidebar.write(f"**Model:** {model_info['base_model']}")
    st.sidebar.write(f"**Training Epochs:** {model_info['training_epochs']}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 How to Use")
st.sidebar.markdown("""
1. Upload a clear image of the IC/connector
2. Click 'Analyze Component'
3. View the prediction results
4. Check the confidence score and explanation
""")

# Main content area
image = None  # Initialize to avoid unbound variable warning
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Component Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a clear image of the IC or connector you want to analyze"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
with col2:
    st.subheader("🤖 Analysis Results")
    
    if st.button("🔍 Analyze Component", type="primary"):
        with st.spinner("Analyzing image..."):
            if image is None and uploaded_file is not None:
                image = Image.open(uploaded_file)
            try:
                # Initialize detector
                model_path = Path("models/counterfeit_ic_detector.h5")
                if not model_path.exists():
                    st.error(f"Model file not found at {model_path}")
                    st.stop()
                    model_path = Path("models/counterfeit_ic_detector.h5")
                    model_path = Path("models/counterfeit_ic_detector.h5")
                    if not model_path.exists():
                        st.error(f"Model file not found at {model_path}")
                        st.stop()
                    
                    detector = CounterfeitICDetector(model_path)
                    
                    # Make prediction
                    result = detector.predict_single_image(image)
                    explanation = detector.get_prediction_explanation(result)
                    
                    # Display results
                    prediction = result['prediction']
                model_path = Path("models/counterfeit_ic_detector.h5")
                if not model_path.exists():
                    st.error(f"Model file not found at {model_path}")
                    st.stop()

                try:
                    detector = CounterfeitICDetector(model_path)
                    result = detector.predict_single_image(image)
                    explanation = detector.get_prediction_explanation(result)

                    prediction = result['prediction']
                    confidence = result['confidence']

                    if prediction == "Genuine":
                        color = "green"
                        icon = "✅"
                    else:
                        color = "red"
                        icon = "⚠️"

                    st.markdown(f"""
                    <div style="padding: 20px; border-radius: 10px; border: 2px solid {color};">
                        <h3 style="color: {color};">{icon} {prediction}</h3>
                        <p><strong>Confidence:</strong> {confidence:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("### 📋 Detailed Explanation")
                    st.info(explanation)

                    st.markdown("### 📈 Technical Details")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Confidence Score", f"{confidence:.1%}")
                        st.metric("Prediction Class", prediction)
                    with col_b:
                        st.metric("Raw Model Score", f"{result['raw_score']:.3f}")
                        authenticity = "Yes" if result['is_genuine'] else "No"
                        st.metric("Authentic Component", authenticity)

                    if confidence < 0.7:
                        st.warning("⚠️ Low confidence prediction. Consider additional verification methods.")
                    elif confidence < 0.85:
                        st.info("ℹ️ Moderate confidence. Cross-verification recommended for critical applications.")
                    else:
                        st.success("✅ High confidence prediction.")
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
