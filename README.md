# AI-Powered-Counterfeit-IC-Detection-System
An AI-driven image classification system designed to detect counterfeit integrated circuits (ICs) using deep learning and computer vision.
The project applies transfer learning with explainable AI (Grad-CAM) to ensure reliability and transparency in IC authenticity verification.

🚀 Features

🔍 Detects genuine vs counterfeit ICs from image data.

🧠 Uses Transfer Learning (ResNet/EfficientNet) for high accuracy.

📊 Provides explainability via Grad-CAM heatmaps.

💻 Streamlit Web App for easy image upload and live prediction.

🧪 End-to-end workflow: dataset → training → inference → visualization.

🧰 Configurable pipeline with modular code for easy experimentation.

🧩 Tech Stack
Component	Technology
Language	Python 3.10
Deep Learning Framework	PyTorch
Web Framework	Streamlit
Visualization	Matplotlib, Grad-CAM
Data Handling	Pandas, OpenCV, Pillow
Deployment	Docker (optional)
📁 Project Structure
ai-counterfeit-ic-detection/
├── src/
│   ├── dataset.py       # Custom dataset loader
│   ├── model.py         # CNN architecture (ResNet/EfficientNet)
│   ├── train.py         # Training pipeline
│   ├── inference.py     # Prediction script
│   ├── explain.py       # Grad-CAM visualization
│   ├── utils.py         # Utility functions & metrics
│   └── app.py           # Streamlit web interface
├── data/                # Dataset directory (not uploaded)
├── models/              # Trained model weights
├── examples/            # Sample images and CSV
├── requirements.txt
├── Dockerfile
├── LICENSE
└── README.md

⚙️ Installation & Setup
1️⃣ Clone this repository
git clone https://github.com/<your-username>/ai-counterfeit-ic-detection.git
cd ai-counterfeit-ic-detection

2️⃣ Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

🧠 Model Training

Prepare your dataset either in folder structure or CSV format:

Folder format:

data/images/train/genuine/*.jpg
data/images/train/counterfeit/*.jpg
data/images/val/genuine/*.jpg
data/images/val/counterfeit/*.jpg


CSV format:

filepath,label
examples/sample_images/genuine1.jpg,genuine
examples/sample_images/counterfeit1.jpg,counterfeit


Then train the model:

python src/train.py --train_csv examples/sample_labels.csv --val_csv examples/sample_labels.csv --epochs 10 --batch-size 16 --output models/

🔎 Inference (Batch Prediction)

Run predictions on a folder of images:

python src/inference.py --weights models/best.pth --input_dir examples/sample_images --output results.json

🌐 Run the Streamlit App
streamlit run src/app.py


Then open in browser → http://localhost:8501

Upload an IC image to view:

Predicted label (Genuine/Counterfeit)

Prediction probabilities

Grad-CAM heatmap for visual explanation

🧪 Example Output
Image	Prediction	Confidence	Grad-CAM

	Genuine	0.96	


	Counterfeit	0.91	
📊 Results (Sample)
Metric	Value
Accuracy	94.8%
Precision	92.5%
Recall	93.1%
F1-Score	92.8%

(Sample results — varies with dataset quality & size.)

🧰 Future Enhancements

Integrate with IoT hardware camera module for real-time IC scanning.

Add multi-angle and texture-based analysis.

Develop mobile app version (Flutter) for on-site inspection.

Integrate Blockchain-based verification for traceable IC supply chains.

📜 License

This project is licensed under the MIT License — see the LICENSE
 file for details.
