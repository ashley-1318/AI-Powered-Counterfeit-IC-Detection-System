<<<<<<< HEAD
# CircuitCheck: AI-Powered Counterfeit Component Detection System

An advanced system for detecting counterfeit electronic components (ICs & Connectors) using multimodal AI analysis combining image processing and electrical signature verification.

## Project Overview

CircuitCheck addresses the critical challenge of counterfeit electronic components in supply chains by providing:

- **Real-time detection** of counterfeit ICs and connectors
- **Multimodal analysis** using both visual and electrical signatures
- **Explainable results** with confidence scores and anomaly highlighting
- **User-friendly interface** for technicians and quality control personnel

## System Architecture

### Hardware Layer

- Macro/Microscope Camera: Captures detailed component images
- Test Jig (Arduino/ESP32): Performs non-destructive electrical tests
- USB/Serial connection to computer

### AI/ML Layer

- **Image Analysis Module**: CNN/Transformer model for visual inspection
- **Electrical Signature Module**: ML model for electrical characteristics analysis
- **Fusion Engine**: Combines results for final classification

### Software Layer

- **Frontend**: React-based web interface
- **Backend**: Flask API server
- **Database**: PostgreSQL for storing component data and test results

## Directory Structure

```
CircuitCheck/
├── backend/             # Flask API server
│   ├── api/             # API endpoints
│   ├── database/        # Database models and connections
│   ├── ml_integration/  # ML model integration
│   └── utils/           # Utility functions
├── frontend/            # React web application
│   ├── public/          # Static assets
│   └── src/             # React components and logic
├── ml_models/           # Machine learning models
│   ├── image_model/     # Image analysis model
│   ├── electrical_model/ # Electrical signature model
│   └── fusion_model/    # Fusion engine model
├── hardware/            # Hardware integration code
│   ├── arduino/         # Arduino test jig firmware
│   └── camera/          # Camera integration scripts
└── docs/                # Documentation
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 16+
- Arduino IDE (for hardware setup)

### Backend Setup

1. Navigate to the backend directory:

```
cd backend
```

2. Create a virtual environment:

```
python -m venv venv
```

3. Activate the virtual environment:

- Windows: `venv\Scripts\activate`
- Unix/MacOS: `source venv/bin/activate`

4. Install dependencies:

```
pip install -r requirements.txt
```

# CircuitCheck: AI-Powered Counterfeit IC & Component Detection System

CircuitCheck is an end-to-end, multimodal AI platform for detecting counterfeit electronic components (Integrated Circuits & Connectors) by fusing visual inspection and electrical signature analysis. It unifies: a hardware test jig (Arduino/ESP32), image acquisition, deep learning models, an electrical signature ML analyzer, a fusion engine, and both a React web UI and Streamlit app for rapid experimentation.

## Key Capabilities

- Real-time classification: Genuine / Suspect / Counterfeit
- Multimodal fusion: Image CNN/Transformer + Electrical signature model
- Explainability: Grad-CAM heatmaps + feature/anomaly highlighting
- Flexible interfaces: React frontend (production) & Streamlit app (rapid prototyping)
- Data export & reporting (backend utilities)
- Scalable architecture ready for PostgreSQL/RDBMS integration

## Architecture Overview

### Hardware Layer
- Macro/Microscope camera for high-detail image capture
- Arduino/ESP32 test jig for non-destructive electrical probing
- USB/Serial communication to host machine

### AI / ML Layer
- Image Analysis: Transfer learning (ResNet/EfficientNet or custom CNN) with Grad-CAM explainability
- Electrical Signature Analysis: ML model comparing measured parameters vs known authentic profiles
- Fusion Engine: Weighted / learned combination of modalities for robust final score

### Software Layer
- Backend: Flask API (`backend/`) exposing auth, upload, analysis, results endpoints
- Frontend: React SPA (`frontend/`) for operators & QC technicians
- Streamlit Prototype: `streamlit_app.py` for quick experimentation
- Database: Configured for PostgreSQL (can fall back to SQLite for local dev) storing components & test runs

## Repository Structure (Selected)
```
backend/          Flask API server (auth, upload, component analysis)
frontend/         React application (dashboard, analysis views)
ml_models/        Model code: image, electrical, fusion
hardware/         Arduino firmware + camera control scripts
docs/             Extended documentation & guides
notebooks/        Exploratory data science / model development notebooks
streamlit_app.py  Streamlit interface for rapid testing
train_model.py    Training script(s) for models
```

## Technology Stack
| Component | Technology |
|-----------|------------|
| Languages | Python 3.10, JavaScript (React) |
| DL Framework | PyTorch |
| Web (Prototype) | Streamlit |
| Web (Prod UI) | React + Fetch/XHR |
| Backend | Flask |
| Explainability | Grad-CAM |
| Data / Image | OpenCV, Pillow, Pandas |
| Optional Container | Docker |

## Setup Instructions

### Prerequisites
- Python 3.10+
- Node.js 16+
- Arduino IDE (if using hardware)

### Backend Setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
python app.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```
Visit: http://localhost:3000

### Streamlit Prototype
```bash
pip install -r streamlit_requirements.txt
streamlit run streamlit_app.py
```
Visit: http://localhost:8501

### Hardware Setup (Optional)
1. Flash `hardware/arduino/circuitcheck_testjig.ino` to Arduino/ESP32.
2. Connect measurement leads & camera per docs.
3. Verify serial port and camera accessibility.

## Data & Training
Organize datasets (not included in repo) following a train/val/test split:
```
data/counterfeit_detection/dataset/train/{genuine,fake}
data/counterfeit_detection/dataset/val/{genuine,fake}
data/counterfeit_detection/dataset/test/{genuine,fake}
```
Run model training:
```bash
python train_model.py --epochs 10 --batch-size 16 --output-dir models/
```

## Inference & Fusion
1. Image model predicts authenticity probability.
2. Electrical model computes deviation scores.
3. Fusion engine aggregates (e.g., weighted average or meta-classifier) -> final label + confidence.

## Explainability
- Grad-CAM heatmaps highlight suspicious regions.
- Electrical anomaly report lists out-of-range parameters.

## Usage Workflow
1. Place component in test jig.
2. Capture image & run electrical sweep.
3. Upload via UI or Streamlit.
4. Receive classification + confidence + explanations.
5. Export report (PDF/JSON) via backend utilities.

## Sample Roadmap
- [ ] Expand labeled dataset (multi-angle, lighting variations)
- [ ] Optimize electrical feature extraction & normalization
- [ ] Enhance fusion with learned stacking model
- [ ] Add role-based access & audit logs
- [ ] Containerize services (Docker / Compose)
- [ ] CI pipeline with automated tests & linting

## Future Enhancements (Vision)
- IoT edge deployment with on-device inference
- Blockchain-based traceability for supply chain events
- Mobile app (Flutter) for on-site captures
- Texture & spectral imaging integration

## License
MIT License. See `LICENSE` (add if missing).

## Contributors
Your Name / Organization (replace placeholder)

## Acknowledgements
Inspired by industry needs for secure electronics supply chains; leverages open-source ML tooling.

---
This README merges prior standalone image-focused project info with the broader multimodal CircuitCheck system description.

