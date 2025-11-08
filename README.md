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

5. Start the Flask server:

```
python app.py
```

### Frontend Setup

1. Navigate to the frontend directory:

```
cd frontend
```

2. Install dependencies:

```
npm install
```

3. Start the development server:

```
npm start
```

### Hardware Setup

1. Connect the Arduino/ESP32 to your computer
2. Upload the firmware from the `hardware/arduino` directory
3. Connect the test jig and camera according to the instructions in `hardware/README.md`

## Usage

1. Place the electronic component in the test jig
2. Capture an image using the connected camera
3. Run electrical tests using the Arduino/ESP32
4. Submit for analysis through the web interface
5. Review the results showing PASS/SUSPECT/FAIL with confidence scores and anomaly highlights

## Development Roadmap

- [ ] Data collection of genuine and counterfeit components
- [ ] Train ML models for image and electrical signature analysis
- [ ] Develop fusion engine for combined analysis
- [ ] Build web interface for user interaction
- [ ] Integrate hardware components (camera and test jig)
- [ ] Deploy for testing in production environment

## License

[MIT License](LICENSE)

## Contributors

- [Your Name/Organization]
