# CircuitCheck Installation Guide

This guide will help you set up the CircuitCheck system for development or production use.

## System Requirements

### Hardware Requirements

- **Computer**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **RAM**: Minimum 8GB (16GB recommended for ML training)
- **Storage**: At least 10GB free space
- **Camera**: USB webcam or microscope camera
- **Arduino/ESP32**: For electrical testing (optional)

### Software Requirements

- Python 3.8 or higher
- Node.js 16 or higher
- PostgreSQL 12 or higher (optional, SQLite can be used for development)
- Git

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/your-organization/circuitcheck.git
cd circuitcheck
```

### 2. Backend Setup

#### Install Python Dependencies

```bash
cd backend
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Unix/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Database Setup

**Option A: PostgreSQL (Production)**

1. Install PostgreSQL
2. Create database:

```sql
CREATE DATABASE circuitcheck;
CREATE USER circuitcheck_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE circuitcheck TO circuitcheck_user;
```

3. Set environment variables:

```bash
export DATABASE_URI="postgresql://circuitcheck_user:your_password@localhost/circuitcheck"
export SECRET_KEY="your-secret-key-here"
```

**Option B: SQLite (Development)**

The application will automatically create a SQLite database if PostgreSQL is not configured.

#### Start Backend Server

```bash
python app.py
```

The backend will be available at `http://localhost:5000`

### 3. Frontend Setup

```bash
cd ../frontend
npm install
npm start
```

The frontend will be available at `http://localhost:3000`

### 4. Hardware Setup (Optional)

#### Arduino Test Jig

1. Install Arduino IDE
2. Connect your Arduino Uno/Nano or ESP32
3. Open `hardware/arduino/circuitcheck_testjig.ino`
4. Install required libraries:
   - ArduinoJson library
5. Upload the sketch to your Arduino
6. Note the COM port/device path

#### Camera Setup

1. Connect your USB camera or microscope
2. Test camera detection:

```bash
cd hardware/camera
python camera_controller.py
```

### 5. ML Model Setup (Optional)

For advanced users who want to train custom models:

```bash
cd ml_models/image_model
# Follow the training instructions in the counterfeit_detector.py file
```

## Environment Variables

Create a `.env` file in the backend directory:

```
# Database
DATABASE_URI=postgresql://user:password@localhost/circuitcheck
# or for SQLite:
# DATABASE_URI=sqlite:///circuitcheck.db

# Security
SECRET_KEY=your-very-secret-key-here
FLASK_ENV=development  # Change to 'production' for production

# Optional: Camera settings
DEFAULT_CAMERA_ID=0

# Optional: Arduino settings
ARDUINO_PORT=COM3  # Windows
# ARDUINO_PORT=/dev/ttyUSB0  # Linux
# ARDUINO_PORT=/dev/tty.usbserial-*  # macOS
```

## Testing the Installation

### 1. Test Backend API

```bash
curl http://localhost:5000/health
```

Should return: `{"status": "ok"}`

### 2. Test Frontend

Open `http://localhost:3000` in your browser. You should see the CircuitCheck interface.

### 3. Test Hardware (Optional)

#### Test Camera

1. Go to the Analysis page
2. Try capturing an image using the camera interface

#### Test Arduino

```bash
cd hardware
python arduino_controller.py
```

This will test the Arduino connection and perform a sample electrical measurement.

## Troubleshooting

### Common Issues

**Backend won't start:**

- Check Python version: `python --version`
- Ensure virtual environment is activated
- Check if port 5000 is already in use

**Frontend won't start:**

- Check Node.js version: `node --version`
- Clear npm cache: `npm cache clean --force`
- Delete `node_modules` and run `npm install` again

**Database connection issues:**

- Verify PostgreSQL is running
- Check connection string in environment variables
- For SQLite, ensure write permissions in the backend directory

**Camera not detected:**

- Check USB connection
- Test with other camera applications
- Try different camera IDs (0, 1, 2, etc.)

**Arduino not connecting:**

- Check COM port/device path
- Ensure Arduino IDE can communicate with the device
- Verify the correct firmware is uploaded

### Getting Help

1. Check the [FAQ](FAQ.md)
2. Review the [API Documentation](API.md)
3. Submit issues on GitHub
4. Contact the development team

## Next Steps

After installation:

1. Read the [User Guide](USER_GUIDE.md)
2. Review the [API Documentation](API.md)
3. Set up your first component analysis
4. Configure hardware if using electrical testing

## Development Setup

For developers working on CircuitCheck:

### Additional Dependencies

```bash
# Backend development
pip install pytest flask-testing black flake8

# Frontend development
npm install --save-dev @testing-library/react @testing-library/jest-dom
```

### Running Tests

```bash
# Backend tests
cd backend
python -m pytest

# Frontend tests
cd frontend
npm test
```

### Code Formatting

```bash
# Backend
black . --line-length=88
flake8 .

# Frontend
npm run format
npm run lint
```

## Production Deployment

For production deployment, see the [Deployment Guide](DEPLOYMENT.md).

## Security Considerations

- Change all default passwords and secret keys
- Use HTTPS in production
- Regularly update dependencies
- Follow security best practices for your deployment environment
