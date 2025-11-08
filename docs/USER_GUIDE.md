# CircuitCheck User Guide

Welcome to CircuitCheck, an AI-powered system for detecting counterfeit electronic components. This guide will help you understand how to use the system effectively.

## Overview

CircuitCheck uses advanced machine learning and electrical testing to identify counterfeit ICs and connectors. The system combines:

- **Visual Analysis**: AI-powered image analysis to detect visual anomalies
- **Electrical Testing**: Measurement of electrical characteristics
- **Fusion Analysis**: Combined results for accurate classification

## Getting Started

### 1. Accessing the System

Open your web browser and navigate to `http://localhost:3000` (or your deployed URL).

### 2. User Registration/Login

1. Click "Login" in the top navigation
2. If you're a new user, click the "Register" tab
3. Fill in your details and create an account
4. Log in with your credentials

## Main Interface

### Dashboard

The dashboard provides an overview of your testing activity:

- **Summary Cards**: Total tests, pass rate, suspect/failed components
- **Charts**: Result distribution and testing trends
- **Recent Results**: Quick view of your latest tests

### Navigation Menu

- **Dashboard**: Overview and statistics
- **Analysis**: Perform component testing
- **Results**: View detailed test history

## Performing Component Analysis

### Step 1: Prepare Your Component

1. Ensure the component is clean and well-lit
2. Position the component flat and stable
3. If using electrical testing, connect the test jig

### Step 2: Capture Component Image

**Option A: Upload Image**

1. Go to the Analysis page
2. Click or drag an image file into the upload area
3. Supported formats: JPG, PNG (max 16MB)

**Option B: Camera Capture** (if camera is connected)

1. Position the component under the camera
2. Adjust lighting and focus
3. Click "Capture Image"

### Step 3: Enter Component Information

1. Enter the part number (e.g., "MC74HC00AN")
2. This helps improve analysis accuracy

### Step 4: Electrical Measurements (Optional)

If you have electrical testing capability:

#### Resistance Measurements

- Measure resistance between component pins
- Enter values in Ohms (Ω)
- Common measurements: pin-to-pin resistance

#### Capacitance Measurements

- Measure capacitance from pins to ground
- Enter values in picofarads (pF)

#### Leakage Current

- Measure leakage current at each pin
- Enter values in microamps (µA)

#### Timing Measurements

- Rise time, fall time, propagation delay
- Enter values in nanoseconds (ns)

**Note**: You can perform analysis with just images, or combine with electrical data for better accuracy.

### Step 5: Start Analysis

1. Click "Start Analysis"
2. Wait for processing (typically 10-30 seconds)
3. Review the results

## Understanding Results

### Classification Results

The system provides three possible classifications:

- **PASS** (Green): Component appears genuine
- **SUSPECT** (Orange): Component shows some anomalies, requires further inspection
- **FAIL** (Red): Component likely counterfeit

### Confidence Scores

- **Overall Confidence**: Combined confidence from all analyses
- **Image Analysis**: Confidence from visual inspection
- **Electrical Analysis**: Confidence from electrical measurements

### Anomaly Detection

#### Visual Anomalies

- Surface defects
- Marking inconsistencies
- Package abnormalities
- Location and severity information

#### Electrical Anomalies

- Measurements outside expected ranges
- Comparison with reference values
- Deviation percentages

### Analysis Explanation

The system provides human-readable explanations of:

- Why the component passed or failed
- What anomalies were detected
- Confidence assessment reasoning

## Managing Results

### Viewing Test History

1. Go to the Results page
2. View all your previous tests in a table format
3. Filter by date, result type, or component

### Detailed Result View

1. Click "Details" on any test result
2. View comprehensive information:
   - Test parameters and settings
   - Complete anomaly information
   - Electrical measurement data
   - Analysis explanations

### Downloading Reports

1. Click "Download Report" or the download icon
2. Generate PDF reports with:
   - Test summary and results
   - Component images (if included)
   - Detailed anomaly information
   - Recommendations

## Best Practices

### Image Quality

**Good Images:**

- Well-lit with even lighting
- Component clearly visible
- Minimal shadows
- Sharp focus on component markings
- Component fills most of the frame

**Avoid:**

- Blurry or out-of-focus images
- Extreme shadows or glare
- Component too small in frame
- Poor lighting conditions

### Electrical Testing

**For Accurate Results:**

- Ensure good electrical contact
- Use appropriate voltage levels
- Allow measurements to stabilize
- Clean component pins before testing
- Use proper test procedures

### Component Handling

- Handle components with anti-static precautions
- Avoid touching component pins directly
- Store components in anti-static bags
- Document component source and batch information

## Troubleshooting

### Common Issues

**"No image file provided"**

- Ensure you've selected or captured an image
- Check file format (JPG, PNG only)
- Verify file size is under 16MB

**"Analysis failed"**

- Check internet connection
- Verify image is readable
- Try with a different image
- Contact support if problem persists

**Poor Results**

- Improve image quality
- Add electrical measurements
- Verify component part number
- Ensure proper component positioning

### Getting Help

1. Check error messages for specific guidance
2. Review this user guide
3. Contact your system administrator
4. Submit feedback through the system

## Advanced Features

### Batch Testing

For testing multiple components:

1. Use consistent naming conventions
2. Maintain detailed records
3. Compare results across batches
4. Look for patterns in failures

### Integration with Quality Systems

CircuitCheck can be integrated with:

- Quality management systems
- Inventory tracking
- Supply chain management
- Automated testing workflows

## Privacy and Security

- Test results are stored securely
- Images are processed locally when possible
- User data is protected according to privacy policies
- Regular security updates are applied

## Feedback and Improvement

Help us improve CircuitCheck:

- Report false positives/negatives
- Suggest new features
- Share challenging components for model training
- Provide feedback on user experience

## Support

For technical support:

- Email: support@circuitcheck.com
- Documentation: [docs.circuitcheck.com](https://docs.circuitcheck.com)
- Training materials: Available in the Help section
- Community forum: [community.circuitcheck.com](https://community.circuitcheck.com)
