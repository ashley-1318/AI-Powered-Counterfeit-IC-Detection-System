import React, { useState, useCallback, useContext } from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Box,
  TextField,
  Alert,
  CircularProgress,
  Chip,
  Divider,
  Paper,
  List,
  ListItem,
  ListItemText
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import { CloudUpload, Camera, PlayArrow, GetApp } from '@mui/icons-material';
import { AuthContext } from '../context/AuthContext';
import analysisService from '../services/analysisService';

const RESULT_COLORS = {
  PASS: '#4caf50',
  SUSPECT: '#ff9800',
  FAIL: '#f44336'
};

function Analysis() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [partNumber, setPartNumber] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const { user } = useContext(AuthContext);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedImage(file);
      
      // Create preview URL
      const reader = new FileReader();
      reader.onload = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png']
    },
    multiple: false
  });

  // Removed electrical data handling

  const handleAnalyze = async () => {
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError('');
    setAnalysisResult(null);

    try {
      // Image-only analysis (electrical data removed)
      const result = await analysisService.analyzeComponent(
        selectedImage,
        null, // No electrical data
        null, // component_id
        partNumber || 'Unknown'
      );

      setAnalysisResult(result);
    } catch (error) {
      console.error('Analysis error:', error);
      setError(error.error || 'Analysis failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadReport = async () => {
    if (!analysisResult?.test_id) return;
    
    try {
      await analysisService.downloadReport(analysisResult.test_id);
    } catch (error) {
      console.error('Download error:', error);
      setError('Failed to download report');
    }
  };

  // Removed electrical inputs rendering function

  const renderAnomalies = (anomalies) => {
    if (!anomalies) return null;

    return (
      <Box sx={{ mt: 2 }}>
        <Typography variant="h6" gutterBottom>
          Detected Anomalies
        </Typography>
        
        {/* Visual Anomalies */}
        {anomalies.visual && anomalies.visual.length > 0 && (
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                Visual Anomalies ({anomalies.visual.length})
              </Typography>
              <List dense>
                {anomalies.visual.slice(0, 5).map((anomaly, index) => (
                  <ListItem key={index}>
                    <ListItemText
                      primary={`${anomaly.type?.replace('_', ' ') || 'Unknown'} anomaly`}
                      secondary={`Severity: ${(anomaly.severity * 100).toFixed(1)}% at (${anomaly.x}, ${anomaly.y})`}
                    />
                  </ListItem>
                ))}
                {anomalies.visual.length > 5 && (
                  <ListItem>
                    <ListItemText
                      secondary={`And ${anomalies.visual.length - 5} more...`}
                    />
                  </ListItem>
                )}
              </List>
            </CardContent>
          </Card>
        )}

        {/* Electrical anomalies section removed */}
      </Box>
    );
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Component Analysis
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Image Upload */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Component Image
              </Typography>
              
              <Paper
                {...getRootProps()}
                sx={{
                  p: 3,
                  border: '2px dashed #ccc',
                  borderColor: isDragActive ? 'primary.main' : 'grey.300',
                  backgroundColor: isDragActive ? 'action.hover' : 'background.paper',
                  cursor: 'pointer',
                  textAlign: 'center',
                  mb: 2
                }}
              >
                <input {...getInputProps()} />
                <CloudUpload sx={{ fontSize: 48, color: 'grey.400', mb: 2 }} />
                <Typography variant="body1" gutterBottom>
                  {isDragActive
                    ? "Drop the image here..."
                    : "Drag & drop a component image here, or click to select"}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Supports: JPG, PNG (Max 16MB)
                </Typography>
              </Paper>

              {imagePreview && (
                <Box sx={{ textAlign: 'center' }}>
                  <img
                    src={imagePreview}
                    alt="Component preview"
                    style={{
                      maxWidth: '100%',
                      maxHeight: 300,
                      borderRadius: 8,
                      boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                    }}
                  />
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    {selectedImage?.name}
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>

          {/* Component Information */}
          <Card sx={{ mt: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Component Information
              </Typography>
              <TextField
                fullWidth
                label="Part Number"
                value={partNumber}
                onChange={(e) => setPartNumber(e.target.value)}
                placeholder="e.g., MC74HC00AN"
                helperText="Optional: Enter the component part number for better analysis"
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Analysis Controls - Replaced electrical measurements */}
        <Grid item xs={12} md={6}>
          <Typography variant="h6" gutterBottom>
            Analysis Controls
          </Typography>
          
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="body1" sx={{ mb: 2 }}>
                Our system now performs component analysis using only visual inspection technology.
                Upload a clear image of your component for best results.
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Providing an accurate part number will improve analysis accuracy.
              </Typography>
            </CardContent>
          </Card>

          <Box sx={{ textAlign: 'center', mt: 3 }}>
            <Button
              variant="contained"
              size="large"
              startIcon={loading ? <CircularProgress size={20} /> : <PlayArrow />}
              onClick={handleAnalyze}
              disabled={loading || !selectedImage}
              sx={{ minWidth: 200 }}
            >
              {loading ? 'Analyzing...' : 'Start Analysis'}
            </Button>
          </Box>
        </Grid>

        {/* Analysis Results */}
        {analysisResult && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">
                    Analysis Results
                  </Typography>
                  <Button
                    startIcon={<GetApp />}
                    onClick={handleDownloadReport}
                  >
                    Download Report
                  </Button>
                </Box>

                {/* Main Result */}
                <Box sx={{ textAlign: 'center', mb: 3 }}>
                  <Chip
                    label={analysisResult.classification}
                    sx={{
                      fontSize: '1.5rem',
                      height: 60,
                      backgroundColor: RESULT_COLORS[analysisResult.classification],
                      color: 'white',
                      fontWeight: 'bold',
                      minWidth: 120
                    }}
                  />
                  <Typography variant="h6" sx={{ mt: 2 }}>
                    Confidence: {(analysisResult.confidence * 100).toFixed(1)}%
                  </Typography>
                </Box>

                <Divider sx={{ mb: 2 }} />

                {/* Detailed Scores */}
                <Grid container spacing={3}>
                  <Grid item xs={12}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1" gutterBottom>
                          Visual Analysis Result
                        </Typography>
                        <Typography variant="h5" sx={{ color: RESULT_COLORS[analysisResult.classification || 'FAIL'] }}>
                          {analysisResult.classification || 'N/A'}
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Confidence: {((analysisResult.confidence || 0) * 100).toFixed(1)}%
                        </Typography>
                        <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                          Part Number: {partNumber || 'Unknown'}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>

                {/* Explanation */}
                {analysisResult.explanation && (
                  <Box sx={{ mt: 3 }}>
                    <Typography variant="h6" gutterBottom>
                      Analysis Explanation
                    </Typography>
                    <Paper sx={{ p: 2, backgroundColor: 'grey.50' }}>
                      {analysisResult.explanation.map((line, index) => (
                        <Typography key={index} variant="body2" sx={{ mb: 1 }}>
                          {line}
                        </Typography>
                      ))}
                    </Paper>
                  </Box>
                )}

                {/* Anomalies */}
                {renderAnomalies(analysisResult.anomalies)}
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Container>
  );
}

export default Analysis;