import React, { useState, useEffect, useContext } from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  Divider
} from '@mui/material';
import { GetApp, Visibility } from '@mui/icons-material';
import { AuthContext } from '../context/AuthContext';
import analysisService from '../services/analysisService';

const RESULT_COLORS = {
  PASS: '#4caf50',
  SUSPECT: '#ff9800',
  FAIL: '#f44336'
};

function Results() {
  const [results, setResults] = useState([]);
  const [selectedResult, setSelectedResult] = useState(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const { user } = useContext(AuthContext);

  useEffect(() => {
    loadResults();
  }, [user]);

  const loadResults = async () => {
    try {
      const data = await analysisService.getRecentResults(50, user?.user_id);
      setResults(data);
    } catch (error) {
      console.error('Error loading results:', error);
      setError('Failed to load test results');
    } finally {
      setLoading(false);
    }
  };

  const handleViewDetails = async (resultId) => {
    try {
      const details = await analysisService.getTestResult(resultId);
      setSelectedResult(details);
      setDetailsOpen(true);
    } catch (error) {
      console.error('Error loading result details:', error);
      setError('Failed to load result details');
    }
  };

  const handleDownloadReport = async (testId) => {
    try {
      await analysisService.downloadReport(testId);
    } catch (error) {
      console.error('Download error:', error);
      setError('Failed to download report');
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
  };

  const renderAnomalies = (anomalies) => {
    if (!anomalies) return <Typography>No anomaly data available</Typography>;

    return (
      <Box>
        {/* Visual Anomalies */}
        {anomalies.visual && anomalies.visual.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Visual Anomalies ({anomalies.visual.length})
            </Typography>
            <List dense>
              {anomalies.visual.slice(0, 3).map((anomaly, index) => (
                <ListItem key={index}>
                  <ListItemText
                    primary={`${anomaly.type?.replace('_', ' ') || 'Unknown'} anomaly`}
                    secondary={`Severity: ${(anomaly.severity * 100).toFixed(1)}% at position (${anomaly.x}, ${anomaly.y})`}
                  />
                </ListItem>
              ))}
              {anomalies.visual.length > 3 && (
                <ListItem>
                  <ListItemText secondary={`And ${anomalies.visual.length - 3} more...`} />
                </ListItem>
              )}
            </List>
          </Box>
        )}

        {/* Electrical Anomalies */}
        {anomalies.electrical && anomalies.electrical.length > 0 && (
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Electrical Anomalies ({anomalies.electrical.length})
            </Typography>
            <List dense>
              {anomalies.electrical.map((anomaly, index) => (
                <ListItem key={index}>
                  <ListItemText
                    primary={anomaly.feature?.replace('_', ' ') || 'Unknown measurement'}
                    secondary={`Expected: ${anomaly.expected?.toFixed(2) || 'N/A'}, Got: ${anomaly.actual?.toFixed(2) || 'N/A'} (${((anomaly.deviation || 0) * 100).toFixed(1)}% deviation)`}
                  />
                </ListItem>
              ))}
            </List>
          </Box>
        )}

        {(!anomalies.visual || anomalies.visual.length === 0) && 
         (!anomalies.electrical || anomalies.electrical.length === 0) && (
          <Typography color="textSecondary">No anomalies detected</Typography>
        )}
      </Box>
    );
  };

  if (loading) {
    return (
      <Container>
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
          <CircularProgress />
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Test Results
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Card>
        <CardContent>
          {results.length > 0 ? (
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Test ID</TableCell>
                    <TableCell>Date</TableCell>
                    <TableCell>Result</TableCell>
                    <TableCell>Confidence</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {results.map((result) => (
                    <TableRow key={result.id} hover>
                      <TableCell>
                        <Typography variant="body2">
                          #{result.id}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {formatDate(result.test_date)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={result.result_class}
                          size="small"
                          sx={{
                            backgroundColor: RESULT_COLORS[result.result_class],
                            color: 'white',
                            fontWeight: 'bold'
                          }}
                        />
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {(result.fusion_score * 100).toFixed(1)}%
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          <Button
                            size="small"
                            startIcon={<Visibility />}
                            onClick={() => handleViewDetails(result.id)}
                          >
                            Details
                          </Button>
                          <Button
                            size="small"
                            startIcon={<GetApp />}
                            onClick={() => handleDownloadReport(result.id)}
                          >
                            Report
                          </Button>
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          ) : (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <Typography variant="h6" color="textSecondary">
                No test results available
              </Typography>
              <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                Perform component analysis to see results here
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Result Details Dialog */}
      <Dialog 
        open={detailsOpen} 
        onClose={() => setDetailsOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Test Result Details - #{selectedResult?.id}
        </DialogTitle>
        <DialogContent>
          {selectedResult && (
            <Grid container spacing={3}>
              {/* Basic Information */}
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  Test Information
                </Typography>
                <Typography variant="body2" gutterBottom>
                  <strong>Date:</strong> {formatDate(selectedResult.test_date)}
                </Typography>
                <Typography variant="body2" gutterBottom>
                  <strong>Component ID:</strong> {selectedResult.component_id || 'N/A'}
                </Typography>
                <Typography variant="body2" gutterBottom>
                  <strong>User ID:</strong> {selectedResult.user_id || 'N/A'}
                </Typography>
                {selectedResult.batch_id && (
                  <Typography variant="body2" gutterBottom>
                    <strong>Batch ID:</strong> {selectedResult.batch_id}
                  </Typography>
                )}
              </Grid>

              <Divider sx={{ width: '100%', my: 2 }} />

              {/* Analysis Results */}
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  Analysis Results
                </Typography>
                <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                  <Chip
                    label={selectedResult.result_class}
                    sx={{
                      backgroundColor: RESULT_COLORS[selectedResult.result_class],
                      color: 'white',
                      fontWeight: 'bold'
                    }}
                  />
                  <Typography variant="body2" sx={{ alignSelf: 'center' }}>
                    Overall Confidence: {(selectedResult.fusion_score * 100).toFixed(1)}%
                  </Typography>
                </Box>

                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Paper sx={{ p: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Image Analysis
                      </Typography>
                      <Typography variant="body2">
                        Score: {(selectedResult.image_score * 100).toFixed(1)}%
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6}>
                    <Paper sx={{ p: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Electrical Analysis
                      </Typography>
                      <Typography variant="body2">
                        Score: {(selectedResult.electrical_score * 100).toFixed(1)}%
                      </Typography>
                    </Paper>
                  </Grid>
                </Grid>
              </Grid>

              {/* Electrical Measurements */}
              {selectedResult.electrical_measurements && (
                <Grid item xs={12}>
                  <Divider sx={{ my: 2 }} />
                  <Typography variant="h6" gutterBottom>
                    Electrical Measurements
                  </Typography>
                  <Paper sx={{ p: 2, backgroundColor: 'grey.50' }}>
                    <pre style={{ fontSize: '0.875rem', margin: 0, whiteSpace: 'pre-wrap' }}>
                      {JSON.stringify(selectedResult.electrical_measurements, null, 2)}
                    </pre>
                  </Paper>
                </Grid>
              )}

              {/* Anomalies */}
              {selectedResult.anomaly_data && (
                <Grid item xs={12}>
                  <Divider sx={{ my: 2 }} />
                  <Typography variant="h6" gutterBottom>
                    Detected Anomalies
                  </Typography>
                  {renderAnomalies(selectedResult.anomaly_data)}
                </Grid>
              )}

              {/* Notes */}
              {selectedResult.notes && (
                <Grid item xs={12}>
                  <Divider sx={{ my: 2 }} />
                  <Typography variant="h6" gutterBottom>
                    Notes
                  </Typography>
                  <Typography variant="body2">
                    {selectedResult.notes}
                  </Typography>
                </Grid>
              )}
            </Grid>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => handleDownloadReport(selectedResult?.id)}>
            Download Report
          </Button>
          <Button onClick={() => setDetailsOpen(false)}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}

export default Results;