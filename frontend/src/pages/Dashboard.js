import React, { useState, useEffect, useContext } from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  List,
  ListItem,
  ListItemText,
  CircularProgress,
  Alert
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { AuthContext } from '../context/AuthContext';
import analysisService from '../services/analysisService';

const RESULT_COLORS = {
  PASS: '#4caf50',
  SUSPECT: '#ff9800',
  FAIL: '#f44336'
};

function Dashboard() {
  const [recentResults, setRecentResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const { user } = useContext(AuthContext);

  useEffect(() => {
    loadRecentResults();
  }, [user]);

  const loadRecentResults = async () => {
    try {
      const results = await analysisService.getRecentResults(20, user?.user_id);
      setRecentResults(results);
    } catch (error) {
      console.error('Error loading recent results:', error);
      setError('Failed to load recent results');
    } finally {
      setLoading(false);
    }
  };

  const getResultStats = () => {
    const stats = recentResults.reduce(
      (acc, result) => {
        acc[result.result_class] = (acc[result.result_class] || 0) + 1;
        acc.total++;
        return acc;
      },
      { PASS: 0, SUSPECT: 0, FAIL: 0, total: 0 }
    );

    return [
      { name: 'Pass', value: stats.PASS, color: RESULT_COLORS.PASS },
      { name: 'Suspect', value: stats.SUSPECT, color: RESULT_COLORS.SUSPECT },
      { name: 'Fail', value: stats.FAIL, color: RESULT_COLORS.FAIL }
    ];
  };

  const getRecentTestsData = () => {
    // Group results by date for the last 7 days
    const today = new Date();
    const last7Days = [];
    
    for (let i = 6; i >= 0; i--) {
      const date = new Date(today);
      date.setDate(date.getDate() - i);
      const dateStr = date.toISOString().split('T')[0];
      
      const dayResults = recentResults.filter(result => 
        result.test_date.startsWith(dateStr)
      );
      
      const dayStats = dayResults.reduce(
        (acc, result) => {
          acc[result.result_class] = (acc[result.result_class] || 0) + 1;
          return acc;
        },
        { PASS: 0, SUSPECT: 0, FAIL: 0 }
      );
      
      last7Days.push({
        date: date.toLocaleDateString('en-US', { weekday: 'short' }),
        ...dayStats
      });
    }
    
    return last7Days;
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
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

  const resultStats = getResultStats();
  const recentTestsData = getRecentTestsData();

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Summary Cards */}
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Tests
              </Typography>
              <Typography variant="h5">
                {recentResults.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Pass Rate
              </Typography>
              <Typography variant="h5" style={{ color: RESULT_COLORS.PASS }}>
                {recentResults.length > 0 
                  ? Math.round((resultStats[0].value / recentResults.length) * 100)
                  : 0
                }%
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Suspect Components
              </Typography>
              <Typography variant="h5" style={{ color: RESULT_COLORS.SUSPECT }}>
                {resultStats[1].value}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Failed Components
              </Typography>
              <Typography variant="h5" style={{ color: RESULT_COLORS.FAIL }}>
                {resultStats[2].value}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Result Distribution Chart */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Result Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={resultStats}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, value, percent }) => 
                      `${name}: ${value} (${(percent * 100).toFixed(0)}%)`
                    }
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {resultStats.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Tests Trend */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Tests Last 7 Days
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={recentTestsData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="PASS" stackId="a" fill={RESULT_COLORS.PASS} />
                  <Bar dataKey="SUSPECT" stackId="a" fill={RESULT_COLORS.SUSPECT} />
                  <Bar dataKey="FAIL" stackId="a" fill={RESULT_COLORS.FAIL} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Results List */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Test Results
              </Typography>
              {recentResults.length > 0 ? (
                <List>
                  {recentResults.slice(0, 10).map((result) => (
                    <ListItem key={result.id} divider>
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography>Test #{result.id}</Typography>
                            <Chip
                              label={result.result_class}
                              size="small"
                              sx={{
                                backgroundColor: RESULT_COLORS[result.result_class],
                                color: 'white',
                                fontWeight: 'bold'
                              }}
                            />
                          </Box>
                        }
                        secondary={
                          <Box>
                            <Typography variant="body2" color="textSecondary">
                              {formatDate(result.test_date)}
                            </Typography>
                            <Typography variant="body2" color="textSecondary">
                              Confidence: {(result.fusion_score * 100).toFixed(1)}%
                            </Typography>
                          </Box>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              ) : (
                <Typography color="textSecondary" align="center">
                  No recent test results available
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
}

export default Dashboard;