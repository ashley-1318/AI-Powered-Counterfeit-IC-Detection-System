import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_URL,
  timeout: 30000, // 30 second timeout for analysis operations
});

// Add request interceptor to include auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

const analysisService = {
  analyzeComponent: async (imageFile, electricalData = null, componentId = null, partNumber = 'Unknown') => {
    try {
      const formData = new FormData();
      formData.append('image', imageFile);
      // Electrical data is no longer needed, but kept parameter for backward compatibility
      formData.append('part_number', partNumber);
      
      if (componentId) {
        formData.append('component_id', componentId);
      }

      const response = await apiClient.post('/analysis/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  uploadImage: async (imageFile) => {
    try {
      const formData = new FormData();
      formData.append('file', imageFile);

      const response = await apiClient.post('/upload/image', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  getTestResult: async (testId) => {
    try {
      const response = await apiClient.get(`/results/test/${testId}`);
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  getRecentResults: async (limit = 10, userId = null) => {
    try {
      const params = { limit };
      if (userId) {
        params.user_id = userId;
      }

      const response = await apiClient.get('/results/recent', { params });
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  downloadReport: async (testId) => {
    try {
      const response = await apiClient.get(`/results/report/${testId}`, {
        responseType: 'blob',
      });
      
      // Create blob link to download
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `CircuitCheck_Report_${testId}.pdf`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      return true;
    } catch (error) {
      throw error.response?.data || error;
    }
  },
};

export default analysisService;