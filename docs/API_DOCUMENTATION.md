# CircuitCheck API Documentation

## Overview

CircuitCheck provides a RESTful API for integrating counterfeit component detection into your applications and workflows.

**Base URL**: `http://localhost:5000/api`

**Authentication**: Bearer token (JWT)

## Authentication

### Register User

**Endpoint**: `POST /auth/register`

**Request Body**:

```json
{
  "username": "string",
  "email": "string",
  "password": "string"
}
```

**Response** (200):

```json
{
  "message": "User registered successfully",
  "user": {
    "id": 1,
    "username": "john_doe",
    "email": "john@example.com"
  }
}
```

**Error Responses**:

- `400`: Missing required fields
- `409`: Username or email already exists

### Login

**Endpoint**: `POST /auth/login`

**Request Body**:

```json
{
  "username": "string",
  "password": "string"
}
```

**Response** (200):

```json
{
  "message": "Login successful",
  "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "user": {
    "id": 1,
    "username": "john_doe",
    "email": "john@example.com"
  }
}
```

**Error Responses**:

- `401`: Invalid credentials
- `400`: Missing username or password

### Protected Routes

For all protected endpoints, include the JWT token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

## Image Upload

### Upload Component Image

**Endpoint**: `POST /upload/image`

**Content-Type**: `multipart/form-data`

**Request**:

```
FormData:
- image: File (JPG, PNG, max 16MB)
```

**Response** (200):

```json
{
  "message": "Image uploaded successfully",
  "filename": "component_12345.jpg",
  "path": "/uploads/component_12345.jpg"
}
```

**Error Responses**:

- `400`: No image provided or invalid format
- `413`: File too large

## Component Analysis

### Analyze Component

**Endpoint**: `POST /analysis/analyze`

**Headers**:

```
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body**:

```json
{
  "image_path": "string",
  "part_number": "string",
  "electrical_data": {
    "resistance": {
      "pin1_pin2": 1000.0,
      "pin1_gnd": 5000000.0
    },
    "capacitance": {
      "pin1_gnd": 15.2,
      "pin2_gnd": 12.8
    },
    "leakage_current": {
      "pin1": 0.001,
      "pin2": 0.002
    },
    "timing": {
      "rise_time": 2.5,
      "fall_time": 1.8,
      "propagation_delay": 4.2
    }
  }
}
```

**Response** (200):

```json
{
  "result": {
    "classification": "PASS|SUSPECT|FAIL",
    "confidence": 0.95,
    "analysis": {
      "image_analysis": {
        "confidence": 0.92,
        "anomalies": [
          {
            "type": "marking_inconsistency",
            "location": [100, 150],
            "severity": "medium",
            "description": "Font style differs from expected"
          }
        ]
      },
      "electrical_analysis": {
        "confidence": 0.98,
        "anomalies": [
          {
            "measurement": "resistance_pin1_pin2",
            "expected": 1000.0,
            "actual": 1050.0,
            "deviation": 5.0,
            "severity": "low"
          }
        ]
      },
      "fusion_analysis": {
        "weights": {
          "image": 0.6,
          "electrical": 0.4
        },
        "explanation": "Component passes electrical tests but shows minor visual anomalies"
      }
    },
    "test_id": "test_12345",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

**Error Responses**:

- `400`: Invalid input data
- `404`: Image not found
- `500`: Analysis failed

## Test Results

### Get User Results

**Endpoint**: `GET /results/user`

**Headers**:

```
Authorization: Bearer <token>
```

**Query Parameters**:

- `page`: Page number (default: 1)
- `per_page`: Results per page (default: 20, max: 100)
- `classification`: Filter by result (PASS, SUSPECT, FAIL)
- `start_date`: Filter from date (YYYY-MM-DD)
- `end_date`: Filter to date (YYYY-MM-DD)

**Response** (200):

```json
{
  "results": [
    {
      "id": 1,
      "test_id": "test_12345",
      "part_number": "MC74HC00AN",
      "classification": "PASS",
      "confidence": 0.95,
      "created_at": "2024-01-15T10:30:00Z",
      "image_path": "/uploads/component_12345.jpg"
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 150,
    "pages": 8
  }
}
```

### Get Specific Test Result

**Endpoint**: `GET /results/<test_id>`

**Headers**:

```
Authorization: Bearer <token>
```

**Response** (200):

```json
{
  "result": {
    "id": 1,
    "test_id": "test_12345",
    "part_number": "MC74HC00AN",
    "classification": "PASS",
    "confidence": 0.95,
    "created_at": "2024-01-15T10:30:00Z",
    "image_path": "/uploads/component_12345.jpg",
    "analysis_details": {
      "image_analysis": {
        "confidence": 0.92,
        "anomalies": []
      },
      "electrical_analysis": {
        "confidence": 0.98,
        "measurements": {
          "resistance": {
            "pin1_pin2": 1000.0
          }
        }
      },
      "explanation": "Component passes all checks"
    }
  }
}
```

**Error Responses**:

- `404`: Test result not found
- `403`: Access denied

## Camera Control

### Capture Image

**Endpoint**: `POST /camera/capture`

**Headers**:

```
Authorization: Bearer <token>
```

**Response** (200):

```json
{
  "message": "Image captured successfully",
  "filename": "capture_12345.jpg",
  "path": "/uploads/capture_12345.jpg"
}
```

**Error Responses**:

- `500`: Camera not available
- `503`: Camera capture failed

### Get Camera Status

**Endpoint**: `GET /camera/status`

**Response** (200):

```json
{
  "available": true,
  "resolution": "1920x1080",
  "frame_rate": 30
}
```

## Hardware Integration

### Get Hardware Status

**Endpoint**: `GET /hardware/status`

**Headers**:

```
Authorization: Bearer <token>
```

**Response** (200):

```json
{
  "test_jig": {
    "connected": true,
    "port": "COM3",
    "version": "1.0.0"
  },
  "camera": {
    "connected": true,
    "model": "USB Camera"
  }
}
```

### Run Electrical Test

**Endpoint**: `POST /hardware/test`

**Headers**:

```
Authorization: Bearer <token>
```

**Request Body**:

```json
{
  "test_type": "resistance|capacitance|leakage|timing",
  "pins": [1, 2, 3, 4],
  "parameters": {
    "voltage": 3.3,
    "frequency": 1000
  }
}
```

**Response** (200):

```json
{
  "measurements": {
    "pin1_pin2": 1000.0,
    "pin1_gnd": 5000000.0
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Error Handling

### Standard Error Response Format

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": "Additional details if applicable"
  }
}
```

### Common Error Codes

- `INVALID_TOKEN`: JWT token is invalid or expired
- `MISSING_AUTHENTICATION`: No authentication token provided
- `INVALID_REQUEST`: Request data is malformed
- `RESOURCE_NOT_FOUND`: Requested resource doesn't exist
- `PERMISSION_DENIED`: User doesn't have required permissions
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `INTERNAL_ERROR`: Server-side error occurred

## Rate Limiting

API endpoints have rate limiting:

- **Authentication**: 5 requests per minute
- **Analysis**: 10 requests per minute
- **Results**: 60 requests per minute
- **Other endpoints**: 30 requests per minute

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 9
X-RateLimit-Reset: 1642248000
```

## Webhooks

### Configure Webhook

**Endpoint**: `POST /webhooks/configure`

**Headers**:

```
Authorization: Bearer <token>
```

**Request Body**:

```json
{
  "url": "https://your-app.com/webhook",
  "events": ["analysis_complete", "test_failed"],
  "secret": "your-webhook-secret"
}
```

### Webhook Events

**analysis_complete**:

```json
{
  "event": "analysis_complete",
  "test_id": "test_12345",
  "classification": "PASS",
  "confidence": 0.95,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## SDK Examples

### Python

```python
import requests

class CircuitCheckAPI:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}

    def analyze_component(self, image_path, part_number, electrical_data=None):
        data = {
            "image_path": image_path,
            "part_number": part_number,
            "electrical_data": electrical_data
        }
        response = requests.post(
            f"{self.base_url}/analysis/analyze",
            json=data,
            headers=self.headers
        )
        return response.json()

# Usage
api = CircuitCheckAPI("http://localhost:5000/api", "your-token")
result = api.analyze_component("/uploads/component.jpg", "MC74HC00AN")
```

### JavaScript

```javascript
class CircuitCheckAPI {
  constructor(baseURL, token) {
    this.baseURL = baseURL;
    this.headers = {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    };
  }

  async analyzeComponent(imagePath, partNumber, electricalData = null) {
    const response = await fetch(`${this.baseURL}/analysis/analyze`, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify({
        image_path: imagePath,
        part_number: partNumber,
        electrical_data: electricalData,
      }),
    });
    return response.json();
  }
}

// Usage
const api = new CircuitCheckAPI("http://localhost:5000/api", "your-token");
const result = await api.analyzeComponent(
  "/uploads/component.jpg",
  "MC74HC00AN"
);
```

## Testing

### Test Environment

Use the test environment for development:

- **Base URL**: `http://localhost:5000/api`
- **Test credentials**: username `testuser`, password `testpass123`

### Sample Test Data

Example electrical measurements for testing:

```json
{
  "resistance": {
    "pin1_pin2": 1000.0,
    "pin1_gnd": 5000000.0,
    "pin2_gnd": 5000000.0
  },
  "capacitance": {
    "pin1_gnd": 15.2,
    "pin2_gnd": 12.8
  },
  "leakage_current": {
    "pin1": 0.001,
    "pin2": 0.002
  }
}
```

## Support

For API support:

- Documentation: [api.circuitcheck.com/docs](https://api.circuitcheck.com/docs)
- Email: api-support@circuitcheck.com
- Response time: 24-48 hours
- Status page: [status.circuitcheck.com](https://status.circuitcheck.com)
