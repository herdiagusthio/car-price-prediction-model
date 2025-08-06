# Car Price Prediction API Documentation

This document provides comprehensive API documentation with curl examples for the Car Price Prediction API.

## Base URL

```
http://localhost:8000
```

## Endpoints Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Detailed health check |
| `/predict/` | POST | Predict car price |

---

## 1. Health Check

**Endpoint:** `GET /health`

**Description:** Detailed health check that verifies the API is running and the model is loaded correctly.

### Curl Example

```bash
curl -X GET "http://localhost:8000/health" \
  -H "accept: application/json"
```

### Response Example

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_file": "models/best_model.onnx",
  "feature_count": 74,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## 2. Car Price Prediction

**Endpoint:** `POST /predict/`

**Description:** Predict the price of a car based on its features.

### Request Body

The request requires a JSON object with all car features:

```json
{
  "symboling": 0,
  "CarName": "toyota corolla",
  "fueltype": "gas",
  "aspiration": "std",
  "doornumber": "four",
  "carbody": "sedan",
  "drivewheel": "fwd",
  "enginelocation": "front",
  "wheelbase": 100.4,
  "carlength": 176.6,
  "carwidth": 66.2,
  "carheight": 54.3,
  "curbweight": 2337,
  "enginetype": "ohc",
  "cylindernumber": "four",
  "enginesize": 109,
  "fuelsystem": "mpfi",
  "boreratio": 3.19,
  "stroke": 3.4,
  "compressionratio": 10.0,
  "horsepower": 102,
  "peakrpm": 5500,
  "citympg": 24,
  "highwaympg": 30
}
```

### Curl Example

```bash
curl -X POST "http://localhost:8000/predict/" \
  -H "Content-Type: application/json" \
  -H "accept: application/json" \
  -d '{
    "symboling": 0,
    "CarName": "toyota corolla",
    "fueltype": "gas",
    "aspiration": "std",
    "doornumber": "four",
    "carbody": "sedan",
    "drivewheel": "fwd",
    "enginelocation": "front",
    "wheelbase": 100.4,
    "carlength": 176.6,
    "carwidth": 66.2,
    "carheight": 54.3,
    "curbweight": 2337,
    "enginetype": "ohc",
    "cylindernumber": "four",
    "enginesize": 109,
    "fuelsystem": "mpfi",
    "boreratio": 3.19,
    "stroke": 3.4,
    "compressionratio": 10.0,
    "horsepower": 102,
    "peakrpm": 5500,
    "citympg": 24,
    "highwaympg": 30
  }'
```

### Response Example

```json
{
  "predicted_price": 12500.75,
  "model_confidence": 0.92,
  "input_features": {
    "brand": "toyota",
    "processed_features_count": 74
  },
  "prediction_id": "pred_20240115_103045_abc123",
  "timestamp": "2024-01-15T10:30:45Z"
}
```

---

## Sample Test Data

Here are some additional sample car configurations you can use for testing:

### Luxury Car Example

```bash
curl -X POST "http://localhost:8000/predict/" \
  -H "Content-Type: application/json" \
  -d '{
    "symboling": -1,
    "CarName": "bmw 320i",
    "fueltype": "gas",
    "aspiration": "std",
    "doornumber": "four",
    "carbody": "sedan",
    "drivewheel": "rwd",
    "enginelocation": "front",
    "wheelbase": 101.2,
    "carlength": 176.8,
    "carwidth": 64.8,
    "carheight": 54.3,
    "curbweight": 2395,
    "enginetype": "ohc",
    "cylindernumber": "four",
    "enginesize": 108,
    "fuelsystem": "mpfi",
    "boreratio": 3.5,
    "stroke": 2.8,
    "compressionratio": 8.8,
    "horsepower": 101,
    "peakrpm": 5800,
    "citympg": 23,
    "highwaympg": 29
  }'
```

### Economy Car Example

```bash
curl -X POST "http://localhost:8000/predict/" \
  -H "Content-Type: application/json" \
  -d '{
    "symboling": 1,
    "CarName": "honda civic",
    "fueltype": "gas",
    "aspiration": "std",
    "doornumber": "four",
    "carbody": "hatchback",
    "drivewheel": "fwd",
    "enginelocation": "front",
    "wheelbase": 93.7,
    "carlength": 150.0,
    "carwidth": 63.3,
    "carheight": 50.8,
    "curbweight": 1713,
    "enginetype": "ohc",
    "cylindernumber": "four",
    "enginesize": 79,
    "fuelsystem": "2bbl",
    "boreratio": 2.91,
    "stroke": 3.07,
    "compressionratio": 9.6,
    "horsepower": 58,
    "peakrpm": 4800,
    "citympg": 49,
    "highwaympg": 54
  }'
```

---

## Error Handling

### Common Error Responses

#### 422 Validation Error

```json
{
  "detail": [
    {
      "loc": ["body", "horsepower"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

#### 500 Internal Server Error

```json
{
  "detail": "Internal server error during prediction"
}
```

### Error Testing

```bash
# Test with missing required field
curl -X POST "http://localhost:8000/predict/" \
  -H "Content-Type: application/json" \
  -d '{
    "symboling": 0,
    "CarName": "toyota corolla"
  }'

# Test with invalid data type
curl -X POST "http://localhost:8000/predict/" \
  -H "Content-Type: application/json" \
  -d '{
    "symboling": "invalid",
    "CarName": "toyota corolla",
    "horsepower": "not_a_number"
  }'
```

---

## Interactive Documentation

For a more interactive experience, you can access the auto-generated API documentation:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

These interfaces allow you to:
- Test endpoints directly from the browser
- View detailed request/response schemas
- Download OpenAPI specifications

---

## Troubleshooting

### Common Issues

1. **Connection Refused**
   ```bash
   curl: (7) Failed to connect to localhost port 8000: Connection refused
   ```
   **Solution**: Make sure the API server is running with `python3 main.py`

2. **Model Not Found**
   ```json
   {"detail": "Model file not found"}
   ```
   **Solution**: Ensure model files exist in the `models/` directory

3. **Invalid JSON**
   ```json
   {"detail": "Invalid JSON in request body"}
   ```
   **Solution**: Validate your JSON syntax before sending the request

### Health Check First

Always start by testing the health endpoint:

```bash
curl http://localhost:8000/health
```

This will confirm the API is running and the model is loaded correctly before attempting predictions.