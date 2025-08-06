import unittest
import json
from unittest.mock import patch, MagicMock
import sys
import os
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

# Add the parent directory to the path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

class TestAPI(unittest.TestCase):
    """
    Test suite for FastAPI endpoints.
    """
    
    def setUp(self):
        """Set up test client and sample data."""
        self.client = TestClient(app)
        
        # Sample car data for testing
        self.sample_car_data = {
            "CarName": "toyota corolla",
            "fueltype": "gas",
            "aspiration": "std",
            "doornumber": "four",
            "carbody": "sedan",
            "drivewheel": "fwd",
            "enginelocation": "front",
            "wheelbase": 88.6,
            "carlength": 141.1,
            "carwidth": 60.3,
            "carheight": 47.8,
            "curbweight": 1488,
            "enginetype": "ohc",
            "cylindernumber": "four",
            "enginesize": 61,
            "fuelsystem": "2bbl",
            "boreratio": 2.91,
            "stroke": 3.03,
            "compressionratio": 9.0,
            "horsepower": 48,
            "peakrpm": 5000,
            "citympg": 47,
            "highwaympg": 53
        }
        
        self.invalid_car_data = {
            "CarName": "toyota corolla",
            "fueltype": "gas",
            # Missing required fields
        }
    
    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("Car Price Prediction API", data["message"])
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")
        self.assertIn("timestamp", data)
    
    @patch('main.make_prediction')
    def test_predict_endpoint_success(self, mock_predict):
        """Test successful prediction endpoint."""
        # Mock prediction function
        mock_predict.return_value = [12500.0]
        
        response = self.client.post("/predict/", json=self.sample_car_data)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check response structure
        self.assertIn("predicted_price", data)
        self.assertIn("confidence", data)
        self.assertIn("model_version", data)
        self.assertIn("prediction_timestamp", data)
        
        # Check prediction value
        self.assertEqual(data["predicted_price"], 12500.0)
        self.assertIsInstance(data["confidence"], str)
        
        # Verify mock was called
        mock_predict.assert_called_once()
    
    @patch('main.make_prediction')
    def test_predict_endpoint_multiple_predictions(self, mock_predict):
        """Test prediction endpoint with multiple predictions."""
        # Mock prediction function returning multiple values
        mock_predict.return_value = [12500.0, 15000.0]
        
        response = self.client.post("/predict/", json=self.sample_car_data)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Should return the first prediction
        self.assertEqual(data["predicted_price"], 12500.0)
    
    def test_predict_endpoint_invalid_data(self):
        """Test prediction endpoint with invalid data."""
        response = self.client.post("/predict/", json=self.invalid_car_data)
        
        # Should return validation error
        self.assertEqual(response.status_code, 422)
    
    def test_predict_endpoint_missing_fields(self):
        """Test prediction endpoint with missing required fields."""
        incomplete_data = {"CarName": "toyota corolla"}
        
        response = self.client.post("/predict/", json=incomplete_data)
        
        # Should return validation error
        self.assertEqual(response.status_code, 422)
    
    def test_predict_endpoint_invalid_types(self):
        """Test prediction endpoint with invalid data types."""
        invalid_type_data = self.sample_car_data.copy()
        invalid_type_data["wheelbase"] = "not_a_number"
        
        response = self.client.post("/predict/", json=invalid_type_data)
        
        # Should return validation error
        self.assertEqual(response.status_code, 422)
    
    @patch('main.make_prediction')
    def test_predict_endpoint_prediction_error(self, mock_predict):
        """Test prediction endpoint when prediction fails."""
        # Mock prediction function to raise an error
        mock_predict.side_effect = Exception("Model prediction failed")
        
        response = self.client.post("/predict/", json=self.sample_car_data)
        
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("detail", data)
    
    @patch('main.make_prediction')
    def test_predict_endpoint_negative_prediction(self, mock_predict):
        """Test prediction endpoint with negative prediction."""
        # Mock prediction function returning negative value
        mock_predict.return_value = [-1000.0]
        
        response = self.client.post("/predict/", json=self.sample_car_data)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Should handle negative predictions gracefully
        self.assertIn("predicted_price", data)
        self.assertIn("confidence", data)
        self.assertEqual(data["confidence"], "low")
    
    def test_model_info_endpoint(self):
        """Test the model info endpoint."""
        response = self.client.get("/model/info")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check response structure
        expected_fields = [
            "model_name", "model_version", "model_type", 
            "features_count", "training_date", "performance_metrics"
        ]
        
        for field in expected_fields:
            self.assertIn(field, data)
        
        # Check specific values
        self.assertEqual(data["model_name"], "Car Price Prediction Model")
        self.assertEqual(data["model_type"], "Random Forest Regressor")
        self.assertIsInstance(data["features_count"], int)
    
    def test_predict_endpoint_edge_values(self):
        """Test prediction endpoint with edge values."""
        edge_data = self.sample_car_data.copy()
        
        # Test with extreme values
        edge_data.update({
            "wheelbase": 200.0,  # Very high
            "horsepower": 500,   # Very high
            "citympg": 5,       # Very low
            "highwaympg": 60    # Very high
        })
        
        with patch('main.make_prediction') as mock_predict:
            mock_predict.return_value = [25000.0]
            
            response = self.client.post("/predict/", json=edge_data)
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("predicted_price", data)
    
    def test_predict_endpoint_boundary_values(self):
        """Test prediction endpoint with boundary values."""
        boundary_data = self.sample_car_data.copy()
        
        # Test with zero values where applicable
        boundary_data.update({
            "wheelbase": 0.1,    # Near zero
            "horsepower": 1,     # Minimum reasonable value
            "enginesize": 1      # Minimum reasonable value
        })
        
        with patch('main.make_prediction') as mock_predict:
            mock_predict.return_value = [5000.0]
            
            response = self.client.post("/predict/", json=boundary_data)
            
            self.assertEqual(response.status_code, 200)
    
    def test_api_response_headers(self):
        """Test that API responses have correct headers."""
        response = self.client.get("/")
        
        # Check content type
        self.assertEqual(response.headers["content-type"], "application/json")
    
    def test_api_cors_headers(self):
        """Test CORS headers if implemented."""
        response = self.client.options("/predict/")
        
        # This test depends on CORS middleware being configured
        # Adjust based on your CORS configuration
        self.assertIn(response.status_code, [200, 405])  # 405 if OPTIONS not explicitly handled
    
    @patch('main.make_prediction')
    def test_predict_endpoint_performance(self, mock_predict):
        """Test prediction endpoint performance."""
        mock_predict.return_value = [12500.0]
        
        import time
        
        start_time = time.time()
        response = self.client.post("/predict/", json=self.sample_car_data)
        end_time = time.time()
        
        # Response should be fast (adjust threshold as needed)
        self.assertLess(end_time - start_time, 2.0)  # 2 seconds
        self.assertEqual(response.status_code, 200)
    
    def test_api_documentation_endpoints(self):
        """Test that API documentation endpoints are accessible."""
        # Test OpenAPI schema
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        
        # Test Swagger UI (if enabled)
        response = self.client.get("/docs")
        self.assertEqual(response.status_code, 200)
        
        # Test ReDoc (if enabled)
        response = self.client.get("/redoc")
        self.assertEqual(response.status_code, 200)

class TestAPIIntegration(unittest.TestCase):
    """
    Integration tests for the API with real-like scenarios.
    """
    
    def setUp(self):
        """Set up integration test client."""
        self.client = TestClient(app)
        
        # Multiple car samples for batch testing
        self.car_samples = [
            {
                "CarName": "toyota corolla",
                "fueltype": "gas",
                "aspiration": "std",
                "doornumber": "four",
                "carbody": "sedan",
                "drivewheel": "fwd",
                "enginelocation": "front",
                "wheelbase": 88.6,
                "carlength": 141.1,
                "carwidth": 60.3,
                "carheight": 47.8,
                "curbweight": 1488,
                "enginetype": "ohc",
                "cylindernumber": "four",
                "enginesize": 61,
                "fuelsystem": "2bbl",
                "boreratio": 2.91,
                "stroke": 3.03,
                "compressionratio": 9.0,
                "horsepower": 48,
                "peakrpm": 5000,
                "citympg": 47,
                "highwaympg": 53
            },
            {
                "CarName": "bmw 320i",
                "fueltype": "gas",
                "aspiration": "turbo",
                "doornumber": "four",
                "carbody": "sedan",
                "drivewheel": "rwd",
                "enginelocation": "front",
                "wheelbase": 101.2,
                "carlength": 176.2,
                "carwidth": 66.2,
                "carheight": 54.2,
                "curbweight": 2734,
                "enginetype": "ohc",
                "cylindernumber": "six",
                "enginesize": 164,
                "fuelsystem": "mpfi",
                "boreratio": 3.31,
                "stroke": 3.40,
                "compressionratio": 8.4,
                "horsepower": 121,
                "peakrpm": 4250,
                "citympg": 21,
                "highwaympg": 28
            }
        ]
    
    @patch('main.make_prediction')
    def test_multiple_predictions_consistency(self, mock_predict):
        """Test that multiple predictions are consistent."""
        # Mock different predictions for different cars
        mock_predict.side_effect = [[8500.0], [18500.0]]
        
        predictions = []
        for car_data in self.car_samples:
            response = self.client.post("/predict/", json=car_data)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            predictions.append(data["predicted_price"])
        
        # BMW should be more expensive than Toyota
        self.assertGreater(predictions[1], predictions[0])
    
    @patch('main.make_prediction')
    def test_api_workflow(self, mock_predict):
        """Test complete API workflow."""
        mock_predict.return_value = [12500.0]
        
        # 1. Check API health
        health_response = self.client.get("/health")
        self.assertEqual(health_response.status_code, 200)
        
        # 2. Get model info
        info_response = self.client.get("/model/info")
        self.assertEqual(info_response.status_code, 200)
        
        # 3. Make prediction
        predict_response = self.client.post("/predict/", json=self.car_samples[0])
        self.assertEqual(predict_response.status_code, 200)
        
        # 4. Verify prediction response
        data = predict_response.json()
        self.assertIn("predicted_price", data)
        self.assertGreater(data["predicted_price"], 0)
    
    def test_api_error_handling(self):
        """Test API error handling with various invalid inputs."""
        invalid_inputs = [
            {},  # Empty object
            {"invalid_field": "value"},  # Wrong fields
            {"CarName": None},  # Null values
            {"CarName": ""},  # Empty strings
        ]
        
        for invalid_input in invalid_inputs:
            with self.subTest(input=invalid_input):
                response = self.client.post("/predict/", json=invalid_input)
                self.assertEqual(response.status_code, 422)
    
    @patch('main.make_prediction')
    def test_concurrent_requests(self, mock_predict):
        """Test handling of concurrent requests."""
        mock_predict.return_value = [12500.0]
        
        import concurrent.futures
        import threading
        
        def make_request():
            return self.client.post("/predict/", json=self.car_samples[0])
        
        # Make multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for response in responses:
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("predicted_price", data)

if __name__ == '__main__':
    unittest.main()