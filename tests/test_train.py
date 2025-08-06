import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import tempfile
import json
from sklearn.ensemble import RandomForestRegressor

# Add the parent directory to the path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.train import run_training, save_model_artifacts, evaluate_model

class TestTrain(unittest.TestCase):
    """
    Test suite for training functions.
    """
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)  # For reproducible tests
        
        # Create synthetic training data
        n_samples = 100
        self.sample_data = pd.DataFrame({
            'car_ID': range(1, n_samples + 1),
            'CarName': ['toyota corolla'] * n_samples,
            'brand': ['toyota'] * n_samples,
            'fueltype': np.random.choice(['gas', 'diesel'], n_samples),
            'aspiration': np.random.choice(['std', 'turbo'], n_samples),
            'doornumber': np.random.choice(['two', 'four'], n_samples),
            'carbody': np.random.choice(['sedan', 'hatchback', 'wagon'], n_samples),
            'drivewheel': np.random.choice(['fwd', 'rwd', '4wd'], n_samples),
            'enginelocation': ['front'] * n_samples,
            'wheelbase': np.random.normal(100, 10, n_samples),
            'carlength': np.random.normal(170, 15, n_samples),
            'carwidth': np.random.normal(65, 5, n_samples),
            'carheight': np.random.normal(55, 5, n_samples),
            'curbweight': np.random.normal(2500, 500, n_samples),
            'enginetype': np.random.choice(['ohc', 'ohcv', 'l'], n_samples),
            'cylindernumber': np.random.choice(['four', 'six', 'eight'], n_samples),
            'enginesize': np.random.normal(120, 30, n_samples),
            'fuelsystem': np.random.choice(['mpfi', '2bbl', 'spdi'], n_samples),
            'boreratio': np.random.normal(3.2, 0.3, n_samples),
            'stroke': np.random.normal(3.3, 0.3, n_samples),
            'compressionratio': np.random.normal(9.5, 1.0, n_samples),
            'horsepower': np.random.normal(100, 30, n_samples),
            'peakrpm': np.random.normal(5500, 500, n_samples),
            'citympg': np.random.normal(25, 5, n_samples),
            'highwaympg': np.random.normal(32, 6, n_samples),
            'price': np.random.normal(15000, 5000, n_samples)
        })
        
        # Ensure positive values for certain columns
        self.sample_data['price'] = np.abs(self.sample_data['price'])
        self.sample_data['horsepower'] = np.abs(self.sample_data['horsepower'])
        self.sample_data['enginesize'] = np.abs(self.sample_data['enginesize'])
    
    def test_evaluate_model(self):
        """Test model evaluation function."""
        # Create a simple model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        
        # Prepare simple test data
        X = np.random.randn(50, 5)
        y = np.random.randn(50) * 1000 + 15000
        
        # Train the model
        model.fit(X, y)
        
        # Evaluate the model
        metrics = evaluate_model(model, X, y)
        
        # Check that all expected metrics are present
        expected_metrics = ['mse', 'rmse', 'mae', 'r2']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float, np.number))
        
        # Check that R² is reasonable (should be close to 1 for training data)
        self.assertGreaterEqual(metrics['r2'], 0.0)
        self.assertLessEqual(metrics['r2'], 1.0)
        
        # Check that error metrics are non-negative
        self.assertGreaterEqual(metrics['mse'], 0)
        self.assertGreaterEqual(metrics['rmse'], 0)
        self.assertGreaterEqual(metrics['mae'], 0)
    
    @patch('app.train.skl2onnx.convert_sklearn')
    @patch('app.train.onnx.save_model')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_model_artifacts(self, mock_file, mock_onnx_save, mock_convert):
        """Test saving model artifacts."""
        # Create a mock model
        model = MagicMock()
        columns = ['feature1', 'feature2', 'feature3']
        
        # Mock ONNX conversion
        mock_onnx_model = MagicMock()
        mock_convert.return_value = mock_onnx_model
        
        # Test saving
        save_model_artifacts(model, columns)
        
        # Verify ONNX conversion was called
        mock_convert.assert_called_once()
        mock_onnx_save.assert_called_once()
        
        # Verify column file was written
        mock_file.assert_called()
        mock_file().write.assert_called()
    
    @patch('app.train.save_model_artifacts')
    @patch('app.train.preprocess_data')
    @patch('app.train.pd.read_csv')
    @patch('app.train.os.path.exists')
    def test_run_training_success(self, mock_exists, mock_read_csv, mock_preprocess, mock_save):
        """Test successful training run."""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock data loading
        mock_read_csv.return_value = self.sample_data
        
        # Mock preprocessing
        processed_data = self.sample_data.copy()
        processed_data = processed_data.drop(['car_ID', 'CarName'], axis=1, errors='ignore')
        mock_preprocess.return_value = processed_data
        
        # Run training
        result = run_training()
        
        # Check that training completed successfully
        self.assertIsInstance(result, dict)
        self.assertIn('model', result)
        self.assertIn('metrics', result)
        self.assertIn('training_time', result)
        
        # Verify that save_model_artifacts was called
        mock_save.assert_called_once()
    
    @patch('app.train.os.path.exists')
    def test_run_training_missing_data(self, mock_exists):
        """Test training with missing data file."""
        mock_exists.return_value = False
        
        with self.assertRaises(FileNotFoundError):
            run_training()
    
    @patch('app.train.save_model_artifacts')
    @patch('app.train.preprocess_data')
    @patch('app.train.pd.read_csv')
    @patch('app.train.os.path.exists')
    def test_run_training_with_parameters(self, mock_exists, mock_read_csv, mock_preprocess, mock_save):
        """Test training with custom parameters."""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock data loading
        mock_read_csv.return_value = self.sample_data
        
        # Mock preprocessing
        processed_data = self.sample_data.copy()
        processed_data = processed_data.drop(['car_ID', 'CarName'], axis=1, errors='ignore')
        mock_preprocess.return_value = processed_data
        
        # Run training with custom parameters
        result = run_training(
            test_size=0.3,
            random_state=123,
            n_estimators=50,
            max_depth=10
        )
        
        # Check that training completed successfully
        self.assertIsInstance(result, dict)
        self.assertIn('model', result)
        self.assertIn('metrics', result)
        
        # Check that the model has the correct parameters
        model = result['model']
        self.assertEqual(model.n_estimators, 50)
        self.assertEqual(model.max_depth, 10)
        self.assertEqual(model.random_state, 123)
    
    @patch('app.train.save_model_artifacts')
    @patch('app.train.preprocess_data')
    @patch('app.train.pd.read_csv')
    @patch('app.train.os.path.exists')
    def test_run_training_validation_metrics(self, mock_exists, mock_read_csv, mock_preprocess, mock_save):
        """Test that training produces reasonable validation metrics."""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock data loading
        mock_read_csv.return_value = self.sample_data
        
        # Mock preprocessing
        processed_data = self.sample_data.copy()
        processed_data = processed_data.drop(['car_ID', 'CarName'], axis=1, errors='ignore')
        mock_preprocess.return_value = processed_data
        
        # Run training
        result = run_training(test_size=0.2, random_state=42)
        
        # Check validation metrics
        metrics = result['metrics']
        
        # Validation metrics should be present
        self.assertIn('train_metrics', metrics)
        self.assertIn('test_metrics', metrics)
        
        # Check that test metrics are reasonable
        test_metrics = metrics['test_metrics']
        self.assertIn('r2', test_metrics)
        self.assertIn('rmse', test_metrics)
        
        # R² should be between 0 and 1 for a reasonable model
        self.assertGreaterEqual(test_metrics['r2'], 0.0)
        self.assertLessEqual(test_metrics['r2'], 1.0)
    
    @patch('app.train.json.dump')
    @patch('builtins.open', new_callable=mock_open)
    @patch('app.train.save_model_artifacts')
    @patch('app.train.preprocess_data')
    @patch('app.train.pd.read_csv')
    @patch('app.train.os.path.exists')
    def test_run_training_saves_metadata(self, mock_exists, mock_read_csv, mock_preprocess, 
                                       mock_save, mock_file, mock_json_dump):
        """Test that training saves metadata."""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock data loading
        mock_read_csv.return_value = self.sample_data
        
        # Mock preprocessing
        processed_data = self.sample_data.copy()
        processed_data = processed_data.drop(['car_ID', 'CarName'], axis=1, errors='ignore')
        mock_preprocess.return_value = processed_data
        
        # Run training
        result = run_training()
        
        # Verify that metadata was saved
        mock_json_dump.assert_called()
        
        # Check the metadata structure
        call_args = mock_json_dump.call_args[0][0]  # First argument to json.dump
        self.assertIn('training_timestamp', call_args)
        self.assertIn('model_parameters', call_args)
        self.assertIn('data_info', call_args)
        self.assertIn('performance_metrics', call_args)
    
    @patch('app.train.logger')
    @patch('app.train.save_model_artifacts')
    @patch('app.train.preprocess_data')
    @patch('app.train.pd.read_csv')
    @patch('app.train.os.path.exists')
    def test_training_logging(self, mock_exists, mock_read_csv, mock_preprocess, 
                            mock_save, mock_logger):
        """Test that training functions log appropriately."""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock data loading
        mock_read_csv.return_value = self.sample_data
        
        # Mock preprocessing
        processed_data = self.sample_data.copy()
        processed_data = processed_data.drop(['car_ID', 'CarName'], axis=1, errors='ignore')
        mock_preprocess.return_value = processed_data
        
        # Run training
        run_training()
        
        # Verify that logging calls were made
        self.assertTrue(mock_logger.info.called)
    
    @patch('app.train.save_model_artifacts')
    @patch('app.train.preprocess_data')
    @patch('app.train.pd.read_csv')
    @patch('app.train.os.path.exists')
    def test_run_training_reproducibility(self, mock_exists, mock_read_csv, mock_preprocess, mock_save):
        """Test that training is reproducible with the same random state."""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock data loading
        mock_read_csv.return_value = self.sample_data
        
        # Mock preprocessing
        processed_data = self.sample_data.copy()
        processed_data = processed_data.drop(['car_ID', 'CarName'], axis=1, errors='ignore')
        mock_preprocess.return_value = processed_data
        
        # Run training twice with the same random state
        result1 = run_training(random_state=42, n_estimators=10)
        result2 = run_training(random_state=42, n_estimators=10)
        
        # Results should be similar (allowing for small numerical differences)
        metrics1 = result1['metrics']['test_metrics']
        metrics2 = result2['metrics']['test_metrics']
        
        # R² scores should be very close
        self.assertAlmostEqual(metrics1['r2'], metrics2['r2'], places=3)
    
    @patch('app.train.save_model_artifacts')
    @patch('app.train.preprocess_data')
    @patch('app.train.pd.read_csv')
    @patch('app.train.os.path.exists')
    def test_run_training_invalid_parameters(self, mock_exists, mock_read_csv, mock_preprocess, mock_save):
        """Test training with invalid parameters."""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock data loading
        mock_read_csv.return_value = self.sample_data
        
        # Mock preprocessing
        processed_data = self.sample_data.copy()
        processed_data = processed_data.drop(['car_ID', 'CarName'], axis=1, errors='ignore')
        mock_preprocess.return_value = processed_data
        
        # Test with invalid test_size
        with self.assertRaises(ValueError):
            run_training(test_size=1.5)  # test_size > 1
        
        with self.assertRaises(ValueError):
            run_training(test_size=-0.1)  # negative test_size
        
        # Test with invalid n_estimators
        with self.assertRaises(ValueError):
            run_training(n_estimators=0)  # zero estimators
    
    def test_evaluate_model_edge_cases(self):
        """Test model evaluation with edge cases."""
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        
        # Test with perfect predictions (R² should be 1)
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([10, 20, 30, 40, 50])
        model.fit(X, y)
        
        # For this simple linear relationship, the model should perform well
        metrics = evaluate_model(model, X, y)
        self.assertGreaterEqual(metrics['r2'], 0.8)  # Should be high for training data
        
        # Test with single sample
        X_single = np.array([[1]])
        y_single = np.array([10])
        model_single = RandomForestRegressor(n_estimators=5, random_state=42)
        model_single.fit(X_single, y_single)
        
        metrics_single = evaluate_model(model_single, X_single, y_single)
        self.assertIn('mse', metrics_single)
        self.assertIn('r2', metrics_single)

class TestTrainingIntegration(unittest.TestCase):
    """
    Integration tests for the training pipeline.
    """
    
    def setUp(self):
        """Set up integration test data."""
        # Create a more realistic dataset
        np.random.seed(42)
        n_samples = 200
        
        self.integration_data = pd.DataFrame({
            'car_ID': range(1, n_samples + 1),
            'CarName': ['toyota corolla'] * (n_samples // 4) + 
                     ['honda civic'] * (n_samples // 4) + 
                     ['bmw 320i'] * (n_samples // 4) + 
                     ['audi a4'] * (n_samples // 4),
            'fueltype': np.random.choice(['gas', 'diesel'], n_samples),
            'aspiration': np.random.choice(['std', 'turbo'], n_samples),
            'doornumber': np.random.choice(['two', 'four'], n_samples),
            'carbody': np.random.choice(['sedan', 'hatchback', 'wagon'], n_samples),
            'drivewheel': np.random.choice(['fwd', 'rwd', '4wd'], n_samples),
            'enginelocation': ['front'] * n_samples,
            'wheelbase': np.random.normal(100, 10, n_samples),
            'carlength': np.random.normal(170, 15, n_samples),
            'carwidth': np.random.normal(65, 5, n_samples),
            'carheight': np.random.normal(55, 5, n_samples),
            'curbweight': np.random.normal(2500, 500, n_samples),
            'enginetype': np.random.choice(['ohc', 'ohcv', 'l'], n_samples),
            'cylindernumber': np.random.choice(['four', 'six', 'eight'], n_samples),
            'enginesize': np.random.normal(120, 30, n_samples),
            'fuelsystem': np.random.choice(['mpfi', '2bbl', 'spdi'], n_samples),
            'boreratio': np.random.normal(3.2, 0.3, n_samples),
            'stroke': np.random.normal(3.3, 0.3, n_samples),
            'compressionratio': np.random.normal(9.5, 1.0, n_samples),
            'horsepower': np.random.normal(100, 30, n_samples),
            'peakrpm': np.random.normal(5500, 500, n_samples),
            'citympg': np.random.normal(25, 5, n_samples),
            'highwaympg': np.random.normal(32, 6, n_samples)
        })
        
        # Create realistic price based on features
        base_price = 10000
        price_factors = (
            self.integration_data['horsepower'] * 50 +
            self.integration_data['enginesize'] * 30 +
            self.integration_data['curbweight'] * 2 +
            np.random.normal(0, 2000, n_samples)
        )
        self.integration_data['price'] = base_price + price_factors
        self.integration_data['price'] = np.abs(self.integration_data['price'])
    
    @patch('app.train.save_model_artifacts')
    @patch('app.train.preprocess_data')
    @patch('app.train.pd.read_csv')
    @patch('app.train.os.path.exists')
    def test_end_to_end_training(self, mock_exists, mock_read_csv, mock_preprocess, mock_save):
        """Test end-to-end training pipeline."""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock data loading
        mock_read_csv.return_value = self.integration_data
        
        # Mock preprocessing (simulate real preprocessing)
        processed_data = self.integration_data.copy()
        processed_data['brand'] = processed_data['CarName'].str.split().str[0]
        processed_data = processed_data.drop(['car_ID', 'CarName'], axis=1, errors='ignore')
        mock_preprocess.return_value = processed_data
        
        # Run training
        result = run_training(test_size=0.2, n_estimators=20, random_state=42)
        
        # Verify training completed successfully
        self.assertIsInstance(result, dict)
        self.assertIn('model', result)
        self.assertIn('metrics', result)
        
        # Check model quality
        test_metrics = result['metrics']['test_metrics']
        self.assertGreater(test_metrics['r2'], 0.3)  # Should achieve reasonable performance
        self.assertLess(test_metrics['rmse'], 10000)  # RMSE should be reasonable
        
        # Verify model artifacts were saved
        mock_save.assert_called_once()

if __name__ == '__main__':
    unittest.main()