import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import tempfile
import json

# Add the parent directory to the path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.predict import make_prediction, load_model_artifacts

class TestPredict(unittest.TestCase):
    """
    Test suite for prediction functions.
    """
    
    def setUp(self):
        """Set up test data and mocks."""
        self.sample_features = pd.DataFrame({
            'brand': ['toyota', 'honda', 'bmw'],
            'fueltype': ['gas', 'gas', 'gas'],
            'aspiration': ['std', 'std', 'turbo'],
            'doornumber': ['four', 'four', 'four'],
            'carbody': ['sedan', 'sedan', 'sedan'],
            'drivewheel': ['fwd', 'fwd', 'rwd'],
            'enginelocation': ['front', 'front', 'front'],
            'wheelbase': [88.6, 93.7, 101.2],
            'carlength': [141.1, 158.8, 176.2],
            'carwidth': [60.3, 63.9, 66.2],
            'carheight': [47.8, 63.4, 54.2],
            'curbweight': [1488, 2017, 2734],
            'enginetype': ['ohc', 'ohc', 'ohc'],
            'cylindernumber': ['four', 'four', 'six'],
            'enginesize': [61, 92, 164],
            'fuelsystem': ['2bbl', 'mpfi', 'mpfi'],
            'boreratio': [2.91, 2.97, 3.31],
            'stroke': [3.03, 3.23, 3.40],
            'compressionratio': [9.0, 9.4, 8.4],
            'horsepower': [48, 76, 121],
            'peakrpm': [5000, 6000, 4250],
            'citympg': [47, 31, 21],
            'highwaympg': [53, 38, 28]
        })
        
        self.expected_columns = [
            'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
            'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',
            'peakrpm', 'citympg', 'highwaympg', 'brand_toyota', 'brand_honda',
            'brand_bmw', 'fueltype_gas', 'aspiration_std', 'aspiration_turbo'
        ]
    
    @patch('app.predict.ort')
    @patch('app.predict.os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_model_artifacts_success(self, mock_file, mock_exists, mock_ort):
        """Test successful loading of model artifacts."""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock column file content
        mock_file.return_value.read.return_value = 'col1\ncol2\ncol3\n'
        
        # Mock ONNX session
        mock_session = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session
        
        session, columns = load_model_artifacts()
        
        self.assertEqual(session, mock_session)
        self.assertEqual(columns, ['col1', 'col2', 'col3'])
        mock_ort.InferenceSession.assert_called_once()
    
    @patch('app.predict.os.path.exists')
    def test_load_model_artifacts_missing_files(self, mock_exists):
        """Test loading model artifacts when files are missing."""
        mock_exists.return_value = False
        
        with self.assertRaises(FileNotFoundError):
            load_model_artifacts()
    
    @patch('app.predict.ort')
    @patch('app.predict.load_model_artifacts')
    @patch('app.predict.preprocess_data')
    def test_make_prediction_success(self, mock_preprocess, mock_load, mock_ort):
        """Test successful prediction making."""
        # Mock model artifacts
        mock_session = MagicMock()
        mock_columns = ['feature1', 'feature2', 'feature3']
        mock_load.return_value = (mock_session, mock_columns)
        
        # Mock preprocessing
        processed_data = pd.DataFrame({
            'feature1': [1.0, 2.0],
            'feature2': [3.0, 4.0],
            'feature3': [5.0, 6.0]
        })
        mock_preprocess.return_value = processed_data
        
        # Mock ONNX prediction
        mock_session.run.return_value = [np.array([[10000.0], [15000.0]])]
        
        # Test prediction
        input_data = pd.DataFrame({'some': ['data']})
        predictions = make_prediction(input_data)
        
        # Verify results
        expected_predictions = [10000.0, 15000.0]
        np.testing.assert_array_almost_equal(predictions, expected_predictions)
        
        # Verify function calls
        mock_preprocess.assert_called_once_with(input_data)
        mock_session.run.assert_called_once()
    
    @patch('app.predict.load_model_artifacts')
    @patch('app.predict.preprocess_data')
    def test_make_prediction_column_mismatch(self, mock_preprocess, mock_load):
        """Test prediction with column mismatch."""
        # Mock model artifacts
        mock_session = MagicMock()
        mock_columns = ['feature1', 'feature2', 'feature3']
        mock_load.return_value = (mock_session, mock_columns)
        
        # Mock preprocessing with different columns
        processed_data = pd.DataFrame({
            'feature1': [1.0, 2.0],
            'feature4': [3.0, 4.0],  # Different column
            'feature5': [5.0, 6.0]   # Different column
        })
        mock_preprocess.return_value = processed_data
        
        input_data = pd.DataFrame({'some': ['data']})
        
        # Should handle column mismatch gracefully
        with self.assertRaises(Exception):
            make_prediction(input_data)
    
    @patch('app.predict.load_model_artifacts')
    @patch('app.predict.preprocess_data')
    def test_make_prediction_empty_data(self, mock_preprocess, mock_load):
        """Test prediction with empty data."""
        # Mock model artifacts
        mock_session = MagicMock()
        mock_columns = ['feature1', 'feature2']
        mock_load.return_value = (mock_session, mock_columns)
        
        # Mock preprocessing with empty data
        mock_preprocess.return_value = pd.DataFrame()
        
        input_data = pd.DataFrame()
        
        with self.assertRaises(Exception):
            make_prediction(input_data)
    
    @patch('app.predict.ort')
    @patch('app.predict.load_model_artifacts')
    @patch('app.predict.preprocess_data')
    def test_make_prediction_model_error(self, mock_preprocess, mock_load, mock_ort):
        """Test prediction when model throws an error."""
        # Mock model artifacts
        mock_session = MagicMock()
        mock_columns = ['feature1', 'feature2']
        mock_load.return_value = (mock_session, mock_columns)
        
        # Mock preprocessing
        processed_data = pd.DataFrame({
            'feature1': [1.0],
            'feature2': [2.0]
        })
        mock_preprocess.return_value = processed_data
        
        # Mock model error
        mock_session.run.side_effect = Exception("Model inference error")
        
        input_data = pd.DataFrame({'some': ['data']})
        
        with self.assertRaises(Exception):
            make_prediction(input_data)
    
    @patch('app.predict.ort')
    @patch('app.predict.load_model_artifacts')
    @patch('app.predict.preprocess_data')
    def test_make_prediction_data_types(self, mock_preprocess, mock_load, mock_ort):
        """Test that prediction handles different data types correctly."""
        # Mock model artifacts
        mock_session = MagicMock()
        mock_columns = ['feature1', 'feature2']
        mock_load.return_value = (mock_session, mock_columns)
        
        # Mock preprocessing with mixed data types
        processed_data = pd.DataFrame({
            'feature1': [1, 2, 3],  # Integer
            'feature2': [1.5, 2.5, 3.5]  # Float
        })
        mock_preprocess.return_value = processed_data
        
        # Mock ONNX prediction
        mock_session.run.return_value = [np.array([[10000.0], [15000.0], [20000.0]])]
        
        input_data = pd.DataFrame({'some': ['data']})
        predictions = make_prediction(input_data)
        
        # Verify that predictions are returned as expected
        self.assertEqual(len(predictions), 3)
        self.assertIsInstance(predictions, list)
        for pred in predictions:
            self.assertIsInstance(pred, (int, float, np.number))
    
    @patch('app.predict.ort')
    @patch('app.predict.load_model_artifacts')
    @patch('app.predict.preprocess_data')
    def test_make_prediction_single_sample(self, mock_preprocess, mock_load, mock_ort):
        """Test prediction with a single sample."""
        # Mock model artifacts
        mock_session = MagicMock()
        mock_columns = ['feature1', 'feature2']
        mock_load.return_value = (mock_session, mock_columns)
        
        # Mock preprocessing
        processed_data = pd.DataFrame({
            'feature1': [1.0],
            'feature2': [2.0]
        })
        mock_preprocess.return_value = processed_data
        
        # Mock ONNX prediction
        mock_session.run.return_value = [np.array([[12000.0]])]
        
        input_data = pd.DataFrame({'some': ['data']})
        predictions = make_prediction(input_data)
        
        self.assertEqual(len(predictions), 1)
        self.assertAlmostEqual(predictions[0], 12000.0)
    
    @patch('app.predict.ort')
    @patch('app.predict.load_model_artifacts')
    @patch('app.predict.preprocess_data')
    def test_make_prediction_large_batch(self, mock_preprocess, mock_load, mock_ort):
        """Test prediction with a large batch of data."""
        # Mock model artifacts
        mock_session = MagicMock()
        mock_columns = ['feature1', 'feature2']
        mock_load.return_value = (mock_session, mock_columns)
        
        # Mock preprocessing with large batch
        batch_size = 1000
        processed_data = pd.DataFrame({
            'feature1': np.random.randn(batch_size),
            'feature2': np.random.randn(batch_size)
        })
        mock_preprocess.return_value = processed_data
        
        # Mock ONNX prediction
        mock_predictions = np.random.randn(batch_size, 1) * 10000 + 15000
        mock_session.run.return_value = [mock_predictions]
        
        input_data = pd.DataFrame({'some': ['data'] * batch_size})
        predictions = make_prediction(input_data)
        
        self.assertEqual(len(predictions), batch_size)
        self.assertIsInstance(predictions, list)
    
    @patch('app.predict.logger')
    @patch('app.predict.ort')
    @patch('app.predict.load_model_artifacts')
    @patch('app.predict.preprocess_data')
    def test_prediction_logging(self, mock_preprocess, mock_load, mock_ort, mock_logger):
        """Test that prediction functions log appropriately."""
        # Mock model artifacts
        mock_session = MagicMock()
        mock_columns = ['feature1', 'feature2']
        mock_load.return_value = (mock_session, mock_columns)
        
        # Mock preprocessing
        processed_data = pd.DataFrame({
            'feature1': [1.0],
            'feature2': [2.0]
        })
        mock_preprocess.return_value = processed_data
        
        # Mock ONNX prediction
        mock_session.run.return_value = [np.array([[12000.0]])]
        
        input_data = pd.DataFrame({'some': ['data']})
        make_prediction(input_data)
        
        # Verify that logging calls were made
        self.assertTrue(mock_logger.info.called)
    
    def test_prediction_input_validation(self):
        """Test input validation for prediction functions."""
        # Test with None input
        with self.assertRaises((TypeError, AttributeError)):
            make_prediction(None)
        
        # Test with non-DataFrame input
        with self.assertRaises((TypeError, AttributeError)):
            make_prediction("not a dataframe")
        
        # Test with list input
        with self.assertRaises((TypeError, AttributeError)):
            make_prediction([1, 2, 3])

class TestPredictionIntegration(unittest.TestCase):
    """
    Integration tests for prediction pipeline.
    """
    
    def setUp(self):
        """Set up integration test data."""
        self.sample_car_data = pd.DataFrame({
            'CarName': ['toyota corolla'],
            'fueltype': ['gas'],
            'aspiration': ['std'],
            'doornumber': ['four'],
            'carbody': ['sedan'],
            'drivewheel': ['fwd'],
            'enginelocation': ['front'],
            'wheelbase': [88.6],
            'carlength': [141.1],
            'carwidth': [60.3],
            'carheight': [47.8],
            'curbweight': [1488],
            'enginetype': ['ohc'],
            'cylindernumber': ['four'],
            'enginesize': [61],
            'fuelsystem': ['2bbl'],
            'boreratio': [2.91],
            'stroke': [3.03],
            'compressionratio': [9.0],
            'horsepower': [48],
            'peakrpm': [5000],
            'citympg': [47],
            'highwaympg': [53]
        })
    
    @patch('app.predict.ort')
    @patch('app.predict.os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_end_to_end_prediction(self, mock_file, mock_exists, mock_ort):
        """Test end-to-end prediction pipeline."""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock column file content
        expected_columns = [
            'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
            'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',
            'peakrpm', 'citympg', 'highwaympg', 'brand_toyota'
        ]
        mock_file.return_value.read.return_value = '\n'.join(expected_columns) + '\n'
        
        # Mock ONNX session
        mock_session = MagicMock()
        mock_session.run.return_value = [np.array([[8500.0]])]
        mock_ort.InferenceSession.return_value = mock_session
        
        # Test prediction
        predictions = make_prediction(self.sample_car_data)
        
        self.assertEqual(len(predictions), 1)
        self.assertIsInstance(predictions[0], (int, float, np.number))
        self.assertGreater(predictions[0], 0)  # Price should be positive

if __name__ == '__main__':
    unittest.main()