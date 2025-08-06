import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.preprocessing import (
    preprocess_data,
    validate_input_data,
    clean_numeric_features,
    extract_brand_from_carname,
    clean_brand_name,
    validate_categorical_features
)
from app.config import CATEGORICAL_FEATURES, FEATURES_TO_DROP

class TestPreprocessing(unittest.TestCase):
    """
    Test suite for data preprocessing functions.
    """
    
    def setUp(self):
        """Set up test data."""
        self.sample_data = pd.DataFrame({
            'car_ID': [1, 2, 3, 4, 5],
            'CarName': ['toyota corolla', 'honda civic', 'bmw 320i', 'audi a4', 'volkswagen golf'],
            'fueltype': ['gas', 'gas', 'gas', 'gas', 'gas'],
            'aspiration': ['std', 'std', 'turbo', 'std', 'std'],
            'doornumber': ['four', 'four', 'four', 'four', 'two'],
            'carbody': ['sedan', 'sedan', 'sedan', 'sedan', 'hatchback'],
            'drivewheel': ['fwd', 'fwd', 'rwd', 'fwd', 'fwd'],
            'enginelocation': ['front', 'front', 'front', 'front', 'front'],
            'wheelbase': [88.6, 93.7, 101.2, 99.8, 97.3],
            'carlength': [141.1, 158.8, 176.2, 176.6, 158.5],
            'carwidth': [60.3, 63.9, 66.2, 66.4, 64.1],
            'carheight': [47.8, 63.4, 54.2, 54.3, 59.1],
            'curbweight': [1488, 2017, 2734, 2365, 2024],
            'enginetype': ['ohc', 'ohc', 'ohc', 'ohc', 'ohc'],
            'cylindernumber': ['four', 'four', 'six', 'four', 'four'],
            'enginesize': [61, 92, 164, 109, 109],
            'fuelsystem': ['2bbl', 'mpfi', 'mpfi', 'mpfi', 'mpfi'],
            'boreratio': [2.91, 2.97, 3.31, 3.19, 3.19],
            'stroke': [3.03, 3.23, 3.40, 3.40, 3.40],
            'compressionratio': [9.0, 9.4, 8.4, 10.1, 10.1],
            'horsepower': [48, 76, 121, 102, 102],
            'peakrpm': [5000, 6000, 4250, 5500, 5500],
            'citympg': [47, 31, 21, 24, 24],
            'highwaympg': [53, 38, 28, 30, 29],
            'price': [5118, 6695, 16500, 13950, 12764]
        })
        
        self.invalid_data = pd.DataFrame({
            'car_ID': [1, 2],
            'CarName': ['toyota corolla', None],  # Missing value
            'price': [5118, -1000]  # Negative price
        })
    
    def test_validate_input_data_valid(self):
        """Test input data validation with valid data."""
        result = validate_input_data(self.sample_data)
        self.assertTrue(result['is_valid'])
        self.assertEqual(len(result['errors']), 0)
    
    def test_validate_input_data_invalid(self):
        """Test input data validation with invalid data."""
        result = validate_input_data(self.invalid_data)
        self.assertFalse(result['is_valid'])
        self.assertGreater(len(result['errors']), 0)
    
    def test_validate_input_data_empty(self):
        """Test input data validation with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = validate_input_data(empty_df)
        self.assertFalse(result['is_valid'])
        self.assertIn('DataFrame is empty', result['errors'])
    
    def test_clean_numeric_features(self):
        """Test numeric feature cleaning."""
        # Create data with outliers
        data_with_outliers = self.sample_data.copy()
        data_with_outliers.loc[0, 'horsepower'] = 1000  # Extreme outlier
        
        cleaned_data = clean_numeric_features(data_with_outliers)
        
        # Check that outliers are handled
        self.assertLess(cleaned_data.loc[0, 'horsepower'], 1000)
        self.assertFalse(cleaned_data.isnull().any().any())
    
    def test_extract_brand_from_carname(self):
        """Test brand extraction from car names."""
        test_cases = [
            ('toyota corolla', 'toyota'),
            ('honda civic', 'honda'),
            ('bmw 320i', 'bmw'),
            ('audi a4', 'audi'),
            ('volkswagen golf', 'volkswagen')
        ]
        
        for carname, expected_brand in test_cases:
            with self.subTest(carname=carname):
                result = extract_brand_from_carname(carname)
                self.assertEqual(result, expected_brand)
    
    def test_extract_brand_from_carname_edge_cases(self):
        """Test brand extraction with edge cases."""
        edge_cases = [
            ('', 'unknown'),
            (None, 'unknown'),
            ('single', 'single'),
            ('TOYOTA COROLLA', 'toyota')  # Test case sensitivity
        ]
        
        for carname, expected_brand in edge_cases:
            with self.subTest(carname=carname):
                result = extract_brand_from_carname(carname)
                self.assertEqual(result, expected_brand)
    
    def test_clean_brand_name(self):
        """Test brand name cleaning and standardization."""
        test_cases = [
            ('toyouta', 'toyota'),
            ('vokswagen', 'volkswagen'),
            ('maxda', 'mazda'),
            ('porcshce', 'porsche'),
            ('vw', 'volkswagen'),
            ('bmw', 'bmw')  # Should remain unchanged
        ]
        
        for input_brand, expected_brand in test_cases:
            with self.subTest(input_brand=input_brand):
                result = clean_brand_name(input_brand)
                self.assertEqual(result, expected_brand)
    
    def test_validate_categorical_features(self):
        """Test categorical feature validation."""
        valid_data = self.sample_data.copy()
        result = validate_categorical_features(valid_data)
        self.assertTrue(result['is_valid'])
        
        # Test with invalid categorical values
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, 'fueltype'] = 'invalid_fuel'
        result = validate_categorical_features(invalid_data)
        # Should still be valid as we handle unknown categories
        self.assertTrue(result['is_valid'])
    
    def test_preprocess_data_complete_pipeline(self):
        """Test the complete preprocessing pipeline."""
        processed_data = preprocess_data(self.sample_data)
        
        # Check that required columns are present
        self.assertIn('brand', processed_data.columns)
        
        # Check that dropped columns are removed
        for col in FEATURES_TO_DROP:
            if col in self.sample_data.columns:
                self.assertNotIn(col, processed_data.columns)
        
        # Check data types and no missing values in critical columns
        self.assertFalse(processed_data['brand'].isnull().any())
        
        # Check that numeric columns are properly cleaned
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.assertFalse(processed_data[col].isnull().any())
    
    def test_preprocess_data_preserves_target(self):
        """Test that preprocessing preserves the target variable."""
        if 'price' in self.sample_data.columns:
            processed_data = preprocess_data(self.sample_data)
            self.assertIn('price', processed_data.columns)
            self.assertEqual(len(processed_data), len(self.sample_data))
    
    def test_preprocess_data_handles_missing_values(self):
        """Test preprocessing with missing values."""
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0, 'horsepower'] = np.nan
        data_with_missing.loc[1, 'CarName'] = np.nan
        
        processed_data = preprocess_data(data_with_missing)
        
        # Should handle missing values appropriately
        self.assertFalse(processed_data.isnull().any().any())
    
    @patch('app.preprocessing.logger')
    def test_preprocessing_logging(self, mock_logger):
        """Test that preprocessing functions log appropriately."""
        preprocess_data(self.sample_data)
        
        # Verify that logging calls were made
        self.assertTrue(mock_logger.info.called)
    
    def test_preprocessing_data_types(self):
        """Test that preprocessing maintains appropriate data types."""
        processed_data = preprocess_data(self.sample_data)
        
        # Check that categorical features are strings
        for feature in CATEGORICAL_FEATURES:
            if feature in processed_data.columns:
                self.assertTrue(processed_data[feature].dtype == 'object' or 
                              processed_data[feature].dtype.name == 'category')
        
        # Check that numeric features are numeric
        numeric_features = processed_data.select_dtypes(include=[np.number]).columns
        for feature in numeric_features:
            self.assertTrue(pd.api.types.is_numeric_dtype(processed_data[feature]))
    
    def test_preprocessing_reproducibility(self):
        """Test that preprocessing is reproducible."""
        processed_data1 = preprocess_data(self.sample_data.copy())
        processed_data2 = preprocess_data(self.sample_data.copy())
        
        # Results should be identical
        pd.testing.assert_frame_equal(processed_data1, processed_data2)
    
    def test_preprocessing_performance(self):
        """Test preprocessing performance with larger dataset."""
        # Create a larger dataset
        large_data = pd.concat([self.sample_data] * 1000, ignore_index=True)
        
        import time
        start_time = time.time()
        processed_data = preprocess_data(large_data)
        end_time = time.time()
        
        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(end_time - start_time, 10.0)  # 10 seconds
        self.assertEqual(len(processed_data), len(large_data))

if __name__ == '__main__':
    unittest.main()