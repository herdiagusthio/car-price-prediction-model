
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

def validate_input_data(df: pd.DataFrame) -> None:
    """
    Validate input DataFrame for required columns and data types.
    
    Args:
        df (pd.DataFrame): Input DataFrame to validate
        
    Raises:
        ValueError: If validation fails
    """
    required_columns = [
        'CarName', 'symboling', 'fueltype', 'aspiration', 'doornumber',
        'carbody', 'drivewheel', 'enginelocation', 'wheelbase', 'carlength',
        'carwidth', 'carheight', 'curbweight', 'enginetype', 'cylindernumber',
        'enginesize', 'fuelsystem', 'boreratio', 'stroke', 'compressionratio',
        'horsepower', 'peakrpm', 'citympg', 'highwaympg'
    ]
    
    # Check if DataFrame is empty
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for null values in critical columns
    critical_columns = ['CarName', 'symboling', 'fueltype', 'carbody']
    null_counts = df[critical_columns].isnull().sum()
    if null_counts.any():
        null_cols = null_counts[null_counts > 0].index.tolist()
        raise ValueError(f"Null values found in critical columns: {null_cols}")
    
    logger.info(f"Input validation passed for {len(df)} records")

def clean_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate numeric features.
    
    Args:
        df (pd.DataFrame): DataFrame with numeric features
        
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric features
    """
    numeric_columns = [
        'symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight',
        'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio',
        'horsepower', 'peakrpm', 'citympg', 'highwaympg'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            # Convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill NaN values with median for numeric columns
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.warning(f"Filled {df[col].isnull().sum()} NaN values in {col} with median: {median_val}")
    
    return df

def extract_and_clean_brand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract brand from CarName and clean brand names.
    
    Args:
        df (pd.DataFrame): DataFrame with CarName column
        
    Returns:
        pd.DataFrame: DataFrame with cleaned brand column
    """
    # Extract brand from CarName
    df['brand'] = df['CarName'].apply(lambda x: str(x).split(' ')[0].lower().strip())
    
    # Comprehensive brand cleaning
    brand_replacements = {
        'maxda': 'mazda',
        'porcshce': 'porsche', 
        'toyouta': 'toyota',
        'vokswagen': 'volkswagen',
        'vw': 'volkswagen',
        'alfa-romero': 'alfa-romeo',
        'mercedes-benz': 'mercedes',
        'chevroelt': 'chevrolet',
        'chevy': 'chevrolet',
        'bmw': 'bmw',
        'audi': 'audi'
    }
    
    df['brand'] = df['brand'].replace(brand_replacements)
    
    # Log brand distribution
    brand_counts = df['brand'].value_counts()
    logger.info(f"Brand distribution: {brand_counts.to_dict()}")
    
    return df

def validate_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean categorical features.
    
    Args:
        df (pd.DataFrame): DataFrame with categorical features
        
    Returns:
        pd.DataFrame: DataFrame with validated categorical features
    """
    categorical_validations = {
        'fueltype': ['gas', 'diesel'],
        'aspiration': ['std', 'turbo'],
        'doornumber': ['two', 'four'],
        'carbody': ['convertible', 'hatchback', 'sedan', 'wagon', 'hardtop'],
        'drivewheel': ['rwd', 'fwd', '4wd'],
        'enginelocation': ['front', 'rear'],
        'enginetype': ['dohc', 'dohcv', 'l', 'ohc', 'ohcf', 'ohcv', 'rotor'],
        'cylindernumber': ['two', 'three', 'four', 'five', 'six', 'eight', 'twelve'],
        'fuelsystem': ['1bbl', '2bbl', '4bbl', 'idi', 'mfi', 'mpfi', 'spdi', 'spfi']
    }
    
    for col, valid_values in categorical_validations.items():
        if col in df.columns:
            # Convert to lowercase and strip whitespace
            df[col] = df[col].astype(str).str.lower().str.strip()
            
            # Check for invalid values
            invalid_mask = ~df[col].isin(valid_values)
            if invalid_mask.any():
                invalid_values = df.loc[invalid_mask, col].unique()
                logger.warning(f"Invalid values in {col}: {invalid_values}")
                
                # Replace invalid values with most common valid value
                most_common = df.loc[~invalid_mask, col].mode()[0] if not df.loc[~invalid_mask, col].empty else valid_values[0]
                df.loc[invalid_mask, col] = most_common
                logger.info(f"Replaced invalid values in {col} with: {most_common}")
    
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and perform feature engineering on raw DataFrame.

    Args:
        df (pd.DataFrame): Raw DataFrame from car dataset.

    Returns:
        pd.DataFrame: Processed DataFrame ready for training.
        
    Raises:
        ValueError: If input validation fails
    """
    logger.info(f"Starting preprocessing for {len(df)} records")
    
    # Validate input data
    validate_input_data(df)
    
    # Create copy to avoid modifying original DataFrame
    processed_df = df.copy()
    
    # Clean numeric features
    processed_df = clean_numeric_features(processed_df)
    
    # Extract and clean brand information
    processed_df = extract_and_clean_brand(processed_df)
    
    # Validate categorical features
    processed_df = validate_categorical_features(processed_df)
    
    # Remove columns not needed for training
    columns_to_drop = ['car_ID', 'CarName']
    existing_cols_to_drop = [col for col in columns_to_drop if col in processed_df.columns]
    if existing_cols_to_drop:
        processed_df = processed_df.drop(columns=existing_cols_to_drop, axis=1)
        logger.info(f"Dropped columns: {existing_cols_to_drop}")
    
    # Final validation
    if processed_df.empty:
        raise ValueError("Processed DataFrame is empty")
    
    logger.info(f"Preprocessing completed. Final shape: {processed_df.shape}")
    logger.info(f"Final columns: {processed_df.columns.tolist()}")
    
    return processed_df

# Example usage (optional, for testing)
if __name__ == '__main__':
    # This code will only run if you execute 'python app/preprocessing.py'
    # directly from terminal, not when imported.
    try:
        from app.config import DATA_PATH
        raw_df = pd.read_csv(DATA_PATH)
        cleaned_df = preprocess_data(raw_df)
        print("Preprocessing successful. Sample data:")
        print(cleaned_df.head())
        print("\nColumns:")
        print(cleaned_df.columns)
    except FileNotFoundError:
        print(f"File '{DATA_PATH}' not found. Make sure the file exists in the data directory.")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
