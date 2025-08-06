
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import warnings
import logging
import os
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Ignore warnings from skl2onnx
warnings.filterwarnings("ignore", category=FutureWarning)

# Import preprocessing functions and configuration
from preprocessing import preprocess_data
# (We assume config will be handled at the main application level later)

def run_training(test_size=0.2, random_state=42, n_estimators=100, max_depth=None):
    """
    Run the complete training pipeline: load data, process,
    train model, validate, and save artifacts.
    
    Args:
        test_size (float): Proportion of dataset for testing
        random_state (int): Random state for reproducibility
        n_estimators (int): Number of trees in Random Forest
        max_depth (int): Maximum depth of trees
    
    Returns:
        dict: Training metrics and model information
    """
    start_time = datetime.now()
    logger.info(f"Starting training process at {start_time}")
    
    try:
        # 1. Load Raw Data
        data_path = DATA_PATH
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found: {data_path}")
            
        logger.info(f"Loading dataset from {data_path}")
        raw_df = pd.read_csv(data_path)
        logger.info(f"Dataset loaded successfully. Shape: {raw_df.shape}")
        
        # Data validation
        if raw_df.empty:
            raise ValueError("Dataset is empty")
        if 'price' not in raw_df.columns:
            raise ValueError("Target column 'price' not found in dataset")
            
        # 2. Data Preprocessing
        logger.info("Starting data preprocessing")
        processed_df = preprocess_data(raw_df)
        logger.info(f"Data preprocessing completed. Shape: {processed_df.shape}")
        
        # 3. Feature and Target Separation
        X = processed_df.drop('price', axis=1)
        y = processed_df['price']
        logger.info(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
        
        # 4. One-Hot Encoding
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import make_column_transformer

        logger.info("Applying one-hot encoding")
        
        categorical_features = X.select_dtypes(include=['object']).columns
        
        preprocessor = make_column_transformer(
            (OneHotEncoder(handle_unknown='ignore'), categorical_features),
            remainder='passthrough'
        )
        
        X_encoded = preprocessor.fit_transform(X)
        
        # Get feature names after encoding
        try:
            feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
            # Fallback for older sklearn versions
            feature_names = preprocessor.named_transformers_['onehotencoder'].get_feature_names(categorical_features)
            # Manually construct the full list of feature names
            non_categorical_features = X.select_dtypes(exclude=['object']).columns
            feature_names = list(feature_names) + list(non_categorical_features)

        X_encoded = pd.DataFrame(X_encoded, columns=feature_names, index=X.index)

        logger.info(f"One-hot encoding completed. Final features: {X_encoded.shape[1]}")
        
        # 5. Train-Test Split for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=test_size, random_state=random_state
        )
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # 6. Train Random Forest Model
        logger.info(f"Training Random Forest model (n_estimators={n_estimators}, max_depth={max_depth})")
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
        rf_model.fit(X_train, y_train)
        logger.info("Model training completed")
        
        # 7. Model Validation
        logger.info("Evaluating model performance")
        y_pred_train = rf_model.predict(X_train)
        y_pred_test = rf_model.predict(X_test)
        
        # Calculate metrics
        train_metrics = {
            'mse': mean_squared_error(y_train, y_pred_train),
            'rmse': mean_squared_error(y_train, y_pred_train)**0.5,
            'mae': mean_absolute_error(y_train, y_pred_train),
            'r2': r2_score(y_train, y_pred_train)
        }
        
        test_metrics = {
            'mse': mean_squared_error(y_test, y_pred_test),
            'rmse': mean_squared_error(y_test, y_pred_test)**0.5,
            'mae': mean_absolute_error(y_test, y_pred_test),
            'r2': r2_score(y_test, y_pred_test)
        }
        
        logger.info(f"Training Metrics - RMSE: {train_metrics['rmse']:.2f}, R²: {train_metrics['r2']:.4f}")
        logger.info(f"Test Metrics - RMSE: {test_metrics['rmse']:.2f}, R²: {test_metrics['r2']:.4f}")
        
        # 8. Retrain on full dataset for final model
        logger.info("Retraining on full dataset for final model")
        rf_model.fit(X_encoded, y)
        
        # 9. Save the preprocessor
        preprocessor_path = 'preprocessor.joblib'
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"Preprocessor saved to {preprocessor_path}")

        # 10. Convert and Save Model to ONNX
        logger.info("Converting model to ONNX format")
        num_features = X_encoded.shape[1]
        initial_type = [('float_input', FloatTensorType([None, num_features]))]
        onnx_model = convert_sklearn(rf_model, initial_types=initial_type)
        
        model_path = 'best_model.onnx'
        with open(model_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        logger.info(f"Model successfully saved to: {model_path}")
        
        # 10. Save Column List and Preprocessor
        from config import PREPROCESSOR_PATH
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        logger.info(f"Preprocessor saved to: {PREPROCESSOR_PATH}")

        kolom_path = 'kolom_model.txt'
        with open(kolom_path, 'w') as f:
            for col in X_encoded.columns:
                f.write(f"{col}\n")
        logger.info(f"Column list successfully saved to: {kolom_path}")
        
        # 11. Save training metadata
        metadata = {
            'training_date': start_time.isoformat(),
            'dataset_shape': raw_df.shape,
            'final_features': X_encoded.shape[1],
            'model_params': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'random_state': random_state
            },
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_names': X_encoded.columns.tolist()
        }
        
        import json
        with open('model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info("Training metadata saved to model_metadata.json")
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Training completed successfully in {duration}")
        
        return metadata
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == '__main__':
    run_training()
