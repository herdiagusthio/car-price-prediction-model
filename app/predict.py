
import onnxruntime as rt
import pandas as pd
import numpy as np
import os
import sys

# Handle imports for both standalone and module usage
try:
    from app.config import MODEL_PATH, COLUMN_LIST_PATH
except ImportError:
    # When running as standalone script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from app.config import MODEL_PATH, COLUMN_LIST_PATH
    except ImportError:
        # Fallback to hardcoded paths
        MODEL_PATH = 'best_model.onnx'
        COLUMN_LIST_PATH = 'kolom_model.txt'

# 1. Load model and column list when application starts
# This is more efficient than loading them on every request
sess = rt.InferenceSession(MODEL_PATH)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

with open(COLUMN_LIST_PATH, 'r') as f:
    MODEL_COLUMNS = [line.strip() for line in f.readlines()]

import logging
logger = logging.getLogger(__name__)
logger.info(f"Loaded MODEL_COLUMNS type: {type(MODEL_COLUMNS)}")
logger.info(f"Loaded MODEL_COLUMNS sample: {MODEL_COLUMNS[:5]}")

# Load the preprocessor once at startup
try:
    from app.config import PREPROCESSOR_PATH
    import joblib
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except (ImportError, FileNotFoundError):
    preprocessor = None # Fallback if not found


def load_model_artifacts():
    """
    Load model artifacts (ONNX model and column list).
    
    Returns:
        tuple: (session, input_name, label_name, model_columns)
    """
    session = rt.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    
    with open(COLUMN_LIST_PATH, 'r') as f:
        model_columns = [line.strip() for line in f.readlines()]
    
    return session, input_name, label_name, model_columns


def make_prediction(input_data):
    """
    Make car price prediction from input data.

    Args:
        input_data (pd.DataFrame): DataFrame containing one or more rows of new car data.

    Returns:
        np.ndarray: Array containing price prediction results.
    """
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"Input data columns: {input_data.columns.tolist()}")

    if preprocessor:
        transformed = preprocessor.transform(input_data)
        feature_names = preprocessor.get_feature_names_out()
        logger.info(f"Preprocessor feature names: {feature_names}")
        transformed_df = pd.DataFrame(transformed, columns=feature_names)
        logger.info(f"Transformed DataFrame columns: {list(transformed_df.columns)}")
        stripped_model_columns = [f.replace('onehotencoder__', '').replace('remainder__', '') for f in MODEL_COLUMNS]
        import json
        logger.info(f"Stripped MODEL_COLUMNS: {json.dumps(stripped_model_columns)}")
        # Log intersection of MODEL_COLUMNS and transformed features
        matched_features = [f for f in MODEL_COLUMNS if f in transformed_df.columns]
        logger.info(f"Matched features count: {len(matched_features)}")
        logger.info(f"Matched features: {matched_features[:10]}...")  # Show first 10 for brevity
        
        if len(matched_features) > 0:
            final_input = transformed_df[matched_features]
        else:
            logger.error("No features matched between MODEL_COLUMNS and transformed_df.columns")
            logger.info(f"First 10 MODEL_COLUMNS: {MODEL_COLUMNS[:10]}")
            logger.info(f"First 10 transformed columns: {list(transformed_df.columns)[:10]}")
            raise ValueError("Feature mismatch: No columns from MODEL_COLUMNS found in transformed data")
    else:
        final_input = input_data.copy()

    # Log model input details for debugging
    input_meta = sess.get_inputs()[0]
    logger.info(f"Model expected input name: {input_meta.name}")
    logger.info(f"Model expected input shape: {input_meta.shape}")
    logger.info(f"Model expected input type: {input_meta.type}")
    logger.info(f"Actual input shape being passed to model: {final_input.shape}")

    # 4. Convert to ONNX-accepted format (numpy array float32)
    input_np = final_input.to_numpy().astype(np.float32)

    # 5. Run Prediction
    prediction = sess.run([label_name], {input_name: input_np})[0]

    return prediction.flatten()


# Example usage (optional, for testing)
if __name__ == '__main__':
    # Create example new input data (similar to one row from original dataset)
    sample_data = {
        'symboling': 3, 'CarName': 'alfa-romero giulia', 'fueltype': 'gas', 'aspiration': 'std',
        'doornumber': 'two', 'carbody': 'convertible', 'drivewheel': 'rwd',
        'enginelocation': 'front', 'wheelbase': 88.6, 'carlength': 168.8,
        'carwidth': 64.1, 'carheight': 48.8, 'curbweight': 2548,
        'enginetype': 'dohc', 'cylindernumber': 'four', 'enginesize': 130,
        'fuelsystem': 'mpfi', 'boreratio': 3.47, 'stroke': 2.68,
        'compressionratio': 9.0, 'horsepower': 111, 'peakrpm': 5000,
        'citympg': 21, 'highwaympg': 27
    }
    sample_df = pd.DataFrame([sample_data])

    # Perform basic preprocessing (extract brand)
    # In real application, this could be a separate function
    sample_df['brand'] = sample_df['CarName'].apply(lambda x: x.split(' ')[0].lower())
    sample_df = sample_df.drop(columns=['CarName']) # Remove original column

    # Make prediction
    pred_price = make_prediction(sample_df)

    print(f"Example input data:\n{sample_df.to_string()}")
    print("-" * 30)
    print(f"Price Prediction Result: ${pred_price[0]:,.2f}")
