
import os
from pathlib import Path
from typing import List, Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Model artifacts paths
MODEL_PATH = MODELS_DIR / "best_model.onnx"
COLUMN_LIST_PATH = MODELS_DIR / "kolom_model.txt"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

# Data paths
DATA_PATH = os.getenv('DATA_PATH', 'data/CarPrice_Assignment.csv')
TRAINING_LOG_PATH = 'training.log'

# Model configuration
MODEL_CONFIG = {
    'algorithm': 'RandomForest',
    'n_estimators': int(os.getenv('N_ESTIMATORS', '100')),
    'max_depth': int(os.getenv('MAX_DEPTH', '10')) if os.getenv('MAX_DEPTH') else None,
    'random_state': int(os.getenv('RANDOM_STATE', '42')),
    'test_size': float(os.getenv('TEST_SIZE', '0.2')),
    'n_jobs': int(os.getenv('N_JOBS', '-1'))
}

# Feature configuration
TARGET_FEATURE = 'price'

FEATURES_TO_DROP = ['price', 'car_ID']

NUMERIC_FEATURES = [
    'symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight',
    'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio',
    'horsepower', 'peakrpm', 'citympg', 'highwaympg'
]

CATEGORICAL_FEATURES = [
    'fueltype', 'aspiration', 'doornumber', 'carbody',
    'drivewheel', 'enginelocation', 'enginetype',
    'cylindernumber', 'fuelsystem', 'brand'
]

# Validation rules for categorical features
CATEGORICAL_VALIDATIONS = {
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

# Brand cleaning mappings
BRAND_REPLACEMENTS = {
    'maxda': 'mazda',
    'porcshce': 'porsche',
    'toyouta': 'toyota',
    'vokswagen': 'volkswagen',
    'vw': 'volkswagen',
    'alfa-romero': 'alfa-romeo',
    'mercedes-benz': 'mercedes',
    'chevroelt': 'chevrolet',
    'chevy': 'chevrolet'
}

# API configuration
API_CONFIG = {
    'title': 'Car Price Prediction API',
    'description': 'A machine learning API for predicting car prices based on vehicle features',
    'version': '1.0.0',
    'host': os.getenv('API_HOST', '0.0.0.0'),
    'port': int(os.getenv('API_PORT', '8000')),
    'debug': os.getenv('DEBUG', 'False').lower() == 'true'
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'
        }
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'detailed',
            'class': 'logging.FileHandler',
            'filename': 'app.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    'min_r2_score': 0.7,
    'max_rmse': 5000,
    'max_prediction_time_ms': 100,
    'max_model_size_mb': 50
}

# Environment settings
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
IS_PRODUCTION = ENVIRONMENT.lower() == 'production'
IS_DEVELOPMENT = ENVIRONMENT.lower() == 'development'

# Security settings
SECURITY_CONFIG = {
    'allowed_hosts': os.getenv('ALLOWED_HOSTS', '*').split(','),
    'cors_origins': os.getenv('CORS_ORIGINS', '*').split(','),
    'rate_limit': os.getenv('RATE_LIMIT', '100/minute')
}
