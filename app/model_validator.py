import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import logging
import warnings
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime
from app.config import PERFORMANCE_THRESHOLDS

logger = logging.getLogger(__name__)

class ModelValidator:
    """
    Comprehensive model validation and performance monitoring.
    """
    
    def __init__(self, performance_thresholds: Dict[str, float] = None):
        self.thresholds = performance_thresholds or PERFORMANCE_THRESHOLDS
        self.validation_history = []
    
    def validate_model_performance(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Validate model performance against test data.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dict containing validation results
        """
        logger.info("Starting model performance validation")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),  # Compatible with older sklearn versions
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': self._calculate_mape(y_test, y_pred),
            'prediction_std': np.std(y_pred),
            'residual_mean': np.mean(y_test - y_pred),
            'residual_std': np.std(y_test - y_pred)
        }
        
        # Validate against thresholds
        validation_results = {
            'metrics': metrics,
            'thresholds': self.thresholds,
            'passed_validation': self._check_thresholds(metrics),
            'validation_timestamp': datetime.now().isoformat(),
            'test_samples': len(y_test)
        }
        
        # Add to history
        self.validation_history.append(validation_results)
        
        logger.info(f"Validation completed. R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.2f}")
        
        return validation_results
    
    def cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation on the model.
        
        Args:
            model: Model to validate
            X: Features
            y: Target
            cv: Number of cross-validation folds
            
        Returns:
            Dict containing cross-validation results
        """
        logger.info(f"Starting {cv}-fold cross-validation")

        # Suppress specific UserWarning from sklearn's OneHotEncoder during cross-validation
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Found unknown categories in columns.*",
                category=UserWarning
            )
            # Perform cross-validation for different metrics
            cv_scores = {
                'r2': cross_val_score(model, X, y, cv=cv, scoring='r2'),
                'neg_mse': cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error'),
                'neg_mae': cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
            }
        
        # Calculate statistics
        cv_results = {
            'r2_mean': cv_scores['r2'].mean(),
            'r2_std': cv_scores['r2'].std(),
            'mse_mean': -cv_scores['neg_mse'].mean(),
            'mse_std': cv_scores['neg_mse'].std(),
            'mae_mean': -cv_scores['neg_mae'].mean(),
            'mae_std': cv_scores['neg_mae'].std(),
            'cv_folds': cv,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Cross-validation completed. Mean R²: {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
        
        return cv_results
    
    def detect_data_drift(self, training_data: pd.DataFrame, new_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect data drift between training and new data.
        
        Args:
            training_data: Original training dataset
            new_data: New data to compare
            
        Returns:
            Dict containing drift analysis results
        """
        logger.info("Analyzing data drift")
        
        drift_results = {
            'feature_drift': {},
            'overall_drift_score': 0.0,
            'drift_detected': False,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        numeric_features = training_data.select_dtypes(include=[np.number]).columns
        
        for feature in numeric_features:
            if feature in new_data.columns:
                # Calculate statistical differences
                train_mean = training_data[feature].mean()
                new_mean = new_data[feature].mean()
                train_std = training_data[feature].std()
                new_std = new_data[feature].std()
                
                # Simple drift detection using mean and std differences
                mean_diff = abs(train_mean - new_mean) / (train_std + 1e-8)
                std_diff = abs(train_std - new_std) / (train_std + 1e-8)
                
                drift_score = (mean_diff + std_diff) / 2
                
                drift_results['feature_drift'][feature] = {
                    'drift_score': drift_score,
                    'train_mean': train_mean,
                    'new_mean': new_mean,
                    'train_std': train_std,
                    'new_std': new_std,
                    'drift_detected': drift_score > 0.5  # Threshold for drift detection
                }
        
        # Calculate overall drift score
        if drift_results['feature_drift']:
            drift_scores = [info['drift_score'] for info in drift_results['feature_drift'].values()]
            drift_results['overall_drift_score'] = np.mean(drift_scores)
            drift_results['drift_detected'] = drift_results['overall_drift_score'] > 0.3
        
        logger.info(f"Data drift analysis completed. Overall drift score: {drift_results['overall_drift_score']:.4f}")
        
        return drift_results
    
    def validate_prediction_quality(self, predictions: np.ndarray, confidence_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Validate the quality of predictions.
        
        Args:
            predictions: Model predictions
            confidence_threshold: Threshold for prediction confidence
            
        Returns:
            Dict containing prediction quality metrics
        """
        logger.info("Validating prediction quality")
        
        quality_metrics = {
            'prediction_count': len(predictions),
            'mean_prediction': np.mean(predictions),
            'std_prediction': np.std(predictions),
            'min_prediction': np.min(predictions),
            'max_prediction': np.max(predictions),
            'negative_predictions': np.sum(predictions < 0),
            'extreme_predictions': np.sum(predictions > 100000),  # Assuming car prices shouldn't exceed $100k in this dataset
            'validation_timestamp': datetime.now().isoformat()
        }
        
        # Quality flags
        quality_metrics['quality_flags'] = {
            'has_negative_predictions': quality_metrics['negative_predictions'] > 0,
            'has_extreme_predictions': quality_metrics['extreme_predictions'] > 0,
            'reasonable_std': quality_metrics['std_prediction'] < quality_metrics['mean_prediction'],
            'reasonable_range': quality_metrics['max_prediction'] - quality_metrics['min_prediction'] < 80000
        }
        
        # Overall quality score
        quality_score = sum(1 for flag in quality_metrics['quality_flags'].values() if not flag) / len(quality_metrics['quality_flags'])
        quality_metrics['overall_quality_score'] = quality_score
        quality_metrics['quality_passed'] = quality_score >= confidence_threshold
        
        logger.info(f"Prediction quality validation completed. Quality score: {quality_score:.4f}")
        
        return quality_metrics
    
    def generate_validation_report(self, output_path: str = 'validation_report.json') -> Dict[str, Any]:
        """
        Generate a comprehensive validation report.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Dict containing the complete validation report
        """
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'validation_history': self.validation_history,
            'summary': {
                'total_validations': len(self.validation_history),
                'latest_validation': self.validation_history[-1] if self.validation_history else None
            }
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {output_path}")
        
        return report
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        """
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    def _check_thresholds(self, metrics: Dict[str, float]) -> bool:
        """
        Check if metrics pass the defined thresholds.
        """
        checks = [
            metrics['r2'] >= self.thresholds.get('min_r2_score', 0.7),
            metrics['rmse'] <= self.thresholds.get('max_rmse', 5000)
        ]
        
        return all(checks)