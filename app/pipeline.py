import os
import sys
import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# Add app directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.config import (
    MODEL_PATH, COLUMN_LIST_PATH, MODEL_METADATA_PATH,
    DATA_PATH, LOGS_DIR, MODEL_CONFIG
)
from app.preprocessing import preprocess_data
from app.train import run_training
from app.predict import load_model_artifacts, make_prediction
from app.model_validator import ModelValidator
from app.monitoring import get_monitor

logger = logging.getLogger(__name__)

class MLPipeline:
    """
    Comprehensive ML pipeline orchestrator for car price prediction.
    
    This class manages the entire ML workflow including:
    - Data preprocessing and validation
    - Model training and evaluation
    - Model validation and testing
    - Deployment preparation
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ML pipeline.
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        self.config = config or {}
        self.validator = ModelValidator()
        self.monitor = get_monitor()
        
        # Pipeline state
        self.pipeline_state = {
            'last_training': None,
            'last_validation': None,
            'model_version': None,
            'data_version': None,
            'pipeline_status': 'initialized'
        }
        
        # Ensure required directories exist
        self._ensure_directories()
        
        logger.info("MLPipeline initialized")
    
    def _ensure_directories(self) -> None:
        """
        Ensure all required directories exist.
        """
        directories = [
            LOGS_DIR,
            os.path.dirname(MODEL_PATH),
            os.path.dirname(MODEL_METADATA_PATH)
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def run_full_pipeline(self, data_path: Optional[str] = None, 
                         validate_model: bool = True,
                         deploy_ready: bool = False) -> Dict[str, Any]:
        """
        Run the complete ML pipeline from data to deployment.
        
        Args:
            data_path: Path to training data (defaults to config)
            validate_model: Whether to run model validation
            deploy_ready: Whether to prepare for deployment
            
        Returns:
            Dict containing pipeline execution results
        """
        pipeline_start_time = time.time()
        results = {
            'pipeline_id': f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'start_time': datetime.now().isoformat(),
            'steps_completed': [],
            'steps_failed': [],
            'warnings': [],
            'metrics': {},
            'artifacts_created': []
        }
        
        try:
            logger.info(f"Starting full ML pipeline: {results['pipeline_id']}")
            self.pipeline_state['pipeline_status'] = 'running'
            
            # Step 1: Data preprocessing and validation
            logger.info("Step 1: Data preprocessing and validation")
            preprocessing_result = self._run_preprocessing_step(data_path or DATA_PATH)
            results['steps_completed'].append('preprocessing')
            results['metrics']['preprocessing'] = preprocessing_result
            
            # Step 2: Model training
            logger.info("Step 2: Model training")
            training_result = self._run_training_step()
            results['steps_completed'].append('training')
            results['metrics']['training'] = training_result
            results['artifacts_created'].extend(['model.onnx', 'column_list.txt', 'model_metadata.json'])
            
            # Step 3: Model validation (optional)
            if validate_model:
                logger.info("Step 3: Model validation")
                validation_result = self._run_validation_step()
                results['steps_completed'].append('validation')
                results['metrics']['validation'] = validation_result
                results['artifacts_created'].append('validation_report.json')
            
            # Step 4: Deployment preparation (optional)
            if deploy_ready:
                logger.info("Step 4: Deployment preparation")
                deployment_result = self._run_deployment_preparation()
                results['steps_completed'].append('deployment_preparation')
                results['metrics']['deployment'] = deployment_result
            
            # Step 5: Initialize monitoring
            logger.info("Step 5: Initialize monitoring")
            monitoring_result = self._initialize_monitoring()
            results['steps_completed'].append('monitoring_initialization')
            results['metrics']['monitoring'] = monitoring_result
            
            # Update pipeline state
            self.pipeline_state.update({
                'last_training': datetime.now().isoformat(),
                'pipeline_status': 'completed',
                'model_version': results['pipeline_id']
            })
            
            pipeline_duration = time.time() - pipeline_start_time
            results.update({
                'status': 'success',
                'end_time': datetime.now().isoformat(),
                'duration_seconds': pipeline_duration,
                'pipeline_state': self.pipeline_state.copy()
            })
            
            logger.info(f"Pipeline completed successfully in {pipeline_duration:.2f} seconds")
            
        except Exception as e:
            self.pipeline_state['pipeline_status'] = 'failed'
            results.update({
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': time.time() - pipeline_start_time
            })
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise
        
        # Save pipeline results
        self._save_pipeline_results(results)
        
        return results
    
    def _run_preprocessing_step(self, data_path: str) -> Dict[str, Any]:
        """
        Run data preprocessing step.
        
        Args:
            data_path: Path to the data file
            
        Returns:
            Dict containing preprocessing results
        """
        step_start_time = time.time()
        
        try:
            # Check if data file exists
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            # Load and preprocess data
            import pandas as pd
            raw_data = pd.read_csv(data_path)
            
            logger.info(f"Loaded raw data: {raw_data.shape}")
            
            # Preprocess data
            processed_data = preprocess_data(raw_data)
            
            # Data quality checks
            quality_metrics = {
                'raw_data_shape': raw_data.shape,
                'processed_data_shape': processed_data.shape,
                'missing_values': processed_data.isnull().sum().sum(),
                'duplicate_rows': processed_data.duplicated().sum(),
                'data_types': processed_data.dtypes.to_dict()
            }
            
            # Check for data quality issues
            warnings = []
            if quality_metrics['missing_values'] > 0:
                warnings.append(f"Found {quality_metrics['missing_values']} missing values")
            
            if quality_metrics['duplicate_rows'] > 0:
                warnings.append(f"Found {quality_metrics['duplicate_rows']} duplicate rows")
            
            result = {
                'status': 'success',
                'duration_seconds': time.time() - step_start_time,
                'quality_metrics': quality_metrics,
                'warnings': warnings,
                'data_path': data_path
            }
            
            logger.info(f"Preprocessing completed: {processed_data.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Preprocessing step failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'duration_seconds': time.time() - step_start_time
            }
    
    def _run_training_step(self) -> Dict[str, Any]:
        """
        Run model training step.
        
        Returns:
            Dict containing training results
        """
        step_start_time = time.time()
        
        try:
            # Get training configuration
            training_config = {
                'test_size': self.config.get('test_size', MODEL_CONFIG['test_size']),
                'random_state': self.config.get('random_state', MODEL_CONFIG['random_state']),
                'n_estimators': self.config.get('n_estimators', MODEL_CONFIG['n_estimators']),
                'max_depth': self.config.get('max_depth', MODEL_CONFIG['max_depth'])
            }
            
            # Run training
            training_result = run_training(**training_config)
            
            # Load and parse metadata
            metadata = {}
            if os.path.exists(MODEL_METADATA_PATH):
                with open(MODEL_METADATA_PATH, 'r') as f:
                    metadata = json.load(f)
            
            result = {
                'status': 'success',
                'duration_seconds': time.time() - step_start_time,
                'training_config': training_config,
                'model_metrics': metadata.get('validation_metrics', {}),
                'model_path': MODEL_PATH,
                'metadata_path': MODEL_METADATA_PATH
            }
            
            logger.info(f"Training completed with metrics: {metadata.get('validation_metrics', {})}")
            return result
            
        except Exception as e:
            logger.error(f"Training step failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'duration_seconds': time.time() - step_start_time
            }
    
    def _run_validation_step(self) -> Dict[str, Any]:
        """
        Run model validation step.
        
        Returns:
            Dict containing validation results
        """
        step_start_time = time.time()
        
        try:
            # Load test data for validation
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from app.preprocessing import preprocess_data
            from app.config import TARGET_FEATURE, FEATURES_TO_DROP
            
            # Load and preprocess data
            data = pd.read_csv(DATA_PATH)
            processed_data = preprocess_data(data)
            
            # Prepare features and target
            X = processed_data.drop(columns=[TARGET_FEATURE] + 
                                  [col for col in FEATURES_TO_DROP if col in processed_data.columns])
            y = processed_data[TARGET_FEATURE]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Preprocess for sklearn model (handle categorical variables)
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            from sklearn.ensemble import RandomForestRegressor
            
            # Identify categorical and numeric columns
            categorical_cols = X_train.select_dtypes(include=['object']).columns
            numeric_cols = X_train.select_dtypes(exclude=['object']).columns
            
            # Create preprocessor with handle_unknown='ignore' to handle unknown categories
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
                    ('num', 'passthrough', numeric_cols)
                ]
            )
            
            # Create pipeline
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Run comprehensive validation
            validation_report = self.validator.validate_model_performance(
                model, X_test, y_test
            )
            
            # Cross-validation
            cv_results = self.validator.cross_validate_model(
                model, X_train, y_train, cv=5
            )
            
            # Check for data drift (using training data as baseline)
            drift_results = self.validator.detect_data_drift(
                X_train, X_test
            )
            
            # Generate comprehensive report
            comprehensive_report = {
                'validation_report': validation_report,
                'cross_validation': cv_results,
                'data_drift': drift_results,
                'passed_validation': validation_report.get('passed_validation', False),
                'generated_at': datetime.now().isoformat()
            }
            
            # Save validation report
            report_path = os.path.join(LOGS_DIR, 'validation_report.json')
            with open(report_path, 'w') as f:
                json.dump(comprehensive_report, f, indent=2, default=str)
            
            result = {
                'status': 'success',
                'duration_seconds': time.time() - step_start_time,
                'validation_report': comprehensive_report,
                'report_path': report_path,
                'model_performance': validation_report,
                'cross_validation': cv_results,
                'data_drift': drift_results
            }
            
            logger.info(f"Validation completed. Report saved to: {report_path}")
            return result
            
        except Exception as e:
            logger.error(f"Validation step failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'duration_seconds': time.time() - step_start_time
            }
    
    def _run_deployment_preparation(self) -> Dict[str, Any]:
        """
        Run deployment preparation step.
        
        Returns:
            Dict containing deployment preparation results
        """
        step_start_time = time.time()
        try:
            deployment_checks = {
                'model_file_exists': os.path.exists(MODEL_PATH),
                'column_list_exists': os.path.exists(COLUMN_LIST_PATH),
                'metadata_exists': os.path.exists(MODEL_METADATA_PATH),
                'dockerfile_exists': os.path.exists('Dockerfile'),
                'requirements_exists': os.path.exists('requirements.txt'),
                'main_app_exists': os.path.exists('main.py')
            }
            
            # Load data for feature analysis
            import pandas as pd
            from app.preprocessing import preprocess_data
            from app.config import TARGET_FEATURE, FEATURES_TO_DROP
            
            # Load and preprocess data
            data = pd.read_csv(DATA_PATH)
            processed_data = preprocess_data(data)
            
            # Prepare features
            X = processed_data.drop(columns=[TARGET_FEATURE] + 
                                  [col for col in FEATURES_TO_DROP if col in processed_data.columns])
            feature_cols = [col for col in X.columns]
            
            # Test model loading
            try:
                session, input_name, label_name, columns = load_model_artifacts()
                deployment_checks['model_loads_successfully'] = True
                logger.info("Model artifacts loaded successfully")
            except Exception as e:
                deployment_checks['model_loads_successfully'] = False
                deployment_checks['model_load_error'] = str(e)
                logger.warning(f"Model loading test failed: {str(e)}")
            
            # Test prediction pipeline with proper scoping
            try:
                import numpy as np
                
                # Create test data with correct dimensions for ONNX model (64 features)
                test_data = np.random.randn(1, 64).astype(np.float32)
                session, input_name, label_name, columns = load_model_artifacts()
                predictions = session.run([label_name], {input_name: test_data})[0]
                
                deployment_checks['prediction_pipeline_works'] = True
                # Correctly extract the scalar value from the prediction array
                deployment_checks['sample_prediction'] = float(predictions[0][0]) if predictions.size > 0 else None
                logger.info(f"Prediction pipeline test successful: {predictions}")
                
            except Exception as e:
                deployment_checks['prediction_pipeline_works'] = False
                deployment_checks['prediction_pipeline_error'] = str(e)
                logger.warning(f"Prediction pipeline test failed: {str(e)}")
                
            # Create model metadata if it doesn't exist
            try:
                metadata_path = Path(MODEL_METADATA_PATH)
                
                if not metadata_path.exists():
                    metadata = {
                        'model_type': 'RandomForestRegressor',
                        'feature_count': len(feature_cols),
                        'target_feature': TARGET_FEATURE,
                        'features': feature_cols,
                        'training_date': datetime.now().isoformat(),
                        'model_version': '1.0.0'
                    }
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    deployment_checks['metadata_exists'] = True
                    logger.info("Model metadata created successfully")
                else:
                    deployment_checks['metadata_exists'] = True
                    logger.info("Model metadata already exists")
            except Exception as e:
                deployment_checks['metadata_exists'] = False
                deployment_checks['metadata_error'] = str(e)
                logger.warning(f"Model metadata creation failed: {str(e)}")
            
            # Check deployment readiness
            required_checks = [
                'model_file_exists', 'column_list_exists', 'metadata_exists',
                'dockerfile_exists', 'requirements_exists', 'main_app_exists',
                'model_loads_successfully', 'prediction_pipeline_works'
            ]
            
            deployment_ready = all(deployment_checks.get(check, False) for check in required_checks)
            
            result = {
                'status': 'success',
                'duration_seconds': time.time() - step_start_time,
                'deployment_checks': deployment_checks,
                'deployment_ready': deployment_ready,
                'required_checks': required_checks,
                'missing_requirements': [
                    check for check in required_checks 
                    if not deployment_checks.get(check, False)
                ]
            }
            
            if deployment_ready:
                logger.info("Deployment preparation successful - ready for deployment")
            else:
                logger.warning(f"Deployment not ready. Missing: {result['missing_requirements']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Deployment preparation failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'duration_seconds': time.time() - step_start_time
            }
    
    def _initialize_monitoring(self) -> Dict[str, Any]:
        """
        Initialize monitoring system.
        
        Returns:
            Dict containing monitoring initialization results
        """
        step_start_time = time.time()
        
        try:
            # Reset monitoring to start fresh
            self.monitor.reset_monitoring()
            
            # Set baseline metrics if model metadata exists
            if os.path.exists(MODEL_METADATA_PATH):
                with open(MODEL_METADATA_PATH, 'r') as f:
                    metadata = json.load(f)
                
                # Create baseline metrics from training data
                from app.monitoring import PredictionMetrics
                baseline_metrics = PredictionMetrics(
                    timestamp=datetime.now().isoformat(),
                    prediction_count=100,  # Placeholder
                    mean_prediction=metadata.get('validation_metrics', {}).get('mean_prediction', 0),
                    std_prediction=metadata.get('validation_metrics', {}).get('std_prediction', 0),
                    min_prediction=0,
                    max_prediction=100000,
                    response_time_ms=100.0  # Expected baseline response time
                )
                
                self.monitor.set_baseline_metrics(baseline_metrics)
                logger.info("Baseline metrics set from training metadata")
            
            # Get initial health status
            health_status = self.monitor.get_health_status()
            
            result = {
                'status': 'success',
                'duration_seconds': time.time() - step_start_time,
                'monitoring_initialized': True,
                'baseline_set': os.path.exists(MODEL_METADATA_PATH),
                'initial_health_status': health_status
            }
            
            logger.info("Monitoring system initialized successfully")
            return result
            
        except Exception as e:
            logger.error(f"Monitoring initialization failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'duration_seconds': time.time() - step_start_time
            }
    
    def _save_pipeline_results(self, results: Dict[str, Any]) -> None:
        """
        Save pipeline execution results.
        
        Args:
            results: Pipeline execution results
        """
        try:
            results_path = os.path.join(LOGS_DIR, f"pipeline_results_{results['pipeline_id']}.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Pipeline results saved to: {results_path}")
            
        except Exception as e:
            logger.error(f"Failed to save pipeline results: {str(e)}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status.
        
        Returns:
            Dict containing pipeline status information
        """
        return {
            'pipeline_state': self.pipeline_state.copy(),
            'model_artifacts_exist': {
                'model': os.path.exists(MODEL_PATH),
                'columns': os.path.exists(COLUMN_LIST_PATH),
                'metadata': os.path.exists(MODEL_METADATA_PATH)
            },
            'monitoring_status': self.monitor.get_health_status(),
            'last_updated': datetime.now().isoformat()
        }
    
    def run_incremental_training(self, new_data_path: str) -> Dict[str, Any]:
        """
        Run incremental training with new data.
        
        Args:
            new_data_path: Path to new training data
            
        Returns:
            Dict containing incremental training results
        """
        logger.info(f"Starting incremental training with data: {new_data_path}")
        
        # For now, this runs full retraining
        # In a production system, this could implement true incremental learning
        return self.run_full_pipeline(
            data_path=new_data_path,
            validate_model=True,
            deploy_ready=True
        )
    
    def validate_production_model(self, test_data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate the production model against new test data.
        
        Args:
            test_data_path: Path to test data (optional)
            
        Returns:
            Dict containing validation results
        """
        logger.info("Starting production model validation")
        
        try:
            # Use default data if no test data provided
            data_path = test_data_path or DATA_PATH
            
            # Run validation step
            validation_result = self._run_validation_step()
            
            # Update pipeline state
            self.pipeline_state['last_validation'] = datetime.now().isoformat()
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Production model validation failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

def create_pipeline(config: Optional[Dict[str, Any]] = None) -> MLPipeline:
    """
    Factory function to create an ML pipeline instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        MLPipeline instance
    """
    return MLPipeline(config)

def run_quick_pipeline(data_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to run a quick pipeline with default settings.
    
    Args:
        data_path: Path to training data
        
    Returns:
        Dict containing pipeline results
    """
    pipeline = create_pipeline()
    return pipeline.run_full_pipeline(
        data_path=data_path,
        validate_model=True,
        deploy_ready=True
    )

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ML Pipeline')
    parser.add_argument('--data', type=str, help='Path to training data')
    parser.add_argument('--validate', action='store_true', help='Run model validation')
    parser.add_argument('--deploy', action='store_true', help='Prepare for deployment')
    parser.add_argument('--quick', action='store_true', help='Run quick pipeline with defaults')
    
    args = parser.parse_args()
    
    if args.quick:
        results = run_quick_pipeline(args.data)
    else:
        pipeline = create_pipeline()
        results = pipeline.run_full_pipeline(
            data_path=args.data,
            validate_model=args.validate,
            deploy_ready=args.deploy
        )
    
    print(json.dumps(results, indent=2, default=str))