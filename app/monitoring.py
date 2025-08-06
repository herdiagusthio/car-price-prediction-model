import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import threading
from dataclasses import dataclass, asdict
from app.config import PERFORMANCE_THRESHOLDS

logger = logging.getLogger(__name__)

@dataclass
class PredictionMetrics:
    """Data class for storing prediction metrics."""
    timestamp: str
    prediction_count: int
    mean_prediction: float
    std_prediction: float
    min_prediction: float
    max_prediction: float
    response_time_ms: float
    error_count: int = 0
    
class ModelMonitor:
    """
    Real-time monitoring system for ML model performance and data quality.
    """
    
    def __init__(self, window_size: int = 1000, alert_threshold: float = 0.1):
        """
        Initialize the model monitor.
        
        Args:
            window_size: Number of recent predictions to keep in memory
            alert_threshold: Threshold for triggering alerts (e.g., 0.1 = 10% degradation)
        """
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        
        # Thread-safe storage for metrics
        self._lock = threading.Lock()
        self.predictions_history = deque(maxlen=window_size)
        self.metrics_history = deque(maxlen=100)  # Keep last 100 metric snapshots
        self.error_log = deque(maxlen=500)  # Keep last 500 errors
        
        # Performance tracking
        self.baseline_metrics = None
        self.current_metrics = None
        self.alerts = deque(maxlen=50)  # Keep last 50 alerts
        
        # Feature drift tracking
        self.feature_stats = defaultdict(lambda: {'values': deque(maxlen=window_size), 'stats': {}})
        
        logger.info(f"ModelMonitor initialized with window_size={window_size}")
    
    def log_prediction(self, input_features: Dict[str, Any], prediction: float, 
                      response_time_ms: float, error: Optional[str] = None) -> None:
        """
        Log a single prediction with its metadata.
        
        Args:
            input_features: Dictionary of input features
            prediction: Model prediction value
            response_time_ms: Response time in milliseconds
            error: Error message if prediction failed
        """
        with self._lock:
            timestamp = datetime.now().isoformat()
            
            # Log prediction
            prediction_record = {
                'timestamp': timestamp,
                'prediction': prediction,
                'response_time_ms': response_time_ms,
                'features': input_features.copy(),
                'error': error
            }
            
            self.predictions_history.append(prediction_record)
            
            # Log error if present
            if error:
                self.error_log.append({
                    'timestamp': timestamp,
                    'error': error,
                    'features': input_features.copy()
                })
                logger.warning(f"Prediction error logged: {error}")
            
            # Update feature statistics
            self._update_feature_stats(input_features)
            
            # Update current metrics
            self._update_current_metrics()
            
            logger.debug(f"Prediction logged: {prediction:.2f} (response_time: {response_time_ms:.2f}ms)")
    
    def _update_feature_stats(self, features: Dict[str, Any]) -> None:
        """
        Update feature statistics for drift detection.
        
        Args:
            features: Dictionary of input features
        """
        for feature_name, value in features.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                self.feature_stats[feature_name]['values'].append(value)
                
                # Update statistics if we have enough data
                values = list(self.feature_stats[feature_name]['values'])
                if len(values) >= 10:
                    self.feature_stats[feature_name]['stats'] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
    
    def _update_current_metrics(self) -> None:
        """
        Update current performance metrics based on recent predictions.
        """
        if not self.predictions_history:
            return
        
        # Get recent predictions (last hour or last 100 predictions, whichever is smaller)
        recent_predictions = list(self.predictions_history)[-100:]
        valid_predictions = [p for p in recent_predictions if p['error'] is None]
        
        if not valid_predictions:
            return
        
        predictions = [p['prediction'] for p in valid_predictions]
        response_times = [p['response_time_ms'] for p in valid_predictions]
        error_count = len(recent_predictions) - len(valid_predictions)
        
        self.current_metrics = PredictionMetrics(
            timestamp=datetime.now().isoformat(),
            prediction_count=len(valid_predictions),
            mean_prediction=np.mean(predictions),
            std_prediction=np.std(predictions),
            min_prediction=np.min(predictions),
            max_prediction=np.max(predictions),
            response_time_ms=np.mean(response_times),
            error_count=error_count
        )
        
        # Add to metrics history
        self.metrics_history.append(self.current_metrics)
        
        # Check for alerts
        self._check_alerts()
    
    def _check_alerts(self) -> None:
        """
        Check for performance degradation and trigger alerts.
        """
        if not self.current_metrics or not self.baseline_metrics:
            return
        
        alerts = []
        
        # Check prediction distribution drift
        mean_drift = abs(self.current_metrics.mean_prediction - self.baseline_metrics.mean_prediction)
        if mean_drift > self.baseline_metrics.mean_prediction * self.alert_threshold:
            alerts.append({
                'type': 'prediction_drift',
                'severity': 'warning',
                'message': f'Mean prediction drift detected: {mean_drift:.2f}',
                'current_value': self.current_metrics.mean_prediction,
                'baseline_value': self.baseline_metrics.mean_prediction
            })
        
        # Check response time degradation
        response_time_increase = (self.current_metrics.response_time_ms - 
                                self.baseline_metrics.response_time_ms)
        if response_time_increase > self.baseline_metrics.response_time_ms * self.alert_threshold:
            alerts.append({
                'type': 'response_time_degradation',
                'severity': 'warning',
                'message': f'Response time increased by {response_time_increase:.2f}ms',
                'current_value': self.current_metrics.response_time_ms,
                'baseline_value': self.baseline_metrics.response_time_ms
            })
        
        # Check error rate
        error_rate = self.current_metrics.error_count / max(self.current_metrics.prediction_count, 1)
        if error_rate > 0.05:  # 5% error rate threshold
            alerts.append({
                'type': 'high_error_rate',
                'severity': 'critical',
                'message': f'High error rate detected: {error_rate:.2%}',
                'current_value': error_rate,
                'baseline_value': 0.0
            })
        
        # Log and store alerts
        for alert in alerts:
            alert['timestamp'] = datetime.now().isoformat()
            self.alerts.append(alert)
            logger.warning(f"Alert triggered: {alert['message']}")
    
    def set_baseline_metrics(self, metrics: Optional[PredictionMetrics] = None) -> None:
        """
        Set baseline metrics for comparison.
        
        Args:
            metrics: Baseline metrics. If None, uses current metrics as baseline.
        """
        if metrics is None:
            metrics = self.current_metrics
        
        if metrics:
            self.baseline_metrics = metrics
            logger.info(f"Baseline metrics set: mean_prediction={metrics.mean_prediction:.2f}")
        else:
            logger.warning("Cannot set baseline metrics: no current metrics available")
    
    def detect_feature_drift(self, feature_name: str, baseline_stats: Dict[str, float]) -> Dict[str, Any]:
        """
        Detect drift in a specific feature.
        
        Args:
            feature_name: Name of the feature to check
            baseline_stats: Baseline statistics for comparison
            
        Returns:
            Dict containing drift analysis results
        """
        if feature_name not in self.feature_stats:
            return {
                'feature_name': feature_name,
                'drift_detected': False,
                'error': 'Feature not found in monitoring data'
            }
        
        current_stats = self.feature_stats[feature_name]['stats']
        if not current_stats:
            return {
                'feature_name': feature_name,
                'drift_detected': False,
                'error': 'Insufficient data for drift detection'
            }
        
        # Calculate drift metrics
        mean_drift = abs(current_stats['mean'] - baseline_stats.get('mean', 0))
        std_drift = abs(current_stats['std'] - baseline_stats.get('std', 0))
        
        # Normalize by baseline values
        baseline_mean = baseline_stats.get('mean', 1)
        baseline_std = baseline_stats.get('std', 1)
        
        normalized_mean_drift = mean_drift / (abs(baseline_mean) + 1e-8)
        normalized_std_drift = std_drift / (baseline_std + 1e-8)
        
        # Determine if drift is significant
        drift_threshold = 0.2  # 20% change threshold
        drift_detected = (normalized_mean_drift > drift_threshold or 
                         normalized_std_drift > drift_threshold)
        
        return {
            'feature_name': feature_name,
            'drift_detected': drift_detected,
            'mean_drift': mean_drift,
            'std_drift': std_drift,
            'normalized_mean_drift': normalized_mean_drift,
            'normalized_std_drift': normalized_std_drift,
            'current_stats': current_stats,
            'baseline_stats': baseline_stats,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive performance summary.
        
        Returns:
            Dict containing performance metrics and statistics
        """
        with self._lock:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'monitoring_window_size': self.window_size,
                'total_predictions': len(self.predictions_history),
                'current_metrics': asdict(self.current_metrics) if self.current_metrics else None,
                'baseline_metrics': asdict(self.baseline_metrics) if self.baseline_metrics else None,
                'recent_alerts': list(self.alerts)[-10:],  # Last 10 alerts
                'error_summary': self._get_error_summary(),
                'feature_drift_summary': self._get_feature_drift_summary(),
                'performance_trends': self._get_performance_trends()
            }
        
        return summary
    
    def _get_error_summary(self) -> Dict[str, Any]:
        """
        Get summary of recent errors.
        
        Returns:
            Dict containing error statistics
        """
        if not self.error_log:
            return {'total_errors': 0, 'error_rate': 0.0, 'recent_errors': []}
        
        total_predictions = len(self.predictions_history)
        total_errors = len(self.error_log)
        error_rate = total_errors / max(total_predictions, 1)
        
        # Group errors by type
        error_types = defaultdict(int)
        for error in self.error_log:
            error_types[error['error']] += 1
        
        return {
            'total_errors': total_errors,
            'error_rate': error_rate,
            'error_types': dict(error_types),
            'recent_errors': list(self.error_log)[-5:]  # Last 5 errors
        }
    
    def _get_feature_drift_summary(self) -> Dict[str, Any]:
        """
        Get summary of feature drift across all monitored features.
        
        Returns:
            Dict containing feature drift statistics
        """
        drift_summary = {
            'monitored_features': len(self.feature_stats),
            'features_with_data': 0,
            'feature_statistics': {}
        }
        
        for feature_name, feature_data in self.feature_stats.items():
            if feature_data['stats']:
                drift_summary['features_with_data'] += 1
                drift_summary['feature_statistics'][feature_name] = feature_data['stats']
        
        return drift_summary
    
    def _get_performance_trends(self) -> Dict[str, Any]:
        """
        Get performance trends over time.
        
        Returns:
            Dict containing trend analysis
        """
        if len(self.metrics_history) < 2:
            return {'trend_analysis': 'Insufficient data for trend analysis'}
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 snapshots
        
        # Calculate trends
        response_times = [m.response_time_ms for m in recent_metrics]
        prediction_means = [m.mean_prediction for m in recent_metrics]
        error_counts = [m.error_count for m in recent_metrics]
        
        return {
            'response_time_trend': {
                'current': response_times[-1],
                'average': np.mean(response_times),
                'trend': 'increasing' if response_times[-1] > np.mean(response_times[:-1]) else 'stable'
            },
            'prediction_stability': {
                'current_mean': prediction_means[-1],
                'std_deviation': np.std(prediction_means),
                'trend': 'stable' if np.std(prediction_means) < np.mean(prediction_means) * 0.1 else 'volatile'
            },
            'error_trend': {
                'current_errors': error_counts[-1],
                'average_errors': np.mean(error_counts),
                'trend': 'increasing' if error_counts[-1] > np.mean(error_counts[:-1]) else 'stable'
            }
        }
    
    def export_metrics(self, filepath: str, format: str = 'json') -> None:
        """
        Export monitoring metrics to file.
        
        Args:
            filepath: Path to save the metrics
            format: Export format ('json' or 'csv')
        """
        summary = self.get_performance_summary()
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        elif format.lower() == 'csv':
            # Export predictions history as CSV
            if self.predictions_history:
                df = pd.DataFrame(list(self.predictions_history))
                df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Metrics exported to {filepath} in {format} format")
    
    def reset_monitoring(self) -> None:
        """
        Reset all monitoring data.
        """
        with self._lock:
            self.predictions_history.clear()
            self.metrics_history.clear()
            self.error_log.clear()
            self.alerts.clear()
            self.feature_stats.clear()
            self.current_metrics = None
            self.baseline_metrics = None
        
        logger.info("Monitoring data reset")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get overall health status of the model.
        
        Returns:
            Dict containing health status and recommendations
        """
        if not self.current_metrics:
            return {
                'status': 'unknown',
                'message': 'Insufficient data for health assessment',
                'recommendations': ['Start making predictions to gather monitoring data']
            }
        
        health_score = 100  # Start with perfect score
        issues = []
        recommendations = []
        
        # Check error rate
        error_rate = self.current_metrics.error_count / max(self.current_metrics.prediction_count, 1)
        if error_rate > 0.05:
            health_score -= 30
            issues.append(f'High error rate: {error_rate:.2%}')
            recommendations.append('Investigate prediction errors and model stability')
        
        # Check response time
        if self.current_metrics.response_time_ms > 1000:  # 1 second threshold
            health_score -= 20
            issues.append(f'Slow response time: {self.current_metrics.response_time_ms:.2f}ms')
            recommendations.append('Optimize model inference or infrastructure')
        
        # Check for recent alerts
        recent_alerts = [a for a in self.alerts if a.get('severity') == 'critical']
        if recent_alerts:
            health_score -= 25
            issues.append(f'{len(recent_alerts)} critical alerts in recent history')
            recommendations.append('Address critical alerts immediately')
        
        # Check prediction stability
        if self.current_metrics.std_prediction > self.current_metrics.mean_prediction * 0.5:
            health_score -= 15
            issues.append('High prediction variance detected')
            recommendations.append('Review model consistency and input data quality')
        
        # Determine status
        if health_score >= 90:
            status = 'excellent'
        elif health_score >= 75:
            status = 'good'
        elif health_score >= 50:
            status = 'warning'
        else:
            status = 'critical'
        
        return {
            'status': status,
            'health_score': max(0, health_score),
            'issues': issues,
            'recommendations': recommendations,
            'last_updated': datetime.now().isoformat(),
            'metrics_summary': {
                'total_predictions': self.current_metrics.prediction_count,
                'error_rate': error_rate,
                'avg_response_time': self.current_metrics.response_time_ms,
                'prediction_range': {
                    'min': self.current_metrics.min_prediction,
                    'max': self.current_metrics.max_prediction,
                    'mean': self.current_metrics.mean_prediction
                }
            }
        }

# Global monitor instance
_monitor_instance = None

def get_monitor() -> ModelMonitor:
    """
    Get the global monitor instance (singleton pattern).
    
    Returns:
        ModelMonitor instance
    """
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ModelMonitor()
    return _monitor_instance

def log_prediction(input_features: Dict[str, Any], prediction: float, 
                  response_time_ms: float, error: Optional[str] = None) -> None:
    """
    Convenience function to log a prediction to the global monitor.
    
    Args:
        input_features: Dictionary of input features
        prediction: Model prediction value
        response_time_ms: Response time in milliseconds
        error: Error message if prediction failed
    """
    monitor = get_monitor()
    monitor.log_prediction(input_features, prediction, response_time_ms, error)