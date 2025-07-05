"""
Predictive Analytics Engine

Advanced predictive analytics capabilities for UniMind.
Provides time series forecasting, regression analysis, classification models, and predictive modeling.
"""

import asyncio
import logging
import time
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
from datetime import datetime, timedelta
import hashlib

# Analytics dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from scipy import stats
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
    from sklearn.model_selection import train_test_split, cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ModelType(Enum):
    """Types of predictive models."""
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    LOGISTIC_REGRESSION = "logistic_regression"
    TIME_SERIES = "time_series"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"


class ForecastMethod(Enum):
    """Time series forecasting methods."""
    ARIMA = "arima"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    LINEAR_TREND = "linear_trend"
    SEASONAL_DECOMPOSITION = "seasonal_decomposition"
    NEURAL_FORECAST = "neural_forecast"


@dataclass
class PredictionModel:
    """A predictive model."""
    model_id: str
    model_type: ModelType
    target_variable: str
    features: List[str]
    model_object: Any
    training_data_size: int
    accuracy_metrics: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ForecastResult:
    """Result of time series forecasting."""
    target_variable: str
    forecast_values: List[float]
    confidence_intervals: List[Tuple[float, float]]
    forecast_horizon: int
    method: ForecastMethod
    accuracy_metrics: Dict[str, float]
    seasonality_detected: bool
    trend_direction: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegressionResult:
    """Result of regression analysis."""
    target_variable: str
    features: List[str]
    coefficients: Dict[str, float]
    intercept: float
    r_squared: float
    mse: float
    feature_importance: Dict[str, float]
    predictions: List[float]
    residuals: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassificationResult:
    """Result of classification analysis."""
    target_variable: str
    features: List[str]
    classes: List[str]
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    feature_importance: Dict[str, float]
    predictions: List[str]
    probabilities: List[List[float]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class PredictiveAnalyticsEngine:
    """
    Advanced predictive analytics engine for UniMind.
    
    Provides time series forecasting, regression analysis, classification,
    and comprehensive predictive modeling capabilities.
    """
    
    def __init__(self):
        """Initialize the predictive analytics engine."""
        self.logger = logging.getLogger('PredictiveAnalyticsEngine')
        
        # Model registry
        self.models: Dict[str, PredictionModel] = {}
        self.forecast_models: Dict[str, Any] = {}
        self.regression_models: Dict[str, Any] = {}
        self.classification_models: Dict[str, Any] = {}
        
        # Data storage
        self.time_series_data: Dict[str, pd.DataFrame] = {}
        self.tabular_data: Dict[str, pd.DataFrame] = {}
        
        # Performance metrics
        self.metrics = {
            'total_models': 0,
            'total_forecasts': 0,
            'total_regressions': 0,
            'total_classifications': 0,
            'avg_forecast_accuracy': 0.0,
            'avg_regression_r2': 0.0,
            'avg_classification_accuracy': 0.0
        }
        
        # Threading
        self.lock = threading.RLock()
        
        # Check dependencies
        self.pandas_available = PANDAS_AVAILABLE
        self.scipy_available = SCIPY_AVAILABLE
        self.sklearn_available = SKLEARN_AVAILABLE
        
        if not self.pandas_available:
            self.logger.warning("Pandas not available - predictive analytics capabilities limited")
        
        self.logger.info("Predictive analytics engine initialized")
    
    def add_time_series_data(self, data_id: str, timestamps: List[datetime], 
                           values: List[float], metadata: Dict[str, Any] = None) -> str:
        """Add time series data for forecasting."""
        if not self.pandas_available:
            raise RuntimeError("Pandas not available")
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': values
        })
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        with self.lock:
            self.time_series_data[data_id] = df
        
        self.logger.info(f"Added time series data: {data_id} with {len(values)} points")
        return data_id
    
    def add_tabular_data(self, data_id: str, data: Dict[str, List[Any]], 
                        metadata: Dict[str, Any] = None) -> str:
        """Add tabular data for regression/classification."""
        if not self.pandas_available:
            raise RuntimeError("Pandas not available")
        
        df = pd.DataFrame(data)
        
        with self.lock:
            self.tabular_data[data_id] = df
        
        self.logger.info(f"Added tabular data: {data_id} with {len(df)} rows")
        return data_id
    
    async def forecast_time_series(self, data_id: str, 
                                 forecast_horizon: int = 12,
                                 method: ForecastMethod = ForecastMethod.LINEAR_TREND) -> ForecastResult:
        """Forecast time series data."""
        if not self.pandas_available:
            raise RuntimeError("Pandas not available")
        
        if data_id not in self.time_series_data:
            raise ValueError(f"Data ID {data_id} not found")
        
        start_time = time.time()
        
        df = self.time_series_data[data_id]
        values = df['value'].values
        
        # Perform forecasting based on method
        if method == ForecastMethod.LINEAR_TREND:
            forecast_values, confidence_intervals = self._linear_trend_forecast(values, forecast_horizon)
        elif method == ForecastMethod.EXPONENTIAL_SMOOTHING:
            forecast_values, confidence_intervals = self._exponential_smoothing_forecast(values, forecast_horizon)
        elif method == ForecastMethod.SEASONAL_DECOMPOSITION:
            forecast_values, confidence_intervals = self._seasonal_decomposition_forecast(values, forecast_horizon)
        else:
            forecast_values, confidence_intervals = self._linear_trend_forecast(values, forecast_horizon)
        
        # Calculate accuracy metrics
        accuracy_metrics = self._calculate_forecast_accuracy(values, forecast_values[:len(values)])
        
        # Detect seasonality and trend
        seasonality_detected = self._detect_seasonality(values)
        trend_direction = self._detect_trend_direction(values)
        
        result = ForecastResult(
            target_variable=data_id,
            forecast_values=forecast_values.tolist(),
            confidence_intervals=confidence_intervals,
            forecast_horizon=forecast_horizon,
            method=method,
            accuracy_metrics=accuracy_metrics,
            seasonality_detected=seasonality_detected,
            trend_direction=trend_direction,
            metadata={'data_id': data_id, 'method': method.value}
        )
        
        # Update metrics
        with self.lock:
            self.metrics['total_forecasts'] += 1
            self.metrics['avg_forecast_accuracy'] = (
                (self.metrics['avg_forecast_accuracy'] * (self.metrics['total_forecasts'] - 1) + accuracy_metrics['mape']) /
                self.metrics['total_forecasts']
            )
        
        self.logger.info(f"Forecast completed in {time.time() - start_time:.2f}s")
        return result
    
    def _linear_trend_forecast(self, values: np.ndarray, horizon: int) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """Linear trend forecasting."""
        x = np.arange(len(values))
        
        # Fit linear trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Generate forecast
        future_x = np.arange(len(values), len(values) + horizon)
        forecast = slope * future_x + intercept
        
        # Calculate confidence intervals
        confidence_intervals = []
        for i, pred in enumerate(forecast):
            margin = std_err * np.sqrt(1 + (future_x[i] - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
            confidence_intervals.append((pred - 1.96 * margin, pred + 1.96 * margin))
        
        return forecast, confidence_intervals
    
    def _exponential_smoothing_forecast(self, values: np.ndarray, horizon: int) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """Exponential smoothing forecasting."""
        alpha = 0.3  # Smoothing parameter
        
        # Simple exponential smoothing
        smoothed = [values[0]]
        for i in range(1, len(values)):
            smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[i-1])
        
        # Forecast
        forecast = []
        last_value = smoothed[-1]
        for _ in range(horizon):
            forecast.append(last_value)
        
        # Simple confidence intervals
        confidence_intervals = []
        std_dev = np.std(values)
        for pred in forecast:
            confidence_intervals.append((pred - 1.96 * std_dev, pred + 1.96 * std_dev))
        
        return np.array(forecast), confidence_intervals
    
    def _seasonal_decomposition_forecast(self, values: np.ndarray, horizon: int) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """Seasonal decomposition forecasting."""
        # Simple seasonal decomposition
        n = len(values)
        period = min(12, n // 4)  # Assume seasonal period
        
        if period < 2:
            return self._linear_trend_forecast(values, horizon)
        
        # Calculate trend
        trend = np.convolve(values, np.ones(period)/period, mode='valid')
        
        # Calculate seasonal component
        seasonal = []
        for i in range(period):
            seasonal_values = [values[j] for j in range(i, n, period)]
            seasonal.append(np.mean(seasonal_values))
        
        # Forecast
        forecast = []
        for i in range(horizon):
            trend_val = trend[-1] if len(trend) > 0 else np.mean(values)
            seasonal_val = seasonal[i % period]
            forecast.append(trend_val + seasonal_val)
        
        # Confidence intervals
        confidence_intervals = []
        std_dev = np.std(values)
        for pred in forecast:
            confidence_intervals.append((pred - 1.96 * std_dev, pred + 1.96 * std_dev))
        
        return np.array(forecast), confidence_intervals
    
    def _calculate_forecast_accuracy(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate forecast accuracy metrics."""
        if len(actual) != len(predicted):
            min_len = min(len(actual), len(predicted))
            actual = actual[:min_len]
            predicted = predicted[:min_len]
        
        mse = mean_squared_error(actual, predicted)
        mae = np.mean(np.abs(actual - predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100 if np.any(actual != 0) else 0
        
        return {
            'mse': mse,
            'mae': mae,
            'mape': mape,
            'r2': r2_score(actual, predicted) if len(actual) > 1 else 0
        }
    
    def _detect_seasonality(self, values: np.ndarray) -> bool:
        """Detect seasonality in time series."""
        if len(values) < 24:
            return False
        
        # Simple autocorrelation check
        try:
            autocorr = np.correlate(values, values, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Check for peaks at regular intervals
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(autocorr[:len(autocorr)//2])
            return len(peaks) > 1
        except:
            return False
    
    def _detect_trend_direction(self, values: np.ndarray) -> str:
        """Detect trend direction in time series."""
        if len(values) < 2:
            return "stable"
        
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    async def perform_regression(self, data_id: str, target_variable: str,
                               features: List[str],
                               model_type: ModelType = ModelType.LINEAR_REGRESSION) -> RegressionResult:
        """Perform regression analysis."""
        if not self.sklearn_available or not self.pandas_available:
            raise RuntimeError("Required dependencies not available")
        
        if data_id not in self.tabular_data:
            raise ValueError(f"Data ID {data_id} not found")
        
        start_time = time.time()
        
        df = self.tabular_data[data_id]
        
        # Prepare data
        X = df[features].values
        y = df[target_variable].values
        
        # Handle missing values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        if len(X) < 10:
            raise ValueError("Insufficient data for regression")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        if model_type == ModelType.LINEAR_REGRESSION:
            model = LinearRegression()
        elif model_type == ModelType.RANDOM_FOREST:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = LinearRegression()
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Get coefficients
        coefficients = {}
        if hasattr(model, 'coef_'):
            for i, feature in enumerate(features):
                coefficients[feature] = float(model.coef_[i])
            intercept = float(model.intercept_)
        else:
            coefficients = {feature: 0.0 for feature in features}
            intercept = 0.0
        
        # Feature importance
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            for i, feature in enumerate(features):
                feature_importance[feature] = float(model.feature_importances_[i])
        else:
            feature_importance = {feature: 1.0/len(features) for feature in features}
        
        # Calculate residuals
        residuals = (y_test - y_pred).tolist()
        
        result = RegressionResult(
            target_variable=target_variable,
            features=features,
            coefficients=coefficients,
            intercept=intercept,
            r_squared=r2,
            mse=mse,
            feature_importance=feature_importance,
            predictions=y_pred.tolist(),
            residuals=residuals,
            metadata={'data_id': data_id, 'model_type': model_type.value}
        )
        
        # Store model
        model_id = f"regression_{data_id}_{target_variable}"
        prediction_model = PredictionModel(
            model_id=model_id,
            model_type=model_type,
            target_variable=target_variable,
            features=features,
            model_object=model,
            training_data_size=len(X_train),
            accuracy_metrics={'r2': r2, 'mse': mse}
        )
        
        with self.lock:
            self.models[model_id] = prediction_model
            self.metrics['total_regressions'] += 1
            self.metrics['avg_regression_r2'] = (
                (self.metrics['avg_regression_r2'] * (self.metrics['total_regressions'] - 1) + r2) /
                self.metrics['total_regressions']
            )
        
        self.logger.info(f"Regression completed in {time.time() - start_time:.2f}s")
        return result
    
    async def perform_classification(self, data_id: str, target_variable: str,
                                   features: List[str],
                                   model_type: ModelType = ModelType.LOGISTIC_REGRESSION) -> ClassificationResult:
        """Perform classification analysis."""
        if not self.sklearn_available or not self.pandas_available:
            raise RuntimeError("Required dependencies not available")
        
        if data_id not in self.tabular_data:
            raise ValueError(f"Data ID {data_id} not found")
        
        start_time = time.time()
        
        df = self.tabular_data[data_id]
        
        # Prepare data
        X = df[features].values
        y = df[target_variable].values
        
        # Handle missing values
        mask = ~(np.isnan(X).any(axis=1) | pd.isna(y))
        X = X[mask]
        y = y[mask]
        
        if len(X) < 10:
            raise ValueError("Insufficient data for classification")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        if model_type == ModelType.LOGISTIC_REGRESSION:
            model = LogisticRegression(random_state=42)
        elif model_type == ModelType.RANDOM_FOREST:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = LogisticRegression(random_state=42)
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1 = report['weighted avg']['f1-score']
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrix_list = cm.tolist()
        
        # Feature importance
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            for i, feature in enumerate(features):
                feature_importance[feature] = float(model.feature_importances_[i])
        else:
            feature_importance = {feature: 1.0/len(features) for feature in features}
        
        # Get classes
        classes = list(model.classes_) if hasattr(model, 'classes_') else list(set(y))
        
        result = ClassificationResult(
            target_variable=target_variable,
            features=features,
            classes=classes,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=confusion_matrix_list,
            feature_importance=feature_importance,
            predictions=y_pred.tolist(),
            probabilities=y_prob.tolist(),
            metadata={'data_id': data_id, 'model_type': model_type.value}
        )
        
        # Store model
        model_id = f"classification_{data_id}_{target_variable}"
        prediction_model = PredictionModel(
            model_id=model_id,
            model_type=model_type,
            target_variable=target_variable,
            features=features,
            model_object=model,
            training_data_size=len(X_train),
            accuracy_metrics={'accuracy': accuracy, 'f1': f1}
        )
        
        with self.lock:
            self.models[model_id] = prediction_model
            self.metrics['total_classifications'] += 1
            self.metrics['avg_classification_accuracy'] = (
                (self.metrics['avg_classification_accuracy'] * (self.metrics['total_classifications'] - 1) + accuracy) /
                self.metrics['total_classifications']
            )
        
        self.logger.info(f"Classification completed in {time.time() - start_time:.2f}s")
        return result
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        if model_id not in self.models:
            return None
        
        model = self.models[model_id]
        return {
            'model_id': model.model_id,
            'model_type': model.model_type.value,
            'target_variable': model.target_variable,
            'features': model.features,
            'training_data_size': model.training_data_size,
            'accuracy_metrics': model.accuracy_metrics,
            'created_at': model.created_at.isoformat(),
            'metadata': model.metadata
        }
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all models."""
        with self.lock:
            return [
                {
                    'model_id': model.model_id,
                    'model_type': model.model_type.value,
                    'target_variable': model.target_variable,
                    'features_count': len(model.features),
                    'training_data_size': model.training_data_size,
                    'accuracy': list(model.accuracy_metrics.values())[0] if model.accuracy_metrics else 0.0
                }
                for model in self.models.values()
            ]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get predictive analytics system status."""
        with self.lock:
            return {
                'metrics': self.metrics.copy(),
                'total_models': len(self.models),
                'time_series_datasets': len(self.time_series_data),
                'tabular_datasets': len(self.tabular_data),
                'pandas_available': self.pandas_available,
                'scipy_available': self.scipy_available,
                'sklearn_available': self.sklearn_available
            }


# Global instance
predictive_analytics_engine = PredictiveAnalyticsEngine() 