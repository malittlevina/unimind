"""
Advanced Analytics Engine

Advanced analytics capabilities for UniMind.
Provides predictive analytics, anomaly detection, market intelligence, and business analytics.
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
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class AnalyticsType(Enum):
    """Types of analytics."""
    PREDICTIVE = "predictive"
    ANOMALY_DETECTION = "anomaly_detection"
    MARKET_INTELLIGENCE = "market_intelligence"
    TREND_ANALYSIS = "trend_analysis"
    FORECASTING = "forecasting"
    CLUSTERING = "clustering"
    CORRELATION = "correlation"
    REGRESSION = "regression"


class AnomalyType(Enum):
    """Types of anomalies."""
    POINT_ANOMALY = "point_anomaly"
    CONTEXTUAL_ANOMALY = "contextual_anomaly"
    COLLECTIVE_ANOMALY = "collective_anomaly"
    TREND_ANOMALY = "trend_anomaly"
    SEASONAL_ANOMALY = "seasonal_anomaly"


@dataclass
class TimeSeriesData:
    """Time series data structure."""
    timestamps: List[datetime]
    values: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Result of predictive analysis."""
    target_variable: str
    predicted_values: List[float]
    confidence_intervals: List[Tuple[float, float]]
    model_performance: Dict[str, float]
    feature_importance: Dict[str, float]
    prediction_horizon: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    anomaly_type: AnomalyType
    anomaly_indices: List[int]
    anomaly_scores: List[float]
    threshold: float
    confidence: float
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketIntelligenceResult:
    """Result of market intelligence analysis."""
    market_trends: List[Dict[str, Any]]
    competitive_analysis: Dict[str, Any]
    market_opportunities: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    recommendations: List[str]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendAnalysisResult:
    """Result of trend analysis."""
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float
    trend_duration: int
    seasonality: Dict[str, Any]
    breakpoints: List[int]
    forecast: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedAnalyticsEngine:
    """
    Advanced analytics engine for UniMind.
    
    Provides predictive analytics, anomaly detection, market intelligence,
    and comprehensive business analytics capabilities.
    """
    
    def __init__(self):
        """Initialize the advanced analytics engine."""
        self.logger = logging.getLogger('AdvancedAnalyticsEngine')
        
        # Analytics models
        self.prediction_models: Dict[str, Any] = {}
        self.anomaly_detectors: Dict[str, Any] = {}
        self.trend_analyzers: Dict[str, Any] = {}
        
        # Data storage
        self.time_series_data: Dict[str, TimeSeriesData] = {}
        self.market_data: Dict[str, Dict[str, Any]] = {}
        
        # Analysis history
        self.prediction_history: List[PredictionResult] = []
        self.anomaly_history: List[AnomalyResult] = []
        self.market_intelligence_history: List[MarketIntelligenceResult] = []
        
        # Performance metrics
        self.metrics = {
            'total_predictions': 0,
            'total_anomalies_detected': 0,
            'total_market_analyses': 0,
            'avg_prediction_accuracy': 0.0,
            'avg_anomaly_detection_rate': 0.0,
            'avg_analysis_time': 0.0
        }
        
        # Threading
        self.lock = threading.RLock()
        
        # Check dependencies
        self.pandas_available = PANDAS_AVAILABLE
        self.scipy_available = SCIPY_AVAILABLE
        self.sklearn_available = SKLEARN_AVAILABLE
        
        if not self.pandas_available:
            self.logger.warning("Pandas not available - analytics capabilities limited")
        
        self.logger.info("Advanced analytics engine initialized")
    
    def add_time_series_data(self, data_id: str, timestamps: List[datetime], 
                           values: List[float], metadata: Dict[str, Any] = None) -> str:
        """Add time series data for analysis."""
        if not self.pandas_available:
            raise RuntimeError("Pandas not available")
        
        if len(timestamps) != len(values):
            raise ValueError("Timestamps and values must have same length")
        
        time_series = TimeSeriesData(
            timestamps=timestamps,
            values=values,
            metadata=metadata or {}
        )
        
        with self.lock:
            self.time_series_data[data_id] = time_series
        
        self.logger.info(f"Added time series data: {data_id} with {len(values)} points")
        return data_id
    
    async def predict_values(self, data_id: str, target_variable: str = None,
                           prediction_horizon: int = 10, 
                           method: str = "linear_regression") -> PredictionResult:
        """Predict future values using time series data."""
        if not self.pandas_available or not self.sklearn_available:
            raise RuntimeError("Required dependencies not available")
        
        start_time = time.time()
        
        if data_id not in self.time_series_data:
            raise ValueError(f"Data ID {data_id} not found")
        
        time_series = self.time_series_data[data_id]
        
        # Convert to pandas DataFrame
        df = pd.DataFrame({
            'timestamp': time_series.timestamps,
            'value': time_series.values
        })
        
        # Create features for prediction
        df['time_index'] = range(len(df))
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        
        # Prepare training data
        X = df[['time_index', 'day_of_week', 'month', 'year']].values[:-prediction_horizon]
        y = df['value'].values[:-prediction_horizon]
        
        # Train prediction model
        if method == "linear_regression":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        elif method == "random_forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = LinearRegression()  # Default
        
        model.fit(X, y)
        
        # Make predictions
        future_X = df[['time_index', 'day_of_week', 'month', 'year']].values[-prediction_horizon:]
        predicted_values = model.predict(future_X)
        
        # Calculate confidence intervals (simplified)
        confidence_intervals = []
        for pred in predicted_values:
            margin = pred * 0.1  # 10% margin
            confidence_intervals.append((pred - margin, pred + margin))
        
        # Calculate model performance
        y_pred_train = model.predict(X)
        mse = mean_squared_error(y, y_pred_train)
        r2 = r2_score(y, y_pred_train)
        
        # Feature importance (for Random Forest)
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            features = ['time_index', 'day_of_week', 'month', 'year']
            for feature, importance in zip(features, model.feature_importances_):
                feature_importance[feature] = float(importance)
        else:
            feature_importance = {'time_index': 1.0}
        
        result = PredictionResult(
            target_variable=target_variable or data_id,
            predicted_values=predicted_values.tolist(),
            confidence_intervals=confidence_intervals,
            model_performance={'mse': mse, 'r2': r2},
            feature_importance=feature_importance,
            prediction_horizon=prediction_horizon,
            metadata={'method': method, 'data_id': data_id}
        )
        
        # Store result
        with self.lock:
            self.prediction_history.append(result)
            self.metrics['total_predictions'] += 1
            self.metrics['avg_prediction_accuracy'] = (
                (self.metrics['avg_prediction_accuracy'] * (self.metrics['total_predictions'] - 1) + r2) /
                self.metrics['total_predictions']
            )
        
        self.logger.info(f"Prediction completed in {time.time() - start_time:.2f}s")
        return result
    
    async def detect_anomalies(self, data_id: str, 
                             method: str = "isolation_forest",
                             threshold: float = 0.95) -> AnomalyResult:
        """Detect anomalies in time series data."""
        if not self.pandas_available or not self.sklearn_available:
            raise RuntimeError("Required dependencies not available")
        
        start_time = time.time()
        
        if data_id not in self.time_series_data:
            raise ValueError(f"Data ID {data_id} not found")
        
        time_series = self.time_series_data[data_id]
        values = np.array(time_series.values)
        
        # Detect anomalies based on method
        if method == "isolation_forest":
            anomaly_scores, anomaly_indices = self._isolation_forest_detection(values, threshold)
        elif method == "statistical":
            anomaly_scores, anomaly_indices = self._statistical_anomaly_detection(values, threshold)
        elif method == "zscore":
            anomaly_scores, anomaly_indices = self._zscore_anomaly_detection(values, threshold)
        else:
            anomaly_scores, anomaly_indices = self._isolation_forest_detection(values, threshold)
        
        # Determine anomaly type
        anomaly_type = self._classify_anomaly_type(values, anomaly_indices)
        
        # Generate explanation
        explanation = self._generate_anomaly_explanation(anomaly_type, len(anomaly_indices), len(values))
        
        result = AnomalyResult(
            anomaly_type=anomaly_type,
            anomaly_indices=anomaly_indices,
            anomaly_scores=anomaly_scores.tolist(),
            threshold=threshold,
            confidence=len(anomaly_indices) / len(values) if len(values) > 0 else 0.0,
            explanation=explanation,
            metadata={'method': method, 'data_id': data_id}
        )
        
        # Store result
        with self.lock:
            self.anomaly_history.append(result)
            self.metrics['total_anomalies_detected'] += len(anomaly_indices)
        
        self.logger.info(f"Anomaly detection completed in {time.time() - start_time:.2f}s")
        return result
    
    def _isolation_forest_detection(self, values: np.ndarray, threshold: float) -> Tuple[np.ndarray, List[int]]:
        """Detect anomalies using Isolation Forest."""
        # Reshape for sklearn
        X = values.reshape(-1, 1)
        
        # Fit isolation forest
        iso_forest = IsolationForest(contamination=1-threshold, random_state=42)
        anomaly_scores = iso_forest.fit_predict(X)
        
        # Find anomaly indices
        anomaly_indices = np.where(anomaly_scores == -1)[0].tolist()
        
        return anomaly_scores, anomaly_indices
    
    def _statistical_anomaly_detection(self, values: np.ndarray, threshold: float) -> Tuple[np.ndarray, List[int]]:
        """Detect anomalies using statistical methods."""
        # Calculate moving average and standard deviation
        window = min(10, len(values) // 4)
        if window < 2:
            window = 2
        
        moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
        moving_std = np.array([np.std(values[max(0, i-window):i+1]) for i in range(len(values))])
        
        # Calculate z-scores
        z_scores = np.abs((values - moving_avg) / (moving_std + 1e-8))
        
        # Find anomalies
        anomaly_threshold = stats.norm.ppf(threshold)
        anomaly_indices = np.where(z_scores > anomaly_threshold)[0].tolist()
        
        return z_scores, anomaly_indices
    
    def _zscore_anomaly_detection(self, values: np.ndarray, threshold: float) -> Tuple[np.ndarray, List[int]]:
        """Detect anomalies using z-score method."""
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        z_scores = np.abs((values - mean_val) / (std_val + 1e-8))
        
        # Find anomalies
        anomaly_threshold = stats.norm.ppf(threshold)
        anomaly_indices = np.where(z_scores > anomaly_threshold)[0].tolist()
        
        return z_scores, anomaly_indices
    
    def _classify_anomaly_type(self, values: np.ndarray, anomaly_indices: List[int]) -> AnomalyType:
        """Classify the type of anomaly."""
        if not anomaly_indices:
            return AnomalyType.POINT_ANOMALY
        
        # Check for collective anomalies
        if len(anomaly_indices) > len(values) * 0.1:  # More than 10% anomalies
            return AnomalyType.COLLECTIVE_ANOMALY
        
        # Check for trend anomalies
        if len(anomaly_indices) > 1:
            # Check if anomalies are consecutive
            consecutive_count = 0
            for i in range(len(anomaly_indices) - 1):
                if anomaly_indices[i+1] - anomaly_indices[i] == 1:
                    consecutive_count += 1
            
            if consecutive_count > len(anomaly_indices) * 0.5:
                return AnomalyType.TREND_ANOMALY
        
        # Check for seasonal anomalies
        if len(values) > 24:  # Enough data for seasonal analysis
            # Simple seasonal check
            seasonal_pattern = self._check_seasonality(values)
            if seasonal_pattern:
                return AnomalyType.SEASONAL_ANOMALY
        
        return AnomalyType.POINT_ANOMALY
    
    def _check_seasonality(self, values: np.ndarray) -> bool:
        """Check for seasonality in the data."""
        if len(values) < 24:
            return False
        
        # Simple autocorrelation check
        try:
            autocorr = np.correlate(values, values, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Check for peaks at regular intervals
            peaks, _ = find_peaks(autocorr[:len(autocorr)//2])
            if len(peaks) > 1:
                return True
        except:
            pass
        
        return False
    
    def _generate_anomaly_explanation(self, anomaly_type: AnomalyType, 
                                    num_anomalies: int, total_points: int) -> str:
        """Generate explanation for detected anomalies."""
        percentage = (num_anomalies / total_points) * 100 if total_points > 0 else 0
        
        explanations = {
            AnomalyType.POINT_ANOMALY: f"Detected {num_anomalies} point anomalies ({percentage:.1f}% of data)",
            AnomalyType.CONTEXTUAL_ANOMALY: f"Detected {num_anomalies} contextual anomalies",
            AnomalyType.COLLECTIVE_ANOMALY: f"Detected collective anomaly pattern with {num_anomalies} affected points",
            AnomalyType.TREND_ANOMALY: f"Detected trend anomaly affecting {num_anomalies} consecutive points",
            AnomalyType.SEASONAL_ANOMALY: f"Detected seasonal anomaly pattern with {num_anomalies} affected points"
        }
        
        return explanations.get(anomaly_type, f"Detected {num_anomalies} anomalies")
    
    async def analyze_trends(self, data_id: str) -> TrendAnalysisResult:
        """Analyze trends in time series data."""
        if not self.pandas_available:
            raise RuntimeError("Pandas not available")
        
        if data_id not in self.time_series_data:
            raise ValueError(f"Data ID {data_id} not found")
        
        time_series = self.time_series_data[data_id]
        values = np.array(time_series.values)
        
        # Calculate trend direction and strength
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Determine trend direction
        if slope > 0.01:
            trend_direction = "increasing"
        elif slope < -0.01:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
        
        trend_strength = abs(r_value)
        
        # Calculate trend duration
        trend_duration = len(values)
        
        # Analyze seasonality
        seasonality = self._analyze_seasonality(values)
        
        # Find breakpoints (simplified)
        breakpoints = self._find_trend_breakpoints(values)
        
        # Generate forecast
        forecast = self._generate_trend_forecast(values, slope, intercept, 10)
        
        result = TrendAnalysisResult(
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            trend_duration=trend_duration,
            seasonality=seasonality,
            breakpoints=breakpoints,
            forecast=forecast.tolist(),
            metadata={'data_id': data_id, 'slope': slope, 'r_value': r_value}
        )
        
        return result
    
    def _analyze_seasonality(self, values: np.ndarray) -> Dict[str, Any]:
        """Analyze seasonality in the data."""
        seasonality = {
            'has_seasonality': False,
            'seasonal_period': None,
            'seasonal_strength': 0.0
        }
        
        if len(values) < 24:
            return seasonality
        
        try:
            # Simple seasonal analysis
            autocorr = np.correlate(values, values, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peaks
            peaks, _ = find_peaks(autocorr[:len(autocorr)//2])
            
            if len(peaks) > 1:
                seasonality['has_seasonality'] = True
                seasonality['seasonal_period'] = int(np.mean(np.diff(peaks)))
                seasonality['seasonal_strength'] = float(np.max(autocorr[peaks]) / autocorr[0])
        except:
            pass
        
        return seasonality
    
    def _find_trend_breakpoints(self, values: np.ndarray) -> List[int]:
        """Find breakpoints in the trend."""
        breakpoints = []
        
        if len(values) < 10:
            return breakpoints
        
        # Simple breakpoint detection using rolling mean
        window = min(5, len(values) // 4)
        rolling_mean = np.convolve(values, np.ones(window)/window, mode='valid')
        
        # Find points where trend changes significantly
        for i in range(window, len(values) - window):
            before_mean = np.mean(values[i-window:i])
            after_mean = np.mean(values[i:i+window])
            
            if abs(after_mean - before_mean) > np.std(values) * 0.5:
                breakpoints.append(i)
        
        return breakpoints
    
    def _generate_trend_forecast(self, values: np.ndarray, slope: float, 
                               intercept: float, horizon: int) -> np.ndarray:
        """Generate trend-based forecast."""
        last_index = len(values) - 1
        future_indices = np.arange(last_index + 1, last_index + 1 + horizon)
        forecast = slope * future_indices + intercept
        
        return forecast
    
    async def market_intelligence_analysis(self, market_data: Dict[str, Any]) -> MarketIntelligenceResult:
        """Perform market intelligence analysis."""
        start_time = time.time()
        
        # Analyze market trends
        market_trends = self._analyze_market_trends(market_data)
        
        # Competitive analysis
        competitive_analysis = self._analyze_competition(market_data)
        
        # Market opportunities
        market_opportunities = self._identify_opportunities(market_data)
        
        # Risk assessment
        risk_assessment = self._assess_market_risks(market_data)
        
        # Generate recommendations
        recommendations = self._generate_market_recommendations(
            market_trends, competitive_analysis, market_opportunities, risk_assessment
        )
        
        result = MarketIntelligenceResult(
            market_trends=market_trends,
            competitive_analysis=competitive_analysis,
            market_opportunities=market_opportunities,
            risk_assessment=risk_assessment,
            recommendations=recommendations,
            confidence=0.7,  # Base confidence
            metadata={'analysis_timestamp': datetime.now().isoformat()}
        )
        
        # Store result
        with self.lock:
            self.market_intelligence_history.append(result)
            self.metrics['total_market_analyses'] += 1
        
        self.logger.info(f"Market intelligence analysis completed in {time.time() - start_time:.2f}s")
        return result
    
    def _analyze_market_trends(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze market trends."""
        trends = []
        
        # Extract trend information from market data
        for key, value in market_data.items():
            if isinstance(value, (list, tuple)) and len(value) > 1:
                # Analyze trend in this metric
                trend_direction = "stable"
                if value[-1] > value[0] * 1.1:
                    trend_direction = "increasing"
                elif value[-1] < value[0] * 0.9:
                    trend_direction = "decreasing"
                
                trends.append({
                    'metric': key,
                    'trend_direction': trend_direction,
                    'current_value': value[-1],
                    'change_percentage': ((value[-1] - value[0]) / value[0]) * 100 if value[0] != 0 else 0
                })
        
        return trends
    
    def _analyze_competition(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze competitive landscape."""
        competition = {
            'market_share': {},
            'competitive_advantages': [],
            'threats': [],
            'market_position': 'unknown'
        }
        
        # Extract competitive information
        if 'competitors' in market_data:
            competitors = market_data['competitors']
            if isinstance(competitors, list):
                competition['market_share'] = {comp: 1.0/len(competitors) for comp in competitors}
        
        return competition
    
    def _identify_opportunities(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify market opportunities."""
        opportunities = []
        
        # Look for growth indicators
        for key, value in market_data.items():
            if isinstance(value, (list, tuple)) and len(value) > 1:
                growth_rate = ((value[-1] - value[0]) / value[0]) * 100 if value[0] != 0 else 0
                
                if growth_rate > 10:  # 10% growth threshold
                    opportunities.append({
                        'area': key,
                        'growth_rate': growth_rate,
                        'opportunity_type': 'growth_market',
                        'confidence': min(growth_rate / 20, 1.0)
                    })
        
        return opportunities
    
    def _assess_market_risks(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess market risks."""
        risks = {
            'volatility_risk': 'low',
            'competition_risk': 'medium',
            'regulatory_risk': 'low',
            'economic_risk': 'medium',
            'overall_risk_score': 0.3
        }
        
        # Calculate risk scores based on market data
        risk_factors = []
        
        for key, value in market_data.items():
            if isinstance(value, (list, tuple)) and len(value) > 1:
                volatility = np.std(value) / np.mean(value) if np.mean(value) != 0 else 0
                if volatility > 0.5:
                    risk_factors.append(volatility)
        
        if risk_factors:
            risks['overall_risk_score'] = min(np.mean(risk_factors), 1.0)
        
        return risks
    
    def _generate_market_recommendations(self, trends: List[Dict[str, Any]], 
                                       competition: Dict[str, Any],
                                       opportunities: List[Dict[str, Any]],
                                       risks: Dict[str, Any]) -> List[str]:
        """Generate market recommendations."""
        recommendations = []
        
        # Recommendations based on trends
        for trend in trends:
            if trend['trend_direction'] == 'increasing':
                recommendations.append(f"Capitalize on growing {trend['metric']} trend")
            elif trend['trend_direction'] == 'decreasing':
                recommendations.append(f"Address declining {trend['metric']} trend")
        
        # Recommendations based on opportunities
        for opportunity in opportunities:
            recommendations.append(f"Focus on {opportunity['area']} growth opportunity")
        
        # Recommendations based on risks
        if risks['overall_risk_score'] > 0.7:
            recommendations.append("Implement risk mitigation strategies")
        
        return recommendations
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get advanced analytics system status."""
        with self.lock:
            return {
                'metrics': self.metrics.copy(),
                'total_time_series': len(self.time_series_data),
                'total_predictions': len(self.prediction_history),
                'total_anomalies': len(self.anomaly_history),
                'total_market_analyses': len(self.market_intelligence_history),
                'pandas_available': self.pandas_available,
                'scipy_available': self.scipy_available,
                'sklearn_available': self.sklearn_available
            }


# Global instance
advanced_analytics_engine = AdvancedAnalyticsEngine() 