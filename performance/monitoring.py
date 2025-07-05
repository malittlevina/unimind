"""
Performance Monitor

Comprehensive performance monitoring for Unimind.
Tracks system metrics, response times, and performance indicators.
"""

import asyncio
import logging
import time
import psutil
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from collections import deque


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    response_times: List[float]
    error_rate: float
    throughput: float
    active_connections: int
    system_load: float


class PerformanceMonitor:
    """
    Comprehensive performance monitor for Unimind.
    
    Tracks system metrics, response times, and performance indicators
    to provide insights into system performance.
    """
    
    def __init__(self, history_size: int = 1000):
        """Initialize the performance monitor."""
        self.history_size = history_size
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=history_size)
        self.response_times: deque = deque(maxlen=1000)
        self.error_counts: deque = deque(maxlen=1000)
        
        # Monitoring state
        self.is_running = False
        self.monitoring_thread = None
        self.monitoring_interval = 5  # 5 seconds
        
        # Performance thresholds
        self.thresholds = {
            'cpu_warning': 70.0,
            'cpu_critical': 90.0,
            'memory_warning': 90.0,
            'memory_critical': 95.0,
            'response_time_warning': 2.0,
            'response_time_critical': 5.0,
            'error_rate_warning': 0.05,
            'error_rate_critical': 0.10
        }
        
        # Statistics
        self.total_requests = 0
        self.total_errors = 0
        self.start_time = time.time()
        
        # Threading
        self.lock = threading.Lock()
        
        self.logger.info("Performance monitor initialized")
    
    async def start(self) -> None:
        """Start the performance monitor."""
        self.is_running = True
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Performance monitor started")
    
    async def stop(self) -> None:
        """Stop the performance monitor."""
        self.is_running = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Performance monitor stopped")
    
    async def record_response_time(self, response_time: float) -> None:
        """Record a response time measurement."""
        with self.lock:
            self.response_times.append(response_time)
            self.total_requests += 1
    
    async def record_error(self, error_type: str = "general") -> None:
        """Record an error occurrence."""
        with self.lock:
            self.error_counts.append({
                'timestamp': time.time(),
                'type': error_type
            })
            self.total_errors += 1
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        try:
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            network_io = psutil.net_io_counters()
            
            # Response time statistics
            avg_response_time = 0.0
            if self.response_times:
                avg_response_time = sum(self.response_times) / len(self.response_times)
            
            # Error rate
            error_rate = 0.0
            if self.total_requests > 0:
                error_rate = self.total_errors / self.total_requests
            
            # Throughput (requests per second)
            uptime = time.time() - self.start_time
            throughput = self.total_requests / uptime if uptime > 0 else 0
            
            # System load
            system_load = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
            
            metrics = {
                'timestamp': time.time(),
                'cpu_usage': cpu_usage,
                'memory_usage': memory.percent,
                'memory_available': memory.available,
                'memory_total': memory.total,
                'disk_usage': disk.percent,
                'disk_free': disk.free,
                'network_bytes_sent': network_io.bytes_sent,
                'network_bytes_recv': network_io.bytes_recv,
                'avg_response_time': avg_response_time,
                'error_rate': error_rate,
                'throughput': throughput,
                'total_requests': self.total_requests,
                'total_errors': self.total_errors,
                'system_load': system_load,
                'uptime': uptime
            }
            
            # Calculate performance score
            metrics['performance_score'] = self._calculate_performance_score(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting current metrics: {e}")
            return {
                'timestamp': time.time(),
                'error': str(e)
            }
    
    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall performance score (0-100)."""
        try:
            # CPU score (lower is better)
            cpu_score = max(0, 100 - metrics['cpu_usage'])
            
            # Memory score (lower usage is better)
            memory_score = max(0, 100 - metrics['memory_usage'])
            
            # Response time score (faster is better)
            response_time_score = max(0, 100 - (metrics['avg_response_time'] * 20))
            
            # Error rate score (lower is better)
            error_rate_score = max(0, 100 - (metrics['error_rate'] * 1000))
            
            # Throughput score (higher is better, capped at 100)
            throughput_score = min(100, metrics['throughput'] * 10)
            
            # Weighted average
            performance_score = (
                cpu_score * 0.25 +
                memory_score * 0.25 +
                response_time_score * 0.20 +
                error_rate_score * 0.20 +
                throughput_score * 0.10
            )
            
            return max(0, min(100, performance_score))
            
        except Exception:
            return 50.0  # Default score
    
    async def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """Get performance alerts based on thresholds."""
        alerts = []
        current_metrics = await self.get_current_metrics()
        
        # CPU alerts
        if current_metrics.get('cpu_usage', 0) > self.thresholds['cpu_critical']:
            alerts.append({
                'level': 'critical',
                'type': 'cpu_usage',
                'message': f"CPU usage is critically high: {current_metrics['cpu_usage']:.1f}%",
                'value': current_metrics['cpu_usage'],
                'threshold': self.thresholds['cpu_critical']
            })
        elif current_metrics.get('cpu_usage', 0) > self.thresholds['cpu_warning']:
            alerts.append({
                'level': 'warning',
                'type': 'cpu_usage',
                'message': f"CPU usage is high: {current_metrics['cpu_usage']:.1f}%",
                'value': current_metrics['cpu_usage'],
                'threshold': self.thresholds['cpu_warning']
            })
        
        # Memory alerts
        if current_metrics.get('memory_usage', 0) > self.thresholds['memory_critical']:
            alerts.append({
                'level': 'critical',
                'type': 'memory_usage',
                'message': f"Memory usage is critically high: {current_metrics['memory_usage']:.1f}%",
                'value': current_metrics['memory_usage'],
                'threshold': self.thresholds['memory_critical']
            })
        elif current_metrics.get('memory_usage', 0) > self.thresholds['memory_warning']:
            alerts.append({
                'level': 'warning',
                'type': 'memory_usage',
                'message': f"Memory usage is high: {current_metrics['memory_usage']:.1f}%",
                'value': current_metrics['memory_usage'],
                'threshold': self.thresholds['memory_warning']
            })
        
        # Response time alerts
        if current_metrics.get('avg_response_time', 0) > self.thresholds['response_time_critical']:
            alerts.append({
                'level': 'critical',
                'type': 'response_time',
                'message': f"Response time is critically slow: {current_metrics['avg_response_time']:.2f}s",
                'value': current_metrics['avg_response_time'],
                'threshold': self.thresholds['response_time_critical']
            })
        elif current_metrics.get('avg_response_time', 0) > self.thresholds['response_time_warning']:
            alerts.append({
                'level': 'warning',
                'type': 'response_time',
                'message': f"Response time is slow: {current_metrics['avg_response_time']:.2f}s",
                'value': current_metrics['avg_response_time'],
                'threshold': self.thresholds['response_time_warning']
            })
        
        # Error rate alerts
        if current_metrics.get('error_rate', 0) > self.thresholds['error_rate_critical']:
            alerts.append({
                'level': 'critical',
                'type': 'error_rate',
                'message': f"Error rate is critically high: {current_metrics['error_rate']:.2%}",
                'value': current_metrics['error_rate'],
                'threshold': self.thresholds['error_rate_critical']
            })
        elif current_metrics.get('error_rate', 0) > self.thresholds['error_rate_warning']:
            alerts.append({
                'level': 'warning',
                'type': 'error_rate',
                'message': f"Error rate is high: {current_metrics['error_rate']:.2%}",
                'value': current_metrics['error_rate'],
                'threshold': self.thresholds['error_rate_warning']
            })
        
        return alerts
    
    async def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends over time."""
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)
        
        # Filter recent metrics
        recent_metrics = [
            metrics for metrics in self.metrics_history
            if metrics.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {'error': 'No recent metrics available'}
        
        # Calculate trends
        cpu_trend = [m.cpu_usage for m in recent_metrics]
        memory_trend = [m.memory_usage for m in recent_metrics]
        response_time_trend = [m.response_times[-1] if m.response_times else 0 for m in recent_metrics]
        
        return {
            'period_hours': hours,
            'metrics_count': len(recent_metrics),
            'cpu_trend': {
                'min': min(cpu_trend),
                'max': max(cpu_trend),
                'avg': sum(cpu_trend) / len(cpu_trend),
                'trend': 'increasing' if cpu_trend[-1] > cpu_trend[0] else 'decreasing'
            },
            'memory_trend': {
                'min': min(memory_trend),
                'max': max(memory_trend),
                'avg': sum(memory_trend) / len(memory_trend),
                'trend': 'increasing' if memory_trend[-1] > memory_trend[0] else 'decreasing'
            },
            'response_time_trend': {
                'min': min(response_time_trend),
                'max': max(response_time_trend),
                'avg': sum(response_time_trend) / len(response_time_trend),
                'trend': 'increasing' if response_time_trend[-1] > response_time_trend[0] else 'decreasing'
            }
        }
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.is_running:
            try:
                time.sleep(self.monitoring_interval)
                
                if self.is_running:
                    # Run monitoring in event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        metrics_data = loop.run_until_complete(self.get_current_metrics())
                        
                        # Create metrics object
                        metrics = PerformanceMetrics(
                            timestamp=metrics_data['timestamp'],
                            cpu_usage=metrics_data.get('cpu_usage', 0),
                            memory_usage=metrics_data.get('memory_usage', 0),
                            disk_usage=metrics_data.get('disk_usage', 0),
                            network_io={
                                'bytes_sent': metrics_data.get('network_bytes_sent', 0),
                                'bytes_recv': metrics_data.get('network_bytes_recv', 0)
                            },
                            response_times=list(self.response_times)[-10:],  # Last 10
                            error_rate=metrics_data.get('error_rate', 0),
                            throughput=metrics_data.get('throughput', 0),
                            active_connections=0,  # Placeholder
                            system_load=metrics_data.get('system_load', 0)
                        )
                        
                        self.metrics_history.append(metrics)
                        
                        # Check for alerts
                        alerts = loop.run_until_complete(self.get_performance_alerts())
                        if alerts:
                            for alert in alerts:
                                self.logger.warning(f"Performance alert: {alert['message']}")
                        
                    finally:
                        loop.close()
                        
            except Exception as e:
                self.logger.error(f"Performance monitoring loop error: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get performance monitor status."""
        current_metrics = await self.get_current_metrics()
        alerts = await self.get_performance_alerts()
        
        return {
            'is_running': self.is_running,
            'monitoring_interval': self.monitoring_interval,
            'metrics_history_size': len(self.metrics_history),
            'response_times_count': len(self.response_times),
            'current_metrics': current_metrics,
            'active_alerts': len(alerts),
            'alerts': alerts,
            'thresholds': self.thresholds
        }


# Global instance
performance_monitor = PerformanceMonitor() 