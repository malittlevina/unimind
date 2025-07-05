"""
Performance Optimizer

Main coordinator for Phase 3 performance optimization.
Manages caching, memory optimization, and response optimization.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from .cache_manager import CacheManager
from .memory_optimizer import MemoryOptimizer
from .response_optimizer import ResponseOptimizer
from .monitoring import PerformanceMonitor


class OptimizationLevel(Enum):
    """Performance optimization levels."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"


@dataclass
class OptimizationResult:
    """Result of a performance optimization operation."""
    success: bool
    optimization_type: str
    performance_gain: float
    memory_saved: int
    response_time_improvement: float
    details: Dict[str, Any]
    timestamp: float


class PerformanceOptimizer:
    """
    Main performance optimizer for Unimind.
    
    Coordinates caching, memory optimization, and response optimization
    to achieve maximum system performance.
    """
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.STANDARD):
        """Initialize the performance optimizer."""
        self.optimization_level = optimization_level
        self.logger = logging.getLogger(__name__)
        
        # Initialize optimization components
        self.cache_manager = CacheManager(max_size_mb=2048)
        self.memory_optimizer = MemoryOptimizer(max_memory_mb=2048)
        self.response_optimizer = ResponseOptimizer()
        self.performance_monitor = PerformanceMonitor()
        
        # Optimization state
        self.is_running = False
        self.optimization_thread = None
        self.last_optimization = time.time()
        self.optimization_interval = 300  # 5 minutes
        
        # Performance metrics
        self.optimization_history: List[OptimizationResult] = []
        self.total_performance_gain = 0.0
        self.total_memory_saved = 0
        
        # Threading
        self.lock = threading.Lock()
        
        self.logger.info(f"Performance optimizer initialized with level: {optimization_level.value}")
    
    async def start(self) -> None:
        """Start the performance optimizer."""
        self.is_running = True
        
        # Start monitoring
        await self.performance_monitor.start()
        
        # Start optimization thread
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()
        
        # Perform initial optimization
        await self.perform_full_optimization()
        
        self.logger.info("Performance optimizer started")
    
    async def stop(self) -> None:
        """Stop the performance optimizer."""
        self.is_running = False
        
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)
        
        await self.performance_monitor.stop()
        
        self.logger.info("Performance optimizer stopped")
    
    async def perform_full_optimization(self) -> OptimizationResult:
        """Perform a full system optimization."""
        start_time = time.time()
        
        with self.lock:
            try:
                self.logger.info("Starting full system optimization")
                
                # Get baseline metrics
                baseline_metrics = await self.performance_monitor.get_current_metrics()
                
                # Perform optimizations
                cache_result = await self.cache_manager.optimize()
                memory_result = await self.memory_optimizer.optimize()
                response_result = await self.response_optimizer.optimize()
                
                # Get post-optimization metrics
                post_metrics = await self.performance_monitor.get_current_metrics()
                
                # Calculate improvements
                performance_gain = self._calculate_performance_gain(baseline_metrics, post_metrics)
                memory_saved = baseline_metrics.get('memory_usage', 0) - post_metrics.get('memory_usage', 0)
                response_time_improvement = baseline_metrics.get('avg_response_time', 0) - post_metrics.get('avg_response_time', 0)
                
                # Create result
                result = OptimizationResult(
                    success=True,
                    optimization_type="full_system",
                    performance_gain=performance_gain,
                    memory_saved=memory_saved,
                    response_time_improvement=response_time_improvement,
                    details={
                        'cache_optimization': cache_result,
                        'memory_optimization': memory_result,
                        'response_optimization': response_result,
                        'baseline_metrics': baseline_metrics,
                        'post_metrics': post_metrics
                    },
                    timestamp=time.time()
                )
                
                # Update history
                self.optimization_history.append(result)
                self.total_performance_gain += performance_gain
                self.total_memory_saved += memory_saved
                
                self.logger.info(f"Full optimization completed: {performance_gain:.2f}% performance gain, {memory_saved} bytes saved")
                
                return result
                
            except Exception as e:
                self.logger.error(f"Full optimization failed: {e}")
                return OptimizationResult(
                    success=False,
                    optimization_type="full_system",
                    performance_gain=0.0,
                    memory_saved=0,
                    response_time_improvement=0.0,
                    details={'error': str(e)},
                    timestamp=time.time()
                )
    
    async def optimize_query_response(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize response for a specific query."""
        try:
            # Check cache first
            cached_response = await self.cache_manager.get_cached_response(query, context)
            if cached_response:
                return {
                    'response': cached_response,
                    'source': 'cache',
                    'response_time': 0.001,  # Cache hit time
                    'optimization_applied': True
                }
            
            # Apply response optimization
            optimized_response = await self.response_optimizer.optimize_response(query, context)
            
            # Cache the result
            await self.cache_manager.cache_response(query, context, optimized_response)
            
            return {
                'response': optimized_response,
                'source': 'optimized',
                'response_time': optimized_response.get('response_time', 0.0),
                'optimization_applied': True
            }
            
        except Exception as e:
            self.logger.error(f"Query optimization failed: {e}")
            return {
                'response': None,
                'source': 'fallback',
                'response_time': 0.0,
                'optimization_applied': False,
                'error': str(e)
            }
    
    async def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        current_metrics = await self.performance_monitor.get_current_metrics()
        
        return {
            'is_running': self.is_running,
            'optimization_level': self.optimization_level.value,
            'last_optimization': self.last_optimization,
            'total_performance_gain': self.total_performance_gain,
            'total_memory_saved': self.total_memory_saved,
            'optimization_count': len(self.optimization_history),
            'current_metrics': current_metrics,
            'cache_status': await self.cache_manager.get_status(),
            'memory_status': await self.memory_optimizer.get_status(),
            'response_status': await self.response_optimizer.get_status()
        }
    
    def _optimization_loop(self) -> None:
        """Background optimization loop."""
        while self.is_running:
            try:
                time.sleep(self.optimization_interval)
                
                if self.is_running:
                    # Run optimization in event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        loop.run_until_complete(self.perform_full_optimization())
                        self.last_optimization = time.time()
                    finally:
                        loop.close()
                        
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
    
    def _calculate_performance_gain(self, baseline: Dict[str, Any], post: Dict[str, Any]) -> float:
        """Calculate performance gain percentage."""
        try:
            baseline_score = baseline.get('performance_score', 0)
            post_score = post.get('performance_score', 0)
            
            if baseline_score > 0:
                return ((post_score - baseline_score) / baseline_score) * 100
            else:
                return 0.0
        except Exception:
            return 0.0
    
    async def set_optimization_level(self, level: OptimizationLevel) -> None:
        """Set the optimization level."""
        self.optimization_level = level
        
        # Update component levels
        await self.cache_manager.set_optimization_level(level)
        await self.memory_optimizer.set_optimization_level(level)
        await self.response_optimizer.set_optimization_level(level)
        
        self.logger.info(f"Optimization level set to: {level.value}")


# Global instance
performance_optimizer = PerformanceOptimizer() 