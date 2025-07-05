"""
Memory Optimizer

Advanced memory management and optimization for Unimind.
Provides intelligent memory allocation, garbage collection, and optimization.
"""

import asyncio
import logging
import time
import gc
import psutil
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class MemoryLevel(Enum):
    """Memory optimization levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""
    total_memory: int
    available_memory: int
    used_memory: int
    memory_percentage: float
    process_memory: int
    python_memory: int
    gc_objects: int
    gc_collections: int


class MemoryOptimizer:
    """
    Advanced memory optimizer for Unimind.
    
    Provides intelligent memory management, garbage collection,
    and memory optimization strategies.
    """
    
    def __init__(self, max_memory_mb: int = 2048):
        """Initialize the memory optimizer."""
        self.optimization_level = "standard"
        self.logger = logging.getLogger(__name__)
        
        # Memory thresholds
        self.memory_thresholds = {
            'low': 0.3,      # 30% memory usage
            'medium': 0.5,   # 50% memory usage
            'high': 0.7,     # 70% memory usage
            'critical': 0.85 # 85% memory usage
        }
        
        # Optimization state
        self.is_running = False
        self.optimization_thread = None
        self.last_optimization = time.time()
        self.optimization_interval = 120  # 2 minutes
        
        # Memory tracking
        self.memory_history: List[MemoryMetrics] = []
        self.optimization_count = 0
        self.total_memory_freed = 0
        
        # Threading
        self.lock = threading.Lock()
        
        self.logger.info("Memory optimizer initialized")
    
    async def start(self) -> None:
        """Start the memory optimizer."""
        self.is_running = True
        
        # Start optimization thread
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()
        
        # Perform initial optimization
        await self.optimize()
        
        self.logger.info("Memory optimizer started")
    
    async def stop(self) -> None:
        """Stop the memory optimizer."""
        self.is_running = False
        
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)
        
        self.logger.info("Memory optimizer stopped")
    
    async def optimize(self) -> Dict[str, Any]:
        """Perform memory optimization."""
        start_time = time.time()
        
        with self.lock:
            try:
                # Get current memory metrics
                current_metrics = await self._get_memory_metrics()
                self.memory_history.append(current_metrics)
                
                # Determine optimization level
                optimization_level = self._determine_optimization_level(current_metrics)
                
                # Perform optimizations based on level
                optimizations = []
                memory_freed = 0
                
                if optimization_level in [MemoryLevel.MEDIUM, MemoryLevel.HIGH, MemoryLevel.CRITICAL]:
                    # Garbage collection
                    gc_freed = await self._perform_garbage_collection()
                    memory_freed += gc_freed
                    optimizations.append(f"Garbage collection freed {gc_freed} bytes")
                
                if optimization_level in [MemoryLevel.HIGH, MemoryLevel.CRITICAL]:
                    # Memory cleanup
                    cleanup_freed = await self._perform_memory_cleanup()
                    memory_freed += cleanup_freed
                    optimizations.append(f"Memory cleanup freed {cleanup_freed} bytes")
                
                if optimization_level == MemoryLevel.CRITICAL:
                    # Aggressive optimization
                    aggressive_freed = await self._perform_aggressive_optimization()
                    memory_freed += aggressive_freed
                    optimizations.append(f"Aggressive optimization freed {aggressive_freed} bytes")
                
                # Update statistics
                self.optimization_count += 1
                self.total_memory_freed += memory_freed
                self.last_optimization = time.time()
                
                # Get post-optimization metrics
                post_metrics = await self._get_memory_metrics()
                
                result = {
                    'status': 'optimized',
                    'optimization_level': optimization_level.value,
                    'memory_freed': memory_freed,
                    'optimizations_applied': optimizations,
                    'pre_optimization': {
                        'used_memory': current_metrics.used_memory,
                        'memory_percentage': current_metrics.memory_percentage,
                        'process_memory': current_metrics.process_memory
                    },
                    'post_optimization': {
                        'used_memory': post_metrics.used_memory,
                        'memory_percentage': post_metrics.memory_percentage,
                        'process_memory': post_metrics.process_memory
                    },
                    'optimization_time': time.time() - start_time
                }
                
                self.logger.info(f"Memory optimization completed: {memory_freed} bytes freed")
                
                return result
                
            except Exception as e:
                self.logger.error(f"Memory optimization failed: {e}")
                return {
                    'status': 'failed',
                    'error': str(e),
                    'optimization_time': time.time() - start_time
                }
    
    async def _get_memory_metrics(self) -> MemoryMetrics:
        """Get current memory usage metrics."""
        # System memory
        memory = psutil.virtual_memory()
        
        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info().rss
        
        # Python memory (approximate)
        python_memory = process_memory
        
        # Garbage collector stats
        gc_stats = gc.get_stats()
        gc_objects = sum(stat['collections'] for stat in gc_stats)
        gc_collections = len(gc_stats)
        
        return MemoryMetrics(
            total_memory=memory.total,
            available_memory=memory.available,
            used_memory=memory.used,
            memory_percentage=memory.percent / 100.0,
            process_memory=process_memory,
            python_memory=python_memory,
            gc_objects=gc_objects,
            gc_collections=gc_collections
        )
    
    def _determine_optimization_level(self, metrics: MemoryMetrics) -> MemoryLevel:
        """Determine the required optimization level based on memory usage."""
        memory_usage = metrics.memory_percentage
        
        if memory_usage >= self.memory_thresholds['critical']:
            return MemoryLevel.CRITICAL
        elif memory_usage >= self.memory_thresholds['high']:
            return MemoryLevel.HIGH
        elif memory_usage >= self.memory_thresholds['medium']:
            return MemoryLevel.MEDIUM
        else:
            return MemoryLevel.LOW
    
    async def _perform_garbage_collection(self) -> int:
        """Perform garbage collection and return freed memory."""
        # Get memory before GC
        process = psutil.Process()
        memory_before = process.memory_info().rss
        
        # Perform garbage collection
        collected = gc.collect()
        
        # Get memory after GC
        memory_after = process.memory_info().rss
        memory_freed = memory_before - memory_after
        
        self.logger.debug(f"Garbage collection: {collected} objects collected, {memory_freed} bytes freed")
        
        return max(0, memory_freed)
    
    async def _perform_memory_cleanup(self) -> int:
        """Perform memory cleanup operations."""
        memory_freed = 0
        
        try:
            # Clear memory history if too large
            if len(self.memory_history) > 1000:
                removed_count = len(self.memory_history) - 500
                self.memory_history = self.memory_history[-500:]
                memory_freed += removed_count * 100  # Approximate size per entry
            
            # Clear unused imports (if possible)
            memory_freed += await self._clear_unused_imports()
            
            # Optimize memory-intensive components
            memory_freed += await self._optimize_components()
            
        except Exception as e:
            self.logger.warning(f"Memory cleanup error: {e}")
        
        return memory_freed
    
    async def _perform_aggressive_optimization(self) -> int:
        """Perform aggressive memory optimization."""
        memory_freed = 0
        
        try:
            # Force garbage collection multiple times
            for _ in range(3):
                memory_freed += await self._perform_garbage_collection()
                await asyncio.sleep(0.1)
            
            # Clear all caches if available
            memory_freed += await self._clear_caches()
            
            # Reset memory-intensive components
            memory_freed += await self._reset_components()
            
        except Exception as e:
            self.logger.warning(f"Aggressive optimization error: {e}")
        
        return memory_freed
    
    async def _clear_unused_imports(self) -> int:
        """Clear unused imports to free memory."""
        # This is a placeholder - in a real implementation,
        # you would track imported modules and clear unused ones
        return 0
    
    async def _optimize_components(self) -> int:
        """Optimize memory usage in Unimind components."""
        memory_freed = 0
        
        try:
            # Optimize memory systems
            from unimind.memory.unified_memory import unified_memory
            if hasattr(unified_memory, 'optimize'):
                result = unified_memory.optimize()
                if isinstance(result, dict) and 'memory_freed' in result:
                    memory_freed += result['memory_freed']
            
            # Optimize conversation memory
            from unimind.native_models.conversation_memory import conversation_memory
            if hasattr(conversation_memory, 'optimize'):
                result = conversation_memory.optimize()
                if isinstance(result, dict) and 'memory_freed' in result:
                    memory_freed += result['memory_freed']
            
            # Optimize hierarchical memory
            from unimind.memory.hierarchical_memory import hierarchical_memory
            if hasattr(hierarchical_memory, 'optimize'):
                result = hierarchical_memory.optimize()
                if isinstance(result, dict) and 'memory_freed' in result:
                    memory_freed += result['memory_freed']
            
        except Exception as e:
            self.logger.warning(f"Component optimization error: {e}")
        
        return memory_freed
    
    async def _clear_caches(self) -> int:
        """Clear various caches to free memory."""
        memory_freed = 0
        
        try:
            # Clear cache manager if available
            from unimind.performance.cache_manager import cache_manager
            if hasattr(cache_manager, 'clear_all'):
                freed = await cache_manager.clear_all()
                memory_freed += freed
            
            # Clear other caches
            # (Add more cache clearing logic here)
            
        except Exception as e:
            self.logger.warning(f"Cache clearing error: {e}")
        
        return memory_freed
    
    async def _reset_components(self) -> int:
        """Reset memory-intensive components."""
        memory_freed = 0
        
        try:
            # Reset LLM models if they support it
            # (Add model reset logic here)
            pass
            
            # Reset other components
            # (Add component reset logic here)
            pass
            
        except Exception as e:
            self.logger.warning(f"Component reset error: {e}")
        
        return memory_freed
    
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
                        loop.run_until_complete(self.optimize())
                    finally:
                        loop.close()
                        
            except Exception as e:
                self.logger.error(f"Memory optimization loop error: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get memory optimizer status."""
        current_metrics = await self._get_memory_metrics()
        
        return {
            'is_running': self.is_running,
            'optimization_level': self.optimization_level,
            'last_optimization': self.last_optimization,
            'optimization_count': self.optimization_count,
            'total_memory_freed': self.total_memory_freed,
            'current_metrics': {
                'used_memory': current_metrics.used_memory,
                'memory_percentage': current_metrics.memory_percentage,
                'process_memory': current_metrics.process_memory,
                'available_memory': current_metrics.available_memory
            },
            'memory_thresholds': self.memory_thresholds
        }
    
    async def set_optimization_level(self, level: str) -> None:
        """Set the optimization level."""
        self.optimization_level = level
        self.logger.info(f"Memory optimization level set to: {level}")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics (synchronous)."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, return basic stats
                return {
                    "total_optimizations": self.optimization_count,
                    "total_memory_freed": self.total_memory_freed,
                    "last_optimization": self.last_optimization,
                    "is_running": self.is_running,
                    "optimization_level": self.optimization_level
                }
            else:
                # If we're not in an async context, get full stats
                return loop.run_until_complete(self.get_status())
        except RuntimeError:
            # No event loop, return basic stats
            return {
                "total_optimizations": self.optimization_count,
                "total_memory_freed": self.total_memory_freed,
                "last_optimization": self.last_optimization,
                "is_running": self.is_running,
                "optimization_level": self.optimization_level
            }


# Global instance
memory_optimizer = MemoryOptimizer() 