"""
Cache Manager

Advanced caching system for Unimind performance optimization.
Provides intelligent caching of responses, models, and data.
"""

import asyncio
import logging
import time
import hashlib
import json
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import OrderedDict


class CacheType(Enum):
    """Types of cache entries."""
    RESPONSE = "response"
    MODEL = "model"
    DATA = "data"
    CONTEXT = "context"
    COMPUTATION = "computation"


@dataclass
class CacheEntry:
    """A cache entry with metadata."""
    key: str
    value: Any
    cache_type: CacheType
    created_at: float
    last_accessed: float
    access_count: int
    size: int
    ttl: Optional[float] = None
    priority: int = 1


class CacheManager:
    """
    Advanced cache manager for Unimind.
    
    Provides intelligent caching with LRU eviction, TTL support,
    and automatic optimization.
    """
    
    def __init__(self, max_size_mb: int = 2048, optimization_level: str = "standard"):
        """Initialize the cache manager."""
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.optimization_level = optimization_level
        self.logger = logging.getLogger(__name__)
        
        # Cache storage
        self.caches: Dict[CacheType, OrderedDict[str, CacheEntry]] = {
            cache_type: OrderedDict() for cache_type in CacheType
        }
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_size': 0,
            'entries_count': 0
        }
        
        # Threading
        self.lock = threading.RLock()
        
        # Cleanup thread
        self.cleanup_thread = None
        self.is_running = False
        
        self.logger.info(f"Cache manager initialized with {max_size_mb}MB limit")
    
    async def start(self) -> None:
        """Start the cache manager."""
        self.is_running = True
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        self.logger.info("Cache manager started")
    
    async def stop(self) -> None:
        """Stop the cache manager."""
        self.is_running = False
        
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        
        self.logger.info("Cache manager stopped")
    
    async def cache_response(self, query: str, context: Dict[str, Any], response: Dict[str, Any]) -> None:
        """Cache a query response."""
        cache_key = self._generate_cache_key(query, context)
        
        entry = CacheEntry(
            key=cache_key,
            value=response,
            cache_type=CacheType.RESPONSE,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            size=self._estimate_size(response),
            ttl=3600,  # 1 hour TTL for responses
            priority=2
        )
        
        await self._add_entry(entry)
    
    async def get_cached_response(self, query: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get a cached response for a query."""
        cache_key = self._generate_cache_key(query, context)
        return await self._get_entry(cache_key, CacheType.RESPONSE)
    
    async def cache_model(self, model_name: str, model_data: Any, ttl: Optional[float] = None) -> None:
        """Cache a model."""
        entry = CacheEntry(
            key=model_name,
            value=model_data,
            cache_type=CacheType.MODEL,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            size=self._estimate_size(model_data),
            ttl=ttl or 7200,  # 2 hours default TTL for models
            priority=3
        )
        
        await self._add_entry(entry)
    
    async def get_cached_model(self, model_name: str) -> Optional[Any]:
        """Get a cached model."""
        return await self._get_entry(model_name, CacheType.MODEL)
    
    async def cache_computation(self, computation_id: str, result: Any, ttl: Optional[float] = None) -> None:
        """Cache a computation result."""
        entry = CacheEntry(
            key=computation_id,
            value=result,
            cache_type=CacheType.COMPUTATION,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            size=self._estimate_size(result),
            ttl=ttl or 1800,  # 30 minutes default TTL for computations
            priority=1
        )
        
        await self._add_entry(entry)
    
    async def get_cached_computation(self, computation_id: str) -> Optional[Any]:
        """Get a cached computation result."""
        return await self._get_entry(computation_id, CacheType.COMPUTATION)
    
    async def _add_entry(self, entry: CacheEntry) -> None:
        """Add an entry to the cache."""
        with self.lock:
            # Check if we need to evict entries
            while self.stats['total_size'] + entry.size > self.max_size_bytes:
                await self._evict_entries()
            
            # Add entry
            self.caches[entry.cache_type][entry.key] = entry
            self.stats['total_size'] += entry.size
            self.stats['entries_count'] += 1
            
            self.logger.debug(f"Cached {entry.cache_type.value}: {entry.key} ({entry.size} bytes)")
    
    async def _get_entry(self, key: str, cache_type: CacheType) -> Optional[Any]:
        """Get an entry from the cache."""
        with self.lock:
            if key in self.caches[cache_type]:
                entry = self.caches[cache_type][key]
                
                # Check TTL
                if entry.ttl and time.time() - entry.created_at > entry.ttl:
                    await self._remove_entry(key, cache_type)
                    self.stats['misses'] += 1
                    return None
                
                # Update access statistics
                entry.last_accessed = time.time()
                entry.access_count += 1
                
                # Move to end (LRU)
                self.caches[cache_type].move_to_end(key)
                
                self.stats['hits'] += 1
                return entry.value
            else:
                self.stats['misses'] += 1
                return None
    
    async def _remove_entry(self, key: str, cache_type: CacheType) -> None:
        """Remove an entry from the cache."""
        if key in self.caches[cache_type]:
            entry = self.caches[cache_type][key]
            self.stats['total_size'] -= entry.size
            self.stats['entries_count'] -= 1
            del self.caches[cache_type][key]
    
    async def _evict_entries(self) -> None:
        """Evict entries using LRU strategy with priority consideration."""
        # Find the least recently used entry with lowest priority
        oldest_entry = None
        oldest_cache_type = None
        oldest_key = None
        
        for cache_type, cache in self.caches.items():
            if cache:
                # Get the oldest entry (first in OrderedDict)
                key, entry = next(iter(cache.items()))
                
                if oldest_entry is None or (
                    entry.priority < oldest_entry.priority or
                    (entry.priority == oldest_entry.priority and entry.last_accessed < oldest_entry.last_accessed)
                ):
                    oldest_entry = entry
                    oldest_cache_type = cache_type
                    oldest_key = key
        
        if oldest_entry:
            await self._remove_entry(oldest_key, oldest_cache_type)
            self.stats['evictions'] += 1
            
            self.logger.debug(f"Evicted {oldest_entry.cache_type.value}: {oldest_key}")
    
    def _generate_cache_key(self, query: str, context: Dict[str, Any]) -> str:
        """Generate a cache key for a query and context."""
        # Create a hash of the query and relevant context
        context_str = json.dumps(context, sort_keys=True)
        combined = f"{query}:{context_str}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate the size of an object in bytes."""
        try:
            return len(json.dumps(obj, default=str).encode())
        except Exception:
            return 1024  # Default estimate
    
    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self.is_running:
            try:
                time.sleep(60)  # Cleanup every minute
                
                if self.is_running:
                    asyncio.run(self._cleanup_expired())
                    
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
    
    async def _cleanup_expired(self) -> None:
        """Clean up expired entries."""
        current_time = time.time()
        expired_count = 0
        
        with self.lock:
            for cache_type, cache in self.caches.items():
                expired_keys = []
                
                for key, entry in cache.items():
                    if entry.ttl and current_time - entry.created_at > entry.ttl:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    await self._remove_entry(key, cache_type)
                    expired_count += 1
        
        if expired_count > 0:
            self.logger.info(f"Cleaned up {expired_count} expired cache entries")
    
    async def optimize(self) -> Dict[str, Any]:
        """Optimize the cache for better performance."""
        with self.lock:
            # Adjust cache size based on optimization level
            if self.optimization_level == "aggressive":
                self.max_size_bytes = min(self.max_size_bytes * 1.5, 1024 * 1024 * 1024)  # Max 1GB
            elif self.optimization_level == "maximum":
                self.max_size_bytes = min(self.max_size_bytes * 2, 2048 * 1024 * 1024)  # Max 2GB
            
            # Clean up expired entries
            await self._cleanup_expired()
            
            # Optimize TTL values based on access patterns
            await self._optimize_ttl_values()
            
            return {
                'status': 'optimized',
                'max_size_bytes': self.max_size_bytes,
                'current_size_bytes': self.stats['total_size'],
                'entries_count': self.stats['entries_count'],
                'hit_rate': self._calculate_hit_rate()
            }
    
    async def _optimize_ttl_values(self) -> None:
        """Optimize TTL values based on access patterns."""
        current_time = time.time()
        
        for cache_type, cache in self.caches.items():
            for entry in cache.values():
                # Increase TTL for frequently accessed entries
                if entry.access_count > 10:
                    if entry.ttl:
                        entry.ttl = min(entry.ttl * 1.5, 7200)  # Max 2 hours
                
                # Decrease TTL for rarely accessed entries
                elif entry.access_count < 3 and current_time - entry.created_at > 1800:
                    if entry.ttl:
                        entry.ttl = max(entry.ttl * 0.5, 300)  # Min 5 minutes
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.stats['hits'] + self.stats['misses']
        if total_requests > 0:
            return (self.stats['hits'] / total_requests) * 100
        return 0.0
    
    async def get_status(self) -> Dict[str, Any]:
        """Get cache status and statistics."""
        return {
            'is_running': self.is_running,
            'max_size_bytes': self.max_size_bytes,
            'current_size_bytes': self.stats['total_size'],
            'entries_count': self.stats['entries_count'],
            'hit_rate': self._calculate_hit_rate(),
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'evictions': self.stats['evictions'],
            'cache_types': {
                cache_type.value: len(cache) for cache_type, cache in self.caches.items()
            }
        }
    
    async def set_optimization_level(self, level: str) -> None:
        """Set the optimization level."""
        self.optimization_level = level
        self.logger.info(f"Cache optimization level set to: {level}")
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set a simple key-value pair in the cache (synchronous)."""
        entry = CacheEntry(
            key=key,
            value=value,
            cache_type=CacheType.DATA,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            size=self._estimate_size(value),
            ttl=ttl or 3600,  # 1 hour default TTL
            priority=1
        )
        
        # Use threading to handle async operation
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, schedule the operation
                asyncio.create_task(self._add_entry(entry))
            else:
                # If we're not in an async context, run it
                loop.run_until_complete(self._add_entry(entry))
        except RuntimeError:
            # No event loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._add_entry(entry))
            loop.close()
    
    def get(self, key: str) -> Optional[Any]:
        """Get a simple key-value pair from the cache (synchronous)."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, we can't block
                # Return None and log a warning
                self.logger.warning("Cache get called in async context, returning None")
                return None
            else:
                # If we're not in an async context, run it
                return loop.run_until_complete(self._get_entry(key, CacheType.DATA))
        except RuntimeError:
            # No event loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._get_entry(key, CacheType.DATA))
            loop.close()
            return result


# Global instance
cache_manager = CacheManager() 