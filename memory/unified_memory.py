"""
unified_memory.py – Unified memory system for LAM and LLM operations.
Provides temporary, centralized memory storage without permanent persistence.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

class MemoryType(Enum):
    """Types of memory that can be stored."""
    CONVERSATION = "conversation"
    CONTEXT = "context"
    PLANNING = "planning"
    EXECUTION = "execution"
    REASONING = "reasoning"
    TEMPORARY = "temporary"

@dataclass
class MemoryEntry:
    """Represents a single memory entry."""
    id: str
    type: MemoryType
    content: Any
    timestamp: float
    ttl: Optional[float] = None  # Time to live in seconds
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if the memory entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl

@dataclass
class MemoryContext:
    """Represents a memory context for a specific operation."""
    session_id: str
    operation_id: str
    entries: List[MemoryEntry] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    
    def add_entry(self, entry: MemoryEntry) -> None:
        """Add a memory entry to this context."""
        self.entries.append(entry)
        self.last_accessed = time.time()
    
    def get_entries(self, memory_type: Optional[MemoryType] = None, 
                   tags: Optional[List[str]] = None) -> List[MemoryEntry]:
        """Get memory entries filtered by type and tags."""
        filtered_entries = []
        
        for entry in self.entries:
            # Skip expired entries
            if entry.is_expired():
                continue
            
            # Filter by type
            if memory_type and entry.type != memory_type:
                continue
            
            # Filter by tags
            if tags and not any(tag in entry.tags for tag in tags):
                continue
            
            filtered_entries.append(entry)
        
        return filtered_entries
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed entries."""
        original_count = len(self.entries)
        self.entries = [entry for entry in self.entries if not entry.is_expired()]
        return original_count - len(self.entries)

class UnifiedMemory:
    """
    Unified memory system for temporary storage during LAM and LLM operations.
    Provides centralized memory management without permanent persistence.
    """
    
    def __init__(self):
        """Initialize the unified memory system."""
        self.logger = logging.getLogger('UnifiedMemory')
        
        # Active memory contexts
        self.contexts: Dict[str, MemoryContext] = {}
        
        # Global memory entries (not tied to specific contexts)
        self.global_entries: List[MemoryEntry] = []
        
        # Configuration
        self.default_ttl = 3600  # 1 hour default TTL
        self.max_contexts = 100
        self.max_entries_per_context = 1000
        self.cleanup_interval = 300  # 5 minutes
        
        # Last cleanup time
        self.last_cleanup = time.time()
        
        self.logger.info("Unified memory system initialized")
    
    def create_context(self, session_id: str, operation_id: str) -> str:
        """
        Create a new memory context.
        
        Args:
            session_id: Session identifier
            operation_id: Operation identifier
            
        Returns:
            Context ID
        """
        context_id = f"{session_id}:{operation_id}"
        
        # Cleanup old contexts if we're at the limit
        if len(self.contexts) >= self.max_contexts:
            self._cleanup_old_contexts()
        
        self.contexts[context_id] = MemoryContext(
            session_id=session_id,
            operation_id=operation_id
        )
        
        self.logger.info(f"Created memory context: {context_id}")
        return context_id
    
    def add_memory(self, context_id: str, memory_type: MemoryType, 
                   content: Any, tags: List[str] = None, 
                   ttl: Optional[float] = None, metadata: Dict[str, Any] = None) -> str:
        """
        Add a memory entry to a context.
        
        Args:
            context_id: Context identifier
            memory_type: Type of memory
            content: Memory content
            tags: Optional tags for filtering
            ttl: Time to live in seconds
            metadata: Additional metadata
            
        Returns:
            Memory entry ID
        """
        if context_id not in self.contexts:
            raise ValueError(f"Context not found: {context_id}")
        
        # Perform periodic cleanup
        self._periodic_cleanup()
        
        context = self.contexts[context_id]
        
        # Check if we're at the entry limit
        if len(context.entries) >= self.max_entries_per_context:
            self._cleanup_old_entries(context)
        
        # Create memory entry
        entry_id = f"{context_id}:{len(context.entries)}"
        entry = MemoryEntry(
            id=entry_id,
            type=memory_type,
            content=content,
            timestamp=time.time(),
            ttl=ttl or self.default_ttl,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        context.add_entry(entry)
        
        self.logger.debug(f"Added memory entry: {entry_id} ({memory_type.value})")
        return entry_id
    
    def add_global_memory(self, memory_type: MemoryType, content: Any,
                         tags: List[str] = None, ttl: Optional[float] = None,
                         metadata: Dict[str, Any] = None) -> str:
        """
        Add a global memory entry (not tied to specific context).
        
        Args:
            memory_type: Type of memory
            content: Memory content
            tags: Optional tags for filtering
            ttl: Time to live in seconds
            metadata: Additional metadata
            
        Returns:
            Memory entry ID
        """
        entry_id = f"global:{len(self.global_entries)}"
        entry = MemoryEntry(
            id=entry_id,
            type=memory_type,
            content=content,
            timestamp=time.time(),
            ttl=ttl or self.default_ttl,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        self.global_entries.append(entry)
        
        self.logger.debug(f"Added global memory entry: {entry_id} ({memory_type.value})")
        return entry_id
    
    def get_memory(self, context_id: str, memory_type: Optional[MemoryType] = None,
                   tags: Optional[List[str]] = None, limit: Optional[int] = None) -> List[Any]:
        """
        Get memory entries from a context.
        
        Args:
            context_id: Context identifier
            memory_type: Optional memory type filter
            tags: Optional tags filter
            limit: Maximum number of entries to return
            
        Returns:
            List of memory contents
        """
        if context_id not in self.contexts:
            return []
        
        context = self.contexts[context_id]
        entries = context.get_entries(memory_type, tags)
        
        # Sort by timestamp (newest first)
        entries.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            entries = entries[:limit]
        
        return [entry.content for entry in entries]
    
    def get_global_memory(self, memory_type: Optional[MemoryType] = None,
                         tags: Optional[List[str]] = None, limit: Optional[int] = None) -> List[Any]:
        """
        Get global memory entries.
        
        Args:
            memory_type: Optional memory type filter
            tags: Optional tags filter
            limit: Maximum number of entries to return
            
        Returns:
            List of memory contents
        """
        entries = []
        
        for entry in self.global_entries:
            if entry.is_expired():
                continue
            
            if memory_type and entry.type != memory_type:
                continue
            
            if tags and not any(tag in entry.tags for tag in tags):
                continue
            
            entries.append(entry)
        
        # Sort by timestamp (newest first)
        entries.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            entries = entries[:limit]
        
        return [entry.content for entry in entries]
    
    def get_context_summary(self, context_id: str) -> Dict[str, Any]:
        """
        Get a summary of a memory context.
        
        Args:
            context_id: Context identifier
            
        Returns:
            Context summary
        """
        if context_id not in self.contexts:
            return {}
        
        context = self.contexts[context_id]
        
        # Count entries by type
        type_counts = {}
        for entry in context.entries:
            if not entry.is_expired():
                type_counts[entry.type.value] = type_counts.get(entry.type.value, 0) + 1
        
        return {
            "context_id": context_id,
            "session_id": context.session_id,
            "operation_id": context.operation_id,
            "total_entries": len([e for e in context.entries if not e.is_expired()]),
            "type_counts": type_counts,
            "created_at": context.created_at,
            "last_accessed": context.last_accessed,
            "age_seconds": time.time() - context.created_at
        }
    
    def clear_context(self, context_id: str) -> bool:
        """
        Clear a memory context.
        
        Args:
            context_id: Context identifier
            
        Returns:
            True if context was cleared, False if not found
        """
        if context_id in self.contexts:
            del self.contexts[context_id]
            self.logger.info(f"Cleared memory context: {context_id}")
            return True
        return False
    
    def clear_global_memory(self) -> int:
        """
        Clear all global memory entries.
        
        Returns:
            Number of entries cleared
        """
        count = len(self.global_entries)
        self.global_entries.clear()
        self.logger.info(f"Cleared {count} global memory entries")
        return count
    
    def get_all_contexts(self) -> List[str]:
        """Get all active context IDs."""
        return list(self.contexts.keys())
    
    def _cleanup_old_contexts(self) -> int:
        """Remove old contexts to stay within limits."""
        if len(self.contexts) <= self.max_contexts:
            return 0
        
        # Sort contexts by last accessed time (oldest first)
        sorted_contexts = sorted(
            self.contexts.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Remove oldest contexts
        contexts_to_remove = len(self.contexts) - self.max_contexts
        removed_count = 0
        
        for context_id, context in sorted_contexts[:contexts_to_remove]:
            del self.contexts[context_id]
            removed_count += 1
        
        self.logger.info(f"Cleaned up {removed_count} old contexts")
        return removed_count
    
    def _cleanup_old_entries(self, context: MemoryContext) -> int:
        """Remove old entries from a context to stay within limits."""
        if len(context.entries) <= self.max_entries_per_context:
            return 0
        
        # Sort entries by timestamp (oldest first)
        context.entries.sort(key=lambda x: x.timestamp)
        
        # Remove oldest entries
        entries_to_remove = len(context.entries) - self.max_entries_per_context
        context.entries = context.entries[entries_to_remove:]
        
        self.logger.debug(f"Cleaned up {entries_to_remove} old entries from context")
        return entries_to_remove
    
    def _periodic_cleanup(self) -> None:
        """Perform periodic cleanup of expired entries."""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        self.last_cleanup = current_time
        
        # Cleanup expired entries in contexts
        total_cleaned = 0
        for context in self.contexts.values():
            cleaned = context.cleanup_expired()
            total_cleaned += cleaned
        
        # Cleanup expired global entries
        original_global_count = len(self.global_entries)
        self.global_entries = [entry for entry in self.global_entries if not entry.is_expired()]
        global_cleaned = original_global_count - len(self.global_entries)
        
        if total_cleaned > 0 or global_cleaned > 0:
            self.logger.info(f"Periodic cleanup: {total_cleaned} context entries, {global_cleaned} global entries")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        total_context_entries = sum(len(ctx.entries) for ctx in self.contexts.values())
        active_context_entries = sum(
            len([e for e in ctx.entries if not e.is_expired()]) 
            for ctx in self.contexts.values()
        )
        
        return {
            "active_contexts": len(self.contexts),
            "total_context_entries": total_context_entries,
            "active_context_entries": active_context_entries,
            "global_entries": len(self.global_entries),
            "active_global_entries": len([e for e in self.global_entries if not e.is_expired()]),
            "max_contexts": self.max_contexts,
            "max_entries_per_context": self.max_entries_per_context
        }

# Global unified memory instance
unified_memory = UnifiedMemory()

def create_memory_context(session_id: str, operation_id: str) -> str:
    """Create a memory context using the global instance."""
    return unified_memory.create_context(session_id, operation_id)

def add_memory(context_id: str, memory_type: MemoryType, content: Any,
               tags: List[str] = None, ttl: Optional[float] = None,
               metadata: Dict[str, Any] = None) -> str:
    """Add memory using the global instance."""
    return unified_memory.add_memory(context_id, memory_type, content, tags, ttl, metadata)

def get_memory(context_id: str, memory_type: Optional[MemoryType] = None,
               tags: Optional[List[str]] = None, limit: Optional[int] = None) -> List[Any]:
    """Get memory using the global instance."""
    return unified_memory.get_memory(context_id, memory_type, tags, limit)

def clear_memory_context(context_id: str) -> bool:
    """Clear a memory context using the global instance."""
    return unified_memory.clear_context(context_id) 