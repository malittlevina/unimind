"""
context_model.py – Context management and tracking for Unimind native models.
Provides conversation context, session state, and contextual information management.
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from collections import deque

@dataclass
class ContextEntry:
    """Represents a single context entry with metadata."""
    content: str
    timestamp: float
    source: str
    context_type: str
    metadata: Dict[str, Any]

class ContextModel:
    """
    Manages conversation context, session state, and contextual information.
    Provides methods for tracking, updating, and retrieving context.
    """
    
    def __init__(self, max_context_length: int = 100, context_window: int = 10):
        """
        Initialize the context model.
        
        Args:
            max_context_length: Maximum number of context entries to store
            context_window: Number of recent entries to consider for current context
        """
        self.max_context_length = max_context_length
        self.context_window = context_window
        self.context_history = deque(maxlen=max_context_length)
        self.session_data = {}
        self.global_context = {}
        
    def add_context(self, content: str, source: str = "user", context_type: str = "conversation", metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a new context entry.
        
        Args:
            content: The context content
            source: Source of the context (user, system, external, etc.)
            context_type: Type of context (conversation, memory, knowledge, etc.)
            metadata: Additional metadata for the context entry
        """
        entry = ContextEntry(
            content=content,
            timestamp=time.time(),
            source=source,
            context_type=context_type,
            metadata=metadata or {}
        )
        self.context_history.append(entry)
        
    def get_current_context(self, context_type: Optional[str] = None) -> List[ContextEntry]:
        """
        Get the current context window.
        
        Args:
            context_type: Filter by context type (optional)
            
        Returns:
            List of recent context entries
        """
        recent_entries = list(self.context_history)[-self.context_window:]
        if context_type:
            return [entry for entry in recent_entries if entry.context_type == context_type]
        return recent_entries
    
    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current context.
        
        Returns:
            Dictionary containing context summary
        """
        recent_entries = self.get_current_context()
        return {
            "total_entries": len(self.context_history),
            "recent_entries": len(recent_entries),
            "sources": list(set(entry.source for entry in recent_entries)),
            "types": list(set(entry.context_type for entry in recent_entries)),
            "latest_timestamp": recent_entries[-1].timestamp if recent_entries else None
        }
    
    def update_session_data(self, key: str, value: Any) -> None:
        """
        Update session-specific data.
        
        Args:
            key: Session data key
            value: Session data value
        """
        self.session_data[key] = value
        
    def get_session_data(self, key: str, default: Any = None) -> Any:
        """
        Get session-specific data.
        
        Args:
            key: Session data key
            default: Default value if key not found
            
        Returns:
            Session data value
        """
        return self.session_data.get(key, default)
    
    def update_global_context(self, key: str, value: Any) -> None:
        """
        Update global context data.
        
        Args:
            key: Global context key
            value: Global context value
        """
        self.global_context[key] = value
        
    def get_global_context(self, key: str, default: Any = None) -> Any:
        """
        Get global context data.
        
        Args:
            key: Global context key
            default: Default value if key not found
            
        Returns:
            Global context value
        """
        return self.global_context.get(key, default)
    
    def clear_context(self, context_type: Optional[str] = None) -> None:
        """
        Clear context entries.
        
        Args:
            context_type: Clear only entries of this type (optional)
        """
        if context_type:
            self.context_history = deque(
                [entry for entry in self.context_history if entry.context_type != context_type],
                maxlen=self.max_context_length
            )
        else:
            self.context_history.clear()
    
    def export_context(self) -> Dict[str, Any]:
        """
        Export context data for persistence.
        
        Returns:
            Dictionary containing all context data
        """
        return {
            "context_history": [asdict(entry) for entry in self.context_history],
            "session_data": self.session_data,
            "global_context": self.global_context,
            "max_context_length": self.max_context_length,
            "context_window": self.context_window
        }
    
    def import_context(self, context_data: Dict[str, Any]) -> None:
        """
        Import context data from persistence.
        
        Args:
            context_data: Context data to import
        """
        self.context_history = deque(
            [ContextEntry(**entry) for entry in context_data.get("context_history", [])],
            maxlen=context_data.get("max_context_length", self.max_context_length)
        )
        self.session_data = context_data.get("session_data", {})
        self.global_context = context_data.get("global_context", {})
        self.max_context_length = context_data.get("max_context_length", self.max_context_length)
        self.context_window = context_data.get("context_window", self.context_window)

# Module-level context model instance
context_model = ContextModel()

def add_context(content: str, source: str = "user", context_type: str = "conversation", metadata: Optional[Dict[str, Any]] = None) -> None:
    """Add context using the module-level instance."""
    context_model.add_context(content, source, context_type, metadata)

def get_current_context(context_type: Optional[str] = None) -> List[ContextEntry]:
    """Get current context using the module-level instance."""
    return context_model.get_current_context(context_type)
