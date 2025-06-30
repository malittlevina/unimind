"""
feedback_bus.py – Unified feedback system for ThothOS/Unimind.
Handles feedback events from modules and scrolls, with logging and adaptation triggers.
"""

import logging
import time
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime

class FeedbackType(Enum):
    """Types of feedback events."""
    SCROLL_SUCCESS = "scroll_success"
    SCROLL_FAILURE = "scroll_failure"
    MODULE_SUCCESS = "module_success"
    MODULE_FAILURE = "module_failure"
    EMOTION_CHANGE = "emotion_change"
    MEMORY_UPDATE = "memory_update"
    PERSONALITY_UPDATE = "personality_update"
    ETHICAL_CHECK = "ethical_check"
    SECURITY_ALERT = "security_alert"
    PERFORMANCE_METRIC = "performance_metric"
    USER_FEEDBACK = "user_feedback"
    SYSTEM_EVENT = "system_event"

class FeedbackLevel(Enum):
    """Levels of feedback importance."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class FeedbackEvent:
    """Represents a feedback event with metadata."""
    event_type: FeedbackType
    level: FeedbackLevel
    source: str
    message: str
    payload: Dict[str, Any]
    timestamp: float
    event_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "level": self.level.value,
            "source": self.source,
            "message": self.message,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "event_id": self.event_id,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat()
        }

class FeedbackBus:
    """
    Centralized feedback bus for ThothOS/Unimind.
    Handles event emission, subscription, logging, and adaptation triggers.
    """
    
    def __init__(self):
        """Initialize the feedback bus."""
        self.subscribers: List[Callable] = []
        self.feedback_history: List[FeedbackEvent] = []
        self.adaptation_triggers: Dict[str, List[Callable]] = {}
        self.logger = logging.getLogger('FeedbackBus')
        
        # Initialize logging
        self._setup_logging()
        
        # Register default adaptation triggers
        self._register_default_triggers()
    
    def _setup_logging(self):
        """Setup logging for feedback events."""
        # Create a file handler for feedback logs
        import os
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        feedback_handler = logging.FileHandler(f"{log_dir}/feedback.log")
        feedback_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        feedback_handler.setFormatter(formatter)
        
        self.logger.addHandler(feedback_handler)
        self.logger.setLevel(logging.INFO)
    
    def _register_default_triggers(self):
        """Register default adaptation triggers."""
        # Trigger personality updates on significant events
        self.register_adaptation_trigger("personality_update", self._handle_personality_update)
        
        # Trigger memory consolidation on important events
        self.register_adaptation_trigger("memory_update", self._handle_memory_update)
        
        # Trigger emotional adaptation
        self.register_adaptation_trigger("emotion_change", self._handle_emotion_change)
        
        # Trigger performance optimization
        self.register_adaptation_trigger("performance_metric", self._handle_performance_update)
    
    def subscribe(self, callback: Callable[[FeedbackEvent], None]) -> None:
        """
        Subscribe to feedback events.
        
        Args:
            callback: Function to call when feedback events occur
        """
        self.subscribers.append(callback)
        self.logger.info(f"New feedback subscriber registered: {callback.__name__}")
    
    def unsubscribe(self, callback: Callable[[FeedbackEvent], None]) -> bool:
        """
        Unsubscribe from feedback events.
        
        Args:
            callback: Function to remove from subscribers
            
        Returns:
            True if callback was found and removed
        """
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            self.logger.info(f"Feedback subscriber unregistered: {callback.__name__}")
            return True
        return False
    
    def emit(self, event_type: FeedbackType, source: str, message: str,
             payload: Dict[str, Any] = None, level: FeedbackLevel = FeedbackLevel.INFO) -> str:
        """
        Emit a feedback event.
        
        Args:
            event_type: Type of feedback event
            source: Source of the feedback (module/scroll name)
            message: Human-readable message
            payload: Additional data
            level: Importance level
            
        Returns:
            Event ID for tracking
        """
        payload = payload or {}
        
        # Generate unique event ID
        import hashlib
        event_id = hashlib.md5(
            f"{event_type.value}:{source}:{time.time()}".encode()
        ).hexdigest()[:8]
        
        # Create feedback event
        event = FeedbackEvent(
            event_type=event_type,
            level=level,
            source=source,
            message=message,
            payload=payload,
            timestamp=time.time(),
            event_id=event_id
        )
        
        # Store in history
        self.feedback_history.append(event)
        
        # Notify subscribers
        for callback in self.subscribers:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Error in feedback subscriber {callback.__name__}: {e}")
        
        # Check for adaptation triggers
        self._check_adaptation_triggers(event)
        
        # Log the event
        log_message = f"[{event_type.value}] {source}: {message}"
        if level == FeedbackLevel.DEBUG:
            self.logger.debug(log_message)
        elif level == FeedbackLevel.INFO:
            self.logger.info(log_message)
        elif level == FeedbackLevel.WARNING:
            self.logger.warning(log_message)
        elif level == FeedbackLevel.ERROR:
            self.logger.error(log_message)
        elif level == FeedbackLevel.CRITICAL:
            self.logger.critical(log_message)
        
        return event_id
    
    def register_adaptation_trigger(self, trigger_type: str, callback: Callable[[FeedbackEvent], None]) -> None:
        """
        Register an adaptation trigger for specific event types.
        
        Args:
            trigger_type: Type of event to trigger on
            callback: Function to call when trigger is activated
        """
        if trigger_type not in self.adaptation_triggers:
            self.adaptation_triggers[trigger_type] = []
        self.adaptation_triggers[trigger_type].append(callback)
        self.logger.info(f"Registered adaptation trigger: {trigger_type} -> {callback.__name__}")
    
    def _check_adaptation_triggers(self, event: FeedbackEvent) -> None:
        """Check if any adaptation triggers should be activated."""
        trigger_type = event.event_type.value
        
        if trigger_type in self.adaptation_triggers:
            for callback in self.adaptation_triggers[trigger_type]:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"Error in adaptation trigger {callback.__name__}: {e}")
    
    def _handle_personality_update(self, event: FeedbackEvent) -> None:
        """Handle personality update triggers."""
        # This would update the soul/personality based on feedback
        self.logger.info(f"Personality update triggered by: {event.source}")
        
        # Example: Update confidence based on success/failure
        if event.event_type == FeedbackType.SCROLL_SUCCESS:
            # Increase confidence
            pass
        elif event.event_type == FeedbackType.SCROLL_FAILURE:
            # Decrease confidence, trigger reflection
            pass
    
    def _handle_memory_update(self, event: FeedbackEvent) -> None:
        """Handle memory update triggers."""
        # This would consolidate memory based on feedback
        self.logger.info(f"Memory update triggered by: {event.source}")
    
    def _handle_emotion_change(self, event: FeedbackEvent) -> None:
        """Handle emotional adaptation triggers."""
        # This would adjust emotional state based on feedback
        self.logger.info(f"Emotion change triggered by: {event.source}")
    
    def _handle_performance_update(self, event: FeedbackEvent) -> None:
        """Handle performance optimization triggers."""
        # This would optimize performance based on metrics
        self.logger.info(f"Performance update triggered by: {event.source}")
    
    def get_feedback_history(self, event_type: Optional[FeedbackType] = None,
                           source: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get feedback history with optional filtering.
        
        Args:
            event_type: Filter by event type
            source: Filter by source
            limit: Maximum number of events to return
            
        Returns:
            List of feedback events as dictionaries
        """
        filtered_events = self.feedback_history
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
        if source:
            filtered_events = [e for e in filtered_events if e.source == source]
        
        # Sort by timestamp (newest first) and limit
        filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
        filtered_events = filtered_events[:limit]
        
        return [event.to_dict() for event in filtered_events]
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        if not self.feedback_history:
            return {
                "total_events": 0,
                "events_by_type": {},
                "events_by_source": {},
                "events_by_level": {}
            }
        
        # Count events by type
        events_by_type = {}
        events_by_source = {}
        events_by_level = {}
        
        for event in self.feedback_history:
            # By type
            event_type = event.event_type.value
            events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
            
            # By source
            source = event.source
            events_by_source[source] = events_by_source.get(source, 0) + 1
            
            # By level
            level = event.level.value
            events_by_level[level] = events_by_level.get(level, 0) + 1
        
        return {
            "total_events": len(self.feedback_history),
            "events_by_type": events_by_type,
            "events_by_source": events_by_source,
            "events_by_level": events_by_level,
            "subscribers_count": len(self.subscribers),
            "adaptation_triggers_count": len(self.adaptation_triggers)
        }
    
    def clear_history(self) -> None:
        """Clear feedback history."""
        self.feedback_history.clear()
        self.logger.info("Feedback history cleared")

# Global feedback bus instance
feedback_bus = FeedbackBus()

def emit_feedback(event_type: FeedbackType, source: str, message: str,
                 payload: Dict[str, Any] = None, level: FeedbackLevel = FeedbackLevel.INFO) -> str:
    """Emit feedback using the global feedback bus instance."""
    return feedback_bus.emit(event_type, source, message, payload, level)

def subscribe_to_feedback(callback: Callable[[FeedbackEvent], None]) -> None:
    """Subscribe to feedback using the global feedback bus instance."""
    feedback_bus.subscribe(callback)

def get_feedback_history(event_type: Optional[FeedbackType] = None,
                        source: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Get feedback history using the global feedback bus instance."""
    return feedback_bus.get_feedback_history(event_type, source, limit)

def get_feedback_stats() -> Dict[str, Any]:
    """Get feedback statistics using the global feedback bus instance."""
    return feedback_bus.get_feedback_stats() 