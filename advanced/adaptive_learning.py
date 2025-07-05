"""
adaptive_learning.py - Advanced adaptive learning system for UniMind.
Enables real-time learning from interactions, pattern recognition, and continuous improvement.
"""

import logging
import time
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import threading

class LearningType(Enum):
    """Types of learning that can occur."""
    PATTERN_RECOGNITION = "pattern_recognition"
    BEHAVIOR_ADAPTATION = "behavior_adaptation"
    SKILL_ACQUISITION = "skill_acquisition"
    KNOWLEDGE_INTEGRATION = "knowledge_integration"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"

@dataclass
class LearningEvent:
    """Represents a learning event."""
    event_id: str
    learning_type: LearningType
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    success: bool
    confidence: float
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LearningPattern:
    """Represents a learned pattern."""
    pattern_id: str
    pattern_type: str
    frequency: int
    confidence: float
    first_seen: float
    last_seen: float
    success_rate: float
    context_conditions: Dict[str, Any]
    adaptation_rules: List[Dict[str, Any]]

class AdaptiveLearningEngine:
    """
    Advanced adaptive learning engine that enables UniMind to learn and improve continuously.
    """
    
    def __init__(self, learning_rate: float = 0.1, memory_decay: float = 0.95):
        """Initialize the adaptive learning engine."""
        self.logger = logging.getLogger('AdaptiveLearningEngine')
        self.learning_rate = learning_rate
        self.memory_decay = memory_decay
        
        # Learning storage
        self.learning_events: List[LearningEvent] = []
        self.learned_patterns: Dict[str, LearningPattern] = {}
        self.adaptation_rules: Dict[str, List[Dict[str, Any]]] = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_learning_events': 0,
            'successful_adaptations': 0,
            'pattern_recognition_accuracy': 0.0,
            'learning_efficiency': 0.0,
            'adaptation_speed': 0.0
        }
        
        # Threading
        self.lock = threading.RLock()
        self.is_learning = True
        
        # Learning algorithms
        self.pattern_recognizer = PatternRecognizer()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.skill_optimizer = SkillOptimizer()
        
        self.logger.info("Adaptive learning engine initialized")
    
    def learn_from_interaction(self, user_input: str, system_response: str, 
                             success: bool, confidence: float, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Learn from a user interaction.
        
        Args:
            user_input: User's input
            system_response: System's response
            success: Whether the interaction was successful
            confidence: Confidence level of the response
            context: Additional context
            
        Returns:
            Learning insights and adaptations
        """
        with self.lock:
            # Create learning event
            event = LearningEvent(
                event_id=f"event_{int(time.time() * 1000)}",
                learning_type=self._determine_learning_type(user_input, context),
                input_data={"user_input": user_input, "context": context},
                output_data={"system_response": system_response, "confidence": confidence},
                success=success,
                confidence=confidence,
                timestamp=time.time(),
                context=context
            )
            
            self.learning_events.append(event)
            self.performance_metrics['total_learning_events'] += 1
            
            # Analyze for patterns
            patterns = self.pattern_recognizer.analyze_event(event)
            
            # Update learned patterns
            for pattern in patterns:
                self._update_pattern(pattern)
            
            # Generate adaptations
            adaptations = self._generate_adaptations(event, patterns)
            
            # Apply adaptations
            applied_adaptations = self._apply_adaptations(adaptations)
            
            # Update performance metrics
            self._update_performance_metrics(event, adaptations, applied_adaptations)
            
            return {
                "learning_insights": {
                    "patterns_found": len(patterns),
                    "adaptations_generated": len(adaptations),
                    "adaptations_applied": len(applied_adaptations),
                    "learning_efficiency": self.performance_metrics['learning_efficiency']
                },
                "adaptations": applied_adaptations,
                "patterns": [p.pattern_id for p in patterns]
            }
    
    def _determine_learning_type(self, user_input: str, context: Dict[str, Any]) -> LearningType:
        """Determine the type of learning from the interaction."""
        # Analyze input for learning type indicators
        input_lower = user_input.lower()
        
        if any(word in input_lower for word in ['pattern', 'trend', 'similar', 'like']):
            return LearningType.PATTERN_RECOGNITION
        elif any(word in input_lower for word in ['behavior', 'act', 'respond', 'react']):
            return LearningType.BEHAVIOR_ADAPTATION
        elif any(word in input_lower for word in ['skill', 'ability', 'capability', 'learn']):
            return LearningType.SKILL_ACQUISITION
        elif any(word in input_lower for word in ['knowledge', 'information', 'fact', 'data']):
            return LearningType.KNOWLEDGE_INTEGRATION
        else:
            return LearningType.PERFORMANCE_OPTIMIZATION
    
    def _update_pattern(self, pattern: LearningPattern):
        """Update or create a learned pattern."""
        if pattern.pattern_id in self.learned_patterns:
            existing = self.learned_patterns[pattern.pattern_id]
            existing.frequency += 1
            existing.last_seen = time.time()
            existing.confidence = (existing.confidence + pattern.confidence) / 2
        else:
            self.learned_patterns[pattern.pattern_id] = pattern
    
    def _generate_adaptations(self, event: LearningEvent, patterns: List[LearningPattern]) -> List[Dict[str, Any]]:
        """Generate adaptations based on learning event and patterns."""
        adaptations = []
        
        # Behavior adaptation
        if event.learning_type == LearningType.BEHAVIOR_ADAPTATION:
            behavior_adaptation = self.behavior_analyzer.generate_adaptation(event, patterns)
            if behavior_adaptation:
                adaptations.append(behavior_adaptation)
        
        # Skill optimization
        if event.learning_type == LearningType.SKILL_ACQUISITION:
            skill_adaptation = self.skill_optimizer.optimize_skill(event, patterns)
            if skill_adaptation:
                adaptations.append(skill_adaptation)
        
        # Performance optimization
        if event.learning_type == LearningType.PERFORMANCE_OPTIMIZATION:
            performance_adaptation = self._generate_performance_adaptation(event)
            if performance_adaptation:
                adaptations.append(performance_adaptation)
        
        return adaptations
    
    def _apply_adaptations(self, adaptations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply generated adaptations to the system."""
        applied = []
        
        for adaptation in adaptations:
            try:
                # Apply the adaptation
                if self._execute_adaptation(adaptation):
                    applied.append(adaptation)
                    self.performance_metrics['successful_adaptations'] += 1
            except Exception as e:
                self.logger.error(f"Failed to apply adaptation: {e}")
        
        return applied
    
    def _execute_adaptation(self, adaptation: Dict[str, Any]) -> bool:
        """Execute a specific adaptation."""
        adaptation_type = adaptation.get('type')
        
        if adaptation_type == 'behavior_change':
            return self._apply_behavior_change(adaptation)
        elif adaptation_type == 'skill_enhancement':
            return self._apply_skill_enhancement(adaptation)
        elif adaptation_type == 'performance_tuning':
            return self._apply_performance_tuning(adaptation)
        else:
            return False
    
    def _apply_behavior_change(self, adaptation: Dict[str, Any]) -> bool:
        """Apply a behavior change adaptation."""
        # Implementation for behavior changes
        return True
    
    def _apply_skill_enhancement(self, adaptation: Dict[str, Any]) -> bool:
        """Apply a skill enhancement adaptation."""
        # Implementation for skill enhancements
        return True
    
    def _apply_performance_tuning(self, adaptation: Dict[str, Any]) -> bool:
        """Apply a performance tuning adaptation."""
        # Implementation for performance tuning
        return True
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from the learning system."""
        with self.lock:
            return {
                "performance_metrics": self.performance_metrics.copy(),
                "total_patterns": len(self.learned_patterns),
                "recent_learning_events": len([e for e in self.learning_events if time.time() - e.timestamp < 3600]),
                "learning_efficiency": self._calculate_learning_efficiency(),
                "top_patterns": self._get_top_patterns(5)
            }
    
    def _calculate_learning_efficiency(self) -> float:
        """Calculate current learning efficiency."""
        if not self.learning_events:
            return 0.0
        
        recent_events = [e for e in self.learning_events if time.time() - e.timestamp < 3600]
        if not recent_events:
            return 0.0
        
        success_rate = sum(1 for e in recent_events if e.success) / len(recent_events)
        avg_confidence = sum(e.confidence for e in recent_events) / len(recent_events)
        
        return (success_rate + avg_confidence) / 2
    
    def _get_top_patterns(self, count: int) -> List[Dict[str, Any]]:
        """Get the top patterns by frequency and confidence."""
        sorted_patterns = sorted(
            self.learned_patterns.values(),
            key=lambda p: (p.frequency, p.confidence),
            reverse=True
        )
        
        return [
            {
                "pattern_id": p.pattern_id,
                "type": p.pattern_type,
                "frequency": p.frequency,
                "confidence": p.confidence,
                "success_rate": p.success_rate
            }
            for p in sorted_patterns[:count]
        ]

class PatternRecognizer:
    """Recognizes patterns in learning events."""
    
    def analyze_event(self, event: LearningEvent) -> List[LearningPattern]:
        """Analyze an event for patterns."""
        patterns = []
        
        # Simple pattern recognition based on input characteristics
        input_text = event.input_data.get('user_input', '')
        
        # Pattern: Question asking
        if '?' in input_text:
            patterns.append(LearningPattern(
                pattern_id=f"question_pattern_{len(patterns)}",
                pattern_type="question_asking",
                frequency=1,
                confidence=0.8,
                first_seen=event.timestamp,
                last_seen=event.timestamp,
                success_rate=1.0 if event.success else 0.0,
                context_conditions={"contains_question": True},
                adaptation_rules=[]
            ))
        
        # Pattern: Command execution
        if any(word in input_text.lower() for word in ['do', 'execute', 'run', 'perform']):
            patterns.append(LearningPattern(
                pattern_id=f"command_pattern_{len(patterns)}",
                pattern_type="command_execution",
                frequency=1,
                confidence=0.7,
                first_seen=event.timestamp,
                last_seen=event.timestamp,
                success_rate=1.0 if event.success else 0.0,
                context_conditions={"contains_command": True},
                adaptation_rules=[]
            ))
        
        return patterns

class BehaviorAnalyzer:
    """Analyzes and adapts behavior patterns."""
    
    def generate_adaptation(self, event: LearningEvent, patterns: List[LearningPattern]) -> Optional[Dict[str, Any]]:
        """Generate behavior adaptation based on event and patterns."""
        if not event.success:
            return {
                "type": "behavior_change",
                "description": "Improve response quality for failed interactions",
                "action": "increase_confidence_threshold",
                "parameters": {"threshold_increase": 0.1}
            }
        return None

class SkillOptimizer:
    """Optimizes skills based on learning events."""
    
    def optimize_skill(self, event: LearningEvent, patterns: List[LearningPattern]) -> Optional[Dict[str, Any]]:
        """Optimize skills based on event and patterns."""
        if event.confidence < 0.5:
            return {
                "type": "skill_enhancement",
                "description": "Enhance skill for low-confidence responses",
                "action": "improve_skill_accuracy",
                "parameters": {"learning_rate_increase": 0.05}
            }
        return None

# Global instance
adaptive_learning_engine = AdaptiveLearningEngine()

# Convenience functions
def learn_from_interaction(user_input: str, system_response: str, success: bool, 
                          confidence: float, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Learn from a user interaction using the global engine."""
    return adaptive_learning_engine.learn_from_interaction(user_input, system_response, success, confidence, context or {})

def get_learning_insights() -> Dict[str, Any]:
    """Get learning insights using the global engine."""
    return adaptive_learning_engine.get_learning_insights()

# Module exports
__all__ = [
    'AdaptiveLearningEngine', 'LearningEvent', 'LearningPattern', 'LearningType',
    'adaptive_learning_engine', 'learn_from_interaction', 'get_learning_insights'
] 