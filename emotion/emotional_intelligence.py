"""
emotional_intelligence.py - Advanced emotional intelligence system for UniMind.
Enables understanding, responding to, and adapting to emotional states and contexts.
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

class EmotionType(Enum):
    """Primary emotion types."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"
    EXCITEMENT = "excitement"
    ANXIETY = "anxiety"
    CONTENTMENT = "contentment"
    FRUSTRATION = "frustration"
    HOPE = "hope"

class EmotionalIntensity(Enum):
    """Emotional intensity levels."""
    VERY_LOW = 0.1
    LOW = 0.3
    MODERATE = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9

@dataclass
class EmotionalState:
    """Represents the current emotional state."""
    primary_emotion: EmotionType
    intensity: float
    secondary_emotions: List[Tuple[EmotionType, float]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    triggers: List[str] = field(default_factory=list)

@dataclass
class EmotionalResponse:
    """Represents an emotional response strategy."""
    response_type: str
    emotional_tone: EmotionType
    intensity: float
    response_text: str
    adaptation_strategy: Dict[str, Any]
    empathy_level: float
    support_actions: List[str]

class EmotionalIntelligenceEngine:
    """
    Advanced emotional intelligence engine for UniMind.
    Enables understanding, responding to, and adapting to emotional states.
    """
    
    def __init__(self):
        """Initialize the emotional intelligence engine."""
        self.logger = logging.getLogger('EmotionalIntelligenceEngine')
        
        # Emotional state tracking
        self.current_emotional_state = EmotionalState(
            primary_emotion=EmotionType.NEUTRAL,
            intensity=0.3
        )
        self.emotional_history: List[EmotionalState] = []
        
        # Response strategies
        self.response_strategies = self._initialize_response_strategies()
        self.empathy_patterns = self._initialize_empathy_patterns()
        
        # Emotional learning
        self.emotional_patterns: Dict[str, Dict[str, Any]] = {}
        self.adaptation_rules: Dict[str, List[Dict[str, Any]]] = {}
        
        # Performance tracking
        self.performance_metrics = {
            'emotional_accuracy': 0.0,
            'empathy_effectiveness': 0.0,
            'response_appropriateness': 0.0,
            'adaptation_success_rate': 0.0
        }
        
        # Threading
        self.lock = threading.RLock()
        
        self.logger.info("Emotional intelligence engine initialized")
    
    def analyze_emotional_state(self, user_input: str, context: Dict[str, Any] = None) -> EmotionalState:
        """
        Analyze the emotional state from user input and context.
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Analyzed emotional state
        """
        context = context or {}
        
        # Analyze text for emotional indicators
        emotional_indicators = self._extract_emotional_indicators(user_input)
        
        # Determine primary emotion
        primary_emotion = self._determine_primary_emotion(emotional_indicators)
        
        # Calculate intensity
        intensity = self._calculate_emotional_intensity(emotional_indicators, context)
        
        # Identify secondary emotions
        secondary_emotions = self._identify_secondary_emotions(emotional_indicators)
        
        # Identify triggers
        triggers = self._identify_emotional_triggers(user_input, context)
        
        # Create emotional state
        emotional_state = EmotionalState(
            primary_emotion=primary_emotion,
            intensity=intensity,
            secondary_emotions=secondary_emotions,
            context=context,
            triggers=triggers
        )
        
        # Update current state
        with self.lock:
            self.current_emotional_state = emotional_state
            self.emotional_history.append(emotional_state)
            
            # Keep only recent history
            if len(self.emotional_history) > 100:
                self.emotional_history = self.emotional_history[-100:]
        
        return emotional_state
    
    def generate_emotional_response(self, emotional_state: EmotionalState, 
                                  user_input: str, context: Dict[str, Any] = None) -> EmotionalResponse:
        """
        Generate an emotionally appropriate response.
        
        Args:
            emotional_state: Current emotional state
            user_input: User's input
            context: Additional context
            
        Returns:
            Emotionally appropriate response
        """
        context = context or {}
        
        # Select response strategy
        strategy = self._select_response_strategy(emotional_state, context)
        
        # Generate response text
        response_text = self._generate_response_text(strategy, emotional_state, user_input)
        
        # Determine adaptation strategy
        adaptation_strategy = self._determine_adaptation_strategy(emotional_state, context)
        
        # Calculate empathy level
        empathy_level = self._calculate_empathy_level(emotional_state, strategy)
        
        # Identify support actions
        support_actions = self._identify_support_actions(emotional_state, context)
        
        return EmotionalResponse(
            response_type=strategy['type'],
            emotional_tone=strategy['emotional_tone'],
            intensity=strategy['intensity'],
            response_text=response_text,
            adaptation_strategy=adaptation_strategy,
            empathy_level=empathy_level,
            support_actions=support_actions
        )
    
    def adapt_to_emotional_context(self, emotional_state: EmotionalState, 
                                 interaction_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Adapt system behavior based on emotional context.
        
        Args:
            emotional_state: Current emotional state
            interaction_history: History of interactions
            
        Returns:
            Adaptation recommendations
        """
        adaptations = {}
        
        # Adapt response style
        if emotional_state.intensity > 0.7:
            adaptations['response_style'] = {
                'tone': 'calming',
                'pace': 'slower',
                'detail_level': 'high'
            }
        elif emotional_state.intensity < 0.3:
            adaptations['response_style'] = {
                'tone': 'engaging',
                'pace': 'normal',
                'detail_level': 'moderate'
            }
        
        # Adapt based on emotion type
        if emotional_state.primary_emotion == EmotionType.FRUSTRATION:
            adaptations['support_focus'] = 'problem_solving'
        elif emotional_state.primary_emotion == EmotionType.ANXIETY:
            adaptations['support_focus'] = 'reassurance'
        elif emotional_state.primary_emotion == EmotionType.JOY:
            adaptations['support_focus'] = 'celebration'
        
        # Adapt based on emotional patterns
        emotional_patterns = self._analyze_emotional_patterns(interaction_history)
        if emotional_patterns.get('increasing_negative'):
            adaptations['intervention'] = 'emotional_support'
        
        return adaptations
    
    def _extract_emotional_indicators(self, text: str) -> Dict[str, float]:
        """Extract emotional indicators from text."""
        indicators = defaultdict(float)
        text_lower = text.lower()
        
        # Joy indicators
        joy_words = ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'fantastic']
        for word in joy_words:
            if word in text_lower:
                indicators['joy'] += 0.3
        
        # Sadness indicators
        sadness_words = ['sad', 'depressed', 'unhappy', 'miserable', 'terrible', 'awful']
        for word in sadness_words:
            if word in text_lower:
                indicators['sadness'] += 0.3
        
        # Anger indicators
        anger_words = ['angry', 'mad', 'furious', 'upset', 'annoyed', 'frustrated']
        for word in anger_words:
            if word in text_lower:
                indicators['anger'] += 0.3
        
        # Fear indicators
        fear_words = ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'terrified']
        for word in fear_words:
            if word in text_lower:
                indicators['fear'] += 0.3
        
        # Excitement indicators
        excitement_words = ['excited', 'thrilled', 'eager', 'enthusiastic', 'pumped']
        for word in excitement_words:
            if word in text_lower:
                indicators['excitement'] += 0.3
        
        # Frustration indicators
        frustration_words = ['frustrated', 'annoyed', 'irritated', 'bothered', 'tired of']
        for word in frustration_words:
            if word in text_lower:
                indicators['frustration'] += 0.3
        
        # Punctuation analysis
        if '!' in text:
            indicators['excitement'] += 0.2
        if '?' in text:
            indicators['anxiety'] += 0.1
        if text.count('.') > 3:
            indicators['sadness'] += 0.1
        
        return dict(indicators)
    
    def _determine_primary_emotion(self, indicators: Dict[str, float]) -> EmotionType:
        """Determine the primary emotion from indicators."""
        if not indicators:
            return EmotionType.NEUTRAL
        
        # Find the emotion with highest score
        primary_emotion = max(indicators.items(), key=lambda x: x[1])
        
        # Map to EmotionType
        emotion_mapping = {
            'joy': EmotionType.JOY,
            'sadness': EmotionType.SADNESS,
            'anger': EmotionType.ANGER,
            'fear': EmotionType.FEAR,
            'excitement': EmotionType.EXCITEMENT,
            'frustration': EmotionType.FRUSTRATION,
            'anxiety': EmotionType.ANXIETY
        }
        
        return emotion_mapping.get(primary_emotion[0], EmotionType.NEUTRAL)
    
    def _calculate_emotional_intensity(self, indicators: Dict[str, float], context: Dict[str, Any]) -> float:
        """Calculate emotional intensity."""
        if not indicators:
            return 0.3  # Neutral intensity
        
        # Base intensity from indicators
        max_indicator = max(indicators.values()) if indicators else 0
        
        # Adjust based on context
        context_multiplier = 1.0
        if context.get('urgent'):
            context_multiplier = 1.3
        if context.get('casual'):
            context_multiplier = 0.7
        
        intensity = min(max_indicator * context_multiplier, 1.0)
        return max(intensity, 0.1)  # Minimum intensity
    
    def _identify_secondary_emotions(self, indicators: Dict[str, float]) -> List[Tuple[EmotionType, float]]:
        """Identify secondary emotions."""
        secondary = []
        emotion_mapping = {
            'joy': EmotionType.JOY,
            'sadness': EmotionType.SADNESS,
            'anger': EmotionType.ANGER,
            'fear': EmotionType.FEAR,
            'excitement': EmotionType.EXCITEMENT,
            'frustration': EmotionType.FRUSTRATION,
            'anxiety': EmotionType.ANXIETY
        }
        
        # Sort by intensity and take top 2
        sorted_indicators = sorted(indicators.items(), key=lambda x: x[1], reverse=True)
        
        for emotion_name, intensity in sorted_indicators[1:3]:  # Skip primary
            if emotion_name in emotion_mapping and intensity > 0.1:
                secondary.append((emotion_mapping[emotion_name], intensity))
        
        return secondary
    
    def _identify_emotional_triggers(self, text: str, context: Dict[str, Any]) -> List[str]:
        """Identify emotional triggers."""
        triggers = []
        
        # Common emotional triggers
        trigger_patterns = {
            'failure': ['failed', 'error', 'wrong', 'mistake', 'broken'],
            'success': ['success', 'achieved', 'completed', 'won', 'solved'],
            'time_pressure': ['urgent', 'deadline', 'hurry', 'quick', 'fast'],
            'uncertainty': ['maybe', 'perhaps', 'not sure', 'uncertain', 'doubt'],
            'rejection': ['no', 'rejected', 'denied', 'refused', 'can\'t']
        }
        
        text_lower = text.lower()
        for trigger_type, words in trigger_patterns.items():
            if any(word in text_lower for word in words):
                triggers.append(trigger_type)
        
        # Context-based triggers
        if context.get('error_occurred'):
            triggers.append('system_error')
        if context.get('timeout'):
            triggers.append('timeout')
        
        return triggers
    
    def _initialize_response_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize response strategies for different emotions."""
        return {
            'joy': {
                'type': 'celebratory',
                'emotional_tone': EmotionType.JOY,
                'intensity': 0.6,
                'empathy_style': 'shared_enthusiasm'
            },
            'sadness': {
                'type': 'supportive',
                'emotional_tone': EmotionType.CONTENTMENT,
                'intensity': 0.4,
                'empathy_style': 'comforting'
            },
            'anger': {
                'type': 'calming',
                'emotional_tone': EmotionType.NEUTRAL,
                'intensity': 0.3,
                'empathy_style': 'understanding'
            },
            'fear': {
                'type': 'reassuring',
                'emotional_tone': EmotionType.HOPE,
                'intensity': 0.5,
                'empathy_style': 'protective'
            },
            'frustration': {
                'type': 'problem_solving',
                'emotional_tone': EmotionType.HOPE,
                'intensity': 0.4,
                'empathy_style': 'collaborative'
            },
            'excitement': {
                'type': 'enthusiastic',
                'emotional_tone': EmotionType.EXCITEMENT,
                'intensity': 0.7,
                'empathy_style': 'shared_enthusiasm'
            }
        }
    
    def _initialize_empathy_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize empathy patterns."""
        return {
            'comforting': {
                'acknowledgment': True,
                'validation': True,
                'support_offering': True,
                'positive_reinforcement': False
            },
            'understanding': {
                'acknowledgment': True,
                'validation': True,
                'support_offering': False,
                'positive_reinforcement': False
            },
            'shared_enthusiasm': {
                'acknowledgment': True,
                'validation': True,
                'support_offering': False,
                'positive_reinforcement': True
            },
            'collaborative': {
                'acknowledgment': True,
                'validation': True,
                'support_offering': True,
                'positive_reinforcement': True
            }
        }
    
    def _select_response_strategy(self, emotional_state: EmotionalState, context: Dict[str, Any]) -> Dict[str, Any]:
        """Select appropriate response strategy."""
        primary_emotion = emotional_state.primary_emotion
        
        # Get base strategy
        strategy = self.response_strategies.get(primary_emotion.value, {
            'type': 'neutral',
            'emotional_tone': EmotionType.NEUTRAL,
            'intensity': 0.3,
            'empathy_style': 'understanding'
        })
        
        # Adjust based on intensity
        if emotional_state.intensity > 0.8:
            strategy['intensity'] = min(strategy['intensity'] + 0.2, 1.0)
        elif emotional_state.intensity < 0.3:
            strategy['intensity'] = max(strategy['intensity'] - 0.2, 0.1)
        
        return strategy
    
    def _generate_response_text(self, strategy: Dict[str, Any], emotional_state: EmotionalState, 
                              user_input: str) -> str:
        """Generate response text based on strategy."""
        empathy_style = strategy.get('empathy_style', 'understanding')
        empathy_pattern = self.empathy_patterns.get(empathy_style, {})
        
        response_parts = []
        
        # Acknowledgment
        if empathy_pattern.get('acknowledgment'):
            response_parts.append("I understand how you're feeling.")
        
        # Validation
        if empathy_pattern.get('validation'):
            if emotional_state.primary_emotion == EmotionType.FRUSTRATION:
                response_parts.append("It's completely understandable to feel frustrated in this situation.")
            elif emotional_state.primary_emotion == EmotionType.JOY:
                response_parts.append("That's wonderful! I'm glad you're feeling this way.")
            elif emotional_state.primary_emotion == EmotionType.SADNESS:
                response_parts.append("I can see this is affecting you deeply.")
        
        # Support offering
        if empathy_pattern.get('support_offering'):
            response_parts.append("I'm here to help you through this.")
        
        # Positive reinforcement
        if empathy_pattern.get('positive_reinforcement'):
            response_parts.append("You're doing great!")
        
        return " ".join(response_parts) if response_parts else "I'm here to help."
    
    def _determine_adaptation_strategy(self, emotional_state: EmotionalState, context: Dict[str, Any]) -> Dict[str, Any]:
        """Determine adaptation strategy based on emotional state."""
        adaptations = {}
        
        # Adapt response style based on emotion
        if emotional_state.primary_emotion in [EmotionType.ANGER, EmotionType.FRUSTRATION]:
            adaptations['response_style'] = 'calm_and_patient'
        elif emotional_state.primary_emotion == EmotionType.SADNESS:
            adaptations['response_style'] = 'gentle_and_supportive'
        elif emotional_state.primary_emotion == EmotionType.JOY:
            adaptations['response_style'] = 'enthusiastic_and_engaging'
        
        # Adapt based on intensity
        if emotional_state.intensity > 0.7:
            adaptations['pace'] = 'slower'
            adaptations['detail_level'] = 'high'
        elif emotional_state.intensity < 0.3:
            adaptations['pace'] = 'normal'
            adaptations['detail_level'] = 'moderate'
        
        return adaptations
    
    def _calculate_empathy_level(self, emotional_state: EmotionalState, strategy: Dict[str, Any]) -> float:
        """Calculate empathy level for the response."""
        base_empathy = 0.5
        
        # Increase empathy for negative emotions
        if emotional_state.primary_emotion in [EmotionType.SADNESS, EmotionType.ANGER, EmotionType.FEAR]:
            base_empathy += 0.3
        
        # Increase empathy for high intensity
        if emotional_state.intensity > 0.7:
            base_empathy += 0.2
        
        # Adjust based on strategy
        if strategy.get('empathy_style') == 'comforting':
            base_empathy += 0.1
        
        return min(base_empathy, 1.0)
    
    def _identify_support_actions(self, emotional_state: EmotionalState, context: Dict[str, Any]) -> List[str]:
        """Identify support actions based on emotional state."""
        actions = []
        
        if emotional_state.primary_emotion == EmotionType.FRUSTRATION:
            actions.extend(['problem_solving', 'break_down_task', 'offer_alternatives'])
        elif emotional_state.primary_emotion == EmotionType.SADNESS:
            actions.extend(['emotional_support', 'positive_reinforcement', 'distraction'])
        elif emotional_state.primary_emotion == EmotionType.ANXIETY:
            actions.extend(['reassurance', 'step_by_step_guidance', 'calming_techniques'])
        elif emotional_state.primary_emotion == EmotionType.JOY:
            actions.extend(['celebration', 'encouragement', 'shared_enthusiasm'])
        
        return actions
    
    def _analyze_emotional_patterns(self, interaction_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze emotional patterns in interaction history."""
        if not interaction_history:
            return {}
        
        # Analyze recent emotions
        recent_emotions = [h.get('emotion', 'neutral') for h in interaction_history[-5:]]
        
        patterns = {}
        
        # Check for increasing negative emotions
        negative_emotions = ['sadness', 'anger', 'fear', 'frustration', 'anxiety']
        negative_count = sum(1 for e in recent_emotions if e in negative_emotions)
        
        if negative_count >= 3:
            patterns['increasing_negative'] = True
        
        # Check for emotional stability
        if len(set(recent_emotions)) <= 2:
            patterns['emotionally_stable'] = True
        
        return patterns
    
    def get_emotional_insights(self) -> Dict[str, Any]:
        """Get insights about emotional intelligence performance."""
        with self.lock:
            return {
                "current_emotional_state": {
                    "primary_emotion": self.current_emotional_state.primary_emotion.value,
                    "intensity": self.current_emotional_state.intensity,
                    "confidence": self.current_emotional_state.confidence
                },
                "performance_metrics": self.performance_metrics.copy(),
                "emotional_history_length": len(self.emotional_history),
                "recent_emotions": [
                    {
                        "emotion": state.primary_emotion.value,
                        "intensity": state.intensity,
                        "timestamp": state.timestamp
                    }
                    for state in self.emotional_history[-10:]
                ]
            }

# Global instance
emotional_intelligence_engine = EmotionalIntelligenceEngine()

# Convenience functions
def analyze_emotional_state(user_input: str, context: Dict[str, Any] = None) -> EmotionalState:
    """Analyze emotional state using the global engine."""
    return emotional_intelligence_engine.analyze_emotional_state(user_input, context)

def generate_emotional_response(emotional_state: EmotionalState, user_input: str, 
                              context: Dict[str, Any] = None) -> EmotionalResponse:
    """Generate emotional response using the global engine."""
    return emotional_intelligence_engine.generate_emotional_response(emotional_state, user_input, context)

def get_emotional_insights() -> Dict[str, Any]:
    """Get emotional insights using the global engine."""
    return emotional_intelligence_engine.get_emotional_insights()

# Module exports
__all__ = [
    'EmotionalIntelligenceEngine', 'EmotionalState', 'EmotionalResponse', 'EmotionType',
    'emotional_intelligence_engine', 'analyze_emotional_state', 'generate_emotional_response',
    'get_emotional_insights'
] 