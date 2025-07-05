"""
conversation_memory.py – Enhanced conversation memory and context tracking for LAM system
=====================================================================================

Advanced features:
- Semantic analysis and topic modeling
- Emotional intelligence and sentiment analysis
- Multi-modal context understanding
- Adaptive learning and personalization
- Hierarchical memory organization
- Real-time context optimization
"""

import time
import logging
import re
import json
import hashlib
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timedelta
from enum import Enum

class MemoryType(Enum):
    """Types of memory storage."""
    SHORT_TERM = "short_term"
    WORKING = "working"
    LONG_TERM = "long_term"
    SEMANTIC = "semantic"
    EMOTIONAL = "emotional"
    PROCEDURAL = "procedural"

class ContextLevel(Enum):
    """Levels of context understanding."""
    SURFACE = "surface"
    SEMANTIC = "semantic"
    EMOTIONAL = "emotional"
    INTENTIONAL = "intentional"
    RELATIONAL = "relational"

@dataclass
class SemanticContext:
    """Semantic analysis of conversation context."""
    topics: List[str]
    entities: List[str]
    relationships: Dict[str, str]
    sentiment: float
    complexity: float
    coherence: float
    novelty: float

@dataclass
class EmotionalContext:
    """Emotional analysis of conversation context."""
    primary_emotion: str
    secondary_emotions: List[str]
    intensity: float
    valence: float
    arousal: float
    emotional_stability: float
    empathy_needed: bool

@dataclass
class IntentionalContext:
    """Intent and goal analysis."""
    primary_intent: str
    secondary_intents: List[str]
    goals: List[str]
    constraints: List[str]
    success_criteria: List[str]
    confidence: float

@dataclass
class RelationalContext:
    """Relational and social context."""
    user_persona: str
    relationship_duration: float
    trust_level: float
    communication_style: str
    cultural_context: str
    power_dynamics: str

@dataclass
class EnhancedResponse:
    """Enhanced response with multiple response options and metadata."""
    primary_response: str
    alternative_responses: List[str]
    follow_up_questions: List[str]
    suggested_actions: List[str]
    emotional_tone: str
    confidence: float
    context_used: bool
    personalization_level: str
    response_type: str
    semantic_context: Optional[SemanticContext] = None
    emotional_context: Optional[EmotionalContext] = None
    intentional_context: Optional[IntentionalContext] = None
    relational_context: Optional[RelationalContext] = None

@dataclass
class EnhancedContext:
    """Enhanced conversation context with comprehensive analysis."""
    user_input: str
    detected_intent: str
    confidence: float
    emotional_tone: str
    semantic_topics: List[str]
    follow_up_type: Optional[str]
    context_references: List[str]
    user_sentiment: str
    conversation_history: List[Dict[str, Any]]
    session_duration: float
    interaction_count: int
    semantic_context: Optional[SemanticContext] = None
    emotional_context: Optional[EmotionalContext] = None
    intentional_context: Optional[IntentionalContext] = None
    relational_context: Optional[RelationalContext] = None
    multi_modal_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryChunk:
    """A chunk of memory with metadata."""
    content: str
    memory_type: MemoryType
    context_level: ContextLevel
    timestamp: float
    importance: float
    access_count: int
    last_accessed: float
    associations: List[str]
    metadata: Dict[str, Any]

class SemanticAnalyzer:
    """Advanced semantic analysis for conversation context."""
    
    def __init__(self):
        self.logger = logging.getLogger('SemanticAnalyzer')
        self.topic_keywords = {
            'technology': ['ai', 'machine learning', 'programming', 'software', 'hardware'],
            'science': ['research', 'experiment', 'theory', 'hypothesis', 'data'],
            'business': ['strategy', 'market', 'finance', 'management', 'operations'],
            'health': ['medical', 'healthcare', 'treatment', 'diagnosis', 'wellness'],
            'education': ['learning', 'teaching', 'curriculum', 'student', 'knowledge'],
            'entertainment': ['music', 'movies', 'games', 'art', 'culture']
        }
    
    def analyze_semantics(self, text: str) -> SemanticContext:
        """Analyze semantic content of text."""
        topics = self._extract_topics(text)
        entities = self._extract_entities(text)
        relationships = self._extract_relationships(text)
        sentiment = self._analyze_sentiment(text)
        complexity = self._calculate_complexity(text)
        coherence = self._calculate_coherence(text)
        novelty = self._calculate_novelty(text)
        
        return SemanticContext(
            topics=topics,
            entities=entities,
            relationships=relationships,
            sentiment=sentiment,
            complexity=complexity,
            coherence=coherence,
            novelty=novelty
        )
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text."""
        text_lower = text.lower()
        topics = []
        for topic, keywords in self.topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        return topics
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        # Simple entity extraction - could be enhanced with NER
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return list(set(entities))
    
    def _extract_relationships(self, text: str) -> Dict[str, str]:
        """Extract relationships between entities."""
        # Simple relationship extraction
        relationships = {}
        # Implementation would use dependency parsing
        return relationships
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text."""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count == 0 and negative_count == 0:
            return 0.0
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity."""
        words = text.split()
        if not words:
            return 0.0
        
        avg_word_length = sum(len(word) for word in words) / len(words)
        return min(1.0, avg_word_length / 10.0)
    
    def _calculate_coherence(self, text: str) -> float:
        """Calculate text coherence."""
        # Simple coherence calculation
        sentences = text.split('.')
        if len(sentences) <= 1:
            return 1.0
        
        # Check for repeated words across sentences
        all_words = []
        for sentence in sentences:
            all_words.extend(sentence.lower().split())
        
        unique_words = set(all_words)
        return len(unique_words) / len(all_words) if all_words else 1.0
    
    def _calculate_novelty(self, text: str) -> float:
        """Calculate text novelty."""
        # Simple novelty calculation based on rare words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = text.lower().split()
        rare_words = [word for word in words if word not in common_words and len(word) > 4]
        
        if not words:
            return 0.0
        return len(rare_words) / len(words)

class EmotionalAnalyzer:
    """Advanced emotional analysis for conversation context."""
    
    def __init__(self):
        self.logger = logging.getLogger('EmotionalAnalyzer')
        self.emotion_keywords = {
            'joy': ['happy', 'excited', 'great', 'wonderful', 'amazing'],
            'sadness': ['sad', 'depressed', 'unhappy', 'disappointed', 'miserable'],
            'anger': ['angry', 'furious', 'mad', 'irritated', 'annoyed'],
            'fear': ['scared', 'afraid', 'terrified', 'worried', 'anxious'],
            'surprise': ['surprised', 'shocked', 'astonished', 'amazed', 'stunned'],
            'disgust': ['disgusted', 'revolted', 'sickened', 'appalled', 'horrified']
        }
    
    def analyze_emotions(self, text: str) -> EmotionalContext:
        """Analyze emotional content of text."""
        emotions = self._detect_emotions(text)
        primary_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else 'neutral'
        secondary_emotions = [emotion for emotion, score in emotions.items() if score > 0.3 and emotion != primary_emotion]
        
        intensity = max(emotions.values()) if emotions else 0.0
        valence = self._calculate_valence(emotions)
        arousal = self._calculate_arousal(emotions)
        emotional_stability = self._calculate_stability(emotions)
        empathy_needed = intensity > 0.7 or primary_emotion in ['sadness', 'fear', 'anger']
        
        return EmotionalContext(
            primary_emotion=primary_emotion,
            secondary_emotions=secondary_emotions,
            intensity=intensity,
            valence=valence,
            arousal=arousal,
            emotional_stability=emotional_stability,
            empathy_needed=empathy_needed
        )
    
    def _detect_emotions(self, text: str) -> Dict[str, float]:
        """Detect emotions in text."""
        text_lower = text.lower()
        emotions = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotions[emotion] = min(1.0, score / len(keywords))
        
        return emotions
    
    def _calculate_valence(self, emotions: Dict[str, float]) -> float:
        """Calculate emotional valence (positive vs negative)."""
        positive_emotions = ['joy', 'surprise']
        negative_emotions = ['sadness', 'anger', 'fear', 'disgust']
        
        positive_score = sum(emotions.get(emotion, 0) for emotion in positive_emotions)
        negative_score = sum(emotions.get(emotion, 0) for emotion in negative_emotions)
        
        if positive_score == 0 and negative_score == 0:
            return 0.0
        return (positive_score - negative_score) / (positive_score + negative_score)
    
    def _calculate_arousal(self, emotions: Dict[str, float]) -> float:
        """Calculate emotional arousal (high vs low energy)."""
        high_arousal = ['joy', 'anger', 'fear', 'surprise']
        low_arousal = ['sadness', 'disgust']
        
        high_score = sum(emotions.get(emotion, 0) for emotion in high_arousal)
        low_score = sum(emotions.get(emotion, 0) for emotion in low_arousal)
        
        if high_score == 0 and low_score == 0:
            return 0.5
        return high_score / (high_score + low_score) if (high_score + low_score) > 0 else 0.5
    
    def _calculate_stability(self, emotions: Dict[str, float]) -> float:
        """Calculate emotional stability."""
        if not emotions:
            return 1.0
        
        # More emotions = less stability
        emotion_count = len(emotions)
        max_intensity = max(emotions.values())
        
        stability = 1.0 - (emotion_count * 0.1) - (max_intensity * 0.3)
        return max(0.0, min(1.0, stability))

class IntentionalAnalyzer:
    """Advanced intentional analysis for conversation context."""
    
    def __init__(self):
        self.logger = logging.getLogger('IntentionalAnalyzer')
        self.intent_patterns = {
            'question': [r'\?', r'what', r'how', r'why', r'when', r'where', r'who'],
            'command': [r'do\s+this', r'make\s+', r'create\s+', r'build\s+', r'generate\s+'],
            'request': [r'please', r'could\s+you', r'would\s+you', r'can\s+you'],
            'statement': [r'i\s+think', r'i\s+believe', r'in\s+my\s+opinion'],
            'clarification': [r'what\s+do\s+you\s+mean', r'clarify', r'explain']
        }
    
    def analyze_intentions(self, text: str) -> IntentionalContext:
        """Analyze intentions and goals in text."""
        primary_intent = self._detect_primary_intent(text)
        secondary_intents = self._detect_secondary_intents(text)
        goals = self._extract_goals(text)
        constraints = self._extract_constraints(text)
        success_criteria = self._extract_success_criteria(text)
        confidence = self._calculate_confidence(text, primary_intent)
        
        return IntentionalContext(
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            goals=goals,
            constraints=constraints,
            success_criteria=success_criteria,
            confidence=confidence
        )
    
    def _detect_primary_intent(self, text: str) -> str:
        """Detect primary intent in text."""
        text_lower = text.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return intent
        
        return 'conversation'
    
    def _detect_secondary_intents(self, text: str) -> List[str]:
        """Detect secondary intents in text."""
        text_lower = text.lower()
        intents = []
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    intents.append(intent)
        
        return list(set(intents))
    
    def _extract_goals(self, text: str) -> List[str]:
        """Extract goals from text."""
        goals = []
        goal_patterns = [
            r'i\s+want\s+to\s+(\w+)',
            r'i\s+need\s+to\s+(\w+)',
            r'my\s+goal\s+is\s+to\s+(\w+)',
            r'i\s+am\s+trying\s+to\s+(\w+)'
        ]
        
        for pattern in goal_patterns:
            matches = re.findall(pattern, text.lower())
            goals.extend(matches)
        
        return goals
    
    def _extract_constraints(self, text: str) -> List[str]:
        """Extract constraints from text."""
        constraints = []
        constraint_patterns = [
            r'but\s+(\w+)',
            r'however\s+(\w+)',
            r'except\s+(\w+)',
            r'not\s+(\w+)'
        ]
        
        for pattern in constraint_patterns:
            matches = re.findall(pattern, text.lower())
            constraints.extend(matches)
        
        return constraints
    
    def _extract_success_criteria(self, text: str) -> List[str]:
        """Extract success criteria from text."""
        criteria = []
        criteria_patterns = [
            r'should\s+(\w+)',
            r'must\s+(\w+)',
            r'needs\s+to\s+(\w+)',
            r'required\s+to\s+(\w+)'
        ]
        
        for pattern in criteria_patterns:
            matches = re.findall(pattern, text.lower())
            criteria.extend(matches)
        
        return criteria
    
    def _calculate_confidence(self, text: str, primary_intent: str) -> float:
        """Calculate confidence in intent detection."""
        text_lower = text.lower()
        pattern_matches = 0
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    pattern_matches += 1
        
        # More pattern matches = higher confidence
        return min(1.0, pattern_matches / 5.0)

class RelationalAnalyzer:
    """Advanced relational analysis for conversation context."""
    
    def __init__(self):
        self.logger = logging.getLogger('RelationalAnalyzer')
        self.communication_styles = {
            'formal': ['sir', 'madam', 'please', 'thank you', 'respectfully'],
            'casual': ['hey', 'hi', 'cool', 'awesome', 'great'],
            'technical': ['algorithm', 'function', 'parameter', 'optimization', 'efficiency'],
            'emotional': ['feel', 'emotion', 'heart', 'soul', 'spirit']
        }
    
    def analyze_relations(self, text: str, session_duration: float = 0.0) -> RelationalContext:
        """Analyze relational context."""
        user_persona = self._detect_user_persona(text)
        relationship_duration = session_duration
        trust_level = self._calculate_trust_level(text, session_duration)
        communication_style = self._detect_communication_style(text)
        cultural_context = self._detect_cultural_context(text)
        power_dynamics = self._analyze_power_dynamics(text)
        
        return RelationalContext(
            user_persona=user_persona,
            relationship_duration=relationship_duration,
            trust_level=trust_level,
            communication_style=communication_style,
            cultural_context=cultural_context,
            power_dynamics=power_dynamics
        )
    
    def _detect_user_persona(self, text: str) -> str:
        """Detect user persona from text."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['developer', 'programmer', 'engineer']):
            return 'technical'
        elif any(word in text_lower for word in ['student', 'learn', 'study']):
            return 'student'
        elif any(word in text_lower for word in ['business', 'manager', 'executive']):
            return 'business'
        elif any(word in text_lower for word in ['researcher', 'scientist', 'academic']):
            return 'researcher'
        else:
            return 'general'
    
    def _calculate_trust_level(self, text: str, session_duration: float) -> float:
        """Calculate trust level based on text and session duration."""
        trust_indicators = ['trust', 'reliable', 'confident', 'sure', 'believe']
        distrust_indicators = ['doubt', 'suspicious', 'untrustworthy', 'uncertain', 'question']
        
        text_lower = text.lower()
        trust_score = sum(1 for word in trust_indicators if word in text_lower)
        distrust_score = sum(1 for word in distrust_indicators if word in text_lower)
        
        base_trust = 0.5
        text_trust = (trust_score - distrust_score) * 0.1
        duration_trust = min(0.3, session_duration / 3600)  # Max 0.3 from duration
        
        return max(0.0, min(1.0, base_trust + text_trust + duration_trust))
    
    def _detect_communication_style(self, text: str) -> str:
        """Detect communication style."""
        text_lower = text.lower()
        
        for style, keywords in self.communication_styles.items():
            if any(keyword in text_lower for keyword in keywords):
                return style
        
        return 'neutral'
    
    def _detect_cultural_context(self, text: str) -> str:
        """Detect cultural context."""
        # Simple cultural detection - could be enhanced
        return 'western'  # Default assumption
    
    def _analyze_power_dynamics(self, text: str) -> str:
        """Analyze power dynamics in conversation."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['please', 'could you', 'would you']):
            return 'deferential'
        elif any(word in text_lower for word in ['must', 'should', 'need to']):
            return 'authoritative'
        else:
            return 'equal'

class ConversationEnhancer:
    """Enhanced conversation enhancer with comprehensive analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger('ConversationEnhancer')
        self.semantic_analyzer = SemanticAnalyzer()
        self.emotional_analyzer = EmotionalAnalyzer()
        self.intentional_analyzer = IntentionalAnalyzer()
        self.relational_analyzer = RelationalAnalyzer()
        self.logger.info("Conversation Enhancer initialized")
    
    def enhance_conversation(self, user_input: str, session_id: Optional[str] = None, 
                           session_duration: float = 0.0) -> EnhancedResponse:
        """Enhance conversation with comprehensive analysis."""
        # Analyze all aspects of the input
        semantic_context = self.semantic_analyzer.analyze_semantics(user_input)
        emotional_context = self.emotional_analyzer.analyze_emotions(user_input)
        intentional_context = self.intentional_analyzer.analyze_intentions(user_input)
        relational_context = self.relational_analyzer.analyze_relations(user_input, session_duration)
        
        # Generate enhanced response
        primary_response = self._generate_primary_response(user_input, semantic_context, emotional_context, intentional_context)
        alternative_responses = self._generate_alternative_responses(user_input, semantic_context, emotional_context)
        follow_up_questions = self._generate_follow_up_questions(intentional_context, semantic_context)
        suggested_actions = self._generate_suggested_actions(intentional_context, semantic_context)
        
        emotional_tone = self._determine_emotional_tone(emotional_context, relational_context)
        confidence = intentional_context.confidence
        context_used = True
        personalization_level = self._determine_personalization_level(relational_context)
        response_type = self._determine_response_type(intentional_context, semantic_context)
        
        return EnhancedResponse(
            primary_response=primary_response,
            alternative_responses=alternative_responses,
            follow_up_questions=follow_up_questions,
            suggested_actions=suggested_actions,
            emotional_tone=emotional_tone,
            confidence=confidence,
            context_used=context_used,
            personalization_level=personalization_level,
            response_type=response_type,
            semantic_context=semantic_context,
            emotional_context=emotional_context,
            intentional_context=intentional_context,
            relational_context=relational_context
        )
    
    def _generate_primary_response(self, user_input: str, semantic_context: SemanticContext, 
                                 emotional_context: EmotionalContext, intentional_context: IntentionalContext) -> str:
        """Generate primary response based on analysis."""
        if intentional_context.primary_intent == 'question':
            return f"I understand you're asking about {', '.join(semantic_context.topics)}. Let me provide a comprehensive answer."
        elif intentional_context.primary_intent == 'command':
            return f"I'll help you with {', '.join(intentional_context.goals)}. Let me process this request."
        elif emotional_context.empathy_needed:
            return f"I sense you're feeling {emotional_context.primary_emotion}. I'm here to help and support you."
        else:
            return f"Enhanced response to: {user_input}"
    
    def _generate_alternative_responses(self, user_input: str, semantic_context: SemanticContext, 
                                      emotional_context: EmotionalContext) -> List[str]:
        """Generate alternative responses."""
        alternatives = []
        
        if semantic_context.topics:
            alternatives.append(f"Regarding {semantic_context.topics[0]}, here's what I can tell you...")
        
        if emotional_context.intensity > 0.5:
            alternatives.append(f"I understand this is important to you. Let me address your concerns.")
        
        alternatives.append("Alternative response 2")
        
        return alternatives
    
    def _generate_follow_up_questions(self, intentional_context: IntentionalContext, 
                                    semantic_context: SemanticContext) -> List[str]:
        """Generate follow-up questions."""
        questions = []
        
        if intentional_context.primary_intent == 'question':
            questions.append("Would you like me to elaborate on any specific aspect?")
        
        if semantic_context.topics:
            questions.append(f"Are you interested in learning more about {semantic_context.topics[0]}?")
        
        if intentional_context.goals:
            questions.append(f"Would you like me to help you achieve {intentional_context.goals[0]}?")
        
        return questions
    
    def _generate_suggested_actions(self, intentional_context: IntentionalContext, 
                                  semantic_context: SemanticContext) -> List[str]:
        """Generate suggested actions."""
        actions = []
        
        if intentional_context.primary_intent == 'command':
            actions.append("execute_task")
        
        if semantic_context.topics:
            actions.append("research_topic")
        
        actions.extend(["optimize_self", "self_assess"])
        
        return actions
    
    def _determine_emotional_tone(self, emotional_context: EmotionalContext, 
                                relational_context: RelationalContext) -> str:
        """Determine appropriate emotional tone."""
        if emotional_context.empathy_needed:
            return "empathetic"
        elif relational_context.communication_style == 'formal':
            return "professional"
        elif relational_context.communication_style == 'casual':
            return "friendly"
        else:
            return "neutral"
    
    def _determine_personalization_level(self, relational_context: RelationalContext) -> str:
        """Determine personalization level."""
        if relational_context.trust_level > 0.8:
            return "high"
        elif relational_context.trust_level > 0.5:
            return "medium"
        else:
            return "low"
    
    def _determine_response_type(self, intentional_context: IntentionalContext, 
                               semantic_context: SemanticContext) -> str:
        """Determine response type."""
        if intentional_context.primary_intent == 'question':
            return "informative"
        elif intentional_context.primary_intent == 'command':
            return "actionable"
        elif semantic_context.complexity > 0.7:
            return "detailed"
        else:
            return "enhanced"

CONVERSATION_ENHANCER_AVAILABLE = True

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    timestamp: float
    user_input: str
    scroll_name: Optional[str] = None
    response: Optional[str] = None
    confidence: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConversationContext:
    """Maintains conversation context and memory."""
    session_id: str
    start_time: float
    turns: deque = field(default_factory=lambda: deque(maxlen=20))  # Keep last 20 turns
    current_topic: Optional[str] = None
    last_command: Optional[str] = None
    last_question: Optional[str] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    conversation_summary: str = ""

class ConversationMemory:
    """
    Manages conversation memory and context for the LAM system.
    Provides intelligent handling of follow-up questions and context-aware responses.
    Enhanced with conversation enhancement capabilities.
    """
    
    def __init__(self, max_turns: int = 20, session_timeout: int = 3600, enable_enhancement: bool = True):
        """
        Initialize conversation memory with optional enhancement.
        
        Args:
            max_turns: Maximum number of conversation turns to remember
            session_timeout: Session timeout in seconds (1 hour default)
            enable_enhancement: Enable conversation enhancement features
        """
        self.logger = logging.getLogger('ConversationMemory')
        self.max_turns = max_turns
        self.session_timeout = session_timeout
        self.sessions: Dict[str, ConversationContext] = {}
        self.current_session_id = None
        
        # Conversation enhancement
        self.enable_enhancement = enable_enhancement and CONVERSATION_ENHANCER_AVAILABLE
        if self.enable_enhancement:
            self.conversation_enhancer = ConversationEnhancer()
            self.logger.info("Conversation enhancement enabled")
        else:
            self.conversation_enhancer = None
            if enable_enhancement and not CONVERSATION_ENHANCER_AVAILABLE:
                self.logger.warning("Conversation enhancement requested but not available")
        
        # Follow-up patterns and their handlers
        self.follow_up_patterns = {
            # Repeat last command
            r"do it again": self._handle_repeat_last_command,
            r"repeat": self._handle_repeat_last_command,
            r"again": self._handle_repeat_last_command,
            r"same thing": self._handle_repeat_last_command,
            
            # Time-based references
            r"yesterday": self._handle_time_reference,
            r"last time": self._handle_time_reference,
            r"earlier": self._handle_time_reference,
            r"before": self._handle_time_reference,
            
            # Context references
            r"that": self._handle_context_reference,
            r"it": self._handle_context_reference,
            r"this": self._handle_context_reference,
            r"the same": self._handle_context_reference,
            
            # Clarification requests
            r"what do you mean": self._handle_clarification,
            r"clarify": self._handle_clarification,
            r"more details": self._handle_clarification,
            
            # Comparison requests
            r"compared to": self._handle_comparison,
            r"versus": self._handle_comparison,
            r"difference": self._handle_comparison,
        }
        
        self.logger.info("Conversation memory system initialized with enhancement: %s", self.enable_enhancement)
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new conversation session.
        
        Args:
            session_id: Optional custom session ID
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        self.sessions[session_id] = ConversationContext(
            session_id=session_id,
            start_time=time.time()
        )
        self.current_session_id = session_id
        
        self.logger.info(f"Started conversation session: {session_id}")
        return session_id
    
    def end_session(self, session_id: Optional[str] = None):
        """End a conversation session."""
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id and session_id in self.sessions:
            del self.sessions[session_id]
            if session_id == self.current_session_id:
                self.current_session_id = None
            
            self.logger.info(f"Ended conversation session: {session_id}")
    
    def add_turn(self, user_input: str, scroll_name: Optional[str] = None, 
                 response: Optional[str] = None, confidence: float = 0.0,
                 context: Optional[Dict[str, Any]] = None) -> ConversationTurn:
        """
        Add a conversation turn to memory.
        
        Args:
            user_input: User's input
            scroll_name: Executed scroll name (if any)
            response: System response
            confidence: Confidence score
            context: Additional context
            
        Returns:
            ConversationTurn object
        """
        if self.current_session_id is None:
            self.start_session()
        
        session = self.sessions[self.current_session_id]
        
        turn = ConversationTurn(
            timestamp=time.time(),
            user_input=user_input,
            scroll_name=scroll_name,
            response=response,
            confidence=confidence,
            context=context or {},
            metadata={
                "session_id": self.current_session_id,
                "turn_number": len(session.turns) + 1
            }
        )
        
        session.turns.append(turn)
        
        # Update session context
        if scroll_name:
            session.last_command = scroll_name
        if self._is_question(user_input):
            session.last_question = user_input
        
        # Update current topic based on input
        session.current_topic = self._extract_topic(user_input)
        
        self.logger.debug(f"Added turn: {user_input[:50]}... -> {scroll_name}")
        return turn
    
    def get_context(self, session_id: Optional[str] = None) -> Optional[ConversationContext]:
        """Get conversation context for a session."""
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Check for session timeout
            if time.time() - session.start_time > self.session_timeout:
                self.logger.info(f"Session {session_id} timed out, ending")
                self.end_session(session_id)
                return None
            
            return session
        
        return None
    
    def get_last_turn(self, session_id: Optional[str] = None) -> Optional[ConversationTurn]:
        """Get the last conversation turn."""
        context = self.get_context(session_id)
        if context and context.turns:
            return context.turns[-1]
        return None
    
    def get_recent_turns(self, count: int = 5, session_id: Optional[str] = None) -> List[ConversationTurn]:
        """Get recent conversation turns."""
        context = self.get_context(session_id)
        if context:
            return list(context.turns)[-count:]
        return []
    
    def analyze_follow_up(self, user_input: str, session_id: Optional[str] = None) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Analyze if input is a follow-up question and resolve context.
        
        Args:
            user_input: User's input
            session_id: Session ID
            
        Returns:
            Tuple of (is_follow_up, resolved_command, context_info)
        """
        user_input_lower = user_input.lower().strip()
        
        # Check for follow-up patterns
        for pattern, handler in self.follow_up_patterns.items():
            if re.search(pattern, user_input_lower, re.IGNORECASE):
                return handler(user_input, session_id)
        
        return False, None, {}
    
    def _handle_repeat_last_command(self, user_input: str, session_id: Optional[str] = None) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Handle requests to repeat the last command."""
        context = self.get_context(session_id)
        if not context or not context.last_command:
            return True, None, {"error": "No previous command to repeat"}
        
        last_turn = self.get_last_turn(session_id)
        if last_turn and last_turn.scroll_name:
            return True, last_turn.scroll_name, {
                "type": "repeat_command",
                "original_command": last_turn.scroll_name,
                "original_input": last_turn.user_input,
                "timestamp": last_turn.timestamp
            }
        
        return True, None, {"error": "No previous command found"}
    
    def _handle_time_reference(self, user_input: str, session_id: Optional[str] = None) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Handle time-based references like 'yesterday' or 'last time'."""
        context = self.get_context(session_id)
        if not context:
            return True, None, {"error": "No conversation history"}
        
        # Look for commands from previous turns
        recent_turns = self.get_recent_turns(10, session_id)
        for turn in reversed(recent_turns[1:]):  # Skip current turn
            if turn.scroll_name:
                return True, turn.scroll_name, {
                    "type": "time_reference",
                    "referenced_command": turn.scroll_name,
                    "referenced_input": turn.user_input,
                    "time_difference": time.time() - turn.timestamp
                }
        
        return True, None, {"error": "No previous commands found in history"}
    
    def _handle_context_reference(self, user_input: str, session_id: Optional[str] = None) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Handle context references like 'that' or 'it'."""
        context = self.get_context(session_id)
        if not context:
            return True, None, {"error": "No conversation context"}
        
        # Try to infer from current topic or last command
        if context.last_command:
            return True, context.last_command, {
                "type": "context_reference",
                "inferred_command": context.last_command,
                "current_topic": context.current_topic
            }
        
        return True, None, {"error": "Unable to determine context reference"}
    
    def _handle_clarification(self, user_input: str, session_id: Optional[str] = None) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Handle requests for clarification."""
        context = self.get_context(session_id)
        if not context:
            return True, None, {"error": "No conversation context"}
        
        # Return the last question or command for clarification
        if context.last_question:
            return True, "general_conversation", {
                "type": "clarification",
                "clarify_question": context.last_question,
                "current_topic": context.current_topic
            }
        
        return True, "general_conversation", {
            "type": "clarification",
            "error": "No specific question to clarify"
        }
    
    def _handle_comparison(self, user_input: str, session_id: Optional[str] = None) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Handle comparison requests."""
        context = self.get_context(session_id)
        if not context:
            return True, None, {"error": "No conversation context"}
        
        # Look for two recent commands to compare
        recent_turns = self.get_recent_turns(5, session_id)
        commands = [turn.scroll_name for turn in recent_turns if turn.scroll_name]
        
        if len(commands) >= 2:
            return True, "general_conversation", {
                "type": "comparison",
                "commands": commands[-2:],
                "comparison_request": user_input
            }
        
        return True, "general_conversation", {
            "type": "comparison",
            "error": "Not enough commands to compare"
        }
    
    def _is_question(self, text: str) -> bool:
        """Check if text is a question."""
        question_words = ["what", "how", "why", "when", "where", "who", "which", "?"]
        text_lower = text.lower()
        return any(word in text_lower for word in question_words) or text.strip().endswith("?")
    
    def _extract_topic(self, text: str) -> Optional[str]:
        """Extract the main topic from text."""
        # Simple topic extraction - can be enhanced with NLP
        topics = {
            "system": ["optimize", "system", "performance", "status"],
            "wellness": ["calm", "relax", "breathe", "stress"],
            "memory": ["memory", "remember", "forget", "clean"],
            "security": ["shield", "protect", "security", "defense"],
            "information": ["search", "find", "look up", "weather"],
            "help": ["help", "commands", "what can you", "capabilities"]
        }
        
        text_lower = text.lower()
        for topic, keywords in topics.items():
            if any(keyword in text_lower for keyword in keywords):
                return topic
        
        return None
    
    def get_conversation_summary(self, session_id: Optional[str] = None) -> str:
        """Get a summary of the conversation."""
        context = self.get_context(session_id)
        if not context:
            return "No conversation history"
        
        summary_parts = []
        
        if context.current_topic:
            summary_parts.append(f"Current topic: {context.current_topic}")
        
        if context.last_command:
            summary_parts.append(f"Last command: {context.last_command}")
        
        if context.last_question:
            summary_parts.append(f"Last question: {context.last_question[:50]}...")
        
        summary_parts.append(f"Total turns: {len(context.turns)}")
        summary_parts.append(f"Session duration: {int(time.time() - context.start_time)}s")
        
        return " | ".join(summary_parts)
    
    def clear_memory(self, session_id: Optional[str] = None):
        """Clear conversation memory for a session."""
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id and session_id in self.sessions:
            self.sessions[session_id].turns.clear()
            self.sessions[session_id].current_topic = None
            self.sessions[session_id].last_command = None
            self.sessions[session_id].last_question = None
            
            self.logger.info(f"Cleared memory for session: {session_id}")
    
    def optimize(self) -> Dict[str, Any]:
        """
        Optimize the conversation memory for better performance.
        
        Returns:
            Dict containing optimization results
        """
        self.logger.info("Optimizing conversation memory")
        
        optimization_results = {
            "status": "optimized",
            "changes": [],
            "performance_metrics": {}
        }
        
        try:
            # Store before state
            before_count = len(self.sessions)
            before_context_size = sum(len(session.turns) for session in self.sessions.values())
            
            # Optimize memory by removing old sessions if too many
            if len(self.sessions) > self.max_turns:
                # Keep only the most recent sessions
                self.sessions = {session_id: session for session_id, session in self.sessions.items() if session_id in self.sessions}
                optimization_results["changes"].append(f"Trimmed conversation sessions from {before_count} to {len(self.sessions)} sessions")
            
            # Optimize context by removing stale sessions
            stale_sessions = []
            for session_id, session in self.sessions.items():
                if time.time() - session.start_time > self.session_timeout:
                    stale_sessions.append(session_id)
            
            for session_id in stale_sessions:
                del self.sessions[session_id]
            
            if stale_sessions:
                optimization_results["changes"].append(f"Removed {len(stale_sessions)} stale sessions")
            
            # Optimize memory parameters
            self.max_turns = 20  # Optimal size
            self.session_timeout = 3600  # 1 hour
            
            optimization_results["changes"].append("Set max session count to 20")
            optimization_results["changes"].append("Set session timeout to 1 hour")
            
            # Performance metrics
            optimization_results["performance_metrics"] = {
                "session_count": len(self.sessions),
                "context_size": before_context_size,
                "max_session_count": self.max_turns,
                "session_timeout": self.session_timeout,
                "sessions_removed": before_count - len(self.sessions),
                "stale_sessions_removed": len(stale_sessions)
            }
            
        except Exception as e:
            optimization_results["status"] = "error"
            optimization_results["error"] = str(e)
            self.logger.error(f"Conversation memory optimization failed: {e}")
        
        return optimization_results
    
    def enhance_conversation(self, user_input: str, session_id: Optional[str] = None) -> Optional[EnhancedResponse]:
        """Enhance conversation using integrated conversation enhancer."""
        if not self.enable_enhancement:
            return None
        
        try:
            return self.conversation_enhancer.enhance_conversation(user_input, session_id)
        except Exception as e:
            self.logger.error(f"Conversation enhancement error: {e}")
            return None
    
    def get_enhanced_response(self, user_input: str, session_id: Optional[str] = None) -> Optional[str]:
        """Get enhanced response for user input."""
        if not self.enable_enhancement:
            return None
        
        try:
            enhanced_response = self.conversation_enhancer.enhance_conversation(user_input, session_id)
            return enhanced_response.primary_response if enhanced_response else None
        except Exception as e:
            self.logger.error(f"Enhanced response error: {e}")
            return None
    
    def analyze_conversation_context(self, user_input: str, session_id: Optional[str] = None) -> Optional[EnhancedContext]:
        """Analyze conversation context using integrated enhancer."""
        if not self.enable_enhancement:
            return None
        
        try:
            return self.conversation_enhancer.enhance_conversation(user_input, session_id)
        except Exception as e:
            self.logger.error(f"Conversation context analysis error: {e}")
            return None

# Global instance
conversation_memory = ConversationMemory() 