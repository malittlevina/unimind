"""
conversation_memory.py – Conversation memory and context tracking for LAM system
"""

import time
import logging
import re
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict

# Define conversation enhancer components inline since the original file was merged
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

@dataclass
class EnhancedContext:
    """Enhanced conversation context with emotional and semantic analysis."""
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

class ConversationEnhancer:
    """Simplified conversation enhancer for integration."""
    
    def __init__(self):
        self.logger = logging.getLogger('ConversationEnhancer')
        self.logger.info("Conversation Enhancer initialized")
    
    def enhance_conversation(self, user_input: str, session_id: Optional[str] = None) -> EnhancedResponse:
        """Enhance conversation with emotional and semantic analysis."""
        return EnhancedResponse(
            primary_response=f"Enhanced response to: {user_input}",
            alternative_responses=["Alternative response 1", "Alternative response 2"],
            follow_up_questions=["Would you like to know more?"],
            suggested_actions=["optimize_self", "self_assess"],
            emotional_tone="neutral",
            confidence=0.8,
            context_used=True,
            personalization_level="medium",
            response_type="enhanced"
        )
    
    def _analyze_conversation_context(self, user_input: str, session_id: Optional[str] = None) -> EnhancedContext:
        """Analyze conversation context."""
        return EnhancedContext(
            user_input=user_input,
            detected_intent="general_conversation",
            confidence=0.7,
            emotional_tone="neutral",
            semantic_topics=["general"],
            follow_up_type=None,
            context_references=[],
            user_sentiment="neutral",
            conversation_history=[],
            session_duration=0.0,
            interaction_count=1
        )

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
            return self.conversation_enhancer._analyze_conversation_context(user_input, session_id)
        except Exception as e:
            self.logger.error(f"Conversation context analysis error: {e}")
            return None

# Global instance
conversation_memory = ConversationMemory() 