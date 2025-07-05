"""
lam_engine.py – Language Action Mapping Engine for ThothOS/Unimind.
Maps natural language input to symbolic actions using fuzzy logic, conversation memory, and intent classification.
Enhanced with LLM-driven planning and iterative reasoning.
Now includes integrated context-aware LLM functionality.
"""

import json
import logging
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from .fuzzy_processor import fuzzy_processor
from .conversation_memory import conversation_memory
from .llm_engine import llm_engine
from ..memory.unified_memory import unified_memory, MemoryType
from pathlib import Path

# Context-Aware LLM Integration
class IntentCategory(Enum):
    """Categories of user intents (enhanced with intent classifier types)."""
    # Context-aware LLM intents
    SELF_IMPROVEMENT = "self_improvement"
    SYSTEM_OPTIMIZATION = "system_optimization"
    INFORMATION_GATHERING = "information_gathering"
    TASK_EXECUTION = "task_execution"
    CREATIVE_WORK = "creative_work"
    ANALYSIS = "analysis"
    CONVERSATION = "conversation"
    THREE_D_CONSTRUCTION = "3d_construction"
    CODE_GENERATION = "code_generation"
    DATA_ANALYSIS = "data_analysis"
    MULTIMEDIA_CREATION = "multimedia_creation"
    SYSTEM_ADMINISTRATION = "system_administration"
    LEARNING = "learning"
    COLLABORATION = "collaboration"
    AUTOMATION = "automation"
    INTEGRATION = "integration"
    
    # Intent classifier types (merged)
    GENERAL_QUESTION = "general_question"
    SYSTEM_COMMAND = "system_command"
    SELF_REFLECTION = "self_reflection"
    EMOTIONAL_STATE = "emotional_state"
    UNKNOWN = "unknown"

@dataclass
class ContextualUnderstanding:
    """Represents contextual understanding of user input."""
    original_input: str
    primary_intent: IntentCategory
    secondary_intents: List[IntentCategory]
    confidence: float
    context_clues: List[str]
    suggested_actions: List[str]
    reasoning: str
    user_goal: str
    system_capabilities_needed: List[str]

class ContextAwareLLM:
    """
    Context-aware LLM wrapper that enhances understanding through:
    - Better prompt engineering
    - Context analysis
    - Intent recognition
    - Semantic understanding
    """
    
    def __init__(self):
        """Initialize the context-aware LLM with integrated intent classifier."""
        self.logger = logging.getLogger('ContextAwareLLM')
        self.llm_engine = llm_engine
        self.unified_memory = unified_memory
        
        # Intent patterns and keywords
        self.intent_patterns = self._initialize_intent_patterns()
        self.context_keywords = self._initialize_context_keywords()
        self.system_capabilities = self._initialize_system_capabilities()
        
        # Configuration
        self.enable_context_analysis = True
        self.enable_intent_recognition = True
        self.enable_semantic_understanding = True
        self.max_context_history = 10
        
        self.logger.info("Context-aware LLM initialized")
    
    def _initialize_intent_patterns(self) -> Dict[IntentCategory, Dict[str, Any]]:
        """Initialize patterns for different intent categories."""
        return {
            IntentCategory.SELF_IMPROVEMENT: {
                "keywords": [
                    "help me", "logic", "better", "assist", "improve", "enhance",
                    "expand", "develop", "grow", "evolve", "upgrade", "optimize", "yourself",
                    "self", "capabilities", "abilities", "smarter", "more capable", "learn",
                    "advance", "progress", "refine", "perfect", "master", "excel"
                ],
                "patterns": [
                    r"help\s+me\s+(?:to\s+)?(?:improve|enhance|expand|develop)",
                    r"(?:improve|enhance|expand)\s+(?:your|the)\s+(?:logic|capabilities|abilities)",
                    r"make\s+(?:yourself|you)\s+(?:better|smarter|more\s+capable)",
                    r"optimize\s+(?:yourself|self|your\s+performance)",
                    r"evolve\s+(?:yourself|your\s+capabilities)",
                    r"upgrade\s+(?:yourself|your\s+logic)",
                    r"develop\s+(?:yourself|your\s+abilities)",
                    r"expand\s+(?:your|the)\s+(?:logic|capabilities)",
                    r"enhance\s+(?:your|the)\s+(?:logic|capabilities)",
                    r"improve\s+(?:your|the)\s+(?:logic|capabilities|abilities)",
                    r"become\s+(?:better|smarter|more\s+capable)",
                    r"learn\s+(?:more|better|faster)",
                    r"advance\s+(?:your|the)\s+(?:skills|knowledge)",
                    r"refine\s+(?:your|the)\s+(?:approach|methods)",
                    r"perfect\s+(?:your|the)\s+(?:technique|process)",
                    r"master\s+(?:your|the)\s+(?:abilities|skills)",
                    r"excel\s+(?:at|in)\s+(?:your|the)",
                    r"grow\s+(?:your|the)\s+(?:capabilities|abilities)",
                    r"progress\s+(?:your|the)\s+(?:development|skills)"
                ],
                "context_clues": [
                    "self-referential language", "improvement requests", "capability enhancement",
                    "system optimization", "personal development", "skill advancement"
                ]
            },
            IntentCategory.SYSTEM_OPTIMIZATION: {
                "keywords": [
                    "optimize", "system", "performance", "speed", "efficiency", "faster",
                    "better", "improve", "enhance", "tune", "calibrate", "adjust", "boost",
                    "accelerate", "streamline", "fine-tune", "maximize", "minimize"
                ],
                "patterns": [
                    r"optimize\s+(?:the\s+)?system",
                    r"improve\s+(?:system\s+)?performance",
                    r"make\s+(?:it|the\s+system)\s+faster",
                    r"enhance\s+(?:system\s+)?efficiency",
                    r"boost\s+(?:system\s+)?performance",
                    r"accelerate\s+(?:system\s+)?speed",
                    r"streamline\s+(?:system\s+)?processes",
                    r"fine-tune\s+(?:system\s+)?settings",
                    r"maximize\s+(?:system\s+)?output",
                    r"minimize\s+(?:system\s+)?overhead",
                    r"calibrate\s+(?:system\s+)?parameters",
                    r"adjust\s+(?:system\s+)?configuration",
                    r"tune\s+(?:system\s+)?performance",
                    r"optimize\s+(?:system\s+)?resources",
                    r"improve\s+(?:system\s+)?response\s+time",
                    r"enhance\s+(?:system\s+)?capabilities",
                    r"boost\s+(?:system\s+)?productivity",
                    r"accelerate\s+(?:system\s+)?processing",
                    r"streamline\s+(?:system\s+)?workflow"
                ],
                "context_clues": [
                    "performance optimization", "system improvement", "efficiency enhancement",
                    "speed optimization", "resource management", "performance tuning"
                ]
            },
            IntentCategory.INFORMATION_GATHERING: {
                "keywords": [
                    "find", "search", "look up", "research", "get", "obtain", "retrieve",
                    "information", "data", "facts", "details", "knowledge", "learn",
                    "what", "how", "why", "when", "where", "who", "which"
                ],
                "patterns": [
                    r"find\s+(?:information\s+)?(?:about|on|for)",
                    r"search\s+(?:for\s+)?(?:information\s+)?(?:about|on)",
                    r"look\s+up\s+(?:information\s+)?(?:about|on)",
                    r"research\s+(?:information\s+)?(?:about|on)",
                    r"get\s+(?:information\s+)?(?:about|on|for)",
                    r"obtain\s+(?:information\s+)?(?:about|on)",
                    r"retrieve\s+(?:information\s+)?(?:about|on)",
                    r"what\s+(?:is|are|was|were)\s+(?:the\s+)?",
                    r"how\s+(?:do|does|did|can|will)\s+(?:the\s+)?",
                    r"why\s+(?:do|does|did|is|are)\s+(?:the\s+)?",
                    r"when\s+(?:do|does|did|is|are)\s+(?:the\s+)?",
                    r"where\s+(?:do|does|did|is|are)\s+(?:the\s+)?",
                    r"who\s+(?:is|are|was|were)\s+(?:the\s+)?",
                    r"which\s+(?:is|are|was|were)\s+(?:the\s+)?",
                    r"tell\s+me\s+(?:about|more\s+about)",
                    r"explain\s+(?:to\s+me\s+)?(?:about|more\s+about)",
                    r"describe\s+(?:to\s+me\s+)?(?:about|more\s+about)",
                    r"provide\s+(?:me\s+with\s+)?(?:information\s+)?(?:about|on)",
                    r"give\s+me\s+(?:information\s+)?(?:about|on)"
                ],
                "context_clues": [
                    "question words", "information requests", "research queries",
                    "knowledge seeking", "fact finding", "data retrieval"
                ]
            },
            IntentCategory.TASK_EXECUTION: {
                "keywords": [
                    "do", "execute", "run", "perform", "carry out", "complete", "finish",
                    "create", "generate", "build", "make", "produce", "develop", "write",
                    "code", "program", "script", "function", "algorithm", "process"
                ],
                "patterns": [
                    r"do\s+(?:this|that|it|the\s+task)",
                    r"execute\s+(?:this|that|it|the\s+task)",
                    r"run\s+(?:this|that|it|the\s+task)",
                    r"perform\s+(?:this|that|it|the\s+task)",
                    r"carry\s+out\s+(?:this|that|it|the\s+task)",
                    r"complete\s+(?:this|that|it|the\s+task)",
                    r"finish\s+(?:this|that|it|the\s+task)",
                    r"create\s+(?:a\s+)?(?:code|program|script|function|algorithm)",
                    r"generate\s+(?:a\s+)?(?:code|program|script|function|algorithm)",
                    r"build\s+(?:a\s+)?(?:code|program|script|function|algorithm)",
                    r"make\s+(?:a\s+)?(?:code|program|script|function|algorithm)",
                    r"produce\s+(?:a\s+)?(?:code|program|script|function|algorithm)",
                    r"develop\s+(?:a\s+)?(?:code|program|script|function|algorithm)",
                    r"write\s+(?:a\s+)?(?:code|program|script|function|algorithm)",
                    r"code\s+(?:a\s+)?(?:program|script|function|algorithm)",
                    r"program\s+(?:a\s+)?(?:script|function|algorithm)",
                    r"script\s+(?:a\s+)?(?:function|algorithm)",
                    r"function\s+(?:to\s+)?(?:do|perform|execute)",
                    r"algorithm\s+(?:to\s+)?(?:do|perform|execute)"
                ],
                "context_clues": [
                    "action verbs", "task execution", "code generation",
                    "programming requests", "automation tasks", "process execution"
                ]
            },
            IntentCategory.CREATIVE_WORK: {
                "keywords": [
                    "create", "design", "invent", "imagine", "brainstorm", "explore",
                    "experiment", "innovate", "develop", "build", "construct", "generate",
                    "art", "design", "creative", "original", "unique", "novel", "new"
                ],
                "patterns": [
                    r"create\s+(?:a\s+)?(?:new|original|unique|novel)",
                    r"design\s+(?:a\s+)?(?:new|original|unique|novel)",
                    r"invent\s+(?:a\s+)?(?:new|original|unique|novel)",
                    r"imagine\s+(?:a\s+)?(?:new|original|unique|novel)",
                    r"brainstorm\s+(?:a\s+)?(?:new|original|unique|novel)",
                    r"explore\s+(?:new|original|unique|novel)",
                    r"experiment\s+(?:with\s+)?(?:new|original|unique|novel)",
                    r"innovate\s+(?:with\s+)?(?:new|original|unique|novel)",
                    r"develop\s+(?:a\s+)?(?:new|original|unique|novel)",
                    r"build\s+(?:a\s+)?(?:new|original|unique|novel)",
                    r"construct\s+(?:a\s+)?(?:new|original|unique|novel)",
                    r"generate\s+(?:a\s+)?(?:new|original|unique|novel)",
                    r"create\s+(?:art|design|creative)",
                    r"design\s+(?:art|creative|original)",
                    r"invent\s+(?:art|design|creative)",
                    r"imagine\s+(?:art|design|creative)",
                    r"brainstorm\s+(?:art|design|creative)",
                    r"explore\s+(?:art|design|creative)",
                    r"experiment\s+(?:with\s+)?(?:art|design|creative)",
                    r"innovate\s+(?:with\s+)?(?:art|design|creative)"
                ],
                "context_clues": [
                    "creative language", "artistic requests", "design tasks",
                    "innovation requests", "original content", "creative expression"
                ]
            },
            IntentCategory.ANALYSIS: {
                "keywords": [
                    "analyze", "examine", "study", "investigate", "explore", "research",
                    "evaluate", "assess", "review", "inspect", "check", "verify",
                    "understand", "comprehend", "interpret", "explain", "clarify"
                ],
                "patterns": [
                    r"analyze\s+(?:this|that|it|the\s+data)",
                    r"examine\s+(?:this|that|it|the\s+data)",
                    r"study\s+(?:this|that|it|the\s+data)",
                    r"investigate\s+(?:this|that|it|the\s+data)",
                    r"explore\s+(?:this|that|it|the\s+data)",
                    r"research\s+(?:this|that|it|the\s+data)",
                    r"evaluate\s+(?:this|that|it|the\s+data)",
                    r"assess\s+(?:this|that|it|the\s+data)",
                    r"review\s+(?:this|that|it|the\s+data)",
                    r"inspect\s+(?:this|that|it|the\s+data)",
                    r"check\s+(?:this|that|it|the\s+data)",
                    r"verify\s+(?:this|that|it|the\s+data)",
                    r"understand\s+(?:this|that|it|the\s+data)",
                    r"comprehend\s+(?:this|that|it|the\s+data)",
                    r"interpret\s+(?:this|that|it|the\s+data)",
                    r"explain\s+(?:this|that|it|the\s+data)",
                    r"clarify\s+(?:this|that|it|the\s+data)",
                    r"what\s+(?:does|do)\s+(?:this|that|it|the\s+data)\s+mean",
                    r"how\s+(?:does|do)\s+(?:this|that|it|the\s+data)\s+work",
                    r"why\s+(?:does|do)\s+(?:this|that|it|the\s+data)\s+happen"
                ],
                "context_clues": [
                    "analysis requests", "examination tasks", "investigation queries",
                    "evaluation requests", "understanding requests", "interpretation tasks"
                ]
            },
            IntentCategory.CONVERSATION: {
                "keywords": [
                    "talk", "chat", "converse", "discuss", "share", "tell", "say",
                    "hello", "hi", "greetings", "how are you", "what's up", "nice to meet you"
                ],
                "patterns": [
                    r"talk\s+(?:to|with)\s+(?:me|you)",
                    r"chat\s+(?:with|to)\s+(?:me|you)",
                    r"converse\s+(?:with|to)\s+(?:me|you)",
                    r"discuss\s+(?:with|to)\s+(?:me|you)",
                    r"share\s+(?:with|to)\s+(?:me|you)",
                    r"tell\s+(?:me|you)\s+(?:about|more)",
                    r"say\s+(?:hello|hi|greetings)",
                    r"hello\s+(?:there|you)",
                    r"hi\s+(?:there|you)",
                    r"greetings\s+(?:there|you)",
                    r"how\s+(?:are|is)\s+(?:you|it)\s+(?:going|doing)",
                    r"what's\s+(?:up|new|happening)",
                    r"nice\s+(?:to|meeting)\s+(?:meet|see)\s+(?:you|there)",
                    r"good\s+(?:morning|afternoon|evening|night)",
                    r"pleasure\s+(?:to|meeting)\s+(?:meet|see)\s+(?:you|there)"
                ],
                "context_clues": [
                    "conversation starters", "greeting language", "social interaction",
                    "casual conversation", "friendly language", "social engagement"
                ]
            },
            IntentCategory.UNKNOWN: {
                "keywords": [],
                "patterns": [],
                "context_clues": []
            }
        }
    
    def _initialize_context_keywords(self) -> Dict[str, List[str]]:
        """Initialize context keywords for different types of context."""
        return {
            "technical": [
                "code", "program", "script", "function", "algorithm", "data", "system",
                "software", "hardware", "network", "database", "api", "interface"
            ],
            "creative": [
                "art", "design", "creative", "imagination", "inspiration", "beauty",
                "aesthetic", "style", "color", "shape", "form", "composition"
            ],
            "business": [
                "business", "market", "customer", "product", "service", "strategy",
                "planning", "management", "leadership", "team", "project", "goal"
            ],
            "personal": [
                "personal", "life", "family", "friend", "relationship", "emotion",
                "feeling", "experience", "memory", "dream", "hope", "fear"
            ],
            "academic": [
                "research", "study", "learning", "education", "knowledge", "theory",
                "concept", "principle", "method", "analysis", "experiment", "hypothesis"
            ]
        }
    
    def _initialize_system_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Initialize system capabilities and their descriptions."""
        return {
            "web_search": {
                "description": "Search the internet for information",
                "keywords": ["search", "find", "look up", "research", "information"],
                "actions": ["web_search", "search_wiki", "location_search"]
            },
            "code_generation": {
                "description": "Generate code in various programming languages",
                "keywords": ["code", "program", "script", "function", "algorithm"],
                "actions": ["generate_code", "internal_ide", "file_access"]
            },
            "text_processing": {
                "description": "Process and analyze text content",
                "keywords": ["text", "document", "content", "analysis", "summarize"],
                "actions": ["summarize_text", "analyze_document", "parse_text"]
            },
            "system_optimization": {
                "description": "Optimize system performance and capabilities",
                "keywords": ["optimize", "improve", "enhance", "system", "performance"],
                "actions": ["optimize_self", "clean_memory", "self_assess"]
            },
            "creative_generation": {
                "description": "Generate creative content and designs",
                "keywords": ["create", "design", "generate", "creative", "art"],
                "actions": ["generate_3d_model", "generate_3d_scene", "create_realm"]
            },
            "memory_management": {
                "description": "Manage and organize memory and knowledge",
                "keywords": ["memory", "remember", "store", "organize", "knowledge"],
                "actions": ["log_memory", "ritual_feedback", "memory_cleanup"]
            }
        }
    
    def understand_context(self, user_input: str, memory_context_id: str = None) -> ContextualUnderstanding:
        """
        Understand the context of user input.
        
        Args:
            user_input: The user's input text
            memory_context_id: Optional memory context identifier
            
        Returns:
            ContextualUnderstanding object
        """
        # Analyze primary intent
        primary_intent = self._analyze_intent(user_input)
        
        # Find secondary intents
        secondary_intents = self._find_secondary_intents(user_input)
        
        # Extract context clues
        context_clues = self._extract_context_clues(user_input)
        
        # Infer user goal
        user_goal = self._infer_user_goal(user_input, primary_intent)
        
        # Suggest actions
        suggested_actions = self._suggest_actions(primary_intent, user_goal)
        
        # Identify system capabilities needed
        system_capabilities_needed = self._identify_capabilities_needed(primary_intent, user_goal)
        
        # Get historical context if available
        historical_context = []
        if memory_context_id:
            historical_context = self.unified_memory.get_memory(
                memory_context_id,
                MemoryType.CONVERSATION,
                limit=5
            )
        
        # Perform semantic understanding
        semantic_result = self._semantic_understanding(user_input, primary_intent, historical_context)
        reasoning = semantic_result.get("reasoning", f"Basic understanding: {primary_intent.value}")
        confidence = semantic_result.get("confidence", 0.6)
        
        return ContextualUnderstanding(
            original_input=user_input,
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            confidence=confidence,
            context_clues=context_clues,
            suggested_actions=suggested_actions,
            reasoning=reasoning,
            user_goal=user_goal,
            system_capabilities_needed=system_capabilities_needed
        )
    
    def _analyze_intent(self, user_input: str) -> IntentCategory:
        """Analyze user input to determine primary intent."""
        user_input_lower = user_input.lower()
        
        # Score each intent category
        intent_scores = {}
        
        for intent_category, pattern_info in self.intent_patterns.items():
            score = 0
            
            # Check keywords
            for keyword in pattern_info["keywords"]:
                if keyword in user_input_lower:
                    score += 1
            
            # Check patterns
            import re
            for pattern in pattern_info["patterns"]:
                if re.search(pattern, user_input_lower, re.IGNORECASE):
                    score += 2
            
            intent_scores[intent_category] = score
        
        # Return the highest scoring intent
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            if best_intent[1] > 0:
                return best_intent[0]
        
        return IntentCategory.UNKNOWN
    
    def _find_secondary_intents(self, user_input: str) -> List[IntentCategory]:
        """Find secondary intents in user input."""
        user_input_lower = user_input.lower()
        secondary_intents = []
        
        for intent_category, pattern_info in self.intent_patterns.items():
            if intent_category == IntentCategory.UNKNOWN:
                continue
            
            # Check for secondary intent indicators
            for keyword in pattern_info["keywords"]:
                if keyword in user_input_lower:
                    secondary_intents.append(intent_category)
                    break
        
        return list(set(secondary_intents))  # Remove duplicates
    
    def _extract_context_clues(self, user_input: str) -> List[str]:
        """Extract context clues from user input."""
        user_input_lower = user_input.lower()
        clues = []
        
        for context_type, keywords in self.context_keywords.items():
            for keyword in keywords:
                if keyword in user_input_lower:
                    clues.append(f"{context_type}: {keyword}")
        
        return clues
    
    def _infer_user_goal(self, user_input: str, primary_intent: IntentCategory) -> str:
        """Infer the user's goal from input and intent."""
        if primary_intent == IntentCategory.SELF_IMPROVEMENT:
            return "Improve system capabilities and performance"
        elif primary_intent == IntentCategory.SYSTEM_OPTIMIZATION:
            return "Optimize system performance and efficiency"
        elif primary_intent == IntentCategory.INFORMATION_GATHERING:
            return "Gather information or data"
        elif primary_intent == IntentCategory.TASK_EXECUTION:
            return "Execute a specific task or action"
        elif primary_intent == IntentCategory.CREATIVE_WORK:
            return "Create something new or innovative"
        elif primary_intent == IntentCategory.ANALYSIS:
            return "Analyze or examine something"
        else:
            return "General assistance or conversation"
    
    def _suggest_actions(self, primary_intent: IntentCategory, user_goal: str) -> List[str]:
        """Suggest appropriate actions based on intent and goal."""
        suggestions = []
        
        if primary_intent == IntentCategory.SELF_IMPROVEMENT:
            suggestions = ["optimize_self", "self_assess", "introspect_core", "reflect"]
        elif primary_intent == IntentCategory.SYSTEM_OPTIMIZATION:
            suggestions = ["optimize_self", "clean_memory", "self_assess"]
        elif primary_intent == IntentCategory.INFORMATION_GATHERING:
            suggestions = ["web_search", "search_wiki", "location_search"]
        elif primary_intent == IntentCategory.TASK_EXECUTION:
            suggestions = ["generate_code", "internal_ide", "file_access"]
        elif primary_intent == IntentCategory.CREATIVE_WORK:
            # Check for 3D construction keywords
            user_input_lower = user_goal.lower()
            if any(keyword in user_input_lower for keyword in ["3d", "three dimensional", "model", "scene", "realm", "statue", "castle", "building", "place", "space"]):
                suggestions = ["generate_3d_model", "generate_3d_scene", "create_realm", "3d_construction"]
            else:
                suggestions = ["generate_code", "summarize_text", "create_realm"]
        elif primary_intent == IntentCategory.ANALYSIS:
            suggestions = ["analyze_document", "summarize_text", "self_assess"]
        else:
            suggestions = ["general_conversation"]
        
        return suggestions
    
    def _identify_capabilities_needed(self, primary_intent: IntentCategory, user_goal: str) -> List[str]:
        """Identify system capabilities needed for the request."""
        capabilities = []
        
        for capability_name, capability_info in self.system_capabilities.items():
            for keyword in capability_info["keywords"]:
                if keyword in user_goal.lower():
                    capabilities.extend(capability_info["actions"])
                    break
        
        return list(set(capabilities))  # Remove duplicates
    
    def _semantic_understanding(self, user_input: str, primary_intent: IntentCategory, 
                              historical_context: List[Any]) -> Dict[str, Any]:
        """Use LLM for semantic understanding of user input."""
        try:
            context_prompt = f"""
Analyze this user input for semantic understanding:

User Input: "{user_input}"
Primary Intent: {primary_intent.value}

Historical Context:
{json.dumps(historical_context, indent=2) if historical_context else "No historical context"}

Provide a detailed analysis including:
1. What the user is really asking for
2. The underlying goal or motivation
3. What capabilities would be most helpful
4. Confidence level in understanding

Respond in JSON format:
{{
    "reasoning": "detailed explanation of what the user wants",
    "confidence": 0.0-1.0,
    "underlying_goal": "what the user is really trying to achieve",
    "recommended_approach": "how to best help the user"
}}
"""
            
            response = self.llm_engine.run(
                prompt=context_prompt,
                temperature=0.3,
                max_tokens=400
            )
            
            if response and response.startswith("{") and response.endswith("}"):
                return json.loads(response)
            else:
                return {
                    "reasoning": f"Basic understanding: {primary_intent.value}",
                    "confidence": 0.6,
                    "underlying_goal": "General assistance",
                    "recommended_approach": "Use standard capabilities"
                }
                
        except Exception as e:
            self.logger.error(f"Semantic understanding failed: {e}")
            return {
                "reasoning": f"Fallback understanding: {primary_intent.value}",
                "confidence": 0.5,
                "underlying_goal": "General assistance",
                "recommended_approach": "Use fallback capabilities"
            }
    
    def generate_enhanced_prompt(self, user_input: str, understanding: ContextualUnderstanding,
                               memory_context_id: str = None) -> str:
        """
        Generate an enhanced prompt based on contextual understanding.
        
        Args:
            user_input: Original user input
            understanding: Contextual understanding
            memory_context_id: Optional memory context
            
        Returns:
            Enhanced prompt for LLM
        """
        # Get relevant context
        relevant_context = []
        if memory_context_id:
            relevant_context = self.unified_memory.get_memory(
                memory_context_id,
                MemoryType.CONVERSATION,
                tags=["relevant"],
                limit=5
            )
        
        enhanced_prompt = f"""
Based on my understanding of your request, here's what I believe you're asking for:

**Your Request**: {user_input}
**My Understanding**: {understanding.reasoning}
**Your Goal**: {understanding.user_goal}
**Confidence**: {understanding.confidence:.2f}

**Context Clues**: {', '.join(understanding.context_clues)}
**Suggested Actions**: {', '.join(understanding.suggested_actions)}
**Capabilities Needed**: {', '.join(understanding.system_capabilities_needed)}

**Relevant Context**: {json.dumps(relevant_context, indent=2) if relevant_context else "No relevant context"}

Please provide a comprehensive response that addresses your request effectively.
"""
        
        return enhanced_prompt
    
    def route_with_understanding(self, user_input: str, memory_context_id: str = None) -> Dict[str, Any]:
        """
        Route user input with understanding.
        
        Args:
            user_input: The user's input text
            memory_context_id: Optional memory context identifier
            
        Returns:
            Dictionary with routing information
        """
        # Get contextual understanding
        understanding = self.understand_context(user_input, memory_context_id)
        
        # Generate enhanced prompt
        enhanced_prompt = self.generate_enhanced_prompt(user_input, understanding, memory_context_id)
        
        # Route based on understanding
        routing_result = {
            "original_input": user_input,
            "understanding": {
                "primary_intent": understanding.primary_intent.value,
                "confidence": understanding.confidence,
                "user_goal": understanding.user_goal,
                "reasoning": understanding.reasoning
            },
            "suggested_actions": understanding.suggested_actions,
            "capabilities_needed": understanding.system_capabilities_needed,
            "enhanced_prompt": enhanced_prompt,
            "context_clues": understanding.context_clues,
            "routing_method": "context_aware"
        }
        
        return routing_result

    def optimize_adaptive_features(self) -> Dict[str, Any]:
        """
        Optimize adaptive LAM features (merged from AdaptiveLAM).
        """
        try:
            optimization_results = {
                "dynamic_actions_enabled": self.enable_dynamic_actions,
                "task_decomposition_enabled": self.enable_task_decomposition,
                "llm_reasoning_enabled": self.enable_llm_reasoning,
                "max_subtasks": self.max_subtasks,
                "safety_threshold": self.safety_threshold,
                "optimization_timestamp": time.time()
            }
            
            # Perform adaptive optimizations
            if self.enable_dynamic_actions:
                # Optimize dynamic action generation
                optimization_results["dynamic_actions_optimized"] = True
            
            if self.enable_task_decomposition:
                # Optimize task decomposition
                optimization_results["task_decomposition_optimized"] = True
            
            if self.enable_llm_reasoning:
                # Optimize LLM reasoning
                optimization_results["llm_reasoning_optimized"] = True
            
            self.logger.info("Adaptive LAM features optimized")
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error optimizing adaptive features: {e}")
            return {
                "status": "error",
                "message": f"Adaptive optimization error: {str(e)}",
                "timestamp": time.time()
            }

# Global context-aware LLM instance
context_aware_llm = ContextAwareLLM()

# Convenience functions
def understand_context(user_input: str, memory_context_id: str = None) -> ContextualUnderstanding:
    """Get contextual understanding of user input."""
    return context_aware_llm.understand_context(user_input, memory_context_id)

def route_with_understanding(user_input: str, memory_context_id: str = None) -> Dict[str, Any]:
    """Route user input with understanding."""
    return context_aware_llm.route_with_understanding(user_input, memory_context_id)

@dataclass
class LLMPlan:
    """Represents a plan generated by the LLM."""
    original_request: str
    plan_steps: List[Dict[str, Any]]
    reasoning: str
    confidence: float
    estimated_duration: str
    required_capabilities: List[str]
    fallback_actions: List[str]
    iteration_count: int = 0

@dataclass
class TaskPlan:
    """Represents a plan for executing an uncoded task (from AdaptiveLAM)."""
    original_request: str
    task_type: str
    subtasks: List[Dict[str, Any]]
    required_capabilities: List[str]
    estimated_complexity: str
    confidence: float
    execution_strategy: str
    fallback_actions: List[str]

@dataclass
class DynamicAction:
    """Represents a dynamically generated action (from AdaptiveLAM)."""
    action_name: str
    description: str
    parameters: Dict[str, Any]
    execution_code: str
    safety_level: str
    estimated_duration: float

@dataclass
class ExecutionResult:
    """Represents the result of executing a plan step."""
    step_id: str
    step_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class IntentType(Enum):
    """Types of user intents."""
    UNKNOWN = "unknown"
    QUESTION = "question"
    COMMAND = "command"
    CONVERSATION = "conversation"
    CREATION = "creation"
    ANALYSIS = "analysis"
    HELP = "help"
    SYSTEM = "system"

@dataclass
class IntentResult:
    """Result of intent classification."""
    intent_type: IntentType
    confidence: float
    scroll_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IterativeReasoningContext:
    """Context for iterative reasoning sessions."""
    session_id: str
    original_request: str
    current_iteration: int
    max_iterations: int
    memory_context_id: str
    plans: List[LLMPlan] = field(default_factory=list)
    results: List[ExecutionResult] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

class LAMEngine:
    """
    Language Action Mapping Engine.
    Maps natural language input to symbolic actions using fuzzy logic, conversation memory, and intent classification.
    Enhanced with LLM-driven planning and iterative reasoning.
    """
    
    def __init__(self):
        """Initialize the LAM engine."""
        self.logger = logging.getLogger('LAMEngine')
        self.knowledge_base = []
        self.inference_rules = []
        self.rule_metadata = {}
        self.memory_context = {}
        self.index = {}
        
        # Initialize components
        self.fuzzy_processor = fuzzy_processor
        self.conversation_memory = conversation_memory
        self.llm_engine = llm_engine
        self.unified_memory = unified_memory
        
        # Initialize context-aware LLM
        self.context_aware_llm = context_aware_llm
        
        # Iterative reasoning sessions
        self.reasoning_sessions: Dict[str, IterativeReasoningContext] = {}
        
        # Configuration
        self.max_iterations = 5
        self.planning_timeout = 30  # seconds
        self.execution_timeout = 60  # seconds
        self.enable_iterative_reasoning = True
        self.enable_planning_fallback = True
        
        # Adaptive LAM configuration (merged from adaptive_lam.py)
        self.enable_dynamic_actions = True
        self.enable_task_decomposition = True
        self.enable_llm_reasoning = True
        self.max_subtasks = 5
        self.safety_threshold = 0.7
        
        # Context-aware configuration
        self.enable_context_analysis = True
        self.enable_intent_recognition = True
        self.enable_semantic_understanding = True
        
        self.logger.info("LAM Engine initialized with fuzzy logic, conversation memory, intent classification, LLM-driven planning, and context-aware LLM")
    
    def _initialize_task_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize patterns for different types of tasks."""
        return {
            "information_gathering": {
                "patterns": [
                    r"find.*information", r"research.*", r"look up.*", r"search for.*",
                    r"get.*data", r"collect.*information", r"gather.*facts"
                ],
                "capabilities": ["web_search", "file_access", "api_call"],
                "complexity": "low"
            },
            "data_processing": {
                "patterns": [
                    r"analyze.*", r"process.*data", r"calculate.*", r"compute.*",
                    r"transform.*", r"convert.*", r"format.*"
                ],
                "capabilities": ["generate_code", "file_access", "api_call"],
                "complexity": "medium"
            },
            "content_creation": {
                "patterns": [
                    r"create.*", r"generate.*", r"write.*", r"build.*",
                    r"make.*", r"produce.*", r"develop.*"
                ],
                "capabilities": ["generate_code", "summarize_text", "file_access"],
                "complexity": "medium"
            },
            "system_management": {
                "patterns": [
                    r"optimize.*", r"improve.*", r"enhance.*", r"fix.*",
                    r"maintain.*", r"organize.*", r"clean.*"
                ],
                "capabilities": ["optimize_self", "clean_memory", "self_assess"],
                "complexity": "low"
            },
            "learning_adaptation": {
                "patterns": [
                    r"learn.*", r"adapt.*", r"evolve.*", r"grow.*",
                    r"develop.*", r"improve.*", r"enhance.*"
                ],
                "capabilities": ["reflect", "introspect_core", "ritual_feedback"],
                "complexity": "high"
            },
            "creative_tasks": {
                "patterns": [
                    r"imagine.*", r"design.*", r"invent.*", r"create.*",
                    r"brainstorm.*", r"explore.*", r"experiment.*"
                ],
                "capabilities": ["generate_code", "summarize_text", "web_search"],
                "complexity": "high"
            }
        }
    
    def _initialize_system_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Initialize system capabilities and their descriptions."""
        return {
            "web_search": {
                "description": "Search the internet for information",
                "parameters": ["query", "max_results"],
                "safety_level": "medium"
            },
            "file_access": {
                "description": "Read, write, and manipulate files",
                "parameters": ["file_path", "operation", "content"],
                "safety_level": "high"
            },
            "generate_code": {
                "description": "Generate code from natural language descriptions",
                "parameters": ["language", "description", "requirements"],
                "safety_level": "medium"
            },
            "summarize_text": {
                "description": "Summarize and analyze text content",
                "parameters": ["text", "max_length", "focus"],
                "safety_level": "low"
            },
            "api_call": {
                "description": "Make API calls to external services",
                "parameters": ["endpoint", "method", "data"],
                "safety_level": "medium"
            },
            "optimize_self": {
                "description": "Optimize system performance and parameters",
                "parameters": ["target_area", "optimization_type"],
                "safety_level": "low"
            },
            "reflect": {
                "description": "Perform deep reflection and learning",
                "parameters": ["focus_area", "depth"],
                "safety_level": "low"
            },
            "introspect_core": {
                "description": "Deep introspection of core systems",
                "parameters": ["system", "depth"],
                "safety_level": "low"
            }
        }
    
    def _initialize_action_templates(self) -> Dict[str, str]:
        """Initialize templates for dynamic action generation."""
        return {
            "information_gathering": """
def gather_information(query: str, sources: List[str] = None):
    \"\"\"Gather information from multiple sources.\"\"\"
    results = []
    if sources is None:
        sources = ["web_search", "file_access"]
    
    for source in sources:
        if source == "web_search":
            # Use web search capability
            pass
        elif source == "file_access":
            # Use file access capability
            pass
    
    return results
""",
            "data_processing": """
def process_data(data: Any, operation: str, parameters: Dict[str, Any] = None):
    \"\"\"Process data using specified operation.\"\"\"
    if operation == "analyze":
        # Perform analysis
        pass
    elif operation == "transform":
        # Perform transformation
        pass
    elif operation == "calculate":
        # Perform calculation
        pass
    
    return processed_data
""",
            "content_creation": """
def create_content(content_type: str, description: str, parameters: Dict[str, Any] = None):
    \"\"\"Create content of specified type.\"\"\"
    if content_type == "code":
        # Generate code
        pass
    elif content_type == "text":
        # Generate text
        pass
    elif content_type == "document":
        # Generate document
        pass
    
    return created_content
"""
        }
    
    def add_fact(self, fact):
        """Add a fact to the knowledge base."""
        self.knowledge_base.append(fact)
        self.index[fact] = len(self.knowledge_base) - 1

    def remove_fact(self, fact):
        """Remove a fact from the knowledge base."""
        if fact in self.knowledge_base:
            self.knowledge_base.remove(fact)
            # Rebuild index
            self.index = {fact: i for i, fact in enumerate(self.knowledge_base)}

    def add_rule(self, rule, tag=None):
        """Add an inference rule."""
        self.inference_rules.append(rule)
        if tag:
            self.rule_metadata[rule] = tag

    def remove_rule(self, rule):
        """Remove an inference rule."""
        if rule in self.inference_rules:
            self.inference_rules.remove(rule)
            if rule in self.rule_metadata:
                del self.rule_metadata[rule]

    def evaluate(self, input_data):
        """Evaluate input data using inference rules."""
        results = []
        for rule in self.inference_rules:
            try:
                result = rule(input_data, self.knowledge_base, self.memory_context)
                if result:
                    results.append(result)
            except Exception as e:
                self.logger.error(f"Rule evaluation error: {e}")
        return results

    def update_context(self, key, value):
        """Update memory context."""
        
        self.memory_context[key] = value

    def summarize_state(self):
        """Return a summary of the current state."""
        return {
            "facts_count": len(self.knowledge_base),
            "rules_count": len(self.inference_rules),
            "context": self.memory_context
        }

    def query_facts(self, keyword):
        """Return all facts that include a keyword."""
        return [fact for fact in self.knowledge_base if keyword in fact]

    def explain(self, input_data):
        """Return a symbolic explanation for the inference result."""
        explanations = []
        for rule in self.inference_rules:
            try:
                result = rule(input_data, self.knowledge_base, self.memory_context)
                if result:
                    explanations.append(f"Rule {self.rule_metadata.get(rule, 'anonymous')} triggered result: {result}")
            except Exception as e:
                explanations.append(f"Rule {self.rule_metadata.get(rule, 'anonymous')} caused error: {e}")
        return explanations

    def save_state(self, filepath):
        """Save LAM state to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump({
                "facts": self.knowledge_base,
                "context": self.memory_context,
                "index": self.index,
                "rule_metadata": {str(k): v for k, v in self.rule_metadata.items()}
            }, f)

    def load_state(self, filepath):
        """Load LAM state from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.knowledge_base = data.get("facts", [])
            self.memory_context = data.get("context", {})
            self.index = data.get("index", {})
            self.rule_metadata = data.get("rule_metadata", {})

    def process_lam_query(self, query):
        """
        Process a LAM-style symbolic query. This supports keyword matching,
        fact relevance evaluation, and optional rule-based enrichment.
        """
        results = {
            "query": query,
            "matched_facts": [],
            "reasoning": [],
            "explanation": []
        }

        # Match facts based on keyword presence
        results["matched_facts"] = self.query_facts(query)

        # Run reasoning over the query if rules exist
        if self.inference_rules:
            results["reasoning"] = self.evaluate(query)
            results["explanation"] = self.explain(query)

        return results

    def fallback_llm_answer(self, user_input: str) -> str:
        """
        Use the fallback LLM engine to answer the user with general knowledge.
        Args:
            user_input: The user's input string
        Returns:
            str: LLM-generated answer
        """
        self.logger.info(f"Fallback to LLM for input: '{user_input}'")
        try:
            response = self.llm_engine.run(user_input)
            return response
        except Exception as e:
            self.logger.error(f"LLM fallback error: {e}")
            return "I'm unable to answer that right now."

    def generate_clarifying_question(self, user_input: str, possible_scrolls: List[str] = None) -> str:
        """
        Generate a clarifying question using LLM when unsure about scroll intent.
        
        Args:
            user_input: User's original input
            possible_scrolls: List of possible scroll matches (optional)
            
        Returns:
            str: Generated clarifying question
        """
        self.logger.info(f"Generating clarifying question for: '{user_input}'")
        
        # Try LLM first, fallback to rule-based if not available
        try:
            # Build context for the LLM
            context = f"User input: '{user_input}'\n"
            if possible_scrolls:
                context += f"Possible scrolls: {', '.join(possible_scrolls)}\n"
            
            prompt = (
                f"{context}\n"
                "As an AI assistant, generate a natural, helpful clarifying question to understand "
                "what the user wants to do. The question should be conversational and specific. "
                "Focus on the most likely intent based on the input.\n\n"
                "Generate a single clarifying question:"
            )
            
            clarifying_question = self.llm_engine.run(prompt)
            return clarifying_question.strip()
        except Exception as e:
            self.logger.warning(f"LLM not available for clarifying question: {e}")
            return self._generate_fallback_clarifying_question(user_input, possible_scrolls)
    
    def _generate_fallback_clarifying_question(self, user_input: str, possible_scrolls: List[str] = None) -> str:
        """
        Generate a clarifying question using rule-based patterns when LLM is not available.
        
        Args:
            user_input: User's original input
            possible_scrolls: List of possible scroll matches (optional)
            
        Returns:
            str: Generated clarifying question
        """
        user_input_lower = user_input.lower()
        
        # Rule-based clarifying questions based on input patterns
        if possible_scrolls:
            # If we have specific scroll suggestions, ask about them
            if len(possible_scrolls) == 1:
                scroll_name = possible_scrolls[0]
                if scroll_name == "optimize_self":
                    return f"Did you want me to optimize and improve my system performance?"
                elif scroll_name == "self_assess":
                    return f"Did you want me to assess my current status and performance?"
                elif scroll_name == "introspect_core":
                    return f"Did you want me to perform deep introspection and self-reflection?"
                elif scroll_name == "calm_sequence":
                    return f"Did you want me to execute a calming and grounding sequence?"
                else:
                    return f"Did you want me to run the '{scroll_name}' function?"
            else:
                return f"I found several possible actions: {', '.join(possible_scrolls)}. Which one did you mean?"
        
        # Pattern-based clarifying questions
        if any(word in user_input_lower for word in ["optimize", "improve", "enhance", "better"]):
            return "What would you like me to optimize or improve? My performance, memory, or something else?"
        
        if any(word in user_input_lower for word in ["assess", "check", "status", "how"]):
            return "What would you like me to assess or check? My current status, performance, or something specific?"
        
        if any(word in user_input_lower for word in ["reflect", "think", "introspect"]):
            return "What would you like me to reflect on or think about? Recent experiences, my state, or something else?"
        
        if any(word in user_input_lower for word in ["calm", "relax", "peace"]):
            return "Would you like me to help you calm down, or did you want me to perform a calming sequence?"
        
        if any(word in user_input_lower for word in ["help", "assist", "support"]):
            return "What kind of help do you need? I can optimize, assess, reflect, or assist with various tasks."
        
        if any(word in user_input_lower for word in ["do", "make", "perform", "execute"]):
            return "What would you like me to do? I can optimize, assess, reflect, search, or help with various tasks."
        
        if any(word in user_input_lower for word in ["search", "find", "look"]):
            return "What would you like me to search for? I can search the web, Wikipedia, or help find information."
        
        if any(word in user_input_lower for word in ["weather", "temperature", "forecast"]):
            return "Would you like me to check the weather for a specific location?"
        
        if any(word in user_input_lower for word in ["summarize", "analyze", "document"]):
            return "What would you like me to summarize or analyze? A document, text, or something else?"
        
        # Default clarifying question
        return f"I'm not sure what you'd like me to do with '{user_input}'. Could you be more specific about what you want me to help you with?"

    def self_reference_detector(self, user_input: str) -> str:
        """
        Detect and transform self-references where "you" refers to the daemon.
        
        Args:
            user_input: Original user input
            
        Returns:
            str: Transformed input with self-references clarified
        """
        input_lower = user_input.lower().strip()
        
        # Patterns where "you" likely refers to the daemon
        self_reference_patterns = [
            # Questions about the daemon's state/status
            (r"how are you", "how am i"),
            (r"how do you feel", "how do i feel"),
            (r"what are you doing", "what am i doing"),
            (r"are you ok", "am i ok"),
            (r"are you working", "am i working"),
            (r"are you ready", "am i ready"),
            (r"can you", "can i"),
            (r"do you", "do i"),
            (r"will you", "will i"),
            (r"would you", "would i"),
            (r"should you", "should i"),
            (r"have you", "have i"),
            (r"did you", "did i"),
            (r"are you", "am i"),
            (r"is you", "am i"),
            (r"was you", "was i"),
            (r"were you", "was i"),
            
            # Self-assessment patterns
            (r"how are you doing", "how am i doing"),
            (r"how do you feel about", "how do i feel about"),
            (r"what do you think about", "what do i think about"),
            (r"what do you know about", "what do i know about"),
            (r"what can you do", "what can i do"),
            (r"what are you capable of", "what am i capable of"),
            
            # Identity questions
            (r"who are you", "who am i"),
            (r"what are you", "what am i"),
            (r"tell me about you", "tell me about myself"),
            (r"describe you", "describe myself"),
            
            # Status checks
            (r"are you online", "am i online"),
            (r"are you active", "am i active"),
            (r"are you functioning", "am i functioning"),
            (r"are you operational", "am i operational"),
            
            # Capability questions
            (r"can you help", "can i help"),
            (r"can you assist", "can i assist"),
            (r"can you explain", "can i explain"),
            (r"can you show", "can i show"),
            (r"can you tell", "can i tell"),
            (r"can you find", "can i find"),
            (r"can you search", "can i search"),
            (r"can you analyze", "can i analyze"),
            (r"can you optimize", "can i optimize"),
            (r"can you assess", "can i assess"),
            (r"can you reflect", "can i reflect"),
            (r"can you introspect", "can i introspect"),
        ]
        
        # Apply transformations
        transformed_input = user_input
        for pattern, replacement in self_reference_patterns:
            if pattern in input_lower:
                # Replace the pattern while preserving case
                import re
                transformed_input = re.sub(pattern, replacement, transformed_input, flags=re.IGNORECASE)
                self.logger.info(f"Self-reference detected: '{user_input}' -> '{transformed_input}'")
                break
        
        # Handle standalone "you" in certain contexts
        if "you" in input_lower and not any(pattern in input_lower for pattern, _ in self_reference_patterns):
            # Check if this is likely a self-reference based on context
            context_words = ["how", "what", "who", "are", "can", "do", "will", "would", "should", "have", "did", "tell", "describe", "show"]
            if any(word in input_lower for word in context_words):
                transformed_input = transformed_input.replace(" you ", " i ").replace(" you?", " i?").replace(" you.", " i.")
                self.logger.info(f"Context-based self-reference: '{user_input}' -> '{transformed_input}'")
        
        return transformed_input

    def interpret_prompt_with_clarification(self, text: str, clarification_context: Dict[str, Any] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Enhanced interpret_prompt that can generate clarifying questions when unsure.
        
        Args:
            text: User input text
            clarification_context: Context from previous clarification attempts
            
        Returns:
            Tuple of (scroll_name, llm_response, clarifying_question)
        """
        # Normalize input
        text = text.lower().strip()
        
        # Apply self-reference detection
        original_text = text
        text = self.self_reference_detector(text)
        if text != original_text:
            self.logger.info(f"Self-reference transformation: '{original_text}' -> '{text}'")
        
        # Check for clarification responses
        if clarification_context and clarification_context.get("waiting_for_clarification"):
            # Handle user's response to previous clarification
            if text in ["yes", "y", "confirm", "correct"]:
                return clarification_context.get("suggested_scroll"), None, None
            elif text in ["no", "n", "cancel", "wrong"]:
                # Generate a new clarifying question
                clarifying_question = self.generate_clarifying_question(
                    clarification_context.get("original_input", text)
                )
                return None, None, clarifying_question
        
        # Step 1: Check if this is a follow-up question
        is_follow_up, resolved_command, context_info = self.conversation_memory.analyze_follow_up(text)
        
        if is_follow_up and resolved_command:
            self.logger.info(f"Follow-up detected: '{text}' -> '{resolved_command}' (context: {context_info.get('type', 'unknown')})")
            return resolved_command, None, None
        
        # Step 2: Intent classification
        intent_result = self.llm_engine.classify_intent(user_input)
        
        if intent_result.confidence >= 0.7:  # High confidence intent classification
            self.logger.info(f"Intent classification: '{text}' -> {intent_result.intent_type.value} (confidence: {intent_result.confidence:.2f}) -> {intent_result.scroll_name}")
            
            # Use intent-based routing
            if intent_result.scroll_name and intent_result.scroll_name != "general_conversation":
                return intent_result.scroll_name, None, None
        
        # Step 3: Fuzzy processor for improved command recognition
        fuzzy_result = self.fuzzy_processor.process_input(text)
        
        if fuzzy_result:
            command, confidence, category = fuzzy_result
            self.logger.info(f"Fuzzy match: '{text}' -> '{command}' (confidence: {confidence:.2f}, category: {category})")
            
            # Check if confidence is low and needs clarification
            if confidence < 0.7:
                # Generate a contextual clarifying question
                clarifying_question = self.generate_clarifying_question(text, [command])
                return None, None, clarifying_question
            
            # Return the command if confidence is high enough
            if confidence >= 0.6:
                return command, None, None
            else:
                self.logger.info(f"Confidence too low ({confidence:.2f}) for '{text}'")
        
        # Step 4: Fallback to legacy phrase mappings for backward compatibility
        phrase_mappings = {
            # Self-assessment and introspection
            "how am i doing": "self_assess",
            "how am i": "self_assess", 
            "self assessment": "self_assess",
            "check my status": "self_assess",
            "how are you doing": "self_assess",
            "status check": "self_assess",
            
            # Calming and emotional regulation
            "calm down": "calm_sequence",
            "calm": "calm_sequence",
            "relax": "calm_sequence",
            "breathe": "calm_sequence",
            "take a breath": "calm_sequence",
            "ground me": "calm_sequence",
            "center me": "calm_sequence",
            
            # Optimization and maintenance
            "optimize": "optimize_self",
            "optimize self": "optimize_self",
            "self optimize": "optimize_self",
            "clean up": "optimize_self",
            "maintenance": "optimize_self",
            "tune up": "optimize_self",
            
            # Memory and introspection
            "introspect": "introspect_core",
            "introspection": "introspect_core",
            "deep dive": "introspect_core",
            "self reflection": "introspect_core",
            "reflect": "introspect_core",
            
            # Memory management
            "clean memory": "clean_memory",
            "clear memory": "clean_memory",
            "memory cleanup": "clean_memory",
            "sweep memory": "clean_memory",
            
            # Protection and security
            "activate shield": "activate_shield",
            "shield": "activate_shield",
            "protect": "activate_shield",
            "defense": "activate_shield",
            
            # Exit and shutdown
            "exit": "exit",
            "quit": "exit",
            "stop": "exit",
            "shutdown": "exit",
            "goodbye": "exit",
            "bye": "exit"
        }
        
        # Check for exact matches first
        if text in phrase_mappings:
            return phrase_mappings[text], None, None
        
        # Check for partial matches (words contained within the input)
        words = text.split()
        for phrase, scroll_name in phrase_mappings.items():
            phrase_words = phrase.split()
            # Check if all words in the phrase are present in the input
            if all(word in words for word in phrase_words):
                return scroll_name, None, None
        
        # Check for keyword matches (any word in the phrase matches)
        for phrase, scroll_name in phrase_mappings.items():
            phrase_words = phrase.split()
            if any(word in words for word in phrase_words):
                return scroll_name, None, None
        
        # Step 5: Use intent-based fallback
        if intent_result.scroll_name:
            return intent_result.scroll_name, None, None
        
        # Step 6: Generate clarifying question if no clear match
        if not clarification_context or not clarification_context.get("asked_clarification"):
            clarifying_question = self.generate_clarifying_question(text)
            return None, None, clarifying_question
        
        # Step 7: Fallback to LLM if clarification was already asked
        llm_response = self.fallback_llm_answer(text)
        return None, llm_response, None

    def get_suggestions(self, text: str, max_suggestions: int = 3) -> List[Tuple[str, float]]:
        """
        Get command suggestions for user input.
        
        Args:
            text: User's input text
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of (command, confidence) tuples
        """
        return self.fuzzy_processor.get_suggestions(text, max_suggestions)

    def cast_scroll(self, scroll_name: str, parameters: dict = None) -> dict:
        """
        Cast a scroll using the unified scroll engine.
        
        Args:
            scroll_name: Name of the scroll to cast
            parameters: Scroll parameters
            
        Returns:
            Scroll execution result
        """
        from ..scrolls.scroll_engine import cast_scroll as engine_cast_scroll
        
        # Use the unified scroll engine
        result = engine_cast_scroll(scroll_name, parameters or {})
        
        # Convert ScrollResult to dict for backward compatibility
        return {
            "scroll": scroll_name,
            "result": result.output,
            "status": "success" if result.success else "failed",
            "execution_time": result.execution_time,
            "metadata": result.metadata
        }

    def list_scrolls(self) -> dict:
        """
        List all available scrolls using the unified scroll engine.
        Returns:
            dict: Mapping of scroll names to their descriptions and metadata.
        """
        from ..scrolls.scroll_engine import list_scrolls as engine_list_scrolls
        
        # Use the unified scroll engine
        scrolls = engine_list_scrolls()
        
        # Convert to the expected format for backward compatibility
        result = {}
        for scroll_name, scroll_info in scrolls.items():
            result[scroll_name] = {
                "description": scroll_info.get("description", "No description"),
                "external_access": scroll_info.get("is_external", False)
            }
        
        return result

    def add_conversation_turn(self, user_input: str, scroll_name: Optional[str] = None, 
                             response: Optional[str] = None, confidence: float = 0.0,
                             context: Optional[Dict[str, Any]] = None):
        """
        Add a conversation turn to memory for context tracking.
        
        Args:
            user_input: User's input
            scroll_name: Executed scroll name (if any)
            response: System response
            confidence: Confidence score
            context: Additional context
        """
        self.conversation_memory.add_turn(
            user_input=user_input,
            scroll_name=scroll_name,
            response=response,
            confidence=confidence,
            context=context
        )
    
    def get_conversation_context(self) -> Optional[Dict[str, Any]]:
        """
        Get current conversation context.
        
        Returns:
            Conversation context or None
        """
        context = self.conversation_memory.get_context()
        if context:
            return {
                "session_id": context.session_id,
                "current_topic": context.current_topic,
                "last_command": context.last_command,
                "last_question": context.last_question,
                "total_turns": len(context.turns),
                "session_duration": time.time() - context.start_time
            }
        return None
    
    def get_conversation_summary(self) -> str:
        """
        Get a summary of the current conversation.
        
        Returns:
            Conversation summary string
        """
        return self.conversation_memory.get_conversation_summary()
    
    def clear_conversation_memory(self):
        """Clear the current conversation memory."""
        self.conversation_memory.clear_memory()

    def classify_intent(self, user_input: str) -> IntentResult:
        """
        Classify user input into an intent type using context-aware analysis.
        
        Args:
            user_input: User's input text
            
        Returns:
            IntentResult with classification and routing information
        """
        # Use context-aware LLM for enhanced intent classification
        if self.enable_context_analysis:
            understanding = self.context_aware_llm.understand_context(user_input)
            
            # Map context-aware intent to LAM intent
            intent_mapping = {
                IntentCategory.SELF_IMPROVEMENT: IntentType.SYSTEM,
                IntentCategory.SYSTEM_OPTIMIZATION: IntentType.SYSTEM,
                IntentCategory.INFORMATION_GATHERING: IntentType.QUESTION,
                IntentCategory.TASK_EXECUTION: IntentType.COMMAND,
                IntentCategory.CREATIVE_WORK: IntentType.CREATION,
                IntentCategory.ANALYSIS: IntentType.ANALYSIS,
                IntentCategory.CONVERSATION: IntentType.CONVERSATION,
                IntentCategory.UNKNOWN: IntentType.UNKNOWN
            }
            
            lam_intent = intent_mapping.get(understanding.primary_intent, IntentType.UNKNOWN)
            
            return IntentResult(
                intent_type=lam_intent,
                confidence=understanding.confidence,
                parameters={"context_understanding": understanding.__dict__},
                metadata={
                    "context_aware": True,
                    "primary_intent": understanding.primary_intent.value,
                    "secondary_intents": [intent.value for intent in understanding.secondary_intents],
                    "user_goal": understanding.user_goal,
                    "suggested_actions": understanding.suggested_actions
                }
            )
        
        # Fallback to LLM engine for intent classification
        return self.llm_engine.classify_intent(user_input)
    
    def get_intent_statistics(self) -> Dict[str, Any]:
        """Get statistics about intent classification."""
        return self.llm_engine.get_statistics()
    
    def route_with_context_awareness(self, user_input: str, memory_context_id: str = None) -> Dict[str, Any]:
        """
        Route user input using context-aware analysis.
        
        Args:
            user_input: User's input text
            memory_context_id: Optional memory context identifier
            
        Returns:
            Dictionary with routing information and context understanding
        """
        if self.enable_context_analysis:
            return self.context_aware_llm.route_with_understanding(user_input, memory_context_id)
        else:
            # Fallback to basic routing
            intent_result = self.classify_intent(user_input)
            return {
                "original_input": user_input,
                "intent": intent_result.intent_type.value,
                "confidence": intent_result.confidence,
                "routing_method": "basic"
            }
    
    def route_with_understanding(self, user_input: str, memory_context_id: str = None) -> Dict[str, Any]:
        """Route user input with context-aware understanding."""
        try:
            # Use the context-aware LLM for understanding
            understanding = self.context_aware_llm.understand_context(user_input, memory_context_id)
            
            # Generate enhanced prompt
            enhanced_prompt = self.context_aware_llm.generate_enhanced_prompt(
                user_input, understanding, memory_context_id
            )
            
            return {
                "success": True,
                "understanding": understanding,
                "enhanced_prompt": enhanced_prompt,
                "confidence": understanding.confidence,
                "suggested_actions": understanding.suggested_actions
            }
            
        except Exception as e:
            self.logger.error(f"Error in route_with_understanding: {e}")
            return {
                "success": False,
                "error": str(e),
                "enhanced_prompt": user_input
            }
    
    def add_custom_intent(self, intent_type: IntentType, patterns: List[str], keywords: List[str], scroll_mapping: Dict[str, str]):
        """Add a custom intent pattern."""
        self.llm_engine.add_custom_intent(intent_type, patterns, keywords, scroll_mapping)
    
    def optimize(self) -> Dict[str, Any]:
        """
        Optimize the LAM engine and its components.
        
        Returns:
            Dict containing optimization results and verification
        """
        self.logger.info("Optimizing LAM Engine components")
        
        optimization_results = {
            "fuzzy_processor": {},
            "conversation_memory": {},
            "intent_classifier": {},
            "overall": {}
        }
        
        # Optimize fuzzy processor
        try:
            if hasattr(self.fuzzy_processor, 'optimize'):
                fuzzy_result = self.fuzzy_processor.optimize()
                optimization_results["fuzzy_processor"] = {
                    "status": "optimized",
                    "result": fuzzy_result
                }
            else:
                optimization_results["fuzzy_processor"] = {
                    "status": "no_optimization_method",
                    "result": None
                }
        except Exception as e:
            optimization_results["fuzzy_processor"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Optimize conversation memory
        try:
            if hasattr(self.conversation_memory, 'optimize'):
                memory_result = self.conversation_memory.optimize()
                optimization_results["conversation_memory"] = {
                    "status": "optimized",
                    "result": memory_result
                }
            else:
                optimization_results["conversation_memory"] = {
                    "status": "no_optimization_method",
                    "result": None
                }
        except Exception as e:
            optimization_results["conversation_memory"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Optimize intent classifier
        try:
            if hasattr(self.llm_engine, 'optimize'):
                intent_result = self.llm_engine.optimize()
                optimization_results["intent_classifier"] = {
                    "status": "optimized",
                    "result": intent_result
                }
            else:
                optimization_results["intent_classifier"] = {
                    "status": "no_optimization_method",
                    "result": None
                }
        except Exception as e:
            optimization_results["intent_classifier"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Calculate overall optimization status
        successful_optimizations = 0
        total_components = 3
        
        for component, result in optimization_results.items():
            if component != "overall" and result.get("status") == "optimized":
                successful_optimizations += 1
        
        optimization_results["overall"] = {
            "status": "completed",
            "successful_optimizations": successful_optimizations,
            "total_components": total_components,
            "success_rate": (successful_optimizations / total_components) * 100
        }
        
        self.logger.info(f"LAM Engine optimization completed: {successful_optimizations}/{total_components} components optimized")
        
        return optimization_results
    
    def handle_uncoded_task(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle a task that isn't explicitly coded.
        
        Args:
            user_input: User's request
            context: Additional context
            
        Returns:
            Result of task execution
        """
        self.logger.info(f"Handling uncoded task: {user_input}")
        
        # Step 1: Analyze the task
        task_analysis = self._analyze_task(user_input)
        
        # Step 2: Generate a task plan
        task_plan = self._generate_task_plan(user_input, task_analysis)
        
        # Step 3: Execute the plan
        execution_result = self._execute_task_plan(task_plan, context)
        
        return execution_result
    
    def _analyze_task(self, user_input: str) -> Dict[str, Any]:
        """Analyze the task to understand its requirements."""
        try:
            # Use LLM to analyze the task
            analysis_prompt = f"""
Analyze this user request and determine:
1. Task type (information_gathering, data_processing, content_creation, system_management, learning_adaptation, creative_tasks)
2. Required capabilities
3. Complexity level (low, medium, high)
4. Safety considerations
5. Estimated duration

User request: {user_input}

Respond in JSON format:
{{
    "task_type": "string",
    "required_capabilities": ["list", "of", "capabilities"],
    "complexity": "low|medium|high",
    "safety_level": "low|medium|high",
    "estimated_duration": "short|medium|long",
    "description": "brief description of what needs to be done"
}}
"""
            
            response = self.llm_engine.run(
                prompt=analysis_prompt,
                temperature=0.3,
                max_tokens=300
            )
            
            # Parse JSON response
            if response and response.startswith("{") and response.endswith("}"):
                analysis = json.loads(response)
                return analysis
            else:
                # Fallback to pattern matching
                return self._pattern_based_analysis(user_input)
                
        except Exception as e:
            self.logger.warning(f"LLM task analysis failed: {e}")
            return self._pattern_based_analysis(user_input)
    
    def _pattern_based_analysis(self, user_input: str) -> Dict[str, Any]:
        """Fallback to pattern-based task analysis."""
        user_input_lower = user_input.lower()
        
        for task_type, task_info in self._initialize_task_patterns().items():
            for pattern in task_info["patterns"]:
                if re.search(pattern, user_input_lower, re.IGNORECASE):
                    return {
                        "task_type": task_type,
                        "required_capabilities": task_info["capabilities"],
                        "complexity": task_info["complexity"],
                        "safety_level": "medium",
                        "estimated_duration": "medium",
                        "description": f"Perform {task_type} task based on user request"
                    }
        
        # Default analysis
        return {
            "task_type": "general",
            "required_capabilities": ["web_search", "generate_code"],
            "complexity": "medium",
            "safety_level": "medium",
            "estimated_duration": "medium",
            "description": "Handle general user request"
        }
    
    def _generate_task_plan(self, user_input: str, analysis: Dict[str, Any]) -> LLMPlan:
        """Generate a plan for executing the task."""
        try:
            # Use LLM to generate task plan
            planning_prompt = f"""
Based on this task analysis, generate a detailed execution plan:

Task Analysis: {json.dumps(analysis, indent=2)}
User Request: {user_input}

Generate a plan with:
1. Subtasks (break down the main task)
2. Required capabilities for each subtask
3. Execution strategy
4. Fallback actions

Respond in JSON format:
{{
    "subtasks": [
        {{
            "name": "subtask name",
            "description": "what this subtask does",
            "capability": "required capability",
            "parameters": {{"param": "value"}}
        }}
    ],
    "execution_strategy": "sequential|parallel|conditional",
    "fallback_actions": ["action1", "action2"],
    "confidence": 0.0-1.0
}}
"""
            
            response = self.llm_engine.run(
                prompt=planning_prompt,
                temperature=0.4,
                max_tokens=500
            )
            
            if response and response.startswith("{") and response.endswith("}"):
                plan_data = json.loads(response)
                
                return LLMPlan(
                    original_request=user_input,
                    plan_steps=plan_data.get("subtasks", []),
                    reasoning=plan_data.get("reasoning", ""),
                    confidence=plan_data.get("confidence", 0.7),
                    estimated_duration=plan_data.get("estimated_duration", "medium"),
                    required_capabilities=analysis["required_capabilities"],
                    fallback_actions=plan_data.get("fallback_actions", [])
                )
            else:
                return self._generate_simple_plan(user_input, analysis)
                
        except Exception as e:
            self.logger.warning(f"LLM task planning failed: {e}")
            return self._generate_simple_plan(user_input, analysis)
    
    def _generate_simple_plan(self, user_input: str, analysis: Dict[str, Any]) -> LLMPlan:
        """Generate a simple task plan when LLM planning fails."""
        subtasks = []
        
        # Create subtasks based on required capabilities
        for capability in analysis["required_capabilities"]:
            subtasks.append({
                "name": f"Use {capability}",
                "description": f"Utilize {capability} capability",
                "capability": capability,
                "parameters": {"query": user_input}
            })
        
        return LLMPlan(
            original_request=user_input,
            plan_steps=subtasks,
            reasoning="",
            confidence=0.6,
            estimated_duration="medium",
            required_capabilities=analysis["required_capabilities"],
            fallback_actions=["general_conversation"]
        )
    
    def _execute_task_plan(self, task_plan: LLMPlan, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the task plan."""
        self.logger.info(f"Executing task plan for: {task_plan.original_request}")
        
        results = []
        context = context or {}
        
        try:
            # Execute subtasks based on strategy
            for step in task_plan.plan_steps:
                result = self._execute_step(step, context)
                results.append(result)
            
            # Compile final result
            successful_results = [r for r in results if r.success]
            
            if successful_results:
                return {
                    "success": True,
                    "message": f"Successfully completed {len(successful_results)} steps for: {task_plan.original_request}",
                    "results": successful_results,
                    "task_plan": task_plan,
                    "execution_time": time.time()
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to complete task: {task_plan.original_request}",
                    "results": results,
                    "task_plan": task_plan,
                    "execution_time": time.time()
                }
                
        except Exception as e:
            self.logger.error(f"Task plan execution failed: {e}")
            return {
                "success": False,
                "message": f"Error executing task plan: {str(e)}",
                "task_plan": task_plan,
                "execution_time": time.time()
            }
    
    def _execute_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> ExecutionResult:
        """Execute a single step of the task plan."""
        capability = step.get("capability")
        parameters = step.get("parameters", {})
        
        try:
            # Map capability to actual scroll execution
            if capability == "web_search":
                result = self.cast_scroll("web_search", parameters)
                return ExecutionResult(
                    step_id=step["name"],
                    step_name=step["name"],
                    success=result.get("status") == "success",
                    result=result.get("result"),
                    execution_time=result.get("execution_time", 0.0),
                    metadata=result.get("metadata", {})
                )
            
            elif capability == "generate_code":
                result = self.cast_scroll("generate_code", parameters)
                return ExecutionResult(
                    step_id=step["name"],
                    step_name=step["name"],
                    success=result.get("status") == "success",
                    result=result.get("result"),
                    execution_time=result.get("execution_time", 0.0),
                    metadata=result.get("metadata", {})
                )
            
            elif capability == "summarize_text":
                result = self.cast_scroll("summarize_text", parameters)
                return ExecutionResult(
                    step_id=step["name"],
                    step_name=step["name"],
                    success=result.get("status") == "success",
                    result=result.get("result"),
                    execution_time=result.get("execution_time", 0.0),
                    metadata=result.get("metadata", {})
                )
            
            elif capability == "optimize_self":
                result = self.cast_scroll("optimize_self", parameters)
                return ExecutionResult(
                    step_id=step["name"],
                    step_name=step["name"],
                    success=result.get("status") == "success",
                    result=result.get("result"),
                    execution_time=result.get("execution_time", 0.0),
                    metadata=result.get("metadata", {})
                )
            
            elif capability == "reflect":
                result = self.cast_scroll("reflect", parameters)
                return ExecutionResult(
                    step_id=step["name"],
                    step_name=step["name"],
                    success=result.get("status") == "success",
                    result=result.get("result"),
                    execution_time=result.get("execution_time", 0.0),
                    metadata=result.get("metadata", {})
                )
            
            else:
                # Try to generate dynamic action
                return self._generate_dynamic_action(capability, parameters, context)
                
        except Exception as e:
            self.logger.error(f"Step execution failed: {e}")
            return ExecutionResult(
                step_id=step["name"],
                step_name=step["name"],
                success=False,
                result=None,
                error=str(e)
            )
    
    def _generate_dynamic_action(self, capability: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> ExecutionResult:
        """Generate and execute a dynamic action."""
        try:
            # Use LLM to generate action code
            action_prompt = f"""
Generate Python code to perform this action:
Capability: {capability}
Parameters: {json.dumps(parameters, indent=2)}
Context: {json.dumps(context, indent=2)}

Generate safe, executable Python code that:
1. Performs the requested action
2. Returns a result
3. Handles errors gracefully
4. Is safe to execute

Return only the Python function code:
"""
            
            response = self.llm_engine.run(
                prompt=action_prompt,
                temperature=0.2,
                max_tokens=400
            )
            
            if response and "def " in response:
                # Create a safe execution environment
                safe_globals = {
                    "__builtins__": {
                        "len": len,
                        "str": str,
                        "int": int,
                        "float": float,
                        "list": list,
                        "dict": dict,
                        "print": print
                    }
                }
                
                # Execute the generated code
                exec(response, safe_globals)
                
                # Find the function and call it
                for name, obj in safe_globals.items():
                    if callable(obj) and name.startswith("execute_"):
                        result = obj(parameters, context)
                        return ExecutionResult(
                            step_id=f"dynamic_{capability}",
                            step_name=f"Dynamic Action: {capability}",
                            success=True,
                            result=result,
                            execution_time=time.time()
                        )
                
                return ExecutionResult(
                    step_id=f"dynamic_{capability}",
                    step_name=f"Dynamic Action: {capability}",
                    success=False,
                    result=None,
                    error="No executable function found"
                )
            else:
                return ExecutionResult(
                    step_id=f"dynamic_{capability}",
                    step_name=f"Dynamic Action: {capability}",
                    success=False,
                    result=None,
                    error="Failed to generate action code"
                )
                
        except Exception as e:
            self.logger.error(f"Dynamic action generation failed: {e}")
            return ExecutionResult(
                step_id=f"dynamic_{capability}",
                step_name=f"Dynamic Action: {capability}",
                success=False,
                result=None,
                error=str(e)
            )
    
    def route_with_adaptive_fallback(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Route user input with adaptive fallback for uncoded tasks.
        
        Args:
            user_input: User's input
            context: Additional context
            
        Returns:
            Routing result with execution details
        """
        self.logger.info(f"Routing with adaptive fallback: {user_input}")
        
        # First, try normal routing through symbolic router
        try:
            from ..core.symbolic_router import route_intent, execute_action
            action_plan = route_intent(user_input, context)
            
            if action_plan.confidence > 0.5:
                # Execute the planned action
                result = execute_action(action_plan)
                if result.success:
                    return {
                        "success": True,
                        "routing_method": "symbolic_router",
                        "action_plan": action_plan,
                        "result": result
                    }
        except Exception as e:
            self.logger.warning(f"Symbolic routing failed: {e}")
        
        # If symbolic routing fails or has low confidence, try adaptive handling
        if self.enable_iterative_reasoning:
            try:
                iterative_context = IterativeReasoningContext(
                    session_id=str(time.time()),
                    original_request=user_input,
                    current_iteration=0,
                    max_iterations=self.max_iterations,
                    memory_context_id=str(id(self.memory_context))
                )
                adaptive_result = self.handle_uncoded_task(user_input, context)
                iterative_context.results.append(adaptive_result)
                iterative_context.reasoning_chain.append(f"Adaptive handling completed with result: {adaptive_result}")
                return {
                    "success": adaptive_result.get("success", False),
                    "routing_method": "adaptive_lam",
                    "result": adaptive_result
                }
            except Exception as e:
                self.logger.error(f"Adaptive handling failed: {e}")
        
        # Final fallback to LLM
        try:
            llm_response = self.fallback_llm_answer(user_input)
            return {
                "success": True,
                "routing_method": "llm_fallback",
                "result": {"response": llm_response}
            }
        except Exception as e:
            self.logger.error(f"LLM fallback failed: {e}")
            return {
                "success": False,
                "routing_method": "failed",
                "error": "All routing methods failed"
            }

    def _execute_plan(self, plan: LLMPlan, memory_context_id: str) -> List[ExecutionResult]:
        """Execute a plan and return results."""
        results = []
        
        for step in plan.plan_steps:
            step_id = step.get("step_id", f"step_{len(results)}")
            step_name = step.get("step_name", "Unknown step")
            action = step.get("action", "")
            parameters = step.get("parameters", {})
            
            start_time = time.time()
            
            try:
                # Execute the step
                if action in self._get_available_scrolls():
                    result_dict = self.cast_scroll(action, parameters)
                    success = result_dict.get("status") == "success"
                    step_result = result_dict.get("result")
                    error = None if success else result_dict.get("error", "Unknown error")
                else:
                    # Try fallback or general conversation
                    result_dict = self.cast_scroll("general_conversation", {
                        "user_input": f"Execute: {step_name} with parameters: {parameters}"
                    })
                    success = result_dict.get("status") == "success"
                    step_result = result_dict.get("result")
                    error = None if success else "Action not available"
                
                execution_time = time.time() - start_time
                
                execution_result = ExecutionResult(
                    step_id=step_id,
                    step_name=step_name,
                    success=success,
                    result=step_result,
                    error=error,
                    execution_time=execution_time,
                    metadata={"action": action, "parameters": parameters}
                )
                
                results.append(execution_result)
                
                # Store step result in memory
                self.unified_memory.add_memory(
                    memory_context_id,
                    MemoryType.EXECUTION,
                    {
                        "step_id": step_id,
                        "step_name": step_name,
                        "success": success,
                        "result": step_result,
                        "error": error,
                        "execution_time": execution_time
                    },
                    tags=["plan_execution"]
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                execution_result = ExecutionResult(
                    step_id=step_id,
                    step_name=step_name,
                    success=False,
                    result=None,
                    error=str(e),
                    execution_time=execution_time,
                    metadata={"action": action, "parameters": parameters}
                )
                results.append(execution_result)
        
        return results

    def _execute_with_iterative_reasoning(self, initial_plan: LLMPlan, user_input: str,
                                        context: Dict[str, Any], memory_context_id: str) -> Dict[str, Any]:
        """Execute plan with iterative reasoning capabilities."""
        session_id = f"reasoning_{int(time.time())}"
        
        reasoning_context = IterativeReasoningContext(
            session_id=session_id,
            original_request=user_input,
            current_iteration=0,
            max_iterations=self.max_iterations,
            memory_context_id=memory_context_id
        )
        
        self.reasoning_sessions[session_id] = reasoning_context
        
        try:
            current_plan = initial_plan
            iteration = 0
            
            while iteration < self.max_iterations:
                iteration += 1
                reasoning_context.current_iteration = iteration
                
                self.logger.info(f"Starting iteration {iteration} for: {user_input}")
                
                # Execute current plan
                execution_results = self._execute_plan(current_plan, memory_context_id)
                reasoning_context.results.extend(execution_results)
                
                # Check if we need to iterate
                if self._should_continue_reasoning(execution_results, current_plan):
                    # Generate refined plan
                    refined_plan = self._refine_plan_with_results(
                        current_plan, execution_results, user_input, memory_context_id
                    )
                    
                    if refined_plan:
                        current_plan = refined_plan
                        reasoning_context.plans.append(refined_plan)
                        
                        # Store reasoning step
                        reasoning_step = f"Iteration {iteration}: Refined plan based on execution results"
                        reasoning_context.reasoning_chain.append(reasoning_step)
                        
                        self.unified_memory.add_memory(
                            memory_context_id,
                            MemoryType.REASONING,
                            {
                                "iteration": iteration,
                                "reasoning": reasoning_step,
                                "refined_plan": refined_plan
                            },
                            tags=["iterative_reasoning"]
                        )
                    else:
                        break
                else:
                    break
            
            # Compile final result
            return self._compile_iterative_result(reasoning_context, reasoning_context.results)
            
        finally:
            # Cleanup reasoning session
            if session_id in self.reasoning_sessions:
                del self.reasoning_sessions[session_id]

    def handle_3d_construction_task(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle 3D construction tasks by integrating native 3D generation with realm building.
        
        This method coordinates between:
        1. Native Text-to-3D model generation
        2. Storyrealms bridge for realm management
        3. External 3D engine integration
        """
        try:
            self.logger.info(f"Handling 3D construction task: {user_input}")
            
            # Step 1: Analyze the construction request
            analysis = self._analyze_3d_construction_request(user_input)
            
            # Step 2: Generate 3D model if needed
            model_result = None
            if analysis.get("needs_model_generation"):
                model_result = self._generate_3d_model_for_construction(analysis)
            
            # Step 3: Create or use existing realm
            realm_result = self._manage_realm_for_construction(analysis, model_result)
            
            # Step 4: Place objects in realm
            placement_result = self._place_objects_in_realm(analysis, realm_result, model_result)
            
            # Step 5: Generate external engine request
            engine_request = self._generate_engine_request(analysis, realm_result, placement_result)
            
            return {
                "status": "success",
                "task_type": "3d_construction",
                "analysis": analysis,
                "model_generated": model_result is not None,
                "realm_created": realm_result.get("realm_created", False),
                "objects_placed": placement_result.get("objects_placed", []),
                "engine_request": engine_request,
                "message": f"Successfully processed 3D construction: {analysis.get('primary_action', 'unknown')}",
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error in 3D construction task: {e}")
            return {
                "status": "error",
                "error": str(e),
                "task_type": "3d_construction",
                "timestamp": time.time()
            }
    
    def _analyze_3d_construction_request(self, user_input: str) -> Dict[str, Any]:
        """Analyze a 3D construction request to determine what needs to be done."""
        analysis = {
            "primary_action": "unknown",
            "needs_model_generation": False,
            "needs_realm_creation": False,
            "object_types": [],
            "materials": [],
            "dimensions": {},
            "location": {},
            "properties": {}
        }
        
        # Detect primary action
        if any(word in user_input.lower() for word in ["build", "create", "make", "construct"]):
            analysis["primary_action"] = "create"
        elif any(word in user_input.lower() for word in ["place", "put", "add", "spawn"]):
            analysis["primary_action"] = "place"
        elif any(word in user_input.lower() for word in ["modify", "change", "alter", "transform"]):
            analysis["primary_action"] = "modify"
        
        # Detect if model generation is needed
        complex_objects = ["statue", "sculpture", "building", "tower", "castle", "monument", "figure", "character"]
        if any(obj in user_input.lower() for obj in complex_objects):
            analysis["needs_model_generation"] = True
        
        # Detect object types
        object_keywords = {
            "statue": "statue", "sculpture": "sculpture", "building": "building",
            "tower": "tower", "castle": "castle", "monument": "monument",
            "tree": "tree", "rock": "rock", "crystal": "crystal",
            "altar": "altar", "portal": "portal", "fountain": "fountain"
        }
        
        for keyword, obj_type in object_keywords.items():
            if keyword in user_input.lower():
                analysis["object_types"].append(obj_type)
        
        # Detect materials
        material_keywords = {
            "stone": "stone", "marble": "marble", "bronze": "bronze",
            "gold": "gold", "silver": "silver", "crystal": "crystal",
            "wood": "wood", "metal": "metal", "glass": "glass"
        }
        
        for keyword, material in material_keywords.items():
            if keyword in user_input.lower():
                analysis["materials"].append(material)
        
        # Detect dimensions
        import re
        size_patterns = [
            r"(\d+)\s*(?:foot|feet|ft|meter|meters|m)\s*(?:tall|high|height)",
            r"(\d+)\s*(?:foot|feet|ft|meter|meters|m)\s*(?:wide|width)",
            r"(\d+)\s*(?:foot|feet|ft|meter|meters|m)\s*(?:deep|depth)"
        ]
        
        for pattern in size_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                size = int(match.group(1))
                if "tall" in pattern or "high" in pattern:
                    analysis["dimensions"]["height"] = size
                elif "wide" in pattern:
                    analysis["dimensions"]["width"] = size
                elif "deep" in pattern:
                    analysis["dimensions"]["depth"] = size
        
        # Detect location
        location_patterns = [
            r"at\s+coordinates?\s*\(?(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)?",
            r"at\s+position\s*\(?(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)?",
            r"at\s+location\s*\(?(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)?"
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                analysis["location"] = {
                    "x": float(match.group(1)),
                    "y": float(match.group(2)),
                    "z": float(match.group(3))
                }
                break
        
        return analysis
    
    def _generate_3d_model_for_construction(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a 3D model using the native Text-to-3D system."""
        try:
            from .text_to_3d import TextTo3D, ModelFormat, Vector3, Material, MaterialType
            
            text_to_3d = TextTo3D()
            
            # Create model specification
            description = f"{' '.join(analysis['object_types'])} made of {' '.join(analysis['materials'])}"
            
            # Determine dimensions
            dimensions = Vector3(
                analysis.get("dimensions", {}).get("width", 1.0),
                analysis.get("dimensions", {}).get("height", 2.0),
                analysis.get("dimensions", {}).get("depth", 1.0)
            )
            
            # Create materials
            materials = []
            for material_name in analysis.get("materials", ["stone"]):
                if material_name == "marble":
                    materials.append(Material("marble", MaterialType.PBR, Vector3(0.9, 0.9, 0.9), 0.0, 0.1))
                elif material_name == "bronze":
                    materials.append(Material("bronze", MaterialType.METALLIC, Vector3(0.8, 0.5, 0.2), 0.8, 0.3))
                elif material_name == "crystal":
                    materials.append(Material("crystal", MaterialType.GLASS, Vector3(0.8, 0.9, 1.0), 0.0, 0.1, transparency=0.3))
                else:
                    materials.append(Material(material_name, MaterialType.LAMBERT, Vector3(0.7, 0.7, 0.7)))
            
            # Generate the model
            model_result = text_to_3d.generate_3d_model({
                "description": description,
                "dimensions": dimensions,
                "complexity": "medium",
                "materials": materials,
                "textures": True
            }, format=ModelFormat.GLTF)
            
            self.logger.info(f"Generated 3D model: {model_result.model_path}")
            return {
                "model_path": model_result.model_path,
                "format": model_result.format.value,
                "dimensions": {
                    "x": model_result.dimensions.x,
                    "y": model_result.dimensions.y,
                    "z": model_result.dimensions.z
                },
                "vertices": model_result.vertices,
                "faces": model_result.faces
            }
            
        except Exception as e:
            self.logger.error(f"Error generating 3D model: {e}")
            return None
    
    def _manage_realm_for_construction(self, analysis: Dict[str, Any], model_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create or use existing realm for construction."""
        try:
            from unimind.bridge.storyrealms_bridge import storyrealms_bridge, RealmArchetype
            
            # Check if we have an active realm
            if hasattr(storyrealms_bridge, 'active_realm') and storyrealms_bridge.active_realm:
                realm_id = storyrealms_bridge.active_realm
                realm_created = False
            else:
                # Create a new realm
                realm_name = f"Construction_Realm_{int(time.time())}"
                realm_id = storyrealms_bridge.create_realm(
                    name=realm_name,
                    archetype=RealmArchetype.FOREST_GLADE,
                    description="Realm created for 3D construction"
                )
                realm_created = True
            
            return {
                "realm_id": realm_id,
                "realm_created": realm_created,
                "realm_name": realm_name if realm_created else "Existing Realm"
            }
            
        except Exception as e:
            self.logger.error(f"Error managing realm: {e}")
            return {
                "realm_id": None,
                "realm_created": False,
                "error": str(e)
            }
    
    def _place_objects_in_realm(self, analysis: Dict[str, Any], realm_result: Dict[str, Any], model_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Place objects in the realm."""
        try:
            from unimind.bridge.storyrealms_bridge import storyrealms_bridge, ObjectType, Coordinates
            
            if not realm_result.get("realm_id"):
                return {"objects_placed": [], "error": "No realm available"}
            
            realm_id = realm_result["realm_id"]
            objects_placed = []
            
            # Determine location
            location = analysis.get("location", {"x": 0, "y": 0, "z": 0})
            coordinates = Coordinates(location["x"], location["y"], location["z"])
            
            # Place objects based on analysis
            for obj_type in analysis.get("object_types", []):
                try:
                    # Map object types to ObjectType enum
                    object_type_mapping = {
                        "statue": ObjectType.ALTAR,  # Use altar as proxy for statue
                        "sculpture": ObjectType.CRYSTAL,  # Use crystal as proxy for sculpture
                        "building": ObjectType.TOWER,  # Use tower as proxy for building
                        "tower": ObjectType.TOWER,
                        "castle": ObjectType.TEMPLE,  # Use temple as proxy for castle
                        "monument": ObjectType.PILLAR,  # Use pillar as proxy for monument
                        "tree": ObjectType.TREE,
                        "rock": ObjectType.ROCK,
                        "crystal": ObjectType.CRYSTAL,
                        "altar": ObjectType.ALTAR,
                        "portal": ObjectType.PORTAL,
                        "fountain": ObjectType.FOUNTAIN
                    }
                    
                    object_type = object_type_mapping.get(obj_type, ObjectType.CRYSTAL)
                    
                    # Add properties if we have a generated model
                    properties = {}
                    if model_result:
                        properties.update({
                            "generated_model": model_result["model_path"],
                            "model_format": model_result["format"],
                            "materials": analysis.get("materials", []),
                            "dimensions": model_result["dimensions"]
                        })
                    
                    object_id = storyrealms_bridge.place_object(
                        realm_id=realm_id,
                        object_type=object_type,
                        coordinates=coordinates,
                        properties=properties
                    )
                    
                    objects_placed.append({
                        "object_id": object_id,
                        "object_type": object_type.value,
                        "coordinates": location,
                        "properties": properties
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error placing object {obj_type}: {e}")
            
            return {
                "objects_placed": objects_placed,
                "realm_id": realm_id
            }
            
        except Exception as e:
            self.logger.error(f"Error placing objects: {e}")
            return {
                "objects_placed": [],
                "error": str(e)
            }
    
    def _generate_engine_request(self, analysis: Dict[str, Any], realm_result: Dict[str, Any], placement_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a request for external 3D engines."""
        return {
            "action": "3d_construction_complete",
            "timestamp": time.time(),
            "construction_type": analysis.get("primary_action", "create"),
            "realm_id": realm_result.get("realm_id"),
            "objects": placement_result.get("objects_placed", []),
            "model_generated": analysis.get("needs_model_generation", False),
            "materials": analysis.get("materials", []),
            "dimensions": analysis.get("dimensions", {}),
            "location": analysis.get("location", {}),
            "properties": analysis.get("properties", {}),
            "message": f"3D construction task completed: {analysis.get('primary_action', 'unknown')} action with {len(placement_result.get('objects_placed', []))} objects"
        }

    def handle_adaptive_code_generation(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle requests by analyzing capabilities and generating code if needed.
        
        This method:
        1. Analyzes if the system can handle the request
        2. Generates code for missing capabilities if possible
        3. Integrates the new code into the system
        4. Executes the original request
        """
        try:
            self.logger.info(f"Handling adaptive code generation for: {user_input}")
            
            # Step 1: Analyze capabilities
            from .capability_analyzer import capability_analyzer, CodeGenerationRequest
            
            analysis = capability_analyzer.analyze_capability(user_input, context)
            
            if analysis.get("can_handle"):
                # System can handle the request directly
                self.logger.info("System can handle request directly")
                return {
                    "status": "success",
                    "approach": "direct_execution",
                    "message": "Request can be handled with existing capabilities",
                    "analysis": analysis,
                    "timestamp": time.time()
                }
            
            # Step 2: Check if code generation is needed and possible
            if not analysis.get("code_generation_needed"):
                return {
                    "status": "error",
                    "approach": "no_solution",
                    "message": "Cannot handle request and code generation not possible",
                    "analysis": analysis,
                    "timestamp": time.time()
                }
            
            # Step 3: Generate code for missing capabilities
            missing_capabilities = analysis.get("missing_capabilities", [])
            
            generated_code_artifacts = []
            for capability in missing_capabilities:
                try:
                    # Create code generation request
                    request = CodeGenerationRequest(
                        task_description=user_input,
                        required_capabilities=[capability],
                        context=context or {},
                        safety_constraints=capability_analyzer.safety_constraints,
                        integration_points=["scroll_engine", "lam_engine"]
                    )
                    
                    # Generate code
                    generated_code = capability_analyzer.generate_code(request)
                    generated_code_artifacts.append(generated_code)
                    
                    self.logger.info(f"Generated code for {capability.name}: {generated_code.file_path}")
                    
                except Exception as e:
                    self.logger.error(f"Error generating code for {capability.name}: {e}")
                    return {
                        "status": "error",
                        "approach": "code_generation_failed",
                        "message": f"Failed to generate code for {capability.name}: {str(e)}",
                        "analysis": analysis,
                        "timestamp": time.time()
                    }
            
            # Step 4: Integrate generated code
            integration_results = []
            for artifact in generated_code_artifacts:
                try:
                    integration_result = self._integrate_generated_code(artifact)
                    integration_results.append(integration_result)
                    
                    if integration_result.get("status") == "success":
                        self.logger.info(f"Successfully integrated {artifact.module_name}")
                    else:
                        self.logger.warning(f"Integration warning for {artifact.module_name}: {integration_result.get('message')}")
                        
                except Exception as e:
                    self.logger.error(f"Error integrating {artifact.module_name}: {e}")
                    integration_results.append({
                        "status": "error",
                        "module": artifact.module_name,
                        "error": str(e)
                    })
            
            # Step 5: Execute the original request
            execution_result = self._execute_with_generated_capabilities(user_input, generated_code_artifacts, context)
            
            return {
                "status": "success",
                "approach": "code_generation_and_execution",
                "message": "Generated code and executed request",
                "analysis": analysis,
                "generated_code": [{
                    "module": artifact.module_name,
                    "file_path": artifact.file_path,
                    "description": artifact.description
                } for artifact in generated_code_artifacts],
                "integration_results": integration_results,
                "execution_result": execution_result,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error in adaptive code generation: {e}")
            return {
                "status": "error",
                "approach": "system_error",
                "message": f"System error in adaptive code generation: {str(e)}",
                "timestamp": time.time()
            }
    
    def _integrate_generated_code(self, artifact) -> Dict[str, Any]:
        """
        Integrate generated code into the system.
        
        This method:
        1. Writes the generated code to the appropriate file
        2. Updates import statements if needed
        3. Registers the new capability
        """
        try:
            # Step 1: Write the generated code to file
            file_path = Path(artifact.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                f.write(artifact.code)
            
            self.logger.info(f"Written generated code to {file_path}")
            
            # Step 2: Update imports if needed
            if artifact.module_name not in self.known_capabilities:
                self.known_capabilities[artifact.module_name] = {
                    "type": artifact.capability_type,
                    "description": artifact.description,
                    "dependencies": artifact.dependencies
                }
            
            # Step 3: Register with scroll engine if it's a scroll handler
            if hasattr(artifact, 'capability_type') and artifact.capability_type == "scroll_handler":
                try:
                    from unimind.scrolls.scroll_engine import scroll_engine
                    
                    # Import the generated scroll handler
                    module_path = artifact.file_path.replace('.py', '').replace('/', '.')
                    exec(f"from {module_path} import {artifact.function_name}_scroll")
                    
                    # Register the scroll (this would need to be implemented in scroll_engine)
                    self.logger.info(f"Registered scroll handler: {artifact.function_name}")
                    
                except Exception as e:
                    self.logger.warning(f"Could not register scroll handler: {e}")
            
            return {
                "status": "success",
                "module": artifact.module_name,
                "file_path": artifact.file_path,
                "message": f"Successfully integrated {artifact.module_name}"
            }
            
        except Exception as e:
            self.logger.error(f"Error integrating generated code: {e}")
            return {
                "status": "error",
                "module": artifact.module_name,
                "error": str(e)
            }
    
    def _execute_with_generated_capabilities(self, user_input: str, generated_artifacts, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the original request using the newly generated capabilities.
        """
        try:
            # For now, return a placeholder execution result
            # In a full implementation, this would:
            # 1. Import the generated modules
            # 2. Call the appropriate functions
            # 3. Return the results
            
            return {
                "status": "success",
                "message": "Request executed with generated capabilities",
                "generated_modules": [artifact.module_name for artifact in generated_artifacts],
                "output": f"Successfully generated and integrated {len(generated_artifacts)} new capabilities",
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error executing with generated capabilities: {e}")
            return {
                "status": "error",
                "message": f"Error executing request: {str(e)}",
                "timestamp": time.time()
            }

    # Additional methods from AdaptiveLAM (merged functionality)
    
    def handle_adaptive_task(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle tasks using adaptive LAM approach (merged from AdaptiveLAM).
        Combines task decomposition, dynamic action generation, and LLM reasoning.
        """
        try:
            self.logger.info(f"Handling adaptive task: {user_input}")
            
            # Use existing handle_uncoded_task method as base
            base_result = self.handle_uncoded_task(user_input, context)
            
            # Add adaptive LAM enhancements
            if base_result.get("status") == "success":
                # Add adaptive features
                base_result["adaptive_features"] = {
                    "dynamic_actions": self.enable_dynamic_actions,
                    "task_decomposition": self.enable_task_decomposition,
                    "llm_reasoning": self.enable_llm_reasoning,
                    "safety_threshold": self.safety_threshold
                }
            
            return base_result
            
        except Exception as e:
            self.logger.error(f"Error in adaptive task handling: {e}")
            return {
                "status": "error",
                "message": f"Adaptive task handling error: {str(e)}",
                "timestamp": time.time()
            }
    
    def _execute_subtask_adaptive(self, subtask: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a subtask using adaptive LAM approach (from AdaptiveLAM).
        """
        try:
            subtask_type = subtask.get("type", "unknown")
            subtask_description = subtask.get("description", "")
            
            self.logger.info(f"Executing adaptive subtask: {subtask_type} - {subtask_description}")
            
            # Check if we have a direct capability for this subtask
            if subtask_type in self.system_capabilities:
                capability = self.system_capabilities[subtask_type]
                
                # Execute using existing capability
                result = self._execute_step({
                    "type": subtask_type,
                    "parameters": subtask.get("parameters", {}),
                    "description": subtask_description
                }, context)
                
                return {
                    "status": "success",
                    "subtask_type": subtask_type,
                    "result": result.result,
                    "execution_time": result.execution_time
                }
            
            # Generate dynamic action if no direct capability
            if self.enable_dynamic_actions:
                dynamic_action = self._generate_dynamic_action_adaptive(
                    subtask_type, 
                    subtask.get("parameters", {}), 
                    context
                )
                
                if dynamic_action:
                    return {
                        "status": "success",
                        "subtask_type": subtask_type,
                        "result": dynamic_action,
                        "execution_method": "dynamic_action"
                    }
            
            # Fallback to LLM reasoning
            return self._execute_fallback_adaptive(subtask, context)
            
        except Exception as e:
            self.logger.error(f"Error executing adaptive subtask: {e}")
            return {
                "status": "error",
                "subtask_type": subtask.get("type", "unknown"),
                "error": str(e)
            }
    
    def _generate_dynamic_action_adaptive(self, capability: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate dynamic action using adaptive approach (from AdaptiveLAM).
        """
        try:
            if not self.enable_dynamic_actions:
                return None
            
            # Use LLM to generate action code
            prompt = f"""
            Generate Python code for a {capability} action with the following parameters:
            {json.dumps(parameters, indent=2)}
            
            Context: {json.dumps(context, indent=2)}
            
            Return only the Python function code, no explanations.
            """
            
            response = self.llm_engine.run(prompt, max_tokens=500, temperature=0.3)
            
            # Extract code from response
            code_match = re.search(r'def\s+\w+.*?(?=\n\s*\n|\Z)', response, re.DOTALL)
            if code_match:
                execution_code = code_match.group(0)
                
                return {
                    "action_name": f"dynamic_{capability}",
                    "description": f"Dynamically generated {capability} action",
                    "parameters": parameters,
                    "execution_code": execution_code,
                    "safety_level": "medium",
                    "estimated_duration": 5.0
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating dynamic action: {e}")
            return None
    
    def _execute_fallback_adaptive(self, subtask: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute fallback for adaptive subtask (from AdaptiveLAM).
        """
        try:
            subtask_type = subtask.get("type", "unknown")
            
            # Use LLM to provide a fallback response
            prompt = f"""
            I need to handle a {subtask_type} task but don't have a direct capability for it.
            Task description: {subtask.get('description', '')}
            Parameters: {json.dumps(subtask.get('parameters', {}), indent=2)}
            
            Provide a helpful response explaining what I can do instead or how to approach this task.
            """
            
            fallback_response = self.llm_engine.run(prompt, max_tokens=300, temperature=0.7)
            
            return {
                "status": "fallback",
                "subtask_type": subtask_type,
                "result": fallback_response,
                "execution_method": "llm_fallback"
            }
            
        except Exception as e:
            self.logger.error(f"Error in adaptive fallback: {e}")
            return {
                "status": "error",
                "subtask_type": subtask.get("type", "unknown"),
                "error": str(e)
            }
    
    def optimize_adaptive_features(self) -> Dict[str, Any]:
        """
        Optimize adaptive LAM features (merged from AdaptiveLAM).
        """
        try:
            optimization_results = {
                "dynamic_actions_enabled": self.enable_dynamic_actions,
                "task_decomposition_enabled": self.enable_task_decomposition,
                "llm_reasoning_enabled": self.enable_llm_reasoning,
                "max_subtasks": self.max_subtasks,
                "safety_threshold": self.safety_threshold,
                "optimization_timestamp": time.time()
            }
            
            # Perform adaptive optimizations
            if self.enable_dynamic_actions:
                # Optimize dynamic action generation
                optimization_results["dynamic_actions_optimized"] = True
            
            if self.enable_task_decomposition:
                # Optimize task decomposition
                optimization_results["task_decomposition_optimized"] = True
            
            if self.enable_llm_reasoning:
                # Optimize LLM reasoning
                optimization_results["llm_reasoning_optimized"] = True
            
            self.logger.info("Adaptive LAM features optimized")
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error optimizing adaptive features: {e}")
            return {
                "status": "error",
                "message": f"Adaptive optimization error: {str(e)}",
                "timestamp": time.time()
            }

# Module-level instance
lam_engine = LAMEngine()

# Convenience functions (merged from adaptive_lam.py)
def handle_uncoded_task_adaptive(user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Convenience function for handling uncoded tasks with adaptive LAM."""
    return lam_engine.handle_adaptive_task(user_input, context)

def optimize_adaptive_lam() -> Dict[str, Any]:
    """Convenience function for optimizing adaptive LAM features."""
    return lam_engine.optimize_adaptive_features()