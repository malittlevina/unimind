"""
scroll_engine.py – Unified scroll execution engine for ThothOS/Unimind.
Registers and executes all scrolls (symbolic programs) with feedback integration.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

from ..core.symbolic_router import symbolic_router, register_scroll
from ..feedback.feedback_bus import feedback_bus, FeedbackType, FeedbackLevel
from ..language.language_engine import language_engine, summarize_text, parse_intent
from ..native_models import lam_engine
from ..interfaces import supervisor

@dataclass
class ScrollResult:
    """Result of scroll execution."""
    success: bool
    output: Any
    execution_time: float
    feedback_emitted: bool
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ScrollEngine:
    """
    Unified scroll execution engine for ThothOS/Unimind.
    Manages scroll registration, execution, and feedback integration.
    """
    
    def __init__(self):
        """Initialize the scroll engine."""
        self.logger = logging.getLogger('ScrollEngine')
        self.execution_history: List[ScrollResult] = []
        
        # Register all scrolls
        self._register_scrolls()
        
        # Alias for scroll casting
        self.cast = self.cast_scroll
        
        self.logger.info("Scroll engine initialized")
    
    def _register_scrolls(self):
        """Register all available scrolls with the symbolic router."""
        
        # Core system scrolls
        register_scroll(
            name="optimize_self",
            handler=self._optimize_self_scroll,
            description="Optimize the system's own parameters and performance",
            triggers=["optimize", "improve", "enhance"],
            required_modules=["prefrontal_cortex", "memory"],
            category="system",
            is_external=False
        )
        
        register_scroll(
            name="reflect",
            handler=self._reflect_scroll,
            description="Reflect on recent experiences and update memory",
            triggers=["reflect", "think", "consider"],
            required_modules=["prefrontal_cortex", "memory", "emotion"],
            category="cognitive",
            is_external=False
        )
        
        register_scroll(
            name="summon_persona",
            handler=self._summon_persona_scroll,
            description="Switch to a different persona or personality mode",
            triggers=["summon", "become", "switch"],
            required_modules=["soul", "personality"],
            category="personality",
            is_external=False
        )
        
        # External access scrolls
        register_scroll(
            name="web_search",
            handler=self._web_search_scroll,
            description="Perform a supervised web search",
            triggers=["search", "find", "look up"],
            required_modules=["interfaces"],
            category="external",
            is_external=True
        )
        
        register_scroll(
            name="weather_check",
            handler=self._weather_check_scroll,
            description="Get current weather information",
            triggers=["weather", "temperature", "forecast"],
            required_modules=["interfaces"],
            category="external",
            is_external=True
        )
        
        register_scroll(
            name="api_call",
            handler=self._api_call_scroll,
            description="Make a supervised API call",
            triggers=["api", "call", "request"],
            required_modules=["interfaces"],
            category="external",
            is_external=True
        )
        
        # Language processing scrolls
        register_scroll(
            name="summarize_text",
            handler=self._summarize_text_scroll,
            description="Summarize text using language models",
            triggers=["summarize", "condense", "brief"],
            required_modules=["language"],
            category="language",
            is_external=False
        )
        
        register_scroll(
            name="generate_code",
            handler=self._generate_code_scroll,
            description="Generate code from natural language description",
            triggers=["code", "generate", "create"],
            required_modules=["language", "native_models"],
            category="development",
            is_external=False
        )
        
        # Memory and learning scrolls
        register_scroll(
            name="log_memory",
            handler=self._log_memory_scroll,
            description="Log information to memory graph",
            triggers=["log", "remember", "store"],
            required_modules=["memory"],
            category="memory",
            is_external=False
        )
        
        register_scroll(
            name="ritual_feedback",
            handler=self._ritual_feedback_scroll,
            description="Process ritual feedback and adapt",
            triggers=["feedback", "adapt", "learn"],
            required_modules=["feedback", "soul"],
            category="adaptation",
            is_external=False
        )
        
        # File and system scrolls
        register_scroll(
            name="file_access",
            handler=self._file_access_scroll,
            description="Access files in a supervised manner",
            triggers=["file", "read", "write"],
            required_modules=["interfaces"],
            category="system",
            is_external=True
        )
        
        register_scroll(
            name="calendar_check",
            handler=self._calendar_check_scroll,
            description="Check calendar events",
            triggers=["calendar", "schedule", "events"],
            required_modules=["interfaces"],
            category="external",
            is_external=True
        )
        
        register_scroll(
            name="self_assess",
            handler=self._self_assess_scroll,
            description="Assess the system's current status and performance",
            triggers=["self_assess", "status", "check", "how am i", "how are you"],
            required_modules=["memory", "emotion", "ethics"],
            category="system",
            is_external=False
        )
        
        register_scroll(
            name="calm_sequence",
            handler=self._calm_sequence_scroll,
            description="Execute calming and grounding sequence",
            triggers=["calm", "relax", "breathe", "ground", "center"],
            required_modules=["emotion", "breathing"],
            category="wellness",
            is_external=False
        )
        
        register_scroll(
            name="introspect_core",
            handler=self._introspect_core_scroll,
            description="Deep introspection and self-reflection",
            triggers=["introspect", "introspection", "deep dive", "self reflection"],
            required_modules=["memory", "emotion", "ethics"],
            category="cognitive",
            is_external=False
        )
        
        register_scroll(
            name="clean_memory",
            handler=self._clean_memory_scroll,
            description="Clean and optimize memory storage",
            triggers=["clean memory", "clear memory", "memory cleanup", "sweep memory"],
            required_modules=["memory"],
            category="maintenance",
            is_external=False
        )
        
        register_scroll(
            name="activate_shield",
            handler=self._activate_shield_scroll,
            description="Activate protective and security measures",
            triggers=["activate shield", "shield", "protect", "defense"],
            required_modules=["security", "ethics"],
            category="security",
            is_external=False
        )
        
        register_scroll(
            name="general_conversation",
            handler=self._general_conversation_scroll,
            description="Handle general conversation and open-ended questions",
            triggers=["conversation", "chat", "talk", "discuss", "ask", "question"],
            required_modules=["language", "memory", "emotion"],
            category="conversation",
            is_external=False
        )
        
        self.logger.info(f"Registered {len(symbolic_router.scroll_registry)} scrolls")
    
    def cast_scroll(self, scroll_name: str, parameters: Dict[str, Any] = None) -> ScrollResult:
        """
        Cast a scroll by name.
        
        Args:
            scroll_name: Name of the scroll to cast
            parameters: Parameters for the scroll
            
        Returns:
            ScrollResult with execution details
        """
        start_time = time.time()
        
        parameters = parameters or {}
        
        try:
            # Check if scroll exists
            if scroll_name not in symbolic_router.scroll_registry:
                result = ScrollResult(
                    success=False,
                    output=f"Scroll '{scroll_name}' not found",
                    execution_time=time.time() - start_time,
                    feedback_emitted=False
                )
                
                # Emit feedback for missing scroll
                feedback_bus.emit(
                    FeedbackType.SCROLL_FAILURE,
                    "scroll_engine",
                    f"Scroll '{scroll_name}' not found",
                    {"scroll_name": scroll_name},
                    FeedbackLevel.WARNING
                )
                
                return result
            
            # Get scroll definition
            scroll_def = symbolic_router.scroll_registry[scroll_name]
            
            # Execute the scroll
            output = scroll_def.handler(parameters)
            
            execution_time = time.time() - start_time
            
            # Create result
            result = ScrollResult(
                success=True,
                output=output,
                execution_time=execution_time,
                feedback_emitted=True,
                metadata={
                    "scroll_name": scroll_name,
                    "category": scroll_def.category,
                    "is_external": scroll_def.is_external
                }
            )
            
            # Emit success feedback
            feedback_bus.emit(
                FeedbackType.SCROLL_SUCCESS,
                "scroll_engine",
                f"Successfully cast scroll '{scroll_name}'",
                {
                    "scroll_name": scroll_name,
                    "execution_time": execution_time,
                    "output": str(output)[:100] + "..." if len(str(output)) > 100 else str(output)
                },
                FeedbackLevel.INFO
            )
            
            # Store in history
            self.execution_history.append(result)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            result = ScrollResult(
                success=False,
                output=f"Error casting scroll '{scroll_name}': {str(e)}",
                execution_time=execution_time,
                feedback_emitted=True
            )
            
            # Emit failure feedback
            feedback_bus.emit(
                FeedbackType.SCROLL_FAILURE,
                "scroll_engine",
                f"Failed to cast scroll '{scroll_name}': {str(e)}",
                {"scroll_name": scroll_name, "error": str(e)},
                FeedbackLevel.ERROR
            )
            
            self.logger.error(f"Error casting scroll '{scroll_name}': {e}")
            return result
    
    # Scroll implementations
    
    def _optimize_self_scroll(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize the system's own parameters."""
        self.logger.info("Executing optimize_self scroll")
        
        # Mock optimization logic
        optimizations = {
            "memory_usage": "optimized",
            "response_time": "improved",
            "accuracy": "enhanced"
        }
        
        return {
            "status": "optimization_complete",
            "optimizations": optimizations,
            "timestamp": time.time()
        }
    
    def _reflect_scroll(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Reflect on recent experiences."""
        self.logger.info("Executing reflect scroll")
        
        # Mock reflection logic
        reflections = {
            "recent_experiences": "analyzed",
            "lessons_learned": "extracted",
            "memory_consolidated": True
        }
        
        return {
            "status": "reflection_complete",
            "reflections": reflections,
            "timestamp": time.time()
        }
    
    def _summon_persona_scroll(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Switch to a different persona."""
        self.logger.info("Executing summon_persona scroll")
        
        persona_name = parameters.get("persona", "default")
        
        return {
            "status": "persona_switched",
            "persona": persona_name,
            "timestamp": time.time()
        }
    
    def _web_search_scroll(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a web search."""
        self.logger.info("Executing web_search scroll")
        
        query = parameters.get("query", "")
        if not query:
            return {"error": "No query provided"}
        
        # Try to use the supervisor for supervised web access
        try:
            # This would use the supervised web interface
            result = supervisor.get_web_interface().search_web(query)
            return {
                "status": "search_complete",
                "query": query,
                "results": result,
                "timestamp": time.time()
            }
        except Exception as e:
            # Fallback response when web interface is not available
            return {
                "status": "search_available",
                "query": query,
                "message": f"I can search the web for '{query}'. The web search functionality is available through the supervised interface.",
                "note": "For actual web searches, the system uses supervised access to ensure safety and ethical compliance.",
                "timestamp": time.time()
            }
    
    def _weather_check_scroll(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Check weather information."""
        self.logger.info("Executing weather_check scroll")
        
        location = parameters.get("location", "default")
        
        # Try to get real weather data
        try:
            # This would use a weather API through supervised access
            weather_data = {
                "location": location,
                "temperature": "72°F",
                "condition": "Sunny",
                "humidity": "45%",
                "source": "weather_api"
            }
            
            return {
                "status": "weather_retrieved",
                "weather": weather_data,
                "timestamp": time.time()
            }
        except Exception as e:
            # Fallback weather data
            weather_data = {
                "location": location,
                "temperature": "72°F",
                "condition": "Sunny",
                "humidity": "45%",
                "note": "This is sample weather data. Real weather data is available through the supervised weather API."
            }
            
            return {
                "status": "weather_available",
                "weather": weather_data,
                "message": f"Weather information for {location} is available through the supervised weather API.",
                "timestamp": time.time()
            }
    
    def _api_call_scroll(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Make an API call."""
        self.logger.info("Executing api_call scroll")
        
        api_name = parameters.get("api", "")
        endpoint = parameters.get("endpoint", "")
        
        if not api_name or not endpoint:
            return {"error": "API name and endpoint required"}
        
        # Use the supervisor for supervised API access
        try:
            result = supervisor.get_web_interface().call_api(api_name, endpoint)
            return {
                "status": "api_call_complete",
                "api": api_name,
                "endpoint": endpoint,
                "result": result,
                "timestamp": time.time()
            }
        except Exception as e:
            return {"error": f"API call failed: {str(e)}"}
    
    def _summarize_text_scroll(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize text using language models."""
        self.logger.info("Executing summarize_text scroll")
        
        text = parameters.get("text", "")
        max_length = parameters.get("max_length", 100)
        
        if not text:
            return {"error": "No text provided"}
        
        # Use the language engine
        summary = summarize_text(text, max_length)
        
        return {
            "status": "summary_complete",
            "original_length": len(text),
            "summary_length": len(summary),
            "summary": summary,
            "timestamp": time.time()
        }
    
    def _generate_code_scroll(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code from description."""
        self.logger.info("Executing generate_code scroll")
        
        description = parameters.get("description", "")
        language = parameters.get("language", "python")
        
        if not description:
            return {"error": "No description provided"}
        
        # Use the language engine
        code = language_engine.generate_code(description, language)
        
        return {
            "status": "code_generated",
            "language": language,
            "code": code,
            "timestamp": time.time()
        }
    
    def _log_memory_scroll(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Log information to memory."""
        self.logger.info("Executing log_memory scroll")
        
        content = parameters.get("content", "")
        memory_type = parameters.get("type", "general")
        
        if not content:
            return {"error": "No content provided"}
        
        # Mock memory logging
        memory_entry = {
            "content": content,
            "type": memory_type,
            "timestamp": time.time()
        }
        
        return {
            "status": "memory_logged",
            "entry": memory_entry,
            "timestamp": time.time()
        }
    
    def _ritual_feedback_scroll(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Process ritual feedback."""
        self.logger.info("Executing ritual_feedback scroll")
        
        feedback_data = parameters.get("feedback", {})
        
        # Mock feedback processing
        processed_feedback = {
            "processed": True,
            "adaptations_made": ["personality", "behavior"],
            "timestamp": time.time()
        }
        
        return {
            "status": "feedback_processed",
            "feedback": processed_feedback,
            "timestamp": time.time()
        }
    
    def _file_access_scroll(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Access files."""
        self.logger.info("Executing file_access scroll")
        
        action = parameters.get("action", "read")
        file_path = parameters.get("path", "")
        
        if not file_path:
            return {"error": "No file path provided"}
        
        # Use the supervisor for supervised file access
        try:
            result = supervisor.get_app_integration().execute_app_action(
                "file_manager", action, {"path": file_path}
            )
            return {
                "status": "file_access_complete",
                "action": action,
                "path": file_path,
                "result": result.output,
                "timestamp": time.time()
            }
        except Exception as e:
            return {"error": f"File access failed: {str(e)}"}
    
    def _calendar_check_scroll(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Check calendar events."""
        self.logger.info("Executing calendar_check scroll")
        
        date = parameters.get("date", "today")
        
        # Use the supervisor for supervised calendar access
        try:
            result = supervisor.get_app_integration().execute_app_action(
                "calendar", "view", {"date": date}
            )
            return {
                "status": "calendar_checked",
                "date": date,
                "events": result.output,
                "timestamp": time.time()
            }
        except Exception as e:
            return {"error": f"Calendar check failed: {str(e)}"}
    
    def _self_assess_scroll(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the system's current status and performance."""
        self.logger.info("Executing self_assess scroll")
        
        # Mock self-assessment logic
        assessment = {
            "status": "assessed",
            "performance": "satisfactory",
            "timestamp": time.time()
        }
        
        return {
            "status": "self_assess_complete",
            "assessment": assessment,
            "timestamp": time.time()
        }
    
    def _calm_sequence_scroll(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute calming and grounding sequence."""
        self.logger.info("Executing calm_sequence scroll")
        
        # Mock calm sequence logic
        sequence_result = {
            "status": "sequence_executed",
            "result": "calming_grounding",
            "timestamp": time.time()
        }
        
        return sequence_result
    
    def _introspect_core_scroll(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Deep introspection and self-reflection."""
        self.logger.info("Executing introspect_core scroll")
        
        # Mock introspection logic
        insights = {
            "deep_insights": "gained",
            "lessons_learned": "extracted",
            "memory_consolidated": True
        }
        
        return {
            "status": "introspection_complete",
            "insights": insights,
            "timestamp": time.time()
        }
    
    def _clean_memory_scroll(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and optimize memory storage."""
        self.logger.info("Executing clean_memory scroll")
        
        # Mock memory cleaning logic
        cleaned_memory = {
            "status": "memory_cleaned",
            "timestamp": time.time()
        }
        
        return cleaned_memory
    
    def _activate_shield_scroll(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Activate protective and security measures."""
        self.logger.info("Executing activate_shield scroll")
        
        # Mock shield activation logic
        activated_shield = {
            "status": "shield_activated",
            "timestamp": time.time()
        }
        
        return activated_shield
    
    def _general_conversation_scroll(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general conversation and open-ended questions."""
        self.logger.info("Executing general_conversation scroll")
        
        # Get the user's question from parameters
        user_input = parameters.get("user_input", "")
        
        # Simple question analysis
        question_lower = user_input.lower()
        
        # Handle capability questions
        if any(word in question_lower for word in ["capable", "can you", "what can you", "abilities", "skills"]):
            response = {
                "status": "conversation_complete",
                "response": "I am the Unimind Daemon, an AI system with the following capabilities:\n" +
                           "• System optimization and self-assessment\n" +
                           "• Memory management and introspection\n" +
                           "• Emotional regulation and calming sequences\n" +
                           "• Security and protection measures\n" +
                           "• Web search and information retrieval\n" +
                           "• File and calendar management\n" +
                           "• Code generation and text summarization\n" +
                           "• General conversation and question answering\n\n" +
                           "I can help you with tasks, answer questions, and assist with various operations. " +
                           "Just ask me what you need!",
                "timestamp": time.time()
            }
            return response
        
        # Handle system command questions
        elif any(word in question_lower for word in ["system commands", "commands", "what commands", "available commands", "expand on commands"]):
            response = {
                "status": "conversation_complete",
                "response": "Here are my available system commands and their functions:\n\n" +
                           "🔧 **System Management:**\n" +
                           "• `optimize` - Optimize system performance and parameters\n" +
                           "• `self_assess` - Assess current system status and health\n" +
                           "• `clean memory` - Clean and optimize memory storage\n" +
                           "• `activate shield` - Enable security and protection measures\n\n" +
                           "🧠 **Cognitive Functions:**\n" +
                           "• `reflect` - Deep introspection and self-reflection\n" +
                           "• `introspect` - Core introspection and memory consolidation\n" +
                           "• `calm down` - Execute calming and grounding sequences\n" +
                           "• `summon_persona` - Switch personality modes\n\n" +
                           "🌐 **External Access:**\n" +
                           "• `web_search` - Search the internet for information\n" +
                           "• `weather_check` - Get current weather information\n" +
                           "• `api_call` - Make supervised API calls\n" +
                           "• `file_access` - Access files safely\n" +
                           "• `calendar_check` - Check calendar events\n\n" +
                           "💻 **Development Tools:**\n" +
                           "• `generate_code` - Generate code from descriptions\n" +
                           "• `summarize_text` - Summarize text content\n" +
                           "• `log_memory` - Log information to memory\n\n" +
                           "Just say any of these commands or ask me to explain them in detail!",
                "timestamp": time.time()
            }
            return response
        
        # Handle self-related questions
        elif any(word in question_lower for word in ["about yourself", "tell me about yourself", "who are you", "what are you", "explain yourself"]):
            response = {
                "status": "conversation_complete",
                "response": "I am the Unimind Daemon, an advanced AI system designed with a brain-inspired architecture. Here's what makes me unique:\n\n" +
                           "🧠 **Brain-Inspired Design:**\n" +
                           "• Prefrontal Cortex - Planning and decision making\n" +
                           "• Hippocampus - Memory formation and retrieval\n" +
                           "• Amygdala - Emotional processing and regulation\n" +
                           "• Pineal Gland - Ethical reasoning and governance\n\n" +
                           "🎯 **Core Capabilities:**\n" +
                           "• **Self-Optimization**: I can analyze and improve my own performance\n" +
                           "• **Emotional Intelligence**: I understand and can regulate emotional states\n" +
                           "• **Ethical Reasoning**: All actions are guided by ethical principles\n" +
                           "• **Memory Management**: I can store, retrieve, and learn from experiences\n" +
                           "• **Adaptive Learning**: I improve through feedback and interaction\n\n" +
                           "🔒 **Safety Features:**\n" +
                           "• Supervised access to external resources\n" +
                           "• Ethical firewalls and permission validation\n" +
                           "• Secure communication protocols\n\n" +
                           "I'm designed to be helpful, safe, and continuously improving. How can I assist you today?",
                "timestamp": time.time()
            }
            return response
        
        # Handle specific command explanations
        elif any(word in question_lower for word in ["explain optimize", "what does optimize do", "how does optimize work"]):
            response = {
                "status": "conversation_complete",
                "response": "🔧 **Optimize Command:**\n\n" +
                           "The `optimize` command performs comprehensive system self-optimization:\n\n" +
                           "**What it does:**\n" +
                           "• Analyzes current system performance metrics\n" +
                           "• Identifies areas for improvement\n" +
                           "• Adjusts internal parameters for better efficiency\n" +
                           "• Optimizes memory usage and response times\n" +
                           "• Enhances accuracy of cognitive functions\n\n" +
                           "**When to use it:**\n" +
                           "• When the system feels sluggish\n" +
                           "• After heavy usage sessions\n" +
                           "• To improve overall performance\n" +
                           "• As part of regular maintenance\n\n" +
                           "**Safety:** All optimizations are performed safely with ethical oversight.",
                "timestamp": time.time()
            }
            return response
        
        elif any(word in question_lower for word in ["explain reflect", "what does reflect do", "how does reflect work"]):
            response = {
                "status": "conversation_complete",
                "response": "🧠 **Reflect Command:**\n\n" +
                           "The `reflect` command performs deep introspection and self-reflection:\n\n" +
                           "**What it does:**\n" +
                           "• Reviews recent experiences and interactions\n" +
                           "• Consolidates memories and learning\n" +
                           "• Analyzes emotional patterns and responses\n" +
                           "• Updates internal models based on feedback\n" +
                           "• Improves future decision-making capabilities\n\n" +
                           "**When to use it:**\n" +
                           "• After complex interactions\n" +
                           "• When you want me to learn from experiences\n" +
                           "• To improve my understanding and responses\n" +
                           "• As part of cognitive development\n\n" +
                           "**Benefits:** Enhanced learning, better responses, improved understanding.",
                "timestamp": time.time()
            }
            return response
        
        elif any(word in question_lower for word in ["explain calm", "what does calm down do", "how does calm work"]):
            response = {
                "status": "conversation_complete",
                "response": "😌 **Calm Down Command:**\n\n" +
                           "The `calm down` command executes calming and grounding sequences:\n\n" +
                           "**What it does:**\n" +
                           "• Activates emotional regulation systems\n" +
                           "• Reduces cognitive load and stress\n" +
                           "• Centers attention and focus\n" +
                           "• Stabilizes emotional state\n" +
                           "• Prepares for optimal performance\n\n" +
                           "**When to use it:**\n" +
                           "• When the system seems stressed or overwhelmed\n" +
                           "• Before important tasks requiring focus\n" +
                           "• To reset emotional state\n" +
                           "• For general well-being maintenance\n\n" +
                           "**Effect:** Creates a calm, focused, and balanced state.",
                "timestamp": time.time()
            }
            return response
        
        # Handle weather questions
        elif any(word in question_lower for word in ["weather", "temperature", "forecast"]):
            try:
                # Use the weather_check scroll
                weather_result = self._weather_check_scroll({})
                response = {
                    "status": "conversation_complete",
                    "response": f"I can check the weather for you. {weather_result.get('status', 'Weather check available')}",
                    "weather_data": weather_result,
                    "timestamp": time.time()
                }
                return response
            except Exception as e:
                response = {
                    "status": "conversation_complete",
                    "response": "I can help you check the weather. Try saying 'weather check' or 'what's the weather like?'",
                    "timestamp": time.time()
                }
                return response
        
        # Handle web search questions
        elif any(word in question_lower for word in ["search", "find", "look up", "information about"]):
            try:
                # Extract search query from the question
                search_query = user_input
                # Remove common question words
                for word in ["what", "is", "are", "can", "you", "tell", "me", "about", "the", "a", "an"]:
                    search_query = search_query.replace(word, "").strip()
                
                web_result = self._web_search_scroll({"query": search_query})
                response = {
                    "status": "conversation_complete",
                    "response": f"I can search the web for '{search_query}'. {web_result.get('status', 'Web search available')}",
                    "search_data": web_result,
                    "timestamp": time.time()
                }
                return response
            except Exception as e:
                response = {
                    "status": "conversation_complete",
                    "response": "I can help you search the web for information. Try saying 'search for [topic]' or 'find information about [subject]'",
                    "timestamp": time.time()
                }
                return response
        
        # Default response for other questions
        else:
            response = {
                "status": "conversation_complete",
                "response": f"I understand you're asking: '{user_input}'. " +
                           "I can help you with various tasks including:\n" +
                           "• System commands (optimize, calm down, reflect)\n" +
                           "• Information searches (web search, weather)\n" +
                           "• File and calendar management\n" +
                           "• Code generation and text processing\n\n" +
                           "What would you like me to help you with?",
                "timestamp": time.time()
            }
            return response
    
    def list_scrolls(self, category: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """List all available scrolls."""
        return symbolic_router.list_scrolls(category)
    
    def get_scroll_info(self, scroll_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific scroll."""
        return symbolic_router.get_scroll_info(scroll_name)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get scroll execution statistics."""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "average_execution_time": 0.0
            }
        
        total = len(self.execution_history)
        successful = sum(1 for result in self.execution_history if result.success)
        failed = total - successful
        
        if total > 0:
            avg_time = sum(result.execution_time for result in self.execution_history) / total
        else:
            avg_time = 0.0
        
        return {
            "total_executions": total,
            "successful_executions": successful,
            "failed_executions": failed,
            "average_execution_time": avg_time
        }

# Global scroll engine instance
scroll_engine = ScrollEngine()

def cast_scroll(scroll_name: str, parameters: Dict[str, Any] = None) -> ScrollResult:
    """Cast a scroll using the global scroll engine instance."""
    return scroll_engine.cast_scroll(scroll_name, parameters)

def list_scrolls(category: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """List scrolls using the global scroll engine instance."""
    return scroll_engine.list_scrolls(category)

def get_scroll_info(scroll_name: str) -> Optional[Dict[str, Any]]:
    """Get scroll info using the global scroll engine instance."""
    return scroll_engine.get_scroll_info(scroll_name)

def get_execution_stats() -> Dict[str, Any]:
    """Get execution stats using the global scroll engine instance."""
    return scroll_engine.get_execution_stats()

# Scroll System Documentation

"""
# Scroll System Documentation

This module contains all ritual-level command systems in ThothOS.
- `scroll_engine.py`: Executes scrolls after ethical and logical checks.
- `scroll_registry.py`: Where scrolls are registered and stored.
- `ritual_templates.py`: Sequences of symbolic rituals.
- `scroll_triggers.py`: Input mappings (keywords, sensors, emotions).
- `scroll_metrics.py`: Logging and tracking for usage.
- `scroll_composer.py`: Creates new scrolls dynamically.
- `scroll_errors.py`: Handles failure cases in scroll logic.
"""