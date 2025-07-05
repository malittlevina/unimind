"""
unified_input_processor.py – Unified input processing pipeline for Unimind.
Implements the flow: LLM Engine (understanding) -> LAM Engine (routing) -> Native Models (execution).
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from .llm_engine import llm_engine, understand_context, classify_intent
from .lam_engine import lam_engine
from .model_registry import model_registry, execute_request, ModelCategory

class ProcessingStage(Enum):
    """Stages of the unified input processing pipeline."""
    LLM_UNDERSTANDING = "llm_understanding"
    LAM_ROUTING = "lam_routing"
    MODEL_EXECUTION = "model_execution"
    RESULT_INTEGRATION = "result_integration"

@dataclass
class ProcessingResult:
    """Result of unified input processing."""
    original_input: str
    llm_understanding: Optional[Dict[str, Any]] = None
    lam_routing: Optional[Dict[str, Any]] = None
    model_execution: Optional[Dict[str, Any]] = None
    final_result: Any = None
    processing_stages: List[ProcessingStage] = field(default_factory=list)
    execution_time: float = 0.0
    success: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class UnifiedInputProcessor:
    """
    Unified input processor that implements the complete flow:
    1. LLM Engine (understanding, context analysis, reasoning)
    2. LAM Engine (action mapping, routing)
    3. Native Models (execution)
    """
    
    def __init__(self):
        """Initialize the unified input processor."""
        self.logger = logging.getLogger('UnifiedInputProcessor')
        
        # Core engines
        self.llm_engine = llm_engine
        self.lam_engine = lam_engine
        self.model_registry = model_registry
        
        # Processing configuration
        self.enable_llm_understanding = True
        self.enable_lam_routing = True
        self.enable_model_execution = True
        self.enable_fallback = True
        
        # Performance tracking
        self.processing_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_processing_time": 0.0,
            "stage_times": {}
        }
        
        self.logger.info("Unified input processor initialized")
    
    def process_input(self, user_input: str, context: Dict[str, Any] = None, 
                     memory_context_id: str = None) -> ProcessingResult:
        """
        Process user input through the complete pipeline.
        
        Args:
            user_input: The user's input text
            context: Additional context information
            memory_context_id: Memory context identifier
            
        Returns:
            ProcessingResult with complete pipeline results
        """
        start_time = time.time()
        result = ProcessingResult(original_input=user_input)
        
        try:
            self.logger.info(f"Processing input: {user_input[:100]}...")
            
            # Stage 1: LLM Understanding
            if self.enable_llm_understanding:
                result.processing_stages.append(ProcessingStage.LLM_UNDERSTANDING)
                result.llm_understanding = self._process_llm_understanding(user_input, context, memory_context_id)
            
            # Stage 2: LAM Routing
            if self.enable_lam_routing:
                result.processing_stages.append(ProcessingStage.LAM_ROUTING)
                result.lam_routing = self._process_lam_routing(user_input, result.llm_understanding, context)
            
            # Stage 3: Model Execution
            if self.enable_model_execution and result.lam_routing:
                result.processing_stages.append(ProcessingStage.MODEL_EXECUTION)
                result.model_execution = self._process_model_execution(user_input, result.lam_routing, context)
            
            # Stage 4: Result Integration
            result.processing_stages.append(ProcessingStage.RESULT_INTEGRATION)
            result.final_result = self._integrate_results(result)
            
            result.success = True
            result.execution_time = time.time() - start_time
            
            # Update statistics
            self._update_stats(result)
            
            self.logger.info(f"Processing completed successfully in {result.execution_time:.2f}s")
            
        except Exception as e:
            result.error = str(e)
            result.execution_time = time.time() - start_time
            self.logger.error(f"Processing failed: {e}")
            
            # Fallback processing if enabled
            if self.enable_fallback:
                result.final_result = self._fallback_processing(user_input, context)
        
        return result
    
    def _process_llm_understanding(self, user_input: str, context: Dict[str, Any], 
                                 memory_context_id: str) -> Dict[str, Any]:
        """Process input through LLM engine for understanding."""
        understanding_result = {}
        
        try:
            # Get contextual understanding
            understanding = understand_context(user_input, memory_context_id)
            if understanding:
                understanding_result["context_understanding"] = {
                    "primary_intent": understanding.primary_intent.value,
                    "confidence": understanding.confidence,
                    "suggested_actions": understanding.suggested_actions,
                    "user_goal": understanding.user_goal
                }
            
            # Get intent classification
            intent_result = classify_intent(user_input)
            if intent_result:
                understanding_result["intent_classification"] = {
                    "intent_type": intent_result.get("intent_type"),
                    "confidence": intent_result.get("confidence"),
                    "scroll_name": intent_result.get("scroll_name")
                }
            
            # Get enhanced language understanding
            language_understanding = self.llm_engine.understand_language(user_input)
            if language_understanding:
                understanding_result["language_understanding"] = {
                    "semantic_meaning": language_understanding.semantic_meaning,
                    "pragmatic_meaning": language_understanding.pragmatic_meaning,
                    "intent": language_understanding.intent,
                    "confidence": language_understanding.confidence
                }
            
            # Perform reasoning if needed
            if self._needs_reasoning(user_input):
                reasoning_result = self.llm_engine.chain_of_thought_reasoning(user_input)
                if reasoning_result:
                    understanding_result["reasoning"] = {
                        "final_answer": reasoning_result.final_answer,
                        "confidence": reasoning_result.confidence,
                        "steps_count": len(reasoning_result.steps)
                    }
            
        except Exception as e:
            self.logger.error(f"LLM understanding failed: {e}")
            understanding_result["error"] = str(e)
        
        return understanding_result
    
    def _process_lam_routing(self, user_input: str, llm_understanding: Dict[str, Any], 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through LAM engine for routing."""
        routing_result = {}
        
        try:
            # Use LAM engine to determine routing
            lam_result = self.lam_engine.route_with_adaptive_fallback(user_input, context)
            
            if lam_result:
                routing_result["lam_routing"] = lam_result
                
                # Extract action information
                if "action" in lam_result:
                    routing_result["action"] = lam_result["action"]
                
                if "scroll_name" in lam_result:
                    routing_result["scroll_name"] = lam_result["scroll_name"]
                
                if "plan" in lam_result:
                    routing_result["plan"] = lam_result["plan"]
            
            # Determine model routing based on understanding
            model_routing = self._determine_model_routing(user_input, llm_understanding)
            if model_routing:
                routing_result["model_routing"] = model_routing
            
        except Exception as e:
            self.logger.error(f"LAM routing failed: {e}")
            routing_result["error"] = str(e)
        
        return routing_result
    
    def _process_model_execution(self, user_input: str, lam_routing: Dict[str, Any], 
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the request using appropriate native models."""
        execution_result = {}
        
        try:
            # Determine which model to use
            model_routing = lam_routing.get("model_routing", {})
            request_type = model_routing.get("request_type", "text_generation")
            
            # Execute using model registry
            model_result = execute_request(request_type, user_input, **context or {})
            
            if model_result["success"]:
                execution_result["model_execution"] = model_result
                execution_result["result"] = model_result["result"]
            else:
                execution_result["error"] = model_result["error"]
            
        except Exception as e:
            self.logger.error(f"Model execution failed: {e}")
            execution_result["error"] = str(e)
        
        return execution_result
    
    def _integrate_results(self, result: ProcessingResult) -> Any:
        """Integrate results from all processing stages."""
        if result.model_execution and result.model_execution.get("result"):
            return result.model_execution["result"]
        
        if result.lam_routing and result.lam_routing.get("result"):
            return result.lam_routing["result"]
        
        if result.llm_understanding:
            # Return understanding summary
            understanding = result.llm_understanding.get("context_understanding", {})
            return f"I understand you want to {understanding.get('user_goal', 'get assistance')}. " \
                   f"Primary intent: {understanding.get('primary_intent', 'unknown')}"
        
        return "Processing completed but no specific result generated."
    
    def _determine_model_routing(self, user_input: str, llm_understanding: Dict[str, Any]) -> Dict[str, Any]:
        """Determine which native model should handle the request."""
        routing = {}
        
        # Check for specific capabilities based on understanding
        if llm_understanding.get("context_understanding"):
            understanding = llm_understanding["context_understanding"]
            primary_intent = understanding.get("primary_intent", "")
            
            # Map intents to request types
            intent_to_request = {
                "code_generation": "code_generation",
                "3d_construction": "3d_creation",
                "multimedia_creation": "video_creation",
                "data_analysis": "data_queries",
                "image_processing": "image_analysis",
                "audio_processing": "audio_processing",
                "emotion_analysis": "emotion_analysis"
            }
            
            if primary_intent in intent_to_request:
                routing["request_type"] = intent_to_request[primary_intent]
                routing["confidence"] = understanding.get("confidence", 0.5)
        
        # Fallback to text generation if no specific routing found
        if not routing:
            routing["request_type"] = "text_generation"
            routing["confidence"] = 0.3
        
        return routing
    
    def _needs_reasoning(self, user_input: str) -> bool:
        """Determine if the input requires sophisticated reasoning."""
        reasoning_keywords = [
            "why", "how", "explain", "analyze", "compare", "evaluate", "solve",
            "problem", "complex", "difficult", "challenge", "figure out"
        ]
        
        user_input_lower = user_input.lower()
        return any(keyword in user_input_lower for keyword in reasoning_keywords)
    
    def _fallback_processing(self, user_input: str, context: Dict[str, Any]) -> str:
        """Fallback processing when the main pipeline fails."""
        try:
            # Simple fallback using LLM engine directly
            return self.llm_engine.run(user_input, max_tokens=200)
        except Exception as e:
            return f"Fallback processing failed: {str(e)}"
    
    def _update_stats(self, result: ProcessingResult):
        """Update processing statistics."""
        self.processing_stats["total_requests"] += 1
        
        if result.success:
            self.processing_stats["successful_requests"] += 1
        else:
            self.processing_stats["failed_requests"] += 1
        
        # Update average processing time
        current_avg = self.processing_stats["avg_processing_time"]
        total_requests = self.processing_stats["total_requests"]
        self.processing_stats["avg_processing_time"] = (
            (current_avg * (total_requests - 1) + result.execution_time) / total_requests
        )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            **self.processing_stats,
            "success_rate": (self.processing_stats["successful_requests"] / 
                           max(self.processing_stats["total_requests"], 1)) * 100
        }
    
    def process(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Simple process method for easy access."""
        result = self.process_input(user_input, context)
        return {
            "success": result.success,
            "result": result.final_result,
            "error": result.error,
            "execution_time": result.execution_time,
            "stages": [stage.value for stage in result.processing_stages]
        }

# Global instance
unified_input_processor = UnifiedInputProcessor()

# Convenience functions
def process_input(user_input: str, context: Dict[str, Any] = None, 
                 memory_context_id: str = None) -> ProcessingResult:
    """Process user input through the unified pipeline."""
    return unified_input_processor.process_input(user_input, context, memory_context_id)

def get_processing_stats() -> Dict[str, Any]:
    """Get processing statistics."""
    return unified_input_processor.get_processing_stats() 