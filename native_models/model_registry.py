"""
model_registry.py – Comprehensive model registry for Unimind native models.
Provides unified model management, routing, and integration capabilities.
"""

import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import time

# Import all native models
from .llm_engine import llm_engine
from .lam_engine import lam_engine
from .text_to_code import text_to_code_engine
from .text_to_3d import text_to_3d_engine
from .vision_model import vision_model
from .emotion_classifier import emotion_engine
from .text_to_text import text_to_text_engine
from .text_to_video import text_to_video_engine
from .text_to_logic import text_to_logic_engine
from .voice_model import voice_model
from .text_to_shell import text_to_shell_engine
from .text_to_sql import text_to_sql_engine
from .context_model import context_engine
from .huggingface_local import huggingface_local_engine

class ModelCategory(Enum):
    """Categories of models."""
    LLM = "llm"
    LAM = "lam"
    TEXT_PROCESSING = "text_processing"
    VISION = "vision"
    AUDIO = "audio"
    CODE_GENERATION = "code_generation"
    DATA_PROCESSING = "data_processing"
    EMOTION = "emotion"
    CONTEXT = "context"

@dataclass
class ModelInfo:
    """Information about a registered model."""
    name: str
    category: ModelCategory
    description: str
    engine: Any
    capabilities: List[str]
    input_types: List[str]
    output_types: List[str]
    is_active: bool = True
    last_used: float = 0.0
    usage_count: int = 0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

class ModelRegistry:
    """
    Comprehensive model registry for managing and routing to native models.
    """
    
    def __init__(self):
        """Initialize the model registry."""
        self.logger = logging.getLogger('ModelRegistry')
        self.models: Dict[str, ModelInfo] = {}
        self.routing_rules: Dict[str, List[str]] = {}
        
        # Initialize all models
        self._register_all_models()
        self._initialize_routing_rules()
        
        self.logger.info(f"Model registry initialized with {len(self.models)} models")
    
    def _register_all_models(self):
        """Register all available native models."""
        # Core engines
        self._register_model("llm_engine", ModelCategory.LLM, llm_engine,
                           "Unified LLM engine with context awareness and reasoning",
                           ["text_generation", "context_understanding", "reasoning", "intent_classification"],
                           ["text", "image", "audio"], ["text"])
        
        self._register_model("lam_engine", ModelCategory.LAM, lam_engine,
                           "Language Action Mapping engine for task routing",
                           ["action_mapping", "task_routing", "planning", "execution"],
                           ["text"], ["action", "plan", "result"])
        
        # Text processing models
        self._register_model("text_to_text", ModelCategory.TEXT_PROCESSING, text_to_text_engine,
                           "Text-to-text processing and transformation",
                           ["text_transformation", "summarization", "translation"],
                           ["text"], ["text"])
        
        self._register_model("text_to_sql", ModelCategory.DATA_PROCESSING, text_to_sql_engine,
                           "Natural language to SQL query generation",
                           ["sql_generation", "query_optimization"],
                           ["text"], ["sql"])
        
        self._register_model("text_to_logic", ModelCategory.CODE_GENERATION, text_to_logic_engine,
                           "Natural language to logical expressions",
                           ["logic_generation", "expression_parsing"],
                           ["text"], ["logic_expression"])
        
        self._register_model("text_to_shell", ModelCategory.CODE_GENERATION, text_to_shell_engine,
                           "Natural language to shell commands",
                           ["shell_command_generation", "system_automation"],
                           ["text"], ["shell_command"])
        
        # Code generation models
        self._register_model("text_to_code", ModelCategory.CODE_GENERATION, text_to_code_engine,
                           "Natural language to code generation",
                           ["code_generation", "programming_language_support"],
                           ["text"], ["code"])
        
        # Creative models
        self._register_model("text_to_3d", ModelCategory.VISION, text_to_3d_engine,
                           "Text to 3D model generation",
                           ["3d_model_generation", "scene_creation"],
                           ["text"], ["3d_model", "scene"])
        
        self._register_model("text_to_video", ModelCategory.VISION, text_to_video_engine,
                           "Text to video generation",
                           ["video_generation", "animation_creation"],
                           ["text"], ["video"])
        
        # Sensory models
        self._register_model("vision_model", ModelCategory.VISION, vision_model,
                           "Computer vision and image processing",
                           ["image_analysis", "object_detection", "image_generation"],
                           ["image", "text"], ["text", "image"])
        
        self._register_model("voice_model", ModelCategory.AUDIO, voice_model,
                           "Voice and audio processing",
                           ["speech_recognition", "text_to_speech", "audio_analysis"],
                           ["audio", "text"], ["text", "audio"])
        
        # Specialized models
        self._register_model("emotion_classifier", ModelCategory.EMOTION, emotion_engine,
                           "Emotion classification and analysis",
                           ["emotion_detection", "sentiment_analysis"],
                           ["text", "image"], ["emotion", "sentiment"])
        
        self._register_model("context_model", ModelCategory.CONTEXT, context_engine,
                           "Context understanding and management",
                           ["context_analysis", "memory_management"],
                           ["text"], ["context", "memory"])
        
        # Local models
        self._register_model("huggingface_local", ModelCategory.LLM, huggingface_local_engine,
                           "Local HuggingFace model inference",
                           ["local_inference", "offline_processing"],
                           ["text"], ["text"])
    
    def _register_model(self, name: str, category: ModelCategory, engine: Any, 
                       description: str, capabilities: List[str], 
                       input_types: List[str], output_types: List[str]):
        """Register a model in the registry."""
        self.models[name] = ModelInfo(
            name=name,
            category=category,
            description=description,
            engine=engine,
            capabilities=capabilities,
            input_types=input_types,
            output_types=output_types
        )
    
    def _initialize_routing_rules(self):
        """Initialize routing rules for different types of requests."""
        self.routing_rules = {
            "text_generation": ["llm_engine", "text_to_text"],
            "code_generation": ["text_to_code", "text_to_logic", "text_to_shell"],
            "3d_creation": ["text_to_3d"],
            "video_creation": ["text_to_video"],
            "image_analysis": ["vision_model"],
            "audio_processing": ["voice_model"],
            "emotion_analysis": ["emotion_classifier"],
            "data_queries": ["text_to_sql"],
            "context_understanding": ["context_model", "llm_engine"],
            "action_mapping": ["lam_engine"],
            "local_processing": ["huggingface_local"]
        }
    
    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get a model by name."""
        return self.models.get(name)
    
    def list_models(self, category: Optional[ModelCategory] = None) -> List[ModelInfo]:
        """List all models, optionally filtered by category."""
        if category:
            return [model for model in self.models.values() if model.category == category]
        return list(self.models.values())
    
    def find_models_by_capability(self, capability: str) -> List[ModelInfo]:
        """Find models that have a specific capability."""
        return [model for model in self.models.values() if capability in model.capabilities]
    
    def find_models_by_input_type(self, input_type: str) -> List[ModelInfo]:
        """Find models that can handle a specific input type."""
        return [model for model in self.models.values() if input_type in model.input_types]
    
    def route_request(self, request_type: str, input_data: Any, **kwargs) -> Optional[ModelInfo]:
        """Route a request to the appropriate model."""
        if request_type in self.routing_rules:
            candidate_models = self.routing_rules[request_type]
            
            # Find the first available model
            for model_name in candidate_models:
                model = self.get_model(model_name)
                if model and model.is_active:
                    return model
        
        # Fallback to LLM engine for unknown request types
        return self.get_model("llm_engine")
    
    def execute_request(self, request_type: str, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Execute a request by routing to the appropriate model."""
        model = self.route_request(request_type, input_data, **kwargs)
        
        if not model:
            return {"success": False, "error": "No suitable model found"}
        
        try:
            # Update model usage statistics
            model.last_used = time.time()
            model.usage_count += 1
            
            # Execute the request
            if hasattr(model.engine, 'run'):
                result = model.engine.run(input_data, **kwargs)
            elif hasattr(model.engine, 'process'):
                result = model.engine.process(input_data, **kwargs)
            elif hasattr(model.engine, 'generate'):
                result = model.engine.generate(input_data, **kwargs)
            else:
                return {"success": False, "error": f"Model {model.name} has no execution method"}
            
            return {
                "success": True,
                "result": result,
                "model_used": model.name,
                "model_category": model.category.value
            }
            
        except Exception as e:
            self.logger.error(f"Error executing request with model {model.name}: {e}")
            return {"success": False, "error": str(e), "model_used": model.name}
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the model registry."""
        stats = {
            "total_models": len(self.models),
            "active_models": len([m for m in self.models.values() if m.is_active]),
            "models_by_category": {},
            "most_used_models": [],
            "recently_used_models": []
        }
        
        # Models by category
        for model in self.models.values():
            category = model.category.value
            if category not in stats["models_by_category"]:
                stats["models_by_category"][category] = 0
            stats["models_by_category"][category] += 1
        
        # Most used models
        sorted_by_usage = sorted(self.models.values(), key=lambda m: m.usage_count, reverse=True)
        stats["most_used_models"] = [(m.name, m.usage_count) for m in sorted_by_usage[:5]]
        
        # Recently used models
        sorted_by_time = sorted(self.models.values(), key=lambda m: m.last_used, reverse=True)
        stats["recently_used_models"] = [(m.name, m.last_used) for m in sorted_by_time[:5] if m.last_used > 0]
        
        return stats

# Global instance
model_registry = ModelRegistry()

# Convenience functions
def get_model(name: str) -> Optional[ModelInfo]:
    """Get a model by name."""
    return model_registry.get_model(name)

def list_models(category: Optional[ModelCategory] = None) -> List[ModelInfo]:
    """List all models, optionally filtered by category."""
    return model_registry.list_models(category)

def execute_request(request_type: str, input_data: Any, **kwargs) -> Dict[str, Any]:
    """Execute a request by routing to the appropriate model."""
    return model_registry.execute_request(request_type, input_data, **kwargs)

def get_registry_stats() -> Dict[str, Any]:
    """Get statistics about the model registry."""
    return model_registry.get_registry_stats() 