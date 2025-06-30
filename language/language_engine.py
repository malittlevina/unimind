"""
language_engine.py – Language abstraction layer for ThothOS/Unimind.
Provides unified interface for all language/LLM/text model operations.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json

class ModelType(Enum):
    """Types of language models."""
    OPENAI = "openai"
    LOCAL_LLM = "local_llm"
    HUGGINGFACE = "huggingface"
    RULE_BASED = "rule_based"
    MOCK = "mock"

class TaskType(Enum):
    """Types of language tasks."""
    SUMMARIZE = "summarize"
    PARSE_INTENT = "parse_intent"
    GENERATE_TEXT = "generate_text"
    CLASSIFY_EMOTION = "classify_emotion"
    TRANSLATE = "translate"
    EXTRACT_ENTITIES = "extract_entities"
    ANSWER_QUESTION = "answer_question"
    CODE_GENERATION = "code_generation"
    LOGIC_REASONING = "logic_reasoning"

@dataclass
class LanguageRequest:
    """Represents a language processing request."""
    task_type: TaskType
    input_text: str
    parameters: Dict[str, Any]
    model_type: ModelType
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}

@dataclass
class LanguageResponse:
    """Represents a language processing response."""
    success: bool
    result: Any
    confidence: float
    model_used: str
    processing_time: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class LanguageEngine:
    """
    Unified language processing engine for ThothOS/Unimind.
    Provides consistent interface for all language/LLM operations.
    """
    
    def __init__(self, default_model: ModelType = ModelType.MOCK):
        """Initialize the language engine."""
        self.default_model = default_model
        self.models: Dict[ModelType, Any] = {}
        self.task_handlers: Dict[TaskType, Dict[ModelType, callable]] = {}
        self.logger = logging.getLogger('LanguageEngine')
        
        # Initialize models and handlers
        self._initialize_models()
        self._initialize_handlers()
    
    def _initialize_models(self):
        """Initialize available language models."""
        # Mock model for testing
        self.models[ModelType.MOCK] = MockLanguageModel()
        
        # Rule-based model for simple tasks
        self.models[ModelType.RULE_BASED] = RuleBasedLanguageModel()
        
        # Try to initialize other models if available
        try:
            # This would initialize actual models
            pass
        except Exception as e:
            self.logger.warning(f"Could not initialize some models: {e}")
    
    def _initialize_handlers(self):
        """Initialize task handlers for different model types."""
        # Initialize handler structure
        for task_type in TaskType:
            self.task_handlers[task_type] = {}
        
        # Register handlers for each task type and model
        self._register_handlers()
    
    def _register_handlers(self):
        """Register handlers for different task types."""
        # Summarize task
        self.task_handlers[TaskType.SUMMARIZE][ModelType.MOCK] = self._mock_summarize
        self.task_handlers[TaskType.SUMMARIZE][ModelType.RULE_BASED] = self._rule_based_summarize
        
        # Parse intent task
        self.task_handlers[TaskType.PARSE_INTENT][ModelType.MOCK] = self._mock_parse_intent
        self.task_handlers[TaskType.PARSE_INTENT][ModelType.RULE_BASED] = self._rule_based_parse_intent
        
        # Generate text task
        self.task_handlers[TaskType.GENERATE_TEXT][ModelType.MOCK] = self._mock_generate_text
        self.task_handlers[TaskType.GENERATE_TEXT][ModelType.RULE_BASED] = self._rule_based_generate_text
        
        # Classify emotion task
        self.task_handlers[TaskType.CLASSIFY_EMOTION][ModelType.MOCK] = self._mock_classify_emotion
        self.task_handlers[TaskType.CLASSIFY_EMOTION][ModelType.RULE_BASED] = self._rule_based_classify_emotion
        
        # Code generation task
        self.task_handlers[TaskType.CODE_GENERATION][ModelType.MOCK] = self._mock_code_generation
        self.task_handlers[TaskType.CODE_GENERATION][ModelType.RULE_BASED] = self._rule_based_code_generation
    
    def process(self, request: LanguageRequest) -> LanguageResponse:
        """
        Process a language request.
        
        Args:
            request: The language processing request
            
        Returns:
            Language processing response
        """
        import time
        start_time = time.time()
        
        try:
            # Get the appropriate handler
            if request.task_type not in self.task_handlers:
                return LanguageResponse(
                    success=False,
                    result="Unknown task type",
                    confidence=0.0,
                    model_used="none",
                    processing_time=time.time() - start_time
                )
            
            if request.model_type not in self.task_handlers[request.task_type]:
                # Fallback to default model
                request.model_type = self.default_model
            
            handler = self.task_handlers[request.task_type][request.model_type]
            
            # Process the request
            result = handler(request)
            
            processing_time = time.time() - start_time
            
            return LanguageResponse(
                success=True,
                result=result,
                confidence=0.8,  # Mock confidence
                model_used=request.model_type.value,
                processing_time=processing_time,
                metadata={"task_type": request.task_type.value}
            )
            
        except Exception as e:
            self.logger.error(f"Error processing language request: {e}")
            return LanguageResponse(
                success=False,
                result=str(e),
                confidence=0.0,
                model_used=request.model_type.value,
                processing_time=time.time() - start_time
            )
    
    def summarize(self, text: str, max_length: int = 100, model_type: ModelType = None) -> str:
        """
        Summarize text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            model_type: Model to use (defaults to default model)
            
        Returns:
            Summarized text
        """
        model_type = model_type or self.default_model
        
        request = LanguageRequest(
            task_type=TaskType.SUMMARIZE,
            input_text=text,
            parameters={"max_length": max_length},
            model_type=model_type
        )
        
        response = self.process(request)
        return response.result if response.success else f"Error: {response.result}"
    
    def parse_intent(self, text: str, model_type: ModelType = None) -> Dict[str, Any]:
        """
        Parse intent from text.
        
        Args:
            text: Text to parse
            model_type: Model to use (defaults to default model)
            
        Returns:
            Parsed intent information
        """
        model_type = model_type or self.default_model
        
        request = LanguageRequest(
            task_type=TaskType.PARSE_INTENT,
            input_text=text,
            parameters={},
            model_type=model_type
        )
        
        response = self.process(request)
        return response.result if response.success else {"intent": "unknown", "confidence": 0.0}
    
    def generate_text(self, prompt: str, max_length: int = 200, model_type: ModelType = None) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Text prompt
            max_length: Maximum length of generated text
            model_type: Model to use (defaults to default model)
            
        Returns:
            Generated text
        """
        model_type = model_type or self.default_model
        
        request = LanguageRequest(
            task_type=TaskType.GENERATE_TEXT,
            input_text=prompt,
            parameters={"max_length": max_length},
            model_type=model_type
        )
        
        response = self.process(request)
        return response.result if response.success else f"Error: {response.result}"
    
    def classify_emotion(self, text: str, model_type: ModelType = None) -> Dict[str, Any]:
        """
        Classify emotion in text.
        
        Args:
            text: Text to classify
            model_type: Model to use (defaults to default model)
            
        Returns:
            Emotion classification
        """
        model_type = model_type or self.default_model
        
        request = LanguageRequest(
            task_type=TaskType.CLASSIFY_EMOTION,
            input_text=text,
            parameters={},
            model_type=model_type
        )
        
        response = self.process(request)
        return response.result if response.success else {"emotion": "neutral", "confidence": 0.0}
    
    def generate_code(self, prompt: str, language: str = "python", model_type: ModelType = None) -> str:
        """
        Generate code from prompt.
        
        Args:
            prompt: Code generation prompt
            language: Programming language
            model_type: Model to use (defaults to default model)
            
        Returns:
            Generated code
        """
        model_type = model_type or self.default_model
        
        request = LanguageRequest(
            task_type=TaskType.CODE_GENERATION,
            input_text=prompt,
            parameters={"language": language},
            model_type=model_type
        )
        
        response = self.process(request)
        return response.result if response.success else f"# Error: {response.result}"
    
    # Mock handlers
    def _mock_summarize(self, request: LanguageRequest) -> str:
        """Mock summarize handler."""
        text = request.input_text
        max_length = request.parameters.get("max_length", 100)
        
        if len(text) <= max_length:
            return text
        
        # Simple truncation for mock
        return text[:max_length] + "..."
    
    def _mock_parse_intent(self, request: LanguageRequest) -> Dict[str, Any]:
        """Mock intent parsing handler."""
        text = request.input_text.lower()
        
        # Simple keyword matching
        if "optimize" in text or "improve" in text:
            return {"intent": "optimize_self", "confidence": 0.8}
        elif "search" in text or "find" in text:
            return {"intent": "web_search", "confidence": 0.7}
        elif "weather" in text:
            return {"intent": "weather_check", "confidence": 0.9}
        else:
            return {"intent": "unknown", "confidence": 0.1}
    
    def _mock_generate_text(self, request: LanguageRequest) -> str:
        """Mock text generation handler."""
        prompt = request.input_text
        max_length = request.parameters.get("max_length", 200)
        
        # Simple mock response
        return f"[MOCK] Generated response to: {prompt[:50]}..."
    
    def _mock_classify_emotion(self, request: LanguageRequest) -> Dict[str, Any]:
        """Mock emotion classification handler."""
        text = request.input_text.lower()
        
        # Simple keyword-based classification
        if any(word in text for word in ["happy", "joy", "excited", "great"]):
            return {"emotion": "joy", "confidence": 0.8}
        elif any(word in text for word in ["sad", "depressed", "unhappy"]):
            return {"emotion": "sadness", "confidence": 0.7}
        elif any(word in text for word in ["angry", "mad", "furious"]):
            return {"emotion": "anger", "confidence": 0.9}
        else:
            return {"emotion": "neutral", "confidence": 0.6}
    
    def _mock_code_generation(self, request: LanguageRequest) -> str:
        """Mock code generation handler."""
        prompt = request.input_text
        language = request.parameters.get("language", "python")
        
        return f"# [MOCK] Generated {language} code for: {prompt[:50]}...\n# TODO: Implement actual code generation"
    
    # Rule-based handlers
    def _rule_based_summarize(self, request: LanguageRequest) -> str:
        """Rule-based summarize handler."""
        return self._mock_summarize(request)  # Same as mock for now
    
    def _rule_based_parse_intent(self, request: LanguageRequest) -> Dict[str, Any]:
        """Rule-based intent parsing handler."""
        return self._mock_parse_intent(request)  # Same as mock for now
    
    def _rule_based_generate_text(self, request: LanguageRequest) -> str:
        """Rule-based text generation handler."""
        return self._mock_generate_text(request)  # Same as mock for now
    
    def _rule_based_classify_emotion(self, request: LanguageRequest) -> Dict[str, Any]:
        """Rule-based emotion classification handler."""
        return self._mock_classify_emotion(request)  # Same as mock for now
    
    def _rule_based_code_generation(self, request: LanguageRequest) -> str:
        """Rule-based code generation handler."""
        return self._mock_code_generation(request)  # Same as mock for now
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return [model.value for model in self.models.keys()]
    
    def get_available_tasks(self) -> List[str]:
        """Get list of available tasks."""
        return [task.value for task in TaskType]

class MockLanguageModel:
    """Mock language model for testing."""
    pass

class RuleBasedLanguageModel:
    """Rule-based language model for simple tasks."""
    pass

# Global language engine instance
language_engine = LanguageEngine()

def summarize_text(text: str, max_length: int = 100, model_type: ModelType = None) -> str:
    """Summarize text using the global language engine."""
    return language_engine.summarize(text, max_length, model_type)

def parse_intent(text: str, model_type: ModelType = None) -> Dict[str, Any]:
    """Parse intent using the global language engine."""
    return language_engine.parse_intent(text, model_type)

def generate_text(prompt: str, max_length: int = 200, model_type: ModelType = None) -> str:
    """Generate text using the global language engine."""
    return language_engine.generate_text(prompt, max_length, model_type)

def classify_emotion(text: str, model_type: ModelType = None) -> Dict[str, Any]:
    """Classify emotion using the global language engine."""
    return language_engine.classify_emotion(text, model_type)

def generate_code(prompt: str, language: str = "python", model_type: ModelType = None) -> str:
    """Generate code using the global language engine."""
    return language_engine.generate_code(prompt, language, model_type) 