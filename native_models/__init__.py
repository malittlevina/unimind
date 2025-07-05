"""
Native models package for Unimind.
Provides various AI/ML model interfaces and utilities.
"""

# Import all native models
from .context_model import ContextModel, context_model, add_context, get_current_context
from .emotion_classifier import EmotionClassifier, EmotionCategory, EmotionResult, emotion_classifier, classify_emotion, analyze_sentiment
from .lam_engine import LAMEngine, lam_engine
from .llm_engine import LLMEngine, llm_engine, run_llm_inference
from .text_to_3d import TextTo3D, ModelFormat, ModelResult, text_to_3d, generate_3d_model, convert_format
from .text_to_code import TextToCodeEngine, SimpleLLM, engine as text_to_code_engine
from .text_to_logic import TextToLogic, LogicType, LogicResult, text_to_logic, text_to_logic_engine, analyze_syntax, interpret_meaning, visualize_concepts
from .text_to_sql import TextToSQL, SQLOperation, SQLResult, text_to_sql, convert_to_sql, set_schema
from .text_to_shell import TextToShell, ShellOperation, ShellResult, text_to_shell, convert_to_shell, execute_command
from .text_to_text import TextToText, TransformationType, TextStyle, TransformationResult, text_to_text, transform_text, analyze_text
from .text_to_video import TextToVideo, VideoFormat, VideoStyle, VideoResult, text_to_video, generate_video

# Import new unified components
from .model_registry import (
    ModelRegistry, ModelInfo, ModelCategory, model_registry,
    get_model, list_models, execute_request, get_registry_stats
)

from .unified_input_processor import (
    UnifiedInputProcessor, ProcessingResult, ProcessingStage, unified_input_processor,
    process_input, get_processing_stats
)

# Import vision model (optional - requires numpy)
try:
    from .vision_model import VisionModel, VisionTask, VisionResult, vision_model, process_image, get_image_info
    VISION_AVAILABLE = True
except ImportError:
    # Fallback if vision_model is not available (e.g., numpy missing)
    VisionModel = None
    VisionTask = None
    VisionResult = None
    vision_model = None
    process_image = None
    get_image_info = None
    VISION_AVAILABLE = False

# Import vision submodules (optional)
try:
    from .vision import (
        scene_classifier,
        object_tracker, 
        object_recognizer,
        emotion_overlay
    )
    VISION_SUBMODULES_AVAILABLE = True
except ImportError:
    # Fallback if vision submodules are not available
    scene_classifier = None
    object_tracker = None
    object_recognizer = None
    emotion_overlay = None
    VISION_SUBMODULES_AVAILABLE = False

# Import voice model (basic implementation)
try:
    from .voice_model import VoiceModel, VoiceTask, VoiceResult, voice_model, speech_to_text, text_to_speech
except ImportError:
    # Fallback if voice_model is not fully implemented
    VoiceModel = None
    VoiceTask = None
    VoiceResult = None
    voice_model = None
    speech_to_text = None
    text_to_speech = None

__all__ = [
    # Core models
    'ContextModel', 'context_model', 'add_context', 'get_current_context',
    'EmotionClassifier', 'EmotionCategory', 'EmotionResult', 'emotion_classifier', 'classify_emotion', 'analyze_sentiment',
    'LAMEngine', 'lam_engine',
    'LLMEngine', 'llm_engine', 'run_llm_inference',
    
    # Text-to-X models
    'TextTo3D', 'ModelFormat', 'ModelResult', 'text_to_3d', 'generate_3d_model', 'convert_format',
    'TextToCodeEngine', 'SimpleLLM', 'text_to_code_engine',
    'TextToLogic', 'LogicType', 'LogicResult', 'text_to_logic', 'text_to_logic_engine', 'analyze_syntax', 'interpret_meaning', 'visualize_concepts',
    'TextToSQL', 'SQLOperation', 'SQLResult', 'text_to_sql', 'convert_to_sql', 'set_schema',
    'TextToShell', 'ShellOperation', 'ShellResult', 'text_to_shell', 'convert_to_shell', 'execute_command',
    'TextToText', 'TransformationType', 'TextStyle', 'TransformationResult', 'text_to_text', 'transform_text', 'analyze_text',
    'TextToVideo', 'VideoFormat', 'VideoStyle', 'VideoResult', 'text_to_video', 'generate_video',
    
    # Voice
    'VoiceModel', 'VoiceTask', 'VoiceResult', 'voice_model', 'speech_to_text', 'text_to_speech',
    
    # Unified components
    'ModelRegistry', 'ModelInfo', 'ModelCategory', 'model_registry',
    'get_model', 'list_models', 'execute_request', 'get_registry_stats',
    'UnifiedInputProcessor', 'ProcessingResult', 'ProcessingStage', 'unified_input_processor',
    'process_input', 'get_processing_stats',
]

# Add vision-related exports only if available
if VISION_AVAILABLE:
    __all__.extend(['VisionModel', 'VisionTask', 'VisionResult', 'vision_model', 'process_image', 'get_image_info'])

if VISION_SUBMODULES_AVAILABLE:
    __all__.extend(['scene_classifier', 'object_tracker', 'object_recognizer', 'emotion_overlay'])
