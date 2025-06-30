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
from .text_to_logic import TextToLogic, LogicType, LogicResult, text_to_logic, analyze_syntax, interpret_meaning, visualize_concepts
from .text_to_sql import TextToSQL, SQLOperation, SQLResult, text_to_sql, convert_to_sql, set_schema
from .text_to_shell import TextToShell, ShellOperation, ShellResult, text_to_shell, convert_to_shell, execute_command
from .text_to_text import TextToText, TransformationType, TextStyle, TransformationResult, text_to_text, transform_text, analyze_text
from .text_to_video import TextToVideo, VideoFormat, VideoStyle, VideoResult, text_to_video, generate_video
from .vision_model import VisionModel, VisionTask, VisionResult, vision_model, process_image, get_image_info

# Import vision submodules
from .vision import (
    scene_classifier,
    object_tracker, 
    object_recognizer,
    emotion_overlay
)

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
    'TextToLogic', 'LogicType', 'LogicResult', 'text_to_logic', 'analyze_syntax', 'interpret_meaning', 'visualize_concepts',
    'TextToSQL', 'SQLOperation', 'SQLResult', 'text_to_sql', 'convert_to_sql', 'set_schema',
    'TextToShell', 'ShellOperation', 'ShellResult', 'text_to_shell', 'convert_to_shell', 'execute_command',
    'TextToText', 'TransformationType', 'TextStyle', 'TransformationResult', 'text_to_text', 'transform_text', 'analyze_text',
    'TextToVideo', 'VideoFormat', 'VideoStyle', 'VideoResult', 'text_to_video', 'generate_video',
    
    # Vision and voice
    'VisionModel', 'VisionTask', 'VisionResult', 'vision_model', 'process_image', 'get_image_info',
    'VoiceModel', 'VoiceTask', 'VoiceResult', 'voice_model', 'speech_to_text', 'text_to_speech',
    
    # Vision submodules
    'scene_classifier', 'object_tracker', 'object_recognizer', 'emotion_overlay'
]
