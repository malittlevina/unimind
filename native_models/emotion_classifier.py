"""
emotion_classifier.py – Emotion analysis and classification for Unimind native models.
Provides emotion detection, sentiment analysis, and emotional state tracking.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

class EmotionCategory(Enum):
    """Enumeration of emotion categories."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"
    LOVE = "love"
    CONTEMPT = "contempt"
    EXCITEMENT = "excitement"
    CALM = "calm"
    ANXIETY = "anxiety"

@dataclass
class EmotionResult:
    """Result of emotion classification."""
    primary_emotion: EmotionCategory
    confidence: float
    secondary_emotions: List[Tuple[EmotionCategory, float]]
    intensity: float  # 0.0 to 1.0
    valence: float   # -1.0 (negative) to 1.0 (positive)
    arousal: float   # 0.0 (calm) to 1.0 (aroused)

class EmotionClassifier:
    """
    Analyzes and classifies emotions in text, audio, and vision inputs.
    Supports multi-modal, multi-language, and SOTA model integration (Hume, OpenAI, Google, DeepFace, etc.).
    Provides emotion detection, sentiment analysis, and emotional state tracking.
    """
    
    def __init__(self, backend: str = "rule-based"):
        self.backend = backend
        # SOTA model stubs (to be implemented)
        self.hume = None
        self.openai = None
        self.google = None
        self.deepface = None
        try:
            from unimind.native_models.free_models.vision.deepface_loader import DeepFaceLoader
            self.deepface = DeepFaceLoader()
        except ImportError:
            pass
        try:
            from unimind.native_models.free_models.audio.hume_loader import HumeLoader
            self.hume = HumeLoader()
        except ImportError:
            pass
        try:
            from unimind.native_models.free_models.text.openai_emotion_loader import OpenAIEmotionLoader
            self.openai = OpenAIEmotionLoader()
        except ImportError:
            pass
        try:
            from unimind.native_models.free_models.text.google_emotion_loader import GoogleEmotionLoader
            self.google = GoogleEmotionLoader()
        except ImportError:
            pass
        
        self.emotion_keywords = {
            EmotionCategory.JOY: ["happy", "joy", "excited", "great", "wonderful", "amazing", "fantastic", "delighted"],
            EmotionCategory.SADNESS: ["sad", "depressed", "melancholy", "grief", "sorrow", "unhappy", "miserable"],
            EmotionCategory.ANGER: ["angry", "furious", "mad", "irritated", "annoyed", "rage", "frustrated"],
            EmotionCategory.FEAR: ["afraid", "scared", "terrified", "anxious", "worried", "fearful", "nervous"],
            EmotionCategory.SURPRISE: ["surprised", "shocked", "amazed", "astonished", "stunned"],
            EmotionCategory.DISGUST: ["disgusted", "revolted", "appalled", "sickened"],
            EmotionCategory.LOVE: ["love", "adore", "cherish", "affection", "tender", "caring"],
            EmotionCategory.CONTEMPT: ["contempt", "disdain", "scorn", "disrespect"],
            EmotionCategory.EXCITEMENT: ["excited", "thrilled", "energized", "pumped", "enthusiastic"],
            EmotionCategory.CALM: ["calm", "peaceful", "serene", "tranquil", "relaxed"],
            EmotionCategory.ANXIETY: ["anxious", "worried", "concerned", "stressed", "tense"]
        }
        
        self.intensity_indicators = {
            "very": 1.5,
            "extremely": 2.0,
            "really": 1.3,
            "so": 1.2,
            "quite": 0.8,
            "somewhat": 0.6,
            "slightly": 0.4,
            "a bit": 0.3
        }
        
    def classify_emotion(self, text: str = None, audio: Any = None, image: Any = None, language: str = "en") -> EmotionResult:
        """
        Classify emotions in text, audio, or image using the selected backend.
        """
        if self.backend == "rule-based" and text:
            return self._classify_emotion_text_rule_based(text, language)
        elif self.backend == "deepface" and image and self.deepface:
            return self.deepface.classify_emotion(image)
        elif self.backend == "hume" and audio and self.hume:
            return self.hume.classify_emotion(audio)
        elif self.backend == "openai" and text and self.openai:
            return self.openai.classify_emotion(text, language)
        elif self.backend == "google" and text and self.google:
            return self.google.classify_emotion(text, language)
        else:
            # Fallback to rule-based
            return self._classify_emotion_text_rule_based(text or "", language)

    def _classify_emotion_text_rule_based(self, text: str, language: str = "en") -> EmotionResult:
        text_lower = text.lower()
        emotion_scores = {}
        
        # Count emotion keywords
        for emotion, keywords in self.emotion_keywords.items():
            score = 0
            for keyword in keywords:
                score += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
            if score > 0:
                emotion_scores[emotion] = score
        
        # Adjust intensity based on modifiers
        for modifier, multiplier in self.intensity_indicators.items():
            if modifier in text_lower:
                for emotion in emotion_scores:
                    emotion_scores[emotion] *= multiplier
        
        if not emotion_scores:
            return EmotionResult(
                primary_emotion=EmotionCategory.NEUTRAL,
                confidence=0.8,
                secondary_emotions=[],
                intensity=0.1,
                valence=0.0,
                arousal=0.1
            )
        
        # Find primary emotion
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        max_score = emotion_scores[primary_emotion]
        
        # Calculate confidence and intensity
        total_score = sum(emotion_scores.values())
        confidence = min(max_score / total_score, 1.0) if total_score > 0 else 0.0
        intensity = min(max_score / 3.0, 1.0)  # Normalize to 0-1
        
        # Calculate valence (positive/negative)
        positive_emotions = {EmotionCategory.JOY, EmotionCategory.LOVE, EmotionCategory.EXCITEMENT, EmotionCategory.CALM}
        negative_emotions = {EmotionCategory.SADNESS, EmotionCategory.ANGER, EmotionCategory.FEAR, EmotionCategory.DISGUST, EmotionCategory.CONTEMPT, EmotionCategory.ANXIETY}
        
        positive_score = sum(emotion_scores.get(e, 0) for e in positive_emotions)
        negative_score = sum(emotion_scores.get(e, 0) for e in negative_emotions)
        
        if positive_score + negative_score > 0:
            valence = (positive_score - negative_score) / (positive_score + negative_score)
        else:
            valence = 0.0
        
        # Calculate arousal (calm vs aroused)
        high_arousal = {EmotionCategory.ANGER, EmotionCategory.FEAR, EmotionCategory.EXCITEMENT, EmotionCategory.SURPRISE}
        low_arousal = {EmotionCategory.SADNESS, EmotionCategory.CALM, EmotionCategory.CONTEMPT}
        
        high_score = sum(emotion_scores.get(e, 0) for e in high_arousal)
        low_score = sum(emotion_scores.get(e, 0) for e in low_arousal)
        
        if high_score + low_score > 0:
            arousal = high_score / (high_score + low_score)
        else:
            arousal = 0.5
        
        # Get secondary emotions
        secondary_emotions = []
        for emotion, score in emotion_scores.items():
            if emotion != primary_emotion and score > 0:
                secondary_emotions.append((emotion, score / total_score))
        
        # Sort by confidence
        secondary_emotions.sort(key=lambda x: x[1], reverse=True)
        
        return EmotionResult(
            primary_emotion=primary_emotion,
            confidence=confidence,
            secondary_emotions=secondary_emotions[:3],  # Top 3 secondary emotions
            intensity=intensity,
            valence=valence,
            arousal=arousal
        )
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing sentiment scores
        """
        emotion_result = self.classify_emotion(text)
        return {
            "valence": emotion_result.valence,
            "arousal": emotion_result.arousal,
            "intensity": emotion_result.intensity,
            "confidence": emotion_result.confidence
        }
    
    def track_emotional_state(self, text_sequence: List[str]) -> List[EmotionResult]:
        """
        Track emotional state changes over a sequence of texts.
        
        Args:
            text_sequence: List of texts to analyze
            
        Returns:
            List of EmotionResult objects for each text
        """
        return [self.classify_emotion(text) for text in text_sequence]
    
    def get_emotion_summary(self, emotion_results: List[EmotionResult]) -> Dict[str, Any]:
        """
        Get a summary of emotional states.
        
        Args:
            emotion_results: List of EmotionResult objects
            
        Returns:
            Dictionary containing emotional summary
        """
        if not emotion_results:
            return {"error": "No emotion results provided"}
        
        primary_emotions = [result.primary_emotion.value for result in emotion_results]
        avg_valence = sum(result.valence for result in emotion_results) / len(emotion_results)
        avg_arousal = sum(result.arousal for result in emotion_results) / len(emotion_results)
        avg_intensity = sum(result.intensity for result in emotion_results) / len(emotion_results)
        
        return {
            "total_analyses": len(emotion_results),
            "most_common_emotion": max(set(primary_emotions), key=primary_emotions.count),
            "average_valence": avg_valence,
            "average_arousal": avg_arousal,
            "average_intensity": avg_intensity,
            "emotional_trajectory": "stable" if abs(avg_valence) < 0.3 else "positive" if avg_valence > 0 else "negative"
        }

# Module-level instance
emotion_classifier = EmotionClassifier()

# Export the engine instance with the expected name
emotion_engine = emotion_classifier

def classify_emotion(text: str) -> EmotionResult:
    """Classify emotion using the module-level instance."""
    return emotion_classifier.classify_emotion(text)

def analyze_sentiment(text: str) -> Dict[str, float]:
    """Analyze sentiment using the module-level instance."""
    return emotion_classifier.analyze_sentiment(text)
