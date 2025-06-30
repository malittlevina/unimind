"""
emotion_classifier.py – Emotion classification for ThothOS/Unimind.
Provides emotion analysis and sentiment detection capabilities.
"""

import logging
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import joblib
import os

class EmotionCategory(Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"

@dataclass
class EmotionResult:
    emotion: EmotionCategory
    confidence: float
    sentiment_score: float
    keywords: List[str]
    metadata: Dict[str, Any]

# Make sklearn optional
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Using fallback emotion classification.")

class EmotionClassifier:
    def __init__(self, model_path='emotion/emotion_model.pkl', vectorizer_path='emotion/vectorizer.pkl'):
        """Initialize the emotion classifier."""
        self.logger = logging.getLogger('EmotionClassifier')
        self.model = None
        self.vectorizer = None
        
        if SKLEARN_AVAILABLE:
            try:
                # Try to load pre-trained models
                if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                    self.model = joblib.load(model_path)
                    self.vectorizer = joblib.load(vectorizer_path)
                    self.logger.info("Loaded pre-trained emotion models")
                else:
                    self.logger.warning("Pre-trained models not found, using fallback")
            except Exception as e:
                self.logger.warning(f"Could not load emotion models: {e}")
        
        # Initialize fallback emotion keywords
        self.emotion_keywords = {
            EmotionCategory.JOY: [
                "happy", "joy", "excited", "great", "wonderful", "amazing", "fantastic",
                "delighted", "thrilled", "ecstatic", "elated", "cheerful", "jubilant"
            ],
            EmotionCategory.SADNESS: [
                "sad", "depressed", "unhappy", "miserable", "sorrowful", "melancholy",
                "gloomy", "down", "blue", "heartbroken", "disappointed", "dejected"
            ],
            EmotionCategory.ANGER: [
                "angry", "mad", "furious", "irritated", "annoyed", "frustrated",
                "enraged", "livid", "outraged", "irate", "heated", "fuming"
            ],
            EmotionCategory.FEAR: [
                "afraid", "scared", "frightened", "terrified", "anxious", "worried",
                "nervous", "panicked", "alarmed", "horrified", "dread", "terror"
            ],
            EmotionCategory.SURPRISE: [
                "surprised", "shocked", "amazed", "astonished", "stunned", "bewildered",
                "startled", "dumbfounded", "flabbergasted", "astounded", "incredible"
            ],
            EmotionCategory.DISGUST: [
                "disgusted", "revolted", "repulsed", "sickened", "appalled", "horrified",
                "nauseated", "offended", "outraged", "disgusting", "revolting"
            ],
            EmotionCategory.NEUTRAL: [
                "neutral", "calm", "peaceful", "serene", "tranquil", "balanced",
                "composed", "collected", "steady", "stable", "normal", "fine"
            ]
        }
        
        self.logger.info("Emotion classifier initialized")
    
    def classify_emotion(self, text: str) -> EmotionResult:
        """
        Classify the emotion in the given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            EmotionResult with classification results
        """
        if not text or not text.strip():
            return EmotionResult(
                emotion=EmotionCategory.NEUTRAL,
                confidence=1.0,
                sentiment_score=0.0,
                keywords=[],
                metadata={"error": "Empty text"}
            )
        
        # Clean and preprocess text
        cleaned_text = self._preprocess_text(text)
        
        if SKLEARN_AVAILABLE and self.model is not None and self.vectorizer is not None:
            try:
                # Use ML model for classification
                features = self.vectorizer.transform([cleaned_text])
                prediction = self.model.predict(features)[0]
                confidence = max(self.model.predict_proba(features)[0])
                
                # Map prediction to emotion category
                emotion = self._map_prediction_to_emotion(prediction)
                
                return EmotionResult(
                    emotion=emotion,
                    confidence=confidence,
                    sentiment_score=self._calculate_sentiment_score(cleaned_text),
                    keywords=self._extract_keywords(cleaned_text),
                    metadata={"method": "ml_model"}
                )
            except Exception as e:
                self.logger.warning(f"ML model failed, using fallback: {e}")
        
        # Fallback to keyword-based classification
        return self._fallback_classification(cleaned_text)
    
    def _fallback_classification(self, text: str) -> EmotionResult:
        """Fallback emotion classification using keyword matching."""
        text_lower = text.lower()
        emotion_scores = {}
        
        # Calculate scores for each emotion based on keyword matches
        for emotion, keywords in self.emotion_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            emotion_scores[emotion] = score
        
        # Find the emotion with the highest score
        if emotion_scores:
            best_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            confidence = min(best_emotion[1] / 3.0, 1.0)  # Normalize confidence
            
            # If no strong emotion detected, default to neutral
            if confidence < 0.3:
                best_emotion = (EmotionCategory.NEUTRAL, 1)
                confidence = 1.0
        else:
            best_emotion = (EmotionCategory.NEUTRAL, 1)
            confidence = 1.0
        
        return EmotionResult(
            emotion=best_emotion[0],
            confidence=confidence,
            sentiment_score=self._calculate_sentiment_score(text),
            keywords=self._extract_keywords(text),
            metadata={"method": "keyword_fallback"}
        )

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for emotion analysis."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _calculate_sentiment_score(self, text: str) -> float:
        """Calculate a simple sentiment score."""
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "happy", "love", "like"]
        negative_words = ["bad", "terrible", "awful", "hate", "dislike", "sad", "angry", "fear"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_words
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract emotion-related keywords from text."""
        keywords = []
        text_lower = text.lower()
        
        for emotion, emotion_keywords in self.emotion_keywords.items():
            for keyword in emotion_keywords:
                if keyword in text_lower:
                    keywords.append(keyword)
        
        return list(set(keywords))  # Remove duplicates
    
    def _map_prediction_to_emotion(self, prediction: str) -> EmotionCategory:
        """Map ML model prediction to emotion category."""
        mapping = {
            "happy": EmotionCategory.JOY,
            "sad": EmotionCategory.SADNESS,
            "angry": EmotionCategory.ANGER,
            "fearful": EmotionCategory.FEAR,
            "surprised": EmotionCategory.SURPRISE,
            "disgusted": EmotionCategory.DISGUST,
            "neutral": EmotionCategory.NEUTRAL
        }
        return mapping.get(prediction, EmotionCategory.NEUTRAL)

# Example usage
if __name__ == "__main__":
    ec = EmotionClassifier()
    print(ec.predict_emotion("I am very frustrated with this situation"))