"""
Advanced Natural Language Processing Engine

Advanced NLP capabilities for UniMind.
Provides multilingual support, sentiment analysis, text generation, language understanding, conversational AI, and advanced text processing.
"""

import asyncio
import logging
import time
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
from datetime import datetime, timedelta
import hashlib
import random
import re

# NLP dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class Language(Enum):
    """Supported languages."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    RUSSIAN = "ru"
    HINDI = "hi"


class SentimentType(Enum):
    """Sentiment types."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class TextType(Enum):
    """Types of text."""
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    CONVERSATIONAL = "conversational"
    FORMAL = "formal"


class EntityType(Enum):
    """Named entity types."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    MONEY = "money"
    PERCENT = "percent"
    TIME = "time"
    QUANTITY = "quantity"


@dataclass
class TextDocument:
    """Text document representation."""
    document_id: str
    content: str
    language: Language
    text_type: TextType
    metadata: Dict[str, Any]
    processing_results: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SentimentAnalysis:
    """Sentiment analysis result."""
    analysis_id: str
    document_id: str
    overall_sentiment: SentimentType
    sentiment_scores: Dict[str, float]
    aspect_sentiments: Dict[str, SentimentType]
    confidence: float
    keywords: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NamedEntity:
    """Named entity recognition result."""
    entity_id: str
    text: str
    entity_type: EntityType
    start_position: int
    end_position: int
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TextGeneration:
    """Text generation result."""
    generation_id: str
    prompt: str
    generated_text: str
    language: Language
    style: str
    length: int
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """Conversation representation."""
    conversation_id: str
    participants: List[str]
    messages: List[Dict[str, Any]]
    context: Dict[str, Any]
    sentiment_trend: List[SentimentType]
    language: Language
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LanguageModel:
    """Language model information."""
    model_id: str
    name: str
    language: Language
    model_type: str  # "transformer", "lstm", "cnn"
    parameters: int
    training_data_size: int
    performance_metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Translation:
    """Translation result."""
    translation_id: str
    source_text: str
    target_text: str
    source_language: Language
    target_language: Language
    confidence: float
    alternatives: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedNLPEngine:
    """
    Advanced natural language processing engine for UniMind.
    
    Provides multilingual support, sentiment analysis, text generation,
    language understanding, conversational AI, and advanced text processing.
    """
    
    def __init__(self):
        """Initialize the advanced NLP engine."""
        self.logger = logging.getLogger('AdvancedNLPEngine')
        
        # NLP data storage
        self.text_documents: Dict[str, TextDocument] = {}
        self.sentiment_analyses: Dict[str, SentimentAnalysis] = {}
        self.named_entities: Dict[str, NamedEntity] = {}
        self.text_generations: Dict[str, TextGeneration] = {}
        self.conversations: Dict[str, Conversation] = {}
        self.language_models: Dict[str, LanguageModel] = {}
        self.translations: Dict[str, Translation] = {}
        
        # NLP models and processors
        self.sentiment_models: Dict[str, Any] = {}
        self.entity_recognizers: Dict[str, Any] = {}
        self.text_generators: Dict[str, Any] = {}
        self.translators: Dict[str, Any] = {}
        
        # Performance metrics
        self.metrics = {
            'total_documents': 0,
            'total_sentiment_analyses': 0,
            'total_entity_extractions': 0,
            'total_text_generations': 0,
            'total_translations': 0,
            'avg_sentiment_confidence': 0.0,
            'avg_translation_confidence': 0.0,
            'supported_languages': len(Language)
        }
        
        # Threading
        self.lock = threading.RLock()
        
        # Check dependencies
        self.pandas_available = PANDAS_AVAILABLE
        self.sklearn_available = SKLEARN_AVAILABLE
        self.nltk_available = NLTK_AVAILABLE
        
        # Initialize NLP knowledge base
        self._initialize_nlp_knowledge()
        
        # Initialize NLP models
        self._initialize_nlp_models()
        
        self.logger.info("Advanced NLP engine initialized")
    
    def _initialize_nlp_knowledge(self):
        """Initialize NLP knowledge base."""
        # Language characteristics
        self.language_characteristics = {
            Language.ENGLISH: {
                'word_order': 'svo',
                'complexity': 'medium',
                'script': 'latin',
                'speakers': 1500000000
            },
            Language.SPANISH: {
                'word_order': 'svo',
                'complexity': 'medium',
                'script': 'latin',
                'speakers': 500000000
            },
            Language.FRENCH: {
                'word_order': 'svo',
                'complexity': 'medium',
                'script': 'latin',
                'speakers': 300000000
            },
            Language.CHINESE: {
                'word_order': 'svo',
                'complexity': 'high',
                'script': 'hanzi',
                'speakers': 1200000000
            },
            Language.JAPANESE: {
                'word_order': 'sov',
                'complexity': 'high',
                'script': 'kanji_hiragana_katakana',
                'speakers': 130000000
            }
        }
        
        # Sentiment keywords
        self.sentiment_keywords = {
            SentimentType.POSITIVE: [
                'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied'
            ],
            SentimentType.NEGATIVE: [
                'bad', 'terrible', 'awful', 'horrible', 'disappointing', 'frustrated',
                'hate', 'dislike', 'angry', 'sad', 'upset', 'annoyed'
            ],
            SentimentType.NEUTRAL: [
                'okay', 'fine', 'normal', 'average', 'standard', 'regular',
                'acceptable', 'adequate', 'satisfactory', 'reasonable'
            ]
        }
        
        # Text type characteristics
        self.text_type_characteristics = {
            TextType.NEWS: {
                'formality': 'high',
                'complexity': 'medium',
                'objectivity': 'high',
                'structure': 'organized'
            },
            TextType.SOCIAL_MEDIA: {
                'formality': 'low',
                'complexity': 'low',
                'objectivity': 'low',
                'structure': 'casual'
            },
            TextType.TECHNICAL: {
                'formality': 'high',
                'complexity': 'high',
                'objectivity': 'high',
                'structure': 'structured'
            },
            TextType.CONVERSATIONAL: {
                'formality': 'low',
                'complexity': 'low',
                'objectivity': 'low',
                'structure': 'natural'
            }
        }
    
    def _initialize_nlp_models(self):
        """Initialize NLP models."""
        if self.sklearn_available:
            # Initialize sentiment analysis model
            self.sentiment_models['basic'] = {
                'vectorizer': TfidfVectorizer(max_features=1000),
                'classifier': MultinomialNB()
            }
        
        if self.nltk_available:
            try:
                # Download required NLTK data
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
            except Exception as e:
                self.logger.warning(f"Failed to download NLTK data: {e}")
    
    async def process_text_document(self, content: str,
                                  language: Language = Language.ENGLISH,
                                  text_type: TextType = TextType.CONVERSATIONAL) -> str:
        """Process a text document."""
        document_id = f"doc_{int(time.time())}"
        
        # Basic text preprocessing
        processed_content = await self._preprocess_text(content, language)
        
        # Detect text type if not specified
        if text_type == TextType.CONVERSATIONAL:
            text_type = await self._detect_text_type(processed_content)
        
        # Perform initial analysis
        processing_results = await self._analyze_text(processed_content, language)
        
        text_document = TextDocument(
            document_id=document_id,
            content=processed_content,
            language=language,
            text_type=text_type,
            metadata={
                'original_length': len(content),
                'processed_length': len(processed_content),
                'word_count': len(processed_content.split())
            },
            processing_results=processing_results
        )
        
        with self.lock:
            self.text_documents[document_id] = text_document
            self.metrics['total_documents'] += 1
        
        self.logger.info(f"Processed text document: {document_id}")
        return document_id
    
    async def _preprocess_text(self, text: str, language: Language) -> str:
        """Preprocess text for analysis."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters (keep basic punctuation)
        text = re.sub(r'[^\w\s\.\,\!\?\;\:]', '', text)
        
        # Basic tokenization
        if self.nltk_available:
            try:
                tokens = word_tokenize(text)
                # Remove stopwords
                stop_words = set(stopwords.words('english'))
                tokens = [token for token in tokens if token not in stop_words]
                text = ' '.join(tokens)
            except Exception as e:
                self.logger.warning(f"Error in NLTK preprocessing: {e}")
        
        return text
    
    async def _detect_text_type(self, text: str) -> TextType:
        """Detect text type based on content."""
        # Simple heuristics for text type detection
        if any(word in text.lower() for word in ['news', 'report', 'announcement']):
            return TextType.NEWS
        elif any(word in text.lower() for word in ['code', 'function', 'algorithm', 'technical']):
            return TextType.TECHNICAL
        elif any(word in text.lower() for word in ['hello', 'hi', 'thanks', 'please']):
            return TextType.CONVERSATIONAL
        else:
            return TextType.CONVERSATIONAL
    
    async def _analyze_text(self, text: str, language: Language) -> Dict[str, Any]:
        """Perform basic text analysis."""
        analysis = {
            'word_count': len(text.split()),
            'sentence_count': len(text.split('.')),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
            'language_detected': language.value,
            'complexity_score': self._calculate_complexity_score(text)
        }
        
        return analysis
    
    def _calculate_complexity_score(self, text: str) -> float:
        """Calculate text complexity score."""
        words = text.split()
        if not words:
            return 0.0
        
        # Factors: average word length, sentence length, vocabulary diversity
        avg_word_length = np.mean([len(word) for word in words])
        sentences = text.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        vocabulary_diversity = len(set(words)) / len(words)
        
        complexity = (avg_word_length * 0.3 + avg_sentence_length * 0.4 + vocabulary_diversity * 0.3)
        return min(1.0, complexity / 10.0)
    
    async def analyze_sentiment(self, document_id: str) -> str:
        """Analyze sentiment of a text document."""
        if document_id not in self.text_documents:
            raise ValueError(f"Document ID {document_id} not found")
        
        document = self.text_documents[document_id]
        
        analysis_id = f"sentiment_{document_id}_{int(time.time())}"
        
        # Perform sentiment analysis
        sentiment_result = await self._perform_sentiment_analysis(document.content)
        
        sentiment_analysis = SentimentAnalysis(
            analysis_id=analysis_id,
            document_id=document_id,
            overall_sentiment=sentiment_result['overall_sentiment'],
            sentiment_scores=sentiment_result['sentiment_scores'],
            aspect_sentiments=sentiment_result['aspect_sentiments'],
            confidence=sentiment_result['confidence'],
            keywords=sentiment_result['keywords']
        )
        
        with self.lock:
            self.sentiment_analyses[analysis_id] = sentiment_analysis
            self.metrics['total_sentiment_analyses'] += 1
            self.metrics['avg_sentiment_confidence'] = (
                (self.metrics['avg_sentiment_confidence'] * (self.metrics['total_sentiment_analyses'] - 1) + 
                 sentiment_result['confidence']) / self.metrics['total_sentiment_analyses']
            )
        
        self.logger.info(f"Analyzed sentiment: {analysis_id}")
        return analysis_id
    
    async def _perform_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Perform sentiment analysis on text."""
        # Simple rule-based sentiment analysis
        words = text.lower().split()
        
        positive_count = sum(1 for word in words if word in self.sentiment_keywords[SentimentType.POSITIVE])
        negative_count = sum(1 for word in words if word in self.sentiment_keywords[SentimentType.NEGATIVE])
        neutral_count = sum(1 for word in words if word in self.sentiment_keywords[SentimentType.NEUTRAL])
        
        total_sentiment_words = positive_count + negative_count + neutral_count
        
        if total_sentiment_words == 0:
            overall_sentiment = SentimentType.NEUTRAL
            confidence = 0.5
        else:
            positive_ratio = positive_count / total_sentiment_words
            negative_ratio = negative_count / total_sentiment_words
            
            if positive_ratio > 0.6:
                overall_sentiment = SentimentType.POSITIVE
                confidence = positive_ratio
            elif negative_ratio > 0.6:
                overall_sentiment = SentimentType.NEGATIVE
                confidence = negative_ratio
            else:
                overall_sentiment = SentimentType.NEUTRAL
                confidence = 0.7
        
        sentiment_scores = {
            'positive': positive_count / max(1, total_sentiment_words),
            'negative': negative_count / max(1, total_sentiment_words),
            'neutral': neutral_count / max(1, total_sentiment_words)
        }
        
        # Extract keywords
        keywords = [word for word in words if word in 
                   self.sentiment_keywords[SentimentType.POSITIVE] + 
                   self.sentiment_keywords[SentimentType.NEGATIVE]]
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_scores': sentiment_scores,
            'aspect_sentiments': {},  # Simplified
            'confidence': confidence,
            'keywords': keywords[:10]  # Top 10 keywords
        }
    
    async def extract_named_entities(self, document_id: str) -> List[str]:
        """Extract named entities from a text document."""
        if document_id not in self.text_documents:
            raise ValueError(f"Document ID {document_id} not found")
        
        document = self.text_documents[document_id]
        
        # Extract entities
        entities = await self._extract_entities(document.content)
        
        entity_ids = []
        for entity_data in entities:
            entity_id = f"entity_{document_id}_{len(entity_ids)}_{int(time.time())}"
            
            named_entity = NamedEntity(
                entity_id=entity_id,
                text=entity_data['text'],
                entity_type=entity_data['type'],
                start_position=entity_data['start'],
                end_position=entity_data['end'],
                confidence=entity_data['confidence']
            )
            
            with self.lock:
                self.named_entities[entity_id] = named_entity
                entity_ids.append(entity_id)
            
            self.metrics['total_entity_extractions'] += 1
        
        self.logger.info(f"Extracted {len(entity_ids)} entities from document: {document_id}")
        return entity_ids
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        entities = []
        
        # Simple pattern-based entity extraction
        patterns = {
            EntityType.PERSON: r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            EntityType.ORGANIZATION: r'\b[A-Z][a-z]+ (Inc|Corp|LLC|Ltd|Company)\b',
            EntityType.LOCATION: r'\b[A-Z][a-z]+, [A-Z]{2}\b',
            EntityType.DATE: r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            EntityType.MONEY: r'\$\d+(\.\d{2})?\b',
            EntityType.PERCENT: r'\d+%\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    'text': match.group(),
                    'type': entity_type,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8
                })
        
        return entities
    
    async def generate_text(self, prompt: str,
                          language: Language = Language.ENGLISH,
                          style: str = "conversational",
                          max_length: int = 100) -> str:
        """Generate text based on prompt."""
        generation_id = f"generation_{int(time.time())}"
        
        # Generate text
        generated_text = await self._generate_text_content(prompt, language, style, max_length)
        
        # Calculate quality score
        quality_score = self._calculate_text_quality(generated_text, prompt)
        
        text_generation = TextGeneration(
            generation_id=generation_id,
            prompt=prompt,
            generated_text=generated_text,
            language=language,
            style=style,
            length=len(generated_text),
            quality_score=quality_score
        )
        
        with self.lock:
            self.text_generations[generation_id] = text_generation
            self.metrics['total_text_generations'] += 1
        
        self.logger.info(f"Generated text: {generation_id}")
        return generation_id
    
    async def _generate_text_content(self, prompt: str,
                                   language: Language,
                                   style: str,
                                   max_length: int) -> str:
        """Generate text content."""
        # Simple template-based text generation
        templates = {
            'conversational': [
                "I understand that {prompt}. Let me help you with that.",
                "Regarding {prompt}, I think it's an interesting topic.",
                "When it comes to {prompt}, there are several aspects to consider."
            ],
            'formal': [
                "The subject of {prompt} requires careful consideration.",
                "Analysis of {prompt} reveals important insights.",
                "The examination of {prompt} demonstrates significant findings."
            ],
            'creative': [
                "Imagine a world where {prompt} becomes reality.",
                "In the realm of {prompt}, possibilities are endless.",
                "The story of {prompt} unfolds in unexpected ways."
            ]
        }
        
        template_list = templates.get(style, templates['conversational'])
        template = random.choice(template_list)
        
        generated_text = template.format(prompt=prompt)
        
        # Add more content if needed
        if len(generated_text) < max_length:
            additional_content = f" This is an important consideration that deserves attention."
            generated_text += additional_content
        
        return generated_text[:max_length]
    
    def _calculate_text_quality(self, generated_text: str, prompt: str) -> float:
        """Calculate text generation quality score."""
        # Simple quality metrics
        relevance_score = 0.8  # Simplified
        coherence_score = 0.9  # Simplified
        grammar_score = 0.95   # Simplified
        
        quality = (relevance_score + coherence_score + grammar_score) / 3
        return quality
    
    async def create_conversation(self, participants: List[str],
                                language: Language = Language.ENGLISH) -> str:
        """Create a new conversation."""
        conversation_id = f"conversation_{int(time.time())}"
        
        conversation = Conversation(
            conversation_id=conversation_id,
            participants=participants,
            messages=[],
            context={},
            sentiment_trend=[],
            language=language
        )
        
        with self.lock:
            self.conversations[conversation_id] = conversation
        
        self.logger.info(f"Created conversation: {conversation_id}")
        return conversation_id
    
    async def add_message_to_conversation(self, conversation_id: str,
                                        sender: str,
                                        message: str) -> str:
        """Add a message to a conversation."""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation ID {conversation_id} not found")
        
        conversation = self.conversations[conversation_id]
        
        message_id = f"message_{conversation_id}_{len(conversation.messages)}_{int(time.time())}"
        
        # Analyze message sentiment
        sentiment_analysis = await self._perform_sentiment_analysis(message)
        
        message_data = {
            'message_id': message_id,
            'sender': sender,
            'content': message,
            'timestamp': datetime.now().isoformat(),
            'sentiment': sentiment_analysis['overall_sentiment'].value,
            'sentiment_confidence': sentiment_analysis['confidence']
        }
        
        with self.lock:
            conversation.messages.append(message_data)
            conversation.sentiment_trend.append(sentiment_analysis['overall_sentiment'])
        
        self.logger.info(f"Added message to conversation: {message_id}")
        return message_id
    
    async def translate_text(self, text: str,
                           source_language: Language,
                           target_language: Language) -> str:
        """Translate text between languages."""
        translation_id = f"translation_{int(time.time())}"
        
        # Perform translation
        translated_text = await self._translate_text_content(text, source_language, target_language)
        
        # Calculate confidence
        confidence = self._calculate_translation_confidence(text, translated_text, source_language, target_language)
        
        # Generate alternatives
        alternatives = await self._generate_translation_alternatives(text, source_language, target_language)
        
        translation = Translation(
            translation_id=translation_id,
            source_text=text,
            target_text=translated_text,
            source_language=source_language,
            target_language=target_language,
            confidence=confidence,
            alternatives=alternatives
        )
        
        with self.lock:
            self.translations[translation_id] = translation
            self.metrics['total_translations'] += 1
            self.metrics['avg_translation_confidence'] = (
                (self.metrics['avg_translation_confidence'] * (self.metrics['total_translations'] - 1) + confidence) /
                self.metrics['total_translations']
            )
        
        self.logger.info(f"Translated text: {translation_id}")
        return translation_id
    
    async def _translate_text_content(self, text: str,
                                    source_language: Language,
                                    target_language: Language) -> str:
        """Translate text content."""
        # Simple dictionary-based translation (simplified)
        translation_dicts = {
            (Language.ENGLISH, Language.SPANISH): {
                'hello': 'hola',
                'goodbye': 'adiós',
                'thank you': 'gracias',
                'please': 'por favor',
                'yes': 'sí',
                'no': 'no'
            },
            (Language.ENGLISH, Language.FRENCH): {
                'hello': 'bonjour',
                'goodbye': 'au revoir',
                'thank you': 'merci',
                'please': 's\'il vous plaît',
                'yes': 'oui',
                'no': 'non'
            }
        }
        
        translation_dict = translation_dicts.get((source_language, target_language), {})
        
        translated_text = text
        for source_word, target_word in translation_dict.items():
            translated_text = translated_text.replace(source_word.lower(), target_word)
        
        return translated_text
    
    def _calculate_translation_confidence(self, source_text: str,
                                        target_text: str,
                                        source_language: Language,
                                        target_language: Language) -> float:
        """Calculate translation confidence."""
        # Simple confidence calculation
        if source_language == target_language:
            return 1.0
        
        # Check if translation changed the text
        if source_text.lower() == target_text.lower():
            return 0.3  # Low confidence if no change
        
        # Higher confidence for known language pairs
        known_pairs = [
            (Language.ENGLISH, Language.SPANISH),
            (Language.ENGLISH, Language.FRENCH),
            (Language.ENGLISH, Language.GERMAN)
        ]
        
        if (source_language, target_language) in known_pairs:
            return 0.8
        else:
            return 0.5
    
    async def _generate_translation_alternatives(self, text: str,
                                               source_language: Language,
                                               target_language: Language) -> List[str]:
        """Generate alternative translations."""
        # Simplified alternative generation
        alternatives = []
        
        # Add slight variations
        base_translation = await self._translate_text_content(text, source_language, target_language)
        alternatives.append(base_translation)
        
        # Add formal/informal variations
        if target_language == Language.SPANISH:
            alternatives.append(base_translation.replace('tú', 'usted'))
        
        return alternatives[:3]  # Return top 3 alternatives
    
    def get_sentiment_analysis_result(self, analysis_id: str) -> Dict[str, Any]:
        """Get sentiment analysis result."""
        if analysis_id not in self.sentiment_analyses:
            return {}
        
        analysis = self.sentiment_analyses[analysis_id]
        
        return {
            'analysis_id': analysis_id,
            'document_id': analysis.document_id,
            'overall_sentiment': analysis.overall_sentiment.value,
            'sentiment_scores': analysis.sentiment_scores,
            'confidence': analysis.confidence,
            'keywords': analysis.keywords
        }
    
    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation summary."""
        if conversation_id not in self.conversations:
            return {}
        
        conversation = self.conversations[conversation_id]
        
        # Calculate conversation metrics
        total_messages = len(conversation.messages)
        participants_count = len(conversation.participants)
        
        # Analyze sentiment trend
        if conversation.sentiment_trend:
            recent_sentiment = conversation.sentiment_trend[-1].value
            sentiment_stability = len(set(conversation.sentiment_trend)) == 1
        else:
            recent_sentiment = 'neutral'
            sentiment_stability = True
        
        return {
            'conversation_id': conversation_id,
            'participants': conversation.participants,
            'total_messages': total_messages,
            'language': conversation.language.value,
            'recent_sentiment': recent_sentiment,
            'sentiment_stability': sentiment_stability,
            'duration': (datetime.now() - conversation.messages[0]['timestamp']).seconds if conversation.messages else 0
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get advanced NLP system status."""
        with self.lock:
            return {
                'metrics': self.metrics.copy(),
                'total_text_documents': len(self.text_documents),
                'total_sentiment_analyses': len(self.sentiment_analyses),
                'total_named_entities': len(self.named_entities),
                'total_text_generations': len(self.text_generations),
                'total_conversations': len(self.conversations),
                'total_translations': len(self.translations),
                'supported_languages': [lang.value for lang in Language],
                'pandas_available': self.pandas_available,
                'sklearn_available': self.sklearn_available,
                'nltk_available': self.nltk_available
            }


# Global instance
advanced_nlp_engine = AdvancedNLPEngine() 