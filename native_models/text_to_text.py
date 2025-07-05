"""
text_to_text.py – Text transformation utilities for Unimind native models.
Provides functions for text processing, transformation, and manipulation.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class TransformationType(Enum):
    """Types of text transformations."""
    CASE = "case"
    FORMAT = "format"
    STYLE = "style"
    TRANSLATE = "translate"
    SUMMARIZE = "summarize"
    PARAPHRASE = "paraphrase"
    EXTRACT = "extract"
    CLEAN = "clean"

class TextStyle(Enum):
    """Text style options."""
    FORMAL = "formal"
    INFORMAL = "informal"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    ACADEMIC = "academic"
    CASUAL = "casual"

@dataclass
class TransformationResult:
    """Result of text transformation."""
    original_text: str
    transformed_text: str
    transformation_type: TransformationType
    confidence: float
    metadata: Dict[str, Any]

class TextToText:
    """
    Transforms and manipulates text content.
    Provides various text processing, transformation, and analysis capabilities.
    """
    
    def __init__(self):
        """Initialize the TextToText transformer."""
        self.transformation_patterns = {
            TransformationType.CASE: {
                "uppercase": str.upper,
                "lowercase": str.lower,
                "title": str.title,
                "capitalize": str.capitalize,
                "camelcase": self._to_camel_case,
                "snake_case": self._to_snake_case,
                "kebab-case": self._to_kebab_case
            },
            TransformationType.FORMAT: {
                "trim": str.strip,
                "normalize_whitespace": self._normalize_whitespace,
                "remove_extra_spaces": self._remove_extra_spaces,
                "normalize_line_breaks": self._normalize_line_breaks
            },
            TransformationType.STYLE: {
                "formal": self._make_formal,
                "informal": self._make_informal,
                "technical": self._make_technical,
                "creative": self._make_creative
            }
        }
        
        self.cleaning_patterns = {
            "html_tags": r'<[^>]+>',
            "urls": r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            "emails": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone_numbers": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "special_chars": r'[^\w\s]'
        }
        
    def transform_text(self, input_text: str, transformation_type: str = "default", style: TextStyle = TextStyle.FORMAL) -> TransformationResult:
        """
        Transform input text based on the specified transformation type.
        
        Args:
            input_text: The text to transform
            transformation_type: Type of transformation to apply
            style: Text style to apply
            
        Returns:
            TransformationResult containing the transformation results
        """
        if not input_text:
            return TransformationResult(
                original_text="",
                transformed_text="",
                transformation_type=TransformationType.FORMAT,
                confidence=0.0,
                metadata={"error": "Empty input text"}
            )
        
        # Apply transformation
        if transformation_type in self.transformation_patterns[TransformationType.CASE]:
            transformed_text = self.transformation_patterns[TransformationType.CASE][transformation_type](input_text)
            trans_type = TransformationType.CASE
        elif transformation_type in self.transformation_patterns[TransformationType.FORMAT]:
            transformed_text = self.transformation_patterns[TransformationType.FORMAT][transformation_type](input_text)
            trans_type = TransformationType.FORMAT
        elif transformation_type in self.transformation_patterns[TransformationType.STYLE]:
            transformed_text = self.transformation_patterns[TransformationType.STYLE][transformation_type](input_text, style)
            trans_type = TransformationType.STYLE
        elif transformation_type == "summarize":
            transformed_text = self._summarize_text(input_text)
            trans_type = TransformationType.SUMMARIZE
        elif transformation_type == "paraphrase":
            transformed_text = self._paraphrase_text(input_text)
            trans_type = TransformationType.PARAPHRASE
        elif transformation_type == "extract_keywords":
            transformed_text = self._extract_keywords(input_text)
            trans_type = TransformationType.EXTRACT
        elif transformation_type == "clean":
            transformed_text = self._clean_text(input_text)
            trans_type = TransformationType.CLEAN
        else:
            # Default transformation - just clean up whitespace
            transformed_text = input_text.strip()
            trans_type = TransformationType.FORMAT
        
        # Calculate confidence
        confidence = self._calculate_confidence(input_text, transformed_text, transformation_type)
        
        return TransformationResult(
            original_text=input_text,
            transformed_text=transformed_text,
            transformation_type=trans_type,
            confidence=confidence,
            metadata={
                "transformation_type": transformation_type,
                "style": style.value,
                "original_length": len(input_text),
                "transformed_length": len(transformed_text)
            }
        )
    
    def _to_camel_case(self, text: str) -> str:
        """Convert text to camelCase."""
        words = text.split()
        if not words:
            return text
        return words[0].lower() + ''.join(word.capitalize() for word in words[1:])
    
    def _to_snake_case(self, text: str) -> str:
        """Convert text to snake_case."""
        return '_'.join(word.lower() for word in text.split())
    
    def _to_kebab_case(self, text: str) -> str:
        """Convert text to kebab-case."""
        return '-'.join(word.lower() for word in text.split())
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        return ' '.join(text.split())
    
    def _remove_extra_spaces(self, text: str) -> str:
        """Remove extra spaces from text."""
        return re.sub(r'\s+', ' ', text).strip()
    
    def _normalize_line_breaks(self, text: str) -> str:
        """Normalize line breaks in text."""
        return re.sub(r'\r\n|\r|\n', '\n', text)
    
    def _make_formal(self, text: str, style: TextStyle) -> str:
        """Make text more formal."""
        # Simple formalization rules
        informal_to_formal = {
            "hey": "hello",
            "hi": "hello",
            "gonna": "going to",
            "wanna": "want to",
            "gotta": "got to",
            "cool": "excellent",
            "awesome": "remarkable"
        }
        
        result = text
        for informal, formal in informal_to_formal.items():
            result = re.sub(r'\b' + re.escape(informal) + r'\b', formal, result, flags=re.IGNORECASE)
        
        return result
    
    def _make_informal(self, text: str, style: TextStyle) -> str:
        """Make text more informal."""
        # Simple informalization rules
        formal_to_informal = {
            "hello": "hey",
            "excellent": "cool",
            "remarkable": "awesome",
            "therefore": "so",
            "furthermore": "also",
            "consequently": "so"
        }
        
        result = text
        for formal, informal in formal_to_informal.items():
            result = re.sub(r'\b' + re.escape(formal) + r'\b', informal, result, flags=re.IGNORECASE)
        
        return result
    
    def _make_technical(self, text: str, style: TextStyle) -> str:
        """Make text more technical."""
        # Add technical terminology
        technical_terms = {
            "thing": "entity",
            "stuff": "materials",
            "work": "function",
            "use": "utilize",
            "make": "generate",
            "get": "retrieve"
        }
        
        result = text
        for simple, technical in technical_terms.items():
            result = re.sub(r'\b' + re.escape(simple) + r'\b', technical, result, flags=re.IGNORECASE)
        
        return result
    
    def _make_creative(self, text: str, style: TextStyle) -> str:
        """Make text more creative."""
        # Add creative elements
        creative_elements = {
            "good": "fantastic",
            "bad": "terrible",
            "big": "enormous",
            "small": "tiny",
            "fast": "lightning-fast",
            "slow": "snail-paced"
        }
        
        result = text
        for plain, creative in creative_elements.items():
            result = re.sub(r'\b' + re.escape(plain) + r'\b', creative, result, flags=re.IGNORECASE)
        
        return result
    
    def _summarize_text(self, text: str) -> str:
        """Summarize text (placeholder implementation)."""
        sentences = text.split('.')
        if len(sentences) <= 2:
            return text
        
        # Simple summarization: take first and last sentence
        summary_sentences = [sentences[0]]
        if len(sentences) > 2:
            summary_sentences.append(sentences[-1])
        
        return '. '.join(summary_sentences) + '.'
    
    def _paraphrase_text(self, text: str) -> str:
        """Paraphrase text (placeholder implementation)."""
        # Simple paraphrasing: replace some words with synonyms
        synonyms = {
            "good": "excellent",
            "bad": "poor",
            "big": "large",
            "small": "little",
            "fast": "quick",
            "slow": "gradual"
        }
        
        result = text
        for word, synonym in synonyms.items():
            result = re.sub(r'\b' + re.escape(word) + r'\b', synonym, result, flags=re.IGNORECASE)
        
        return result
    
    def _extract_keywords(self, text: str) -> str:
        """Extract keywords from text."""
        # Simple keyword extraction: find capitalized words and common nouns
        keywords = re.findall(r'\b[A-Z][a-z]+\b|\b\w{4,}\b', text)
        # Remove common words
        common_words = {"this", "that", "with", "have", "will", "from", "they", "know", "want", "been", "good", "much", "some", "time", "very", "when", "come", "just", "into", "than", "more", "other", "about", "many", "then", "them", "these", "people", "only", "would", "could", "there", "their", "what", "said", "each", "which", "she", "do", "how", "if", "go", "me", "my", "up", "her", "we", "so", "no", "he", "or", "an", "as", "be", "it", "by", "at", "am", "is", "are", "was", "were", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "can", "shall"}
        keywords = [word for word in keywords if word.lower() not in common_words]
        return ', '.join(set(keywords))  # Remove duplicates
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing unwanted elements."""
        cleaned_text = text
        
        # Remove HTML tags
        cleaned_text = re.sub(self.cleaning_patterns["html_tags"], '', cleaned_text)
        
        # Remove URLs
        cleaned_text = re.sub(self.cleaning_patterns["urls"], '[URL]', cleaned_text)
        
        # Remove emails
        cleaned_text = re.sub(self.cleaning_patterns["emails"], '[EMAIL]', cleaned_text)
        
        # Remove phone numbers
        cleaned_text = re.sub(self.cleaning_patterns["phone_numbers"], '[PHONE]', cleaned_text)
        
        # Normalize whitespace
        cleaned_text = self._normalize_whitespace(cleaned_text)
        
        return cleaned_text
    
    def _calculate_confidence(self, original_text: str, transformed_text: str, transformation_type: str) -> float:
        """Calculate confidence score for the transformation."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence for successful transformations
        if transformed_text != original_text:
            confidence += 0.3
        
        # Boost confidence for length-appropriate transformations
        if transformation_type == "summarize" and len(transformed_text) < len(original_text):
            confidence += 0.2
        elif transformation_type == "paraphrase" and len(transformed_text) > 0:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text properties.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing text analysis
        """
        words = text.split()
        sentences = text.split('.')
        
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "character_count": len(text),
            "average_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "average_sentence_length": len(words) / len(sentences) if sentences else 0,
            "unique_words": len(set(words)),
            "vocabulary_diversity": len(set(words)) / len(words) if words else 0,
            "readability_score": self._calculate_readability(text)
        }
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate simple readability score."""
        words = text.split()
        sentences = text.split('.')
        
        if not words or not sentences:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple Flesch Reading Ease approximation
        readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
        return max(0.0, min(100.0, readability))

# Module-level instance
text_to_text = TextToText()

# Export the engine instance with the expected name
text_to_text_engine = text_to_text

def transform_text(input_text: str, transformation_type: str = "default", style: TextStyle = TextStyle.FORMAL) -> TransformationResult:
    """Transform text using the module-level instance."""
    return text_to_text.transform_text(input_text, transformation_type, style)

def analyze_text(text: str) -> Dict[str, Any]:
    """Analyze text using the module-level instance."""
    return text_to_text.analyze_text(text)
