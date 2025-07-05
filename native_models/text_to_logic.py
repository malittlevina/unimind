"""
text_to_logic.py – Logic analysis utilities for Unimind native models.
Provides functions for analyzing syntax, interpreting meaning, and visualizing concepts.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class LogicType(Enum):
    """Types of logical analysis."""
    SYNTACTIC = "syntactic"
    SEMANTIC = "semantic"
    CONCEPTUAL = "conceptual"
    INFERENTIAL = "inferential"

class ComplexityLevel(Enum):
    """Text complexity levels."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    EXPERT = "expert"

@dataclass
class LogicResult:
    """Result of logic analysis."""
    logic_type: LogicType
    confidence: float
    analysis: Dict[str, Any]
    concepts: List[str]
    relationships: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class TextToLogic:
    """
    Analyzes text for logical structure, meaning, and conceptual relationships.
    Provides syntactic analysis, semantic interpretation, and concept visualization.
    """
    
    def __init__(self):
        """Initialize the TextToLogic analyzer."""
        self.logic_connectors = {
            "conjunction": ["and", "also", "moreover", "furthermore", "in addition"],
            "disjunction": ["or", "either", "neither", "nor"],
            "implication": ["if", "then", "implies", "leads to", "results in"],
            "negation": ["not", "no", "never", "none", "neither"],
            "causation": ["because", "since", "as", "due to", "caused by"],
            "comparison": ["like", "similar", "different", "compared to", "versus"]
        }
        
        self.complexity_indicators = {
            "simple": ["basic", "simple", "easy", "clear", "obvious"],
            "medium": ["moderate", "standard", "normal", "typical"],
            "complex": ["complex", "complicated", "advanced", "sophisticated"],
            "expert": ["expert", "specialized", "technical", "professional", "academic"]
        }
        
        self.concept_patterns = {
            "entities": r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",
            "actions": r"\b\w+ing\b|\b\w+ed\b|\b\w+s\b",
            "properties": r"\b\w+ful\b|\b\w+less\b|\b\w+able\b|\b\w+ible\b",
            "quantities": r"\b\d+\b|\b\w+%\b|\b\w+th\b"
        }
        
    def analyze_syntax(self, text: str) -> LogicResult:
        """
        Analyze the syntactic structure of input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            LogicResult containing syntactic analysis
        """
        sentences = self._split_sentences(text)
        words = self._tokenize(text)
        
        # Analyze sentence structure
        sentence_analysis = []
        for sentence in sentences:
            sentence_analysis.append({
                "sentence": sentence,
                "word_count": len(sentence.split()),
                "has_subject": self._has_subject(sentence),
                "has_verb": self._has_verb(sentence),
                "has_object": self._has_object(sentence),
                "structure": self._determine_structure(sentence)
            })
        
        # Calculate complexity
        complexity = self._assess_complexity(text)
        
        # Extract logical connectors
        connectors = self._find_connectors(text)
        
        analysis = {
            "sentence_count": len(sentences),
            "word_count": len(words),
            "average_sentence_length": len(words) / len(sentences) if sentences else 0,
            "complexity": complexity.value,
            "sentence_analysis": sentence_analysis,
            "logical_connectors": connectors,
            "syntactic_patterns": self._find_syntactic_patterns(text)
        }
        
        return LogicResult(
            logic_type=LogicType.SYNTACTIC,
            confidence=self._calculate_syntax_confidence(analysis),
            analysis=analysis,
            concepts=self._extract_concepts(text),
            relationships=self._find_syntactic_relationships(text),
            metadata={"text_length": len(text), "unique_words": len(set(words))}
        )
    
    def interpret_meaning(self, text: str) -> LogicResult:
        """
        Interpret the semantic meaning of input text.
        
        Args:
            text: Input text to interpret
            
        Returns:
            LogicResult containing semantic analysis
        """
        # Analyze sentiment and tone
        sentiment = self._analyze_sentiment(text)
        tone = self._analyze_tone(text)
        
        # Extract topics and themes
        topics = self._extract_topics(text)
        themes = self._identify_themes(text)
        
        # Analyze intent
        intent = self._determine_intent(text)
        
        # Find semantic relationships
        relationships = self._find_semantic_relationships(text)
        
        analysis = {
            "sentiment": sentiment,
            "tone": tone,
            "topics": topics,
            "themes": themes,
            "intent": intent,
            "key_concepts": self._extract_key_concepts(text),
            "semantic_roles": self._extract_semantic_roles(text)
        }
        
        return LogicResult(
            logic_type=LogicType.SEMANTIC,
            confidence=self._calculate_semantic_confidence(analysis),
            analysis=analysis,
            concepts=self._extract_concepts(text),
            relationships=relationships,
            metadata={"text_length": len(text), "complexity": self._assess_complexity(text).value}
        )
    
    def visualize_concepts(self, meaning: Dict[str, Any]) -> LogicResult:
        """
        Generate visual concepts from meaning interpretation.
        
        Args:
            meaning: Meaning dictionary from interpret_meaning
            
        Returns:
            LogicResult containing visual concept descriptions
        """
        # Extract visual elements from meaning
        topics = meaning.get("topics", [])
        themes = meaning.get("themes", [])
        sentiment = meaning.get("sentiment", "neutral")
        
        # Generate visual concepts
        colors = self._generate_colors(sentiment, themes)
        shapes = self._generate_shapes(topics)
        motion = self._determine_motion(sentiment, topics)
        style = self._determine_style(themes, sentiment)
        
        # Create visual relationships
        visual_relationships = self._create_visual_relationships(topics, themes)
        
        analysis = {
            "colors": colors,
            "shapes": shapes,
            "motion": motion,
            "style": style,
            "composition": self._suggest_composition(topics, themes),
            "visual_hierarchy": self._create_visual_hierarchy(topics)
        }
        
        return LogicResult(
            logic_type=LogicType.CONCEPTUAL,
            confidence=0.7,
            analysis=analysis,
            concepts=topics + themes,
            relationships=visual_relationships,
            metadata={"visual_elements": len(colors) + len(shapes)}
        )
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        return re.split(r'[.!?]+', text.strip())
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _has_subject(self, sentence: str) -> bool:
        """Check if sentence has a subject."""
        # Simple heuristic: look for capitalized words or pronouns
        return bool(re.search(r'\b[A-Z][a-z]+\b|\b(I|you|he|she|it|we|they)\b', sentence))
    
    def _has_verb(self, sentence: str) -> bool:
        """Check if sentence has a verb."""
        # Simple heuristic: look for common verb patterns
        return bool(re.search(r'\b\w+ing\b|\b\w+ed\b|\b\w+s\b|\b(is|are|was|were|have|has|had)\b', sentence))
    
    def _has_object(self, sentence: str) -> bool:
        """Check if sentence has an object."""
        # Simple heuristic: look for prepositions or direct objects
        return bool(re.search(r'\b(to|for|with|by|in|on|at)\b', sentence))
    
    def _determine_structure(self, sentence: str) -> str:
        """Determine sentence structure."""
        if self._has_subject(sentence) and self._has_verb(sentence) and self._has_object(sentence):
            return "SVO"
        elif self._has_subject(sentence) and self._has_verb(sentence):
            return "SV"
        else:
            return "fragment"
    
    def _assess_complexity(self, text: str) -> ComplexityLevel:
        """Assess text complexity."""
        words = self._tokenize(text)
        sentences = self._split_sentences(text)
        
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        unique_word_ratio = len(set(words)) / len(words) if words else 0
        
        if avg_sentence_length > 20 or unique_word_ratio > 0.8:
            return ComplexityLevel.EXPERT
        elif avg_sentence_length > 15 or unique_word_ratio > 0.6:
            return ComplexityLevel.COMPLEX
        elif avg_sentence_length > 10 or unique_word_ratio > 0.4:
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.SIMPLE
    
    def _find_connectors(self, text: str) -> Dict[str, List[str]]:
        """Find logical connectors in text."""
        found_connectors = {}
        text_lower = text.lower()
        
        for connector_type, connectors in self.logic_connectors.items():
            found = [connector for connector in connectors if connector in text_lower]
            if found:
                found_connectors[connector_type] = found
        
        return found_connectors
    
    def _find_syntactic_patterns(self, text: str) -> List[str]:
        """Find syntactic patterns in text."""
        patterns = []
        
        # Look for common patterns
        if re.search(r'\bif\s+\w+.*\bthen\b', text, re.IGNORECASE):
            patterns.append("conditional")
        if re.search(r'\bbecause\b.*\bso\b', text, re.IGNORECASE):
            patterns.append("causal")
        if re.search(r'\bon\s+one\s+hand.*\bon\s+the\s+other\s+hand\b', text, re.IGNORECASE):
            patterns.append("contrast")
        
        return patterns
    
    def _calculate_syntax_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence for syntactic analysis."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence for clear sentence structure
        if analysis.get("sentence_count", 0) > 0:
            confidence += 0.2
        
        # Boost confidence for logical connectors
        if analysis.get("logical_connectors"):
            confidence += 0.2
        
        # Boost confidence for syntactic patterns
        if analysis.get("syntactic_patterns"):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract concepts from text."""
        concepts = []
        
        for pattern_name, pattern in self.concept_patterns.items():
            matches = re.findall(pattern, text)
            concepts.extend(matches)
        
        return list(set(concepts))  # Remove duplicates
    
    def _find_syntactic_relationships(self, text: str) -> List[Dict[str, Any]]:
        """Find syntactic relationships in text."""
        relationships = []
        
        # Look for subject-verb-object relationships
        sentences = self._split_sentences(text)
        for sentence in sentences:
            if self._has_subject(sentence) and self._has_verb(sentence):
                relationships.append({
                    "type": "svo",
                    "sentence": sentence,
                    "confidence": 0.7
                })
        
        return relationships
    
    def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of text."""
        positive_words = ["good", "great", "excellent", "wonderful", "amazing", "happy", "love"]
        negative_words = ["bad", "terrible", "awful", "horrible", "sad", "hate", "angry"]
        
        words = self._tokenize(text)
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _analyze_tone(self, text: str) -> str:
        """Analyze tone of text."""
        formal_words = ["therefore", "furthermore", "consequently", "moreover"]
        informal_words = ["hey", "cool", "awesome", "gonna", "wanna"]
        
        text_lower = text.lower()
        formal_count = sum(1 for word in formal_words if word in text_lower)
        informal_count = sum(1 for word in informal_words if word in text_lower)
        
        if formal_count > informal_count:
            return "formal"
        elif informal_count > formal_count:
            return "informal"
        else:
            return "neutral"
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text."""
        # Simple topic extraction based on noun phrases
        topics = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return list(set(topics))
    
    def _identify_themes(self, text: str) -> List[str]:
        """Identify themes in text."""
        # Placeholder theme identification
        return ["general", "information"]
    
    def _determine_intent(self, text: str) -> str:
        """Determine intent of text."""
        if "?" in text:
            return "question"
        elif any(word in text.lower() for word in ["must", "should", "need to"]):
            return "command"
        elif any(word in text.lower() for word in ["because", "since", "as"]):
            return "explanation"
        else:
            return "informative"
    
    def _find_semantic_relationships(self, text: str) -> List[Dict[str, Any]]:
        """Find semantic relationships in text."""
        relationships = []
        
        # Look for cause-effect relationships
        if re.search(r'\bbecause\b', text, re.IGNORECASE):
            relationships.append({
                "type": "cause_effect",
                "confidence": 0.8
            })
        
        return relationships
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text."""
        return self._extract_concepts(text)[:5]  # Top 5 concepts
    
    def _extract_semantic_roles(self, text: str) -> Dict[str, List[str]]:
        """Extract semantic roles from text."""
        return {
            "agents": [],
            "patients": [],
            "instruments": [],
            "locations": []
        }
    
    def _calculate_semantic_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence for semantic analysis."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence for clear topics
        if analysis.get("topics"):
            confidence += 0.2
        
        # Boost confidence for clear intent
        if analysis.get("intent") != "informative":
            confidence += 0.2
        
        # Boost confidence for sentiment analysis
        if analysis.get("sentiment") != "neutral":
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_colors(self, sentiment: str, themes: List[str]) -> List[str]:
        """Generate colors based on sentiment and themes."""
        color_mappings = {
            "positive": ["blue", "green", "yellow"],
            "negative": ["red", "gray", "black"],
            "neutral": ["white", "gray", "blue"]
        }
        return color_mappings.get(sentiment, ["blue", "green"])
    
    def _generate_shapes(self, topics: List[str]) -> List[str]:
        """Generate shapes based on topics."""
        return ["circle", "square", "triangle"]
    
    def _determine_motion(self, sentiment: str, topics: List[str]) -> str:
        """Determine motion based on sentiment and topics."""
        if sentiment == "positive":
            return "dynamic"
        elif sentiment == "negative":
            return "static"
        else:
            return "gentle"
    
    def _determine_style(self, themes: List[str], sentiment: str) -> str:
        """Determine visual style."""
        if sentiment == "positive":
            return "bright"
        elif sentiment == "negative":
            return "dark"
        else:
            return "balanced"
    
    def _create_visual_relationships(self, topics: List[str], themes: List[str]) -> List[Dict[str, Any]]:
        """Create visual relationships."""
        return [{"type": "hierarchy", "elements": topics + themes}]
    
    def _suggest_composition(self, topics: List[str], themes: List[str]) -> str:
        """Suggest visual composition."""
        return "balanced"
    
    def _create_visual_hierarchy(self, topics: List[str]) -> List[str]:
        """Create visual hierarchy."""
        return topics[:3]  # Top 3 topics

# Module-level instance
text_to_logic = TextToLogic()

# Export the engine instance with the expected name
text_to_logic_engine = text_to_logic

def analyze_syntax(input_text: str, logic_type: LogicType = LogicType.SYNTACTIC) -> LogicResult:
    """Analyze syntax using the module-level instance."""
    return text_to_logic.analyze_syntax(input_text)

def interpret_meaning(text: str) -> LogicResult:
    """Interpret meaning using the module-level instance."""
    return text_to_logic.interpret_meaning(text)

def visualize_concepts(meaning: Dict[str, Any]) -> LogicResult:
    """Visualize concepts using the module-level instance."""
    return text_to_logic.visualize_concepts(meaning)
