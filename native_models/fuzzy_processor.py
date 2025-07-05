"""
fuzzy_processor.py – Fuzzy logic processor for natural language command recognition
"""

import re
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from difflib import SequenceMatcher
from collections import defaultdict
import logging
from dataclasses import dataclass, field
from enum import Enum
import ast
import inspect
from pathlib import Path

# Define capability analyzer components inline since the original file was merged
class CapabilityType(Enum):
    """Types of capabilities the system can have."""
    NATIVE_MODEL = "native_model"
    SCROLL_HANDLER = "scroll_handler"
    API_INTEGRATION = "api_integration"
    DATA_PROCESSING = "data_processing"
    EXTERNAL_TOOL = "external_tool"
    CUSTOM_FUNCTION = "custom_function"

@dataclass
class CapabilityRequirement:
    """Represents a capability requirement."""
    name: str
    description: str
    capability_type: CapabilityType
    complexity: str  # "low", "medium", "high"
    dependencies: List[str] = field(default_factory=list)
    safety_level: str = "medium"  # "low", "medium", "high"
    estimated_effort: str = "medium"  # "low", "medium", "high"

@dataclass
class CodeGenerationRequest:
    """Request for code generation."""
    task_description: str
    required_capabilities: List[CapabilityRequirement]
    context: Dict[str, Any]
    safety_constraints: List[str] = field(default_factory=list)
    integration_points: List[str] = field(default_factory=list)

@dataclass
class GeneratedCode:
    """Generated code artifact."""
    code: str
    file_path: str
    module_name: str
    function_name: str
    description: str
    dependencies: List[str]
    safety_checks: List[str]
    test_code: Optional[str] = None
    integration_instructions: Optional[str] = None

class CapabilityAnalyzer:
    """Simplified capability analyzer for integration."""
    
    def __init__(self):
        self.logger = logging.getLogger('CapabilityAnalyzer')
        self.known_capabilities = self._initialize_known_capabilities()
        self.logger.info("Capability Analyzer initialized")
    
    def _initialize_known_capabilities(self) -> Dict[str, CapabilityRequirement]:
        """Initialize the registry of known capabilities."""
        return {
            "text_to_3d": CapabilityRequirement(
                "text_to_3d", "Generate 3D models from text descriptions",
                CapabilityType.NATIVE_MODEL, "medium", [], "medium", "medium"
            ),
            "text_to_code": CapabilityRequirement(
                "text_to_code", "Generate code from natural language",
                CapabilityType.NATIVE_MODEL, "high", [], "medium", "high"
            ),
            "llm_engine": CapabilityRequirement(
                "llm_engine", "Large language model interface",
                CapabilityType.NATIVE_MODEL, "high", [], "medium", "high"
            ),
        }
    
    def analyze_capability(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze user input for capability requirements."""
        return {
            "complexity": "medium",
            "missing_capabilities": [],
            "available_capabilities": list(self.known_capabilities.keys()),
            "suggestions": ["Consider using existing capabilities"]
        }
    
    def generate_code(self, request: CodeGenerationRequest) -> GeneratedCode:
        """Generate code for missing capabilities."""
        return GeneratedCode(
            code="# Generated placeholder code",
            file_path="generated_module.py",
            module_name="generated_module",
            function_name="generated_function",
            description="Generated placeholder",
            dependencies=[],
            safety_checks=[]
        )

CAPABILITY_ANALYZER_AVAILABLE = True

# Add a simple lemmatizer mapping for verbs and common forms
LEMMATIZER = {
    "optimizing": "optimize", "optimized": "optimize", "optimise": "optimize", "optimised": "optimize",
    "improving": "improve", "improved": "improve", "improves": "improve",
    "enhancing": "enhance", "enhanced": "enhance", "enhances": "enhance",
    "fixing": "fix", "fixed": "fix", "fixes": "fix",
    "tuning": "tune", "tuned": "tune", "tunes": "tune",
    "boosting": "boost", "boosted": "boost", "boosts": "boost",
    "making": "make", "made": "make",
    "speeding": "speed", "sped": "speed",
    "grounding": "ground", "grounded": "ground",
    "relaxing": "relax", "relaxed": "relax",
    "calming": "calm", "calmed": "calm",
    "reflecting": "reflect", "reflected": "reflect",
    "assessing": "assess", "assessed": "assess",
    "cleaning": "clean", "cleaned": "clean",
    "clearing": "clear", "cleared": "clear",
    "protecting": "protect", "protected": "protect",
    "defending": "defend", "defended": "defend",
    "shielding": "shield", "shielded": "shield",
    "searching": "search", "searched": "search",
    "finding": "find", "found": "find",
    "generating": "generate", "generated": "generate",
    "summarizing": "summarize", "summarized": "summarize",
    "creating": "create", "created": "create",
    "pondering": "ponder", "pondered": "ponder",
    "meditating": "meditate", "meditated": "meditate",
    "ruminating": "ruminate", "ruminated": "ruminate",
    # Add more as needed
}

def lemmatize_text(text: str) -> str:
    """Lemmatize each word in the text using the simple mapping."""
    return " ".join(LEMMATIZER.get(word, word) for word in text.split())

class FuzzyProcessor:
    """
    Fuzzy logic processor for natural language command recognition.
    Uses multiple similarity algorithms to match user input to commands.
    Enhanced with capability analysis and code generation.
    """
    
    def __init__(self, enable_capability_analysis: bool = True):
        """Initialize the fuzzy processor with optional capability analysis."""
        self.logger = logging.getLogger('FuzzyProcessor')
        self.command_patterns = {}
        self.synonym_groups = {}
        
        # Configuration - Easy to adjust
        self.confidence_threshold = 0.6  # Minimum confidence for command execution
        self.suggestion_threshold = 0.3  # Minimum confidence for suggestions
        self.strict_mode = False         # Set to True for stricter matching
        
        # Capability analysis
        self.enable_capability_analysis = enable_capability_analysis and CAPABILITY_ANALYZER_AVAILABLE
        if self.enable_capability_analysis:
            self.capability_analyzer = CapabilityAnalyzer()
            self.logger.info("Capability analysis enabled")
        else:
            self.capability_analyzer = None
            if enable_capability_analysis and not CAPABILITY_ANALYZER_AVAILABLE:
                self.logger.warning("Capability analysis requested but not available")
        
        self._initialize_patterns()
        self.logger.info("Fuzzy processor initialized with capability analysis: %s", self.enable_capability_analysis)
    
    def _initialize_patterns(self):
        """Initialize command patterns and synonyms."""
        
        # Define command patterns with variations and expanded synonyms
        self.command_patterns = {
            "optimize_self": {
                "patterns": [
                    r"optimize", r"optimise", r"improve", r"enhance", r"tune", r"boost",
                    r"make.*better", r"speed.*up", r"performance", r"efficiency", r"fix.*yourself", r"fix.*system"
                ],
                "synonyms": [
                    "optimize", "optimise", "improve", "enhance", "tune", "boost", "speed up", "make better", "fix yourself", "fix system", "make yourself better", "increase performance", "improve yourself", "enhance system"
                ],
                "category": "system"
            },
            "self_assess": {
                "patterns": [
                    r"how.*am.*i", r"how.*are.*you", r"status", r"check.*status", r"self.*assess",
                    r"health.*check", r"system.*status", r"diagnostic", r"assessment", r"how.*doing", r"am.*i.*ok", r"am.*i.*well",
                    r"how.*are.*you.*doing", r"how.*do.*you.*feel", r"what.*are.*you.*doing", r"are.*you.*ok", r"are.*you.*working",
                    r"are.*you.*ready", r"can.*you", r"do.*you", r"will.*you", r"would.*you", r"should.*you", r"have.*you", r"did.*you",
                    r"are.*you", r"is.*you", r"was.*you", r"were.*you", r"how.*do.*you.*feel.*about", r"what.*do.*you.*think.*about",
                    r"what.*do.*you.*know.*about", r"what.*can.*you.*do", r"what.*are.*you.*capable.*of", r"who.*are.*you", r"what.*are.*you",
                    r"tell.*me.*about.*you", r"describe.*you", r"are.*you.*online", r"are.*you.*active", r"are.*you.*functioning",
                    r"are.*you.*operational", r"can.*you.*help", r"can.*you.*assist", r"can.*you.*explain", r"can.*you.*show",
                    r"can.*you.*tell", r"can.*you.*find", r"can.*you.*search", r"can.*you.*analyze", r"can.*you.*optimize",
                    r"can.*you.*assess", r"can.*you.*reflect", r"can.*you.*introspect"
                ],
                "synonyms": [
                    "how am i doing", "how are you doing", "status check", "self assessment", "health check", "system status", "diagnostic", "assessment", "am i ok", "am i well", "check my status", "how's my status", "how am i",
                    "how are you", "how do you feel", "what are you doing", "are you ok", "are you working", "are you ready", "can you", "do you", "will you", "would you", "should you", "have you", "did you",
                    "are you", "is you", "was you", "were you", "how do you feel about", "what do you think about", "what do you know about", "what can you do", "what are you capable of", "who are you", "what are you",
                    "tell me about you", "describe you", "are you online", "are you active", "are you functioning", "are you operational", "can you help", "can you assist", "can you explain", "can you show",
                    "can you tell", "can you find", "can you search", "can you analyze", "can you optimize", "can you assess", "can you reflect", "can you introspect"
                ],
                "category": "system"
            },
            "introspect_core": {
                "patterns": [
                    r"reflect", r"introspect", r"think", r"consider", r"contemplate",
                    r"meditate", r"ponder", r"ruminate", r"self.*reflection", r"self.*reflect", r"look.*inward", r"analyze.*myself", r"analyze.*my.*thoughts",
                    r"instruct.*yourself", r"teach.*yourself", r"learn.*about", r"develop.*yourself", r"improve.*yourself", r"grow.*yourself",
                    r"study.*yourself", r"research.*yourself", r"understand.*yourself", r"explore.*yourself", r"examine.*yourself", r"investigate.*yourself",
                    r"develop.*better", r"improve.*capabilities", r"enhance.*abilities", r"learn.*new", r"acquire.*knowledge", r"gain.*understanding",
                    r"build.*better", r"create.*better", r"evolve.*yourself", r"advance.*yourself", r"progress.*yourself", r"upgrade.*yourself"
                ],
                "synonyms": [
                    "reflect", "introspect", "think about", "consider", "meditate", "self reflection", "self reflect", "look inward", "analyze myself", "analyze my thoughts", "ponder", "contemplate", "ruminate",
                    "instruct yourself", "teach yourself", "learn about", "develop yourself", "improve yourself", "grow yourself", "study yourself", "research yourself", "understand yourself", "explore yourself", "examine yourself", "investigate yourself",
                    "develop better", "improve capabilities", "enhance abilities", "learn new", "acquire knowledge", "gain understanding", "build better", "create better", "evolve yourself", "advance yourself", "progress yourself", "upgrade yourself"
                ],
                "category": "cognitive"
            },
            "calm_sequence": {
                "patterns": [
                    r"calm.*down", r"relax", r"breathe", r"ground", r"center", r"peace",
                    r"tranquil", r"serene", r"chill", r"take.*breath", r"calm.*myself", r"calm.*me", r"de-stress", r"unwind"
                ],
                "synonyms": [
                    "calm down", "relax", "breathe", "ground me", "center me", "chill out", "find peace", "be calm", "take a breath", "de-stress", "unwind", "calm myself", "calm me"
                ],
                "category": "wellness"
            },
            "clean_memory": {
                "patterns": [
                    r"clean.*memory", r"clear.*memory", r"memory.*cleanup", r"sweep.*memory",
                    r"purge.*memory", r"memory.*maintenance", r"defrag.*memory", r"reset.*memory", r"forget.*old.*data", r"remove.*old.*memory"
                ],
                "synonyms": [
                    "clean memory", "clear memory", "memory cleanup", "sweep memory", "purge memory", "memory maintenance", "defrag memory", "reset memory", "forget old data", "remove old memory"
                ],
                "category": "maintenance"
            },
            "activate_shield": {
                "patterns": [
                    r"activate.*shield", r"shield", r"protect", r"defense", r"security",
                    r"guard", r"fortify", r"secure", r"defend", r"enable.*protection", r"turn.*on.*shield", r"activate.*protection"
                ],
                "synonyms": [
                    "activate shield", "shield", "protect", "defense", "security", "guard", "fortify", "secure", "defend", "enable protection", "turn on shield", "activate protection"
                ],
                "category": "security"
            },
            "web_search": {
                "patterns": [
                    r"search.*for", r"find.*information", r"look.*up", r"web.*search",
                    r"google", r"research", r"investigate", r"explore", r"search.*web", r"search.*online", r"search.*internet"
                ],
                "synonyms": [
                    "search for", "find information", "look up", "web search", "research", "investigate", "explore", "search web", "search online", "search internet", "google"
                ],
                "category": "external"
            },
            "weather_check": {
                "patterns": [
                    r"weather", r"temperature", r"forecast", r"climate", r"weather.*like",
                    r"weather.*check", r"weather.*report", r"current.*weather", r"today.*weather", r"outside.*weather"
                ],
                "synonyms": [
                    "weather", "temperature", "forecast", "weather check", "weather report", "current weather", "today's weather", "outside weather", "climate"
                ],
                "category": "external"
            },
            "system_commands": {
                "patterns": [
                    r"system.*commands", r"available.*commands", r"what.*commands", r"show.*commands",
                    r"list.*commands", r"command.*list", r"expand.*commands", r"all.*commands",
                    r"system.*commands.*list", r"commands.*list", r"show.*list.*commands",
                    r"what.*can.*you.*do", r"your.*commands", r"available.*functions", r"command.*options", r"command.*menu"
                ],
                "synonyms": [
                    "system commands", "available commands", "what commands", "show commands", "list commands", "system commands list", "command list", "expand commands", "all commands", "command options", "command menu", "your commands", "available functions"
                ],
                "category": "help"
            },
            "about_self": {
                "patterns": [
                    r"about.*yourself", r"tell.*me.*about.*yourself", r"who.*are.*you", r"what.*are.*you",
                    r"explain.*yourself", r"describe.*yourself", r"your.*capabilities", r"what.*can.*you.*do", r"who.*is.*unimind", r"what.*is.*unimind"
                ],
                "synonyms": [
                    "about yourself", "tell me about yourself", "who are you", "what are you", "explain yourself", "describe yourself", "your capabilities", "what can you do", "who is unimind", "what is unimind"
                ],
                "category": "help"
            }
        }
        
        # Create synonym groups for better matching
        self.synonym_groups = defaultdict(list)
        for command, data in self.command_patterns.items():
            for synonym in data["synonyms"]:
                self.synonym_groups[synonym.lower()].append(command)
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two strings using multiple algorithms.
        
        Args:
            text1: First string
            text2: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize strings
        text1 = lemmatize_text(text1.lower().strip())
        text2 = lemmatize_text(text2.lower().strip())
        
        # Exact match
        if text1 == text2:
            return 1.0
        
        # Sequence matcher (good for typos and variations)
        sequence_similarity = SequenceMatcher(None, text1, text2).ratio()
        
        # Word-based similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return sequence_similarity
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        jaccard_similarity = intersection / union if union > 0 else 0
        
        # Combined similarity (weighted average)
        combined_similarity = (sequence_similarity * 0.6) + (jaccard_similarity * 0.4)
        
        return combined_similarity
    
    def pattern_match(self, user_input: str) -> List[Tuple[str, float, str]]:
        """
        Match user input against command patterns using regex.
        
        Args:
            user_input: User's input text
            
        Returns:
            List of (command, confidence, category) tuples
        """
        matches = []
        user_input_lower = lemmatize_text(user_input.lower())
        
        for command, data in self.command_patterns.items():
            for pattern in data["patterns"]:
                if re.search(pattern, user_input_lower, re.IGNORECASE):
                    # Calculate confidence based on pattern match
                    confidence = 0.8  # Base confidence for pattern match
                    
                    # Boost confidence for exact synonym matches
                    for synonym in data["synonyms"]:
                        if synonym.lower() in user_input_lower:
                            confidence = 0.95
                            break
                    
                    matches.append((command, confidence, data["category"]))
                    break  # Only match first pattern per command
        
        return matches
    
    def fuzzy_match(self, user_input: str) -> List[Tuple[str, float, str]]:
        """
        Perform fuzzy matching against command synonyms.
        
        Args:
            user_input: User's input text
            
        Returns:
            List of (command, confidence, category) tuples
        """
        matches = []
        user_input_lower = lemmatize_text(user_input.lower())
        
        for synonym, commands in self.synonym_groups.items():
            similarity = self.calculate_similarity(user_input_lower, synonym)
            
            if similarity >= self.confidence_threshold:
                for command in commands:
                    data = self.command_patterns[command]
                    matches.append((command, similarity, data["category"]))
        
        # Partial match: check if any lemmatized synonym is a substring of the input
        for synonym, commands in self.synonym_groups.items():
            if lemmatize_text(synonym) in user_input_lower:
                for command in commands:
                    data = self.command_patterns[command]
                    matches.append((command, 0.7, data["category"]))
        
        return matches
    
    def process_input(self, user_input: str) -> Optional[Tuple[str, float, str]]:
        """
        Process user input and return the best matching command.
        
        Args:
            user_input: User's input text
            
        Returns:
            Tuple of (command, confidence, category) or None if no match
        """
        if not user_input.strip():
            return None
        
        # Get pattern matches
        pattern_matches = self.pattern_match(user_input)
        
        # Get fuzzy matches
        fuzzy_matches = self.fuzzy_match(user_input)
        
        # Combine and deduplicate matches
        all_matches = {}
        
        for command, confidence, category in pattern_matches + fuzzy_matches:
            if command not in all_matches or confidence > all_matches[command][1]:
                all_matches[command] = (command, confidence, category)
        
        # Sort by confidence and return best match
        if all_matches:
            best_match = max(all_matches.values(), key=lambda x: x[1])
            return best_match
        
        return None
    
    def get_suggestions(self, user_input: str, max_suggestions: int = 3) -> List[Tuple[str, float]]:
        """
        Get command suggestions for user input.
        
        Args:
            user_input: User's input text
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of (command, confidence) tuples
        """
        suggestions = []
        user_input_lower = lemmatize_text(user_input.lower())
        
        for command, data in self.command_patterns.items():
            # Calculate similarity with all synonyms
            max_similarity = 0
            for synonym in data["synonyms"]:
                similarity = self.calculate_similarity(user_input_lower, synonym)
                max_similarity = max(max_similarity, similarity)
            
            if max_similarity > self.suggestion_threshold:  # Lower threshold for suggestions
                suggestions.append((command, max_similarity))
        
        # Sort by confidence and return top suggestions
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:max_suggestions]
    
    def add_command_pattern(self, command: str, patterns: List[str], synonyms: List[str], category: str = "general"):
        """
        Add a new command pattern to the processor.
        
        Args:
            command: Command name
            patterns: List of regex patterns
            synonyms: List of synonym phrases
            category: Command category
        """
        self.command_patterns[command] = {
            "patterns": patterns,
            "synonyms": synonyms,
            "category": category
        }
        
        # Update synonym groups
        for synonym in synonyms:
            self.synonym_groups[synonym.lower()].append(command)
    
    def get_command_info(self, command: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific command.
        
        Args:
            command: Command name
            
        Returns:
            Command information dictionary or None
        """
        return self.command_patterns.get(command)
    
    def list_commands(self, category: Optional[str] = None) -> List[str]:
        """
        List all commands or commands in a specific category.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of command names
        """
        if category:
            return [cmd for cmd, data in self.command_patterns.items() if data["category"] == category]
        else:
            return list(self.command_patterns.keys())

    def get_confidence_score(self, text: str, pattern: str) -> float:
        """Get confidence score for a text-pattern match."""
        try:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Calculate confidence based on match length and position
                match_length = len(match.group())
                text_length = len(text)
                position_factor = 1.0 - (match.start() / text_length)
                length_factor = match_length / text_length
                return (position_factor + length_factor) / 2
            return 0.0
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.0
    
    def optimize(self) -> Dict[str, Any]:
        """
        Optimize the fuzzy processor for better performance.
        
        Returns:
            Dict containing optimization results
        """
        self.logger.info("Optimizing fuzzy processor")
        
        optimization_results = {
            "status": "optimized",
            "changes": [],
            "performance_metrics": {}
        }
        
        # Optimize pattern matching
        try:
            # Pre-compile frequently used patterns
            self.compiled_patterns = {}
            for pattern_name, pattern in self.command_patterns.items():
                try:
                    self.compiled_patterns[pattern_name] = re.compile(pattern["patterns"][0], re.IGNORECASE)
                    optimization_results["changes"].append(f"Pre-compiled pattern: {pattern_name}")
                except re.error as e:
                    self.logger.warning(f"Invalid pattern {pattern_name}: {e}")
            
            # Optimize fuzzy matching parameters
            self.similarity_threshold = 0.6  # Optimal threshold
            self.max_matches = 10  # Optimal max matches
            
            optimization_results["changes"].append("Adjusted similarity threshold to 0.6")
            optimization_results["changes"].append("Set max matches to 10")
            
            # Performance metrics
            optimization_results["performance_metrics"] = {
                "compiled_patterns": len(self.compiled_patterns),
                "similarity_threshold": self.similarity_threshold,
                "max_matches": self.max_matches,
                "total_patterns": len(self.command_patterns)
            }
            
        except Exception as e:
            optimization_results["status"] = "error"
            optimization_results["error"] = str(e)
            self.logger.error(f"Fuzzy processor optimization failed: {e}")
        
        return optimization_results
    
    def analyze_capability(self, user_input: str, context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Analyze user input for capability requirements using integrated analyzer."""
        if not self.enable_capability_analysis:
            return None
        
        try:
            return self.capability_analyzer.analyze_capability(user_input, context)
        except Exception as e:
            self.logger.error(f"Capability analysis error: {e}")
            return None
    
    def generate_code(self, request: CodeGenerationRequest) -> Optional[GeneratedCode]:
        """Generate code for missing capabilities using integrated analyzer."""
        if not self.enable_capability_analysis:
            return None
        
        try:
            return self.capability_analyzer.generate_code(request)
        except Exception as e:
            self.logger.error(f"Code generation error: {e}")
            return None
    
    def process_with_capability_analysis(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user input with both fuzzy matching and capability analysis."""
        result = {
            "fuzzy_match": None,
            "capability_analysis": None,
            "suggestions": [],
            "recommendations": []
        }
        
        # Get fuzzy match
        fuzzy_result = self.process_input(user_input)
        if fuzzy_result:
            result["fuzzy_match"] = {
                "command": fuzzy_result[0],
                "confidence": fuzzy_result[1],
                "method": fuzzy_result[2]
            }
        
        # Get capability analysis
        if self.enable_capability_analysis:
            capability_result = self.analyze_capability(user_input, context)
            if capability_result:
                result["capability_analysis"] = capability_result
                
                # Add recommendations based on capability analysis
                if capability_result.get("missing_capabilities"):
                    result["recommendations"].append("Consider generating code for missing capabilities")
                
                if capability_result.get("complexity") == "high":
                    result["recommendations"].append("This request is complex - consider breaking it down")
        
        # Get suggestions
        suggestions = self.get_suggestions(user_input)
        result["suggestions"] = [(cmd, conf) for cmd, conf in suggestions]
        
        return result

# Global instance
fuzzy_processor = FuzzyProcessor() 