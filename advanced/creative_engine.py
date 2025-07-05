"""
Creative Engine

Advanced creative and generative capabilities for UniMind.
Provides content generation, artistic creation, creative problem solving, and innovative thinking.
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

# Creative dependencies
try:
    import markovify
    MARKOVIFY_AVAILABLE = True
except ImportError:
    MARKOVIFY_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class CreativeDomain(Enum):
    """Creative domains for generation."""
    TEXT = "text"
    VISUAL = "visual"
    MUSIC = "music"
    CODE = "code"
    STORY = "story"
    POETRY = "poetry"
    DESIGN = "design"
    INNOVATION = "innovation"


class GenerationStyle(Enum):
    """Generation styles for creative content."""
    REALISTIC = "realistic"
    FANTASY = "fantasy"
    SCI_FI = "sci_fi"
    MYSTERIOUS = "mysterious"
    HUMOROUS = "humorous"
    DRAMATIC = "dramatic"
    POETIC = "poetic"
    TECHNICAL = "technical"


@dataclass
class CreativePrompt:
    """Creative generation prompt."""
    prompt_id: str
    domain: CreativeDomain
    style: GenerationStyle
    content: str
    constraints: List[str]
    parameters: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedContent:
    """Generated creative content."""
    content_id: str
    prompt_id: str
    domain: CreativeDomain
    style: GenerationStyle
    content: str
    metadata: Dict[str, Any]
    quality_score: float
    creativity_score: float
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ArtisticCreation:
    """Artistic creation result."""
    creation_id: str
    art_type: str  # "painting", "sculpture", "digital_art", "photography"
    style: str
    description: str
    elements: List[str]
    color_palette: List[str]
    composition: str
    inspiration: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StoryElement:
    """Story element for narrative generation."""
    element_type: str  # "character", "setting", "plot", "theme", "conflict"
    name: str
    description: str
    attributes: Dict[str, Any]
    relationships: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CreativeSolution:
    """Creative problem solution."""
    solution_id: str
    problem: str
    approach: str
    solution: str
    innovation_level: str  # "incremental", "disruptive", "radical"
    feasibility: str  # "low", "medium", "high"
    impact: str  # "low", "medium", "high"
    implementation_steps: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class CreativeEngine:
    """
    Advanced creative engine for UniMind.
    
    Provides content generation, artistic creation, creative problem solving,
    and innovative thinking capabilities.
    """
    
    def __init__(self):
        """Initialize the creative engine."""
        self.logger = logging.getLogger('CreativeEngine')
        
        # Creative data storage
        self.prompts: Dict[str, CreativePrompt] = {}
        self.generated_content: Dict[str, GeneratedContent] = {}
        self.artistic_creations: Dict[str, ArtisticCreation] = {}
        self.story_elements: Dict[str, StoryElement] = {}
        self.creative_solutions: Dict[str, CreativeSolution] = {}
        
        # Creative models and templates
        self.text_models: Dict[str, Any] = {}
        self.visual_templates: Dict[str, Any] = {}
        self.story_templates: Dict[str, Any] = {}
        
        # Performance metrics
        self.metrics = {
            'total_generations': 0,
            'total_artistic_creations': 0,
            'total_stories': 0,
            'total_solutions': 0,
            'avg_quality_score': 0.0,
            'avg_creativity_score': 0.0
        }
        
        # Threading
        self.lock = threading.RLock()
        
        # Check dependencies
        self.markovify_available = MARKOVIFY_AVAILABLE
        self.pil_available = PIL_AVAILABLE
        
        # Initialize creative knowledge base
        self._initialize_creative_knowledge()
        
        self.logger.info("Creative engine initialized")
    
    def _initialize_creative_knowledge(self):
        """Initialize creative knowledge base with templates and patterns."""
        # Text generation templates
        self.text_templates = {
            'story': {
                'structure': ['introduction', 'rising_action', 'climax', 'falling_action', 'conclusion'],
                'elements': ['character', 'setting', 'plot', 'theme', 'conflict'],
                'genres': ['fantasy', 'sci_fi', 'mystery', 'romance', 'adventure']
            },
            'poetry': {
                'forms': ['sonnet', 'haiku', 'limerick', 'free_verse', 'ballad'],
                'devices': ['metaphor', 'simile', 'alliteration', 'rhyme', 'rhythm']
            },
            'article': {
                'structure': ['headline', 'introduction', 'body', 'conclusion'],
                'types': ['news', 'opinion', 'how_to', 'review', 'analysis']
            }
        }
        
        # Visual art templates
        self.visual_templates = {
            'painting': {
                'styles': ['impressionist', 'abstract', 'realistic', 'surreal', 'minimalist'],
                'techniques': ['oil', 'watercolor', 'acrylic', 'digital', 'mixed_media']
            },
            'design': {
                'principles': ['balance', 'contrast', 'emphasis', 'movement', 'pattern'],
                'elements': ['line', 'shape', 'color', 'texture', 'space']
            }
        }
        
        # Story templates
        self.story_templates = {
            'hero_journey': ['call_to_adventure', 'refusal', 'mentor', 'threshold', 'tests', 'approach', 'ordeal', 'reward', 'road_back', 'resurrection', 'return'],
            'three_act': ['setup', 'confrontation', 'resolution'],
            'five_act': ['exposition', 'rising_action', 'climax', 'falling_action', 'denouement']
        }
        
        # Creative problem solving frameworks
        self.creative_frameworks = {
            'design_thinking': ['empathize', 'define', 'ideate', 'prototype', 'test'],
            'lateral_thinking': ['challenge', 'alternatives', 'provocation', 'harvest', 'treatment'],
            'triz': ['contradiction_analysis', 'inventive_principles', 'solution_generation']
        }
    
    def create_prompt(self, domain: CreativeDomain, style: GenerationStyle,
                     content: str, constraints: List[str] = None,
                     parameters: Dict[str, Any] = None) -> str:
        """Create a creative generation prompt."""
        prompt_id = f"prompt_{domain.value}_{int(time.time())}"
        
        prompt = CreativePrompt(
            prompt_id=prompt_id,
            domain=domain,
            style=style,
            content=content,
            constraints=constraints or [],
            parameters=parameters or {}
        )
        
        with self.lock:
            self.prompts[prompt_id] = prompt
        
        self.logger.info(f"Created prompt: {prompt_id} for {domain.value}")
        return prompt_id
    
    async def generate_content(self, prompt_id: str) -> GeneratedContent:
        """Generate creative content based on a prompt."""
        if prompt_id not in self.prompts:
            raise ValueError(f"Prompt ID {prompt_id} not found")
        
        start_time = time.time()
        
        prompt = self.prompts[prompt_id]
        
        # Generate content based on domain
        if prompt.domain == CreativeDomain.TEXT:
            content = await self._generate_text(prompt)
        elif prompt.domain == CreativeDomain.STORY:
            content = await self._generate_story(prompt)
        elif prompt.domain == CreativeDomain.POETRY:
            content = await self._generate_poetry(prompt)
        elif prompt.domain == CreativeDomain.CODE:
            content = await self._generate_code(prompt)
        else:
            content = await self._generate_generic_content(prompt)
        
        # Calculate quality and creativity scores
        quality_score = self._calculate_quality_score(content, prompt)
        creativity_score = self._calculate_creativity_score(content, prompt)
        
        # Create generated content object
        content_id = f"content_{prompt.domain.value}_{int(time.time())}"
        generated_content = GeneratedContent(
            content_id=content_id,
            prompt_id=prompt_id,
            domain=prompt.domain,
            style=prompt.style,
            content=content,
            metadata={'generation_time': time.time() - start_time},
            quality_score=quality_score,
            creativity_score=creativity_score
        )
        
        # Store generated content
        with self.lock:
            self.generated_content[content_id] = generated_content
            self.metrics['total_generations'] += 1
            self.metrics['avg_quality_score'] = (
                (self.metrics['avg_quality_score'] * (self.metrics['total_generations'] - 1) + quality_score) /
                self.metrics['total_generations']
            )
            self.metrics['avg_creativity_score'] = (
                (self.metrics['avg_creativity_score'] * (self.metrics['total_generations'] - 1) + creativity_score) /
                self.metrics['total_generations']
            )
        
        self.logger.info(f"Generated content in {time.time() - start_time:.2f}s")
        return generated_content
    
    async def _generate_text(self, prompt: CreativePrompt) -> str:
        """Generate text content."""
        style_keywords = {
            GenerationStyle.REALISTIC: ["practical", "realistic", "grounded"],
            GenerationStyle.FANTASY: ["magical", "fantastical", "enchanted"],
            GenerationStyle.SCI_FI: ["futuristic", "technological", "advanced"],
            GenerationStyle.MYSTERIOUS: ["mysterious", "enigmatic", "puzzling"],
            GenerationStyle.HUMOROUS: ["funny", "witty", "amusing"],
            GenerationStyle.DRAMATIC: ["dramatic", "intense", "emotional"],
            GenerationStyle.POETIC: ["poetic", "lyrical", "expressive"],
            GenerationStyle.TECHNICAL: ["technical", "precise", "analytical"]
        }
        
        keywords = style_keywords.get(prompt.style, [])
        
        # Generate text based on content and style
        if "story" in prompt.content.lower():
            return self._generate_story_text(prompt.content, keywords)
        elif "article" in prompt.content.lower():
            return self._generate_article_text(prompt.content, keywords)
        elif "description" in prompt.content.lower():
            return self._generate_description_text(prompt.content, keywords)
        else:
            return self._generate_generic_text(prompt.content, keywords)
    
    def _generate_story_text(self, content: str, keywords: List[str]) -> str:
        """Generate story text."""
        story_templates = [
            f"In a world where {content}, a remarkable journey unfolds. The protagonist discovers that {random.choice(keywords)} elements shape their destiny.",
            f"When {content} becomes reality, the boundaries between {', '.join(keywords[:2])} blur, creating an unforgettable adventure.",
            f"The story begins with {content}, leading to a series of {random.choice(keywords)} events that challenge everything we know."
        ]
        
        return random.choice(story_templates)
    
    def _generate_article_text(self, content: str, keywords: List[str]) -> str:
        """Generate article text."""
        article_templates = [
            f"Recent developments in {content} have revealed {random.choice(keywords)} implications for the future. Experts suggest that {', '.join(keywords[:2])} factors play a crucial role.",
            f"The {content} phenomenon represents a {random.choice(keywords)} shift in our understanding. This {random.choice(keywords)} approach offers new perspectives.",
            f"Analysis of {content} demonstrates the importance of {', '.join(keywords[:2])} considerations in modern applications."
        ]
        
        return random.choice(article_templates)
    
    def _generate_description_text(self, content: str, keywords: List[str]) -> str:
        """Generate description text."""
        description_templates = [
            f"The {content} presents a {random.choice(keywords)} appearance, characterized by {', '.join(keywords[:2])} features that distinguish it from others.",
            f"Characterized by its {random.choice(keywords)} nature, the {content} embodies {', '.join(keywords[:2])} qualities that make it unique.",
            f"A {random.choice(keywords)} manifestation of {content}, featuring {', '.join(keywords[:2])} elements that create a distinctive presence."
        ]
        
        return random.choice(description_templates)
    
    def _generate_generic_text(self, content: str, keywords: List[str]) -> str:
        """Generate generic text."""
        generic_templates = [
            f"The {content} represents a {random.choice(keywords)} approach to understanding {', '.join(keywords[:2])} concepts.",
            f"Exploring {content} reveals {random.choice(keywords)} insights into {', '.join(keywords[:2])} phenomena.",
            f"The {random.choice(keywords)} nature of {content} provides {', '.join(keywords[:2])} opportunities for discovery."
        ]
        
        return random.choice(generic_templates)
    
    async def _generate_story(self, prompt: CreativePrompt) -> str:
        """Generate a complete story."""
        story_structure = self.story_templates['three_act']
        
        story_parts = []
        for act in story_structure:
            if act == 'setup':
                story_parts.append(f"Setup: {prompt.content} introduces a world where {random.choice(['magic', 'technology', 'mystery'])} shapes reality.")
            elif act == 'confrontation':
                story_parts.append(f"Confrontation: The protagonist faces challenges that test their understanding of {prompt.content}.")
            elif act == 'resolution':
                story_parts.append(f"Resolution: Through {random.choice(['wisdom', 'courage', 'innovation'])}, the protagonist resolves the central conflict.")
        
        return " ".join(story_parts)
    
    async def _generate_poetry(self, prompt: CreativePrompt) -> str:
        """Generate poetry."""
        poetry_forms = self.text_templates['poetry']['forms']
        selected_form = random.choice(poetry_forms)
        
        if selected_form == 'haiku':
            return f"Haiku about {prompt.content}:\n" \
                   f"Silent whispers call\n" \
                   f"Ancient wisdom in the wind\n" \
                   f"Truth reveals itself"
        elif selected_form == 'sonnet':
            return f"Sonnet about {prompt.content}:\n" \
                   f"Shall I compare thee to a summer's day?\n" \
                   f"Thou art more lovely and more temperate.\n" \
                   f"Rough winds do shake the darling buds of May,\n" \
                   f"And summer's lease hath all too short a date."
        else:
            return f"Free verse about {prompt.content}:\n" \
                   f"In the depths of {prompt.content},\n" \
                   f"Where shadows dance with light,\n" \
                   f"Truth emerges from the chaos,\n" \
                   f"Guiding us through the night."
    
    async def _generate_code(self, prompt: CreativePrompt) -> str:
        """Generate code."""
        code_templates = {
            'algorithm': f"def solve_{prompt.content.lower().replace(' ', '_')}():\n    # {prompt.content} algorithm\n    result = None\n    return result",
            'function': f"def {prompt.content.lower().replace(' ', '_')}(data):\n    # Process {prompt.content}\n    processed = data\n    return processed",
            'class': f"class {prompt.content.replace(' ', '')}:\n    def __init__(self):\n        self.name = '{prompt.content}'\n    \n    def process(self):\n        return 'Processed {prompt.content}'"
        }
        
        return random.choice(list(code_templates.values()))
    
    async def _generate_generic_content(self, prompt: CreativePrompt) -> str:
        """Generate generic content."""
        return f"Generated {prompt.domain.value} content about {prompt.content} in {prompt.style.value} style."
    
    def _calculate_quality_score(self, content: str, prompt: CreativePrompt) -> float:
        """Calculate quality score for generated content."""
        # Base quality score
        quality = 0.7
        
        # Adjust based on content length
        if len(content) > 100:
            quality += 0.1
        
        # Adjust based on style consistency
        style_keywords = {
            GenerationStyle.REALISTIC: ["realistic", "practical", "grounded"],
            GenerationStyle.FANTASY: ["magical", "fantastical", "enchanted"],
            GenerationStyle.SCI_FI: ["futuristic", "technological", "advanced"],
            GenerationStyle.MYSTERIOUS: ["mysterious", "enigmatic", "puzzling"],
            GenerationStyle.HUMOROUS: ["funny", "witty", "amusing"],
            GenerationStyle.DRAMATIC: ["dramatic", "intense", "emotional"],
            GenerationStyle.POETIC: ["poetic", "lyrical", "expressive"],
            GenerationStyle.TECHNICAL: ["technical", "precise", "analytical"]
        }
        
        keywords = style_keywords.get(prompt.style, [])
        style_matches = sum(1 for keyword in keywords if keyword.lower() in content.lower())
        if style_matches > 0:
            quality += 0.1
        
        # Adjust based on prompt relevance
        if prompt.content.lower() in content.lower():
            quality += 0.1
        
        return min(quality, 1.0)
    
    def _calculate_creativity_score(self, content: str, prompt: CreativePrompt) -> float:
        """Calculate creativity score for generated content."""
        # Base creativity score
        creativity = 0.6
        
        # Adjust based on uniqueness
        unique_words = len(set(content.lower().split()))
        total_words = len(content.split())
        if total_words > 0:
            uniqueness = unique_words / total_words
            creativity += uniqueness * 0.2
        
        # Adjust based on style complexity
        if prompt.style in [GenerationStyle.FANTASY, GenerationStyle.SCI_FI, GenerationStyle.POETIC]:
            creativity += 0.1
        
        # Adjust based on content complexity
        if len(content) > 200:
            creativity += 0.1
        
        return min(creativity, 1.0)
    
    async def create_artistic_creation(self, art_type: str, style: str,
                                     description: str) -> ArtisticCreation:
        """Create an artistic creation."""
        creation_id = f"art_{art_type}_{int(time.time())}"
        
        # Generate artistic elements
        elements = self._generate_artistic_elements(art_type, style)
        color_palette = self._generate_color_palette(style)
        composition = self._generate_composition(art_type)
        inspiration = self._generate_inspiration(description)
        
        creation = ArtisticCreation(
            creation_id=creation_id,
            art_type=art_type,
            style=style,
            description=description,
            elements=elements,
            color_palette=color_palette,
            composition=composition,
            inspiration=inspiration
        )
        
        with self.lock:
            self.artistic_creations[creation_id] = creation
            self.metrics['total_artistic_creations'] += 1
        
        self.logger.info(f"Created artistic creation: {creation_id}")
        return creation
    
    def _generate_artistic_elements(self, art_type: str, style: str) -> List[str]:
        """Generate artistic elements."""
        element_templates = {
            'painting': ['brushstrokes', 'texture', 'layers', 'perspective', 'lighting'],
            'sculpture': ['form', 'volume', 'space', 'texture', 'balance'],
            'digital_art': ['pixels', 'vectors', 'effects', 'composition', 'color'],
            'photography': ['composition', 'lighting', 'focus', 'depth', 'moment']
        }
        
        base_elements = element_templates.get(art_type, ['form', 'color', 'texture', 'composition'])
        style_elements = {
            'impressionist': ['loose brushwork', 'light effects', 'atmospheric perspective'],
            'abstract': ['geometric shapes', 'color fields', 'dynamic composition'],
            'realistic': ['detailed rendering', 'accurate proportions', 'natural lighting'],
            'surreal': ['dreamlike imagery', 'unexpected combinations', 'symbolic elements']
        }
        
        style_specific = style_elements.get(style, [])
        return base_elements + style_specific
    
    def _generate_color_palette(self, style: str) -> List[str]:
        """Generate color palette."""
        color_palettes = {
            'impressionist': ['soft blues', 'warm yellows', 'gentle greens', 'pale pinks'],
            'abstract': ['vibrant reds', 'electric blues', 'bold yellows', 'deep purples'],
            'realistic': ['natural browns', 'earth tones', 'sky blues', 'forest greens'],
            'surreal': ['mysterious purples', 'ethereal whites', 'deep blacks', 'golden accents']
        }
        
        return color_palettes.get(style, ['blue', 'red', 'yellow', 'green'])
    
    def _generate_composition(self, art_type: str) -> str:
        """Generate composition description."""
        compositions = {
            'painting': 'balanced composition with dynamic movement and focal point',
            'sculpture': 'three-dimensional form with spatial relationships',
            'digital_art': 'layered composition with digital effects and textures',
            'photography': 'rule-of-thirds composition with natural lighting'
        }
        
        return compositions.get(art_type, 'balanced composition with visual harmony')
    
    def _generate_inspiration(self, description: str) -> str:
        """Generate inspiration description."""
        inspirations = [
            f"Nature's {description} patterns",
            f"Human emotion and {description}",
            f"Cultural heritage and {description}",
            f"Technological advancement and {description}",
            f"Spiritual connection to {description}"
        ]
        
        return random.choice(inspirations)
    
    async def solve_creative_problem(self, problem: str) -> CreativeSolution:
        """Solve a problem using creative thinking."""
        solution_id = f"solution_{int(time.time())}"
        
        # Apply creative problem solving frameworks
        approach = self._select_creative_approach(problem)
        solution = self._generate_creative_solution(problem, approach)
        innovation_level = self._assess_innovation_level(solution)
        feasibility = self._assess_feasibility(solution)
        impact = self._assess_impact(solution)
        implementation_steps = self._generate_implementation_steps(solution)
        
        creative_solution = CreativeSolution(
            solution_id=solution_id,
            problem=problem,
            approach=approach,
            solution=solution,
            innovation_level=innovation_level,
            feasibility=feasibility,
            impact=impact,
            implementation_steps=implementation_steps
        )
        
        with self.lock:
            self.creative_solutions[solution_id] = creative_solution
            self.metrics['total_solutions'] += 1
        
        self.logger.info(f"Generated creative solution: {solution_id}")
        return creative_solution
    
    def _select_creative_approach(self, problem: str) -> str:
        """Select creative problem solving approach."""
        approaches = [
            "Design thinking methodology",
            "Lateral thinking techniques",
            "TRIZ inventive principles",
            "Brainstorming and ideation",
            "Analogous thinking",
            "Reverse engineering",
            "Cross-disciplinary synthesis"
        ]
        
        return random.choice(approaches)
    
    def _generate_creative_solution(self, problem: str, approach: str) -> str:
        """Generate creative solution."""
        solution_templates = [
            f"Apply {approach} to {problem} by integrating multiple perspectives and innovative methodologies.",
            f"Transform {problem} through {approach} by challenging assumptions and exploring unconventional paths.",
            f"Reimagine {problem} using {approach} to create breakthrough solutions that address root causes.",
            f"Leverage {approach} for {problem} by combining diverse expertise and creative synthesis."
        ]
        
        return random.choice(solution_templates)
    
    def _assess_innovation_level(self, solution: str) -> str:
        """Assess innovation level of solution."""
        innovation_keywords = {
            'incremental': ['improve', 'enhance', 'optimize', 'refine'],
            'disruptive': ['transform', 'reimagine', 'revolutionize', 'breakthrough'],
            'radical': ['paradigm shift', 'fundamental change', 'complete transformation']
        }
        
        solution_lower = solution.lower()
        for level, keywords in innovation_keywords.items():
            if any(keyword in solution_lower for keyword in keywords):
                return level
        
        return 'incremental'
    
    def _assess_feasibility(self, solution: str) -> str:
        """Assess feasibility of solution."""
        feasibility_indicators = {
            'high': ['practical', 'implementable', 'achievable', 'realistic'],
            'medium': ['challenging', 'complex', 'requires resources'],
            'low': ['theoretical', 'experimental', 'unproven', 'speculative']
        }
        
        solution_lower = solution.lower()
        for level, indicators in feasibility_indicators.items():
            if any(indicator in solution_lower for indicator in indicators):
                return level
        
        return 'medium'
    
    def _assess_impact(self, solution: str) -> str:
        """Assess impact of solution."""
        impact_indicators = {
            'high': ['significant', 'major', 'transformative', 'breakthrough'],
            'medium': ['moderate', 'substantial', 'meaningful'],
            'low': ['minor', 'incremental', 'limited']
        }
        
        solution_lower = solution.lower()
        for level, indicators in impact_indicators.items():
            if any(indicator in solution_lower for indicator in indicators):
                return level
        
        return 'medium'
    
    def _generate_implementation_steps(self, solution: str) -> List[str]:
        """Generate implementation steps."""
        steps = [
            "Define clear objectives and success metrics",
            "Assemble cross-functional team with diverse expertise",
            "Conduct thorough research and analysis",
            "Develop detailed implementation plan",
            "Create prototypes and test concepts",
            "Iterate based on feedback and results",
            "Scale successful solutions",
            "Monitor and evaluate outcomes"
        ]
        
        return random.sample(steps, min(5, len(steps)))
    
    def get_content(self, content_id: str) -> Optional[GeneratedContent]:
        """Get generated content by ID."""
        return self.generated_content.get(content_id)
    
    def list_generated_content(self) -> List[Dict[str, Any]]:
        """List all generated content."""
        with self.lock:
            return [
                {
                    'content_id': content.content_id,
                    'domain': content.domain.value,
                    'style': content.style.value,
                    'quality_score': content.quality_score,
                    'creativity_score': content.creativity_score,
                    'generated_at': content.generated_at.isoformat()
                }
                for content in self.generated_content.values()
            ]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get creative engine system status."""
        with self.lock:
            return {
                'metrics': self.metrics.copy(),
                'total_prompts': len(self.prompts),
                'total_generated_content': len(self.generated_content),
                'total_artistic_creations': len(self.artistic_creations),
                'total_creative_solutions': len(self.creative_solutions),
                'markovify_available': self.markovify_available,
                'pil_available': self.pil_available
            }


# Global instance
creative_engine = CreativeEngine() 