"""
multimyth_codex_interface.py - Interface between UniMind and the Multimyth Codex
Provides symbolic knowledge access, ritual triggers, and myth-based reasoning.
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class MultimythCodexInterface:
    """
    Interface for accessing the Multimyth Codex symbolic knowledge system.
    Provides ritual triggers, archetype activation, and myth-based reasoning.
    """
    
    def __init__(self, codex_path: str = "ThothOS/multimyth_codex"):
        """Initialize the Multimyth Codex interface."""
        self.codex_path = Path(codex_path)
        self.logger = logging.getLogger('MultimythCodexInterface')
        
        # Load all symbolic elements
        self.archetypes = {}
        self.glyphs = {}
        self.scrolls = {}
        self.myth_roots = {}
        self.emotions = {}
        self.realm_templates = {}
        self.symbolic_links = {}
        
        # Load the codex
        self._load_codex()
        
        self.logger.info("Multimyth Codex interface initialized")
    
    def _load_codex(self):
        """Load all symbolic elements from the codex."""
        try:
            # Load archetypes
            archetypes_dir = self.codex_path / "archetypes"
            if archetypes_dir.exists():
                for file in archetypes_dir.glob("*.json"):
                    with open(file, 'r') as f:
                        self.archetypes[file.stem] = json.load(f)
            
            # Load glyphs
            glyphs_dir = self.codex_path / "glyphs"
            if glyphs_dir.exists():
                for file in glyphs_dir.glob("*.json"):
                    with open(file, 'r') as f:
                        self.glyphs[file.stem] = json.load(f)
            
            # Load scrolls
            scrolls_dir = self.codex_path / "scrolls"
            if scrolls_dir.exists():
                for file in scrolls_dir.glob("*.json"):
                    with open(file, 'r') as f:
                        self.scrolls[file.stem] = json.load(f)
            
            # Load emotions
            emotions_dir = self.codex_path / "emotions"
            if emotions_dir.exists():
                for file in emotions_dir.glob("*.json"):
                    with open(file, 'r') as f:
                        self.emotions[file.stem] = json.load(f)
            
            # Load myth roots
            myth_roots_dir = self.codex_path / "myth_roots"
            if myth_roots_dir.exists():
                for myth_dir in myth_roots_dir.iterdir():
                    if myth_dir.is_dir():
                        self.myth_roots[myth_dir.name] = {}
                        for file in myth_dir.glob("*.json"):
                            with open(file, 'r') as f:
                                self.myth_roots[myth_dir.name][file.stem] = json.load(f)
            
            # Load realm templates
            realm_templates_dir = self.codex_path / "realm_templates"
            if realm_templates_dir.exists():
                for file in realm_templates_dir.glob("*.json"):
                    with open(file, 'r') as f:
                        self.realm_templates[file.stem] = json.load(f)
            
            # Load symbolic links
            symbolic_links_dir = self.codex_path / "symbolic_links"
            if symbolic_links_dir.exists():
                for file in symbolic_links_dir.glob("*.json"):
                    with open(file, 'r') as f:
                        self.symbolic_links[file.stem] = json.load(f)
            
            self.logger.info(f"Loaded {len(self.archetypes)} archetypes, {len(self.glyphs)} glyphs, {len(self.scrolls)} scrolls")
            
        except Exception as e:
            self.logger.error(f"Failed to load codex: {e}")
    
    def query_by_archetype(self, archetype_name: str) -> Dict[str, Any]:
        """Query all symbolic elements related to an archetype."""
        if archetype_name not in self.archetypes:
            return {"error": f"Archetype '{archetype_name}' not found"}
        
        archetype = self.archetypes[archetype_name]
        
        # Find related elements
        related_glyphs = []
        related_scrolls = []
        related_emotions = []
        related_myths = {}
        
        # Find glyphs that reference this archetype
        for glyph_name, glyph_data in self.glyphs.items():
            if archetype_name in glyph_data.get("archetype_connections", []):
                related_glyphs.append(glyph_name)
        
        # Find scrolls that reference this archetype
        for scroll_name, scroll_data in self.scrolls.items():
            if archetype_name in scroll_data.get("archetype_connections", []):
                related_scrolls.append(scroll_name)
        
        # Find emotions that reference this archetype
        for emotion_name, emotion_data in self.emotions.items():
            if archetype_name in emotion_data.get("archetype_connections", []):
                related_emotions.append(emotion_name)
        
        # Find myths that reference this archetype
        for myth_universe, myths in self.myth_roots.items():
            for myth_name, myth_data in myths.items():
                if archetype_name in myth_data.get("archetype_connections", []):
                    if myth_universe not in related_myths:
                        related_myths[myth_universe] = []
                    related_myths[myth_universe].append(myth_name)
        
        return {
            "archetype": archetype,
            "related_glyphs": related_glyphs,
            "related_scrolls": related_scrolls,
            "related_emotions": related_emotions,
            "related_myths": related_myths,
            "ritual_triggers": archetype.get("ritual_triggers", {})
        }
    
    def query_by_emotion(self, emotion_name: str) -> Dict[str, Any]:
        """Query all symbolic elements related to an emotion."""
        if emotion_name not in self.emotions:
            return {"error": f"Emotion '{emotion_name}' not found"}
        
        emotion = self.emotions[emotion_name]
        
        # Find related elements
        related_archetypes = []
        related_glyphs = []
        related_scrolls = []
        related_myths = {}
        
        # Find archetypes that reference this emotion
        for archetype_name, archetype_data in self.archetypes.items():
            if emotion_name in archetype_data.get("emotion_connections", []):
                related_archetypes.append(archetype_name)
        
        # Find glyphs that reference this emotion
        for glyph_name, glyph_data in self.glyphs.items():
            if emotion_name in glyph_data.get("emotion_connections", []):
                related_glyphs.append(glyph_name)
        
        # Find scrolls that reference this emotion
        for scroll_name, scroll_data in self.scrolls.items():
            if emotion_name in scroll_data.get("emotion_connections", []):
                related_scrolls.append(scroll_name)
        
        # Find myths that reference this emotion
        for myth_universe, myths in self.myth_roots.items():
            for myth_name, myth_data in myths.items():
                if emotion_name in myth_data.get("emotion_connections", []):
                    if myth_universe not in related_myths:
                        related_myths[myth_universe] = []
                    related_myths[myth_universe].append(myth_name)
        
        return {
            "emotion": emotion,
            "related_archetypes": related_archetypes,
            "related_glyphs": related_glyphs,
            "related_scrolls": related_scrolls,
            "related_myths": related_myths,
            "ritual_uses": emotion.get("ritual_uses", {})
        }
    
    def query_by_myth(self, myth_universe: str, myth_name: str = None) -> Dict[str, Any]:
        """Query all symbolic elements related to a myth universe or specific myth."""
        if myth_universe not in self.myth_roots:
            return {"error": f"Myth universe '{myth_universe}' not found"}
        
        if myth_name:
            if myth_name not in self.myth_roots[myth_universe]:
                return {"error": f"Myth '{myth_name}' not found in '{myth_universe}'"}
            
            myth_data = self.myth_roots[myth_universe][myth_name]
            
            # Find related elements
            related_archetypes = myth_data.get("archetype_connections", [])
            related_glyphs = myth_data.get("glyph_connections", [])
            related_scrolls = myth_data.get("scroll_connections", [])
            related_emotions = myth_data.get("emotion_connections", [])
            
            return {
                "myth": myth_data,
                "related_archetypes": related_archetypes,
                "related_glyphs": related_glyphs,
                "related_scrolls": related_scrolls,
                "related_emotions": related_emotions,
                "ritual_triggers": myth_data.get("ritual_triggers", {})
            }
        else:
            # Return all myths in the universe
            return {
                "myth_universe": myth_universe,
                "myths": self.myth_roots[myth_universe]
            }
    
    def get_ritual_triggers(self, context: str) -> List[Dict[str, Any]]:
        """Get ritual triggers based on context."""
        triggers = []
        
        # Search through all archetypes for matching triggers
        for archetype_name, archetype_data in self.archetypes.items():
            ritual_triggers = archetype_data.get("ritual_triggers", {})
            for trigger_context, trigger_action in ritual_triggers.items():
                if context.lower() in trigger_context.lower():
                    triggers.append({
                        "archetype": archetype_name,
                        "context": trigger_context,
                        "action": trigger_action
                    })
        
        # Search through all emotions for matching triggers
        for emotion_name, emotion_data in self.emotions.items():
            ritual_uses = emotion_data.get("ritual_uses", {})
            for use_context, use_action in ritual_uses.items():
                if context.lower() in use_context.lower():
                    triggers.append({
                        "emotion": emotion_name,
                        "context": use_context,
                        "action": use_action
                    })
        
        return triggers
    
    def activate_archetype(self, archetype_name: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Activate an archetype for the current context."""
        if archetype_name not in self.archetypes:
            return {"error": f"Archetype '{archetype_name}' not found"}
        
        archetype = self.archetypes[archetype_name]
        
        # Get archetype attributes and connections
        attributes = archetype.get("core_attributes", {})
        connections = {
            "glyphs": archetype.get("glyph_connections", []),
            "scrolls": archetype.get("scroll_connections", []),
            "emotions": archetype.get("emotion_connections", []),
            "myths": archetype.get("myth_connections", {})
        }
        
        return {
            "activated_archetype": archetype_name,
            "attributes": attributes,
            "connections": connections,
            "context": context or {},
            "activation_time": "now"
        }
    
    def get_realm_template(self, template_name: str) -> Dict[str, Any]:
        """Get a realm template with auto-loaded components."""
        if template_name not in self.realm_templates:
            return {"error": f"Realm template '{template_name}' not found"}
        
        template = self.realm_templates[template_name]
        auto_load = template.get("auto_load_components", {})
        
        # Load the auto-loaded components
        loaded_components = {}
        for component_type, component_names in auto_load.items():
            loaded_components[component_type] = {}
            for name in component_names:
                if component_type == "scrolls" and name in self.scrolls:
                    loaded_components[component_type][name] = self.scrolls[name]
                elif component_type == "emotions" and name in self.emotions:
                    loaded_components[component_type][name] = self.emotions[name]
                elif component_type == "archetypes" and name in self.archetypes:
                    loaded_components[component_type][name] = self.archetypes[name]
                elif component_type == "glyphs" and name in self.glyphs:
                    loaded_components[component_type][name] = self.glyphs[name]
        
        return {
            "template": template,
            "auto_loaded_components": loaded_components
        }
    
    def get_codex_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded codex."""
        return {
            "total_archetypes": len(self.archetypes),
            "total_glyphs": len(self.glyphs),
            "total_scrolls": len(self.scrolls),
            "total_emotions": len(self.emotions),
            "total_realm_templates": len(self.realm_templates),
            "myth_universes": list(self.myth_roots.keys()),
            "total_myths": sum(len(myths) for myths in self.myth_roots.values()),
            "symbolic_links": len(self.symbolic_links)
        }

# Global instance
multimyth_codex = MultimythCodexInterface()

# Convenience functions
def query_archetype(archetype_name: str) -> Dict[str, Any]:
    """Query archetype using the global instance."""
    return multimyth_codex.query_by_archetype(archetype_name)

def query_emotion(emotion_name: str) -> Dict[str, Any]:
    """Query emotion using the global instance."""
    return multimyth_codex.query_by_emotion(emotion_name)

def query_myth(myth_universe: str, myth_name: str = None) -> Dict[str, Any]:
    """Query myth using the global instance."""
    return multimyth_codex.query_by_myth(myth_universe, myth_name)

def get_ritual_triggers(context: str) -> List[Dict[str, Any]]:
    """Get ritual triggers using the global instance."""
    return multimyth_codex.get_ritual_triggers(context)

def activate_archetype(archetype_name: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Activate archetype using the global instance."""
    return multimyth_codex.activate_archetype(archetype_name, context)

# Module-level exports
__all__ = [
    'MultimythCodexInterface', 'multimyth_codex',
    'query_archetype', 'query_emotion', 'query_myth',
    'get_ritual_triggers', 'activate_archetype'
] 