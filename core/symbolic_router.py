"""
symbolic_router.py – Centralized symbolic routing for ThothOS/Unimind.
Routes user intents and system events to appropriate scrolls, rituals, or module actions.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import re

class IntentType(Enum):
    """Types of intents that can be routed."""
    SCROLL = "scroll"
    RITUAL = "ritual"
    MODULE_ACTION = "module_action"
    SYSTEM_COMMAND = "system_command"
    FEEDBACK = "feedback"

@dataclass
class ActionPlan:
    """Represents a planned action with metadata."""
    action_type: IntentType
    target: str
    parameters: Dict[str, Any]
    confidence: float
    description: str
    fallback_actions: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.fallback_actions is None:
            self.fallback_actions = []

@dataclass
class ScrollDefinition:
    """Definition of a scroll with metadata."""
    name: str
    handler: Callable[..., Any]
    description: str
    triggers: List[str]
    required_modules: List[str]
    parameters: Dict[str, Any]
    category: str
    is_external: bool = False
    protected: bool = False
    founder_only: bool = False

class SymbolicRouter:
    """
    Centralized symbolic router for ThothOS/Unimind.
    Routes intents to scrolls, rituals, and module actions with fallback and introspection.
    """
    
    def __init__(self):
        """Initialize the symbolic router."""
        self.scroll_registry: Dict[str, ScrollDefinition] = {}
        self.module_registry: Dict[str, Dict[str, Callable[..., Any]]] = {}
        self.trigger_patterns: Dict[str, List[str]] = {}
        self.logger = logging.getLogger('SymbolicRouter')
        
        # Initialize default patterns
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize default trigger patterns for common intents."""
        self.trigger_patterns = {
            "optimize_self": [
                r"optimize\s+(?:yourself|self|system)",
                r"optimize\s+(?:your|the)\s+(?:llm|lam|communication)",
                r"optimize\s+(?:.*?)\s+(?:llm|lam)",
                r"optimize\s+(?:.*?)\s+(?:communication)",
                r"optimize\s+(?:.*?)",
                r"improve\s+(?:yourself|self|performance)",
                r"improve\s+(?:your|the)\s+(?:llm|lam|communication)",
                r"enhance\s+(?:yourself|self|capabilities)",
                r"enhance\s+(?:your|the)\s+(?:llm|lam|communication)",
                r"expand\s+(?:your|the)\s+(?:logic|capabilities|abilities)",
                r"better\s+(?:assistant|helper|ai)",
                r"upgrade\s+(?:yourself|self)",
                r"evolve\s+(?:yourself|self)",
                r"grow\s+(?:yourself|self)",
                r"develop\s+(?:yourself|self)",
                r"learn\s+(?:more|better)",
                r"become\s+(?:better|smarter|more\s+capable)",
                r"self\s+(?:improvement|development|enhancement)",
                r"help\s+(?:me\s+)?(?:expand|improve|enhance)\s+(?:your|the)\s+(?:logic|capabilities)"
            ],
            "reflect": [
                r"reflect\s+(?:on|about)",
                r"think\s+(?:about|on)",
                r"consider\s+(?:your|the)"
            ],
            "summon_persona": [
                r"summon\s+(?:persona|character)",
                r"become\s+(?:persona|character)",
                r"switch\s+to\s+(?:persona|character)"
            ],
            "web_search": [
                r"search\s+(?:for|about)",
                r"find\s+(?:information|data)",
                r"look\s+up"
            ],
            "location_search": [
                r"find\s+(?:a\s+)?(?:store|shop|place|location)",
                r"nearby\s+(?:store|shop|place)",
                r"where\s+(?:to\s+)?(?:buy|find|get)",
                r"nearest\s+(?:store|shop|place)",
                r"3d\s+printer\s+parts",
                r"electronics\s+store",
                r"hardware\s+store"
            ],
            "weather_check": [
                r"weather\s+(?:in|for|at)",
                r"temperature\s+(?:in|for|at)",
                r"forecast\s+(?:for|in)"
            ],
            "generate_3d_model": [
                r"generate\s+(?:a\s+)?3d\s+model",
                r"create\s+(?:a\s+)?3d\s+model",
                r"make\s+(?:a\s+)?3d\s+model",
                r"build\s+(?:a\s+)?3d\s+model",
                r"3d\s+model\s+(?:of|for)",
                r"model\s+(?:a|an)\s+",
                r"generate\s+(?:a\s+)?model"
            ],
            "generate_3d_scene": [
                r"generate\s+(?:a\s+)?3d\s+scene",
                r"create\s+(?:a\s+)?3d\s+scene",
                r"make\s+(?:a\s+)?3d\s+scene",
                r"build\s+(?:a\s+)?3d\s+scene",
                r"3d\s+scene\s+(?:of|for)",
                r"scene\s+(?:of|with)",
                r"generate\s+(?:a\s+)?scene"
            ],
            "optimize_3d_model": [
                r"optimize\s+(?:3d\s+)?model",
                r"reduce\s+model\s+complexity",
                r"simplify\s+(?:3d\s+)?model",
                r"optimize\s+3d\s+object",
                r"make\s+model\s+faster"
            ],
            "analyze_3d_model": [
                r"analyze\s+(?:3d\s+)?model",
                r"model\s+analysis",
                r"model\s+stats",
                r"model\s+info",
                r"check\s+model\s+properties",
                r"model\s+statistics"
            ],
            "convert_3d_format": [
                r"convert\s+(?:3d\s+)?model\s+format",
                r"convert\s+model\s+to",
                r"change\s+model\s+format",
                r"export\s+model\s+as",
                r"save\s+model\s+as"
            ],
            "create_realm": [
                r"create\s+(?:a\s+)?realm",
                r"build\s+(?:a\s+)?realm",
                r"make\s+(?:a\s+)?realm",
                r"new\s+realm",
                r"generate\s+(?:a\s+)?realm"
            ],
            "place_object": [
                r"place\s+(?:an\s+)?object",
                r"add\s+(?:an\s+)?object",
                r"put\s+(?:an\s+)?object",
                r"spawn\s+(?:an\s+)?object",
                r"insert\s+(?:an\s+)?object"
            ],
            "cast_glyph": [
                r"cast\s+(?:a\s+)?glyph",
                r"cast\s+(?:a\s+)?spell",
                r"magic\s+glyph",
                r"enchant",
                r"magical\s+effect"
            ],
            "list_realms": [
                r"list\s+realms",
                r"show\s+realms",
                r"realms",
                r"available\s+realms",
                r"what\s+realms\s+are\s+there"
            ]
        }
    
    def register_scroll(self, name: str, handler: Callable, description: str = "",
                       triggers: List[str] = None, required_modules: List[str] = None,
                       parameters: Dict[str, Any] = None, category: str = "general",
                       is_external: bool = False, protected: bool = False, 
                       founder_only: bool = False) -> None:
        """
        Register a scroll with the router.
        
        Args:
            name: Name of the scroll
            handler: Function to execute the scroll
            description: Human-readable description
            triggers: List of trigger patterns
            required_modules: Modules required for execution
            parameters: Expected parameters
            category: Category of the scroll
            is_external: Whether scroll requires external access
            protected: Whether scroll requires privileged access
            founder_only: Whether scroll requires founder access
        """
        self.scroll_registry[name] = ScrollDefinition(
            name=name,
            handler=handler,
            description=description,
            triggers=triggers or [],
            required_modules=required_modules or [],
            parameters=parameters or {},
            category=category,
            is_external=is_external,
            protected=protected,
            founder_only=founder_only
        )
        self.logger.info(f"Registered scroll: {name} ({description})")
    
    def register_module_actions(self, module_name: str, actions: Dict[str, Callable]) -> None:
        """
        Register module actions with the router.
        
        Args:
            module_name: Name of the module
            actions: Dictionary of action names to handlers
        """
        self.module_registry[module_name] = actions
        self.logger.info(f"Registered module actions for: {module_name}")
    
    def route_intent(self, intent: str, context: Dict[str, Any] = None) -> ActionPlan:
        """
        Route an intent to the appropriate action.
        
        Args:
            intent: The intent string to route
            context: Additional context for routing
            
        Returns:
            ActionPlan with the planned action
        """
        context = context or {}
        
        # First, try direct scroll match
        if intent in self.scroll_registry:
            scroll_def = self.scroll_registry[intent]
            return ActionPlan(
                action_type=IntentType.SCROLL,
                target=intent,
                parameters=context,
                confidence=1.0,
                description=scroll_def.description
            )
        
        # Try pattern matching for scrolls
        matched_scroll = self._match_patterns(intent)
        if matched_scroll:
            scroll_def = self.scroll_registry[matched_scroll]
            return ActionPlan(
                action_type=IntentType.SCROLL,
                target=matched_scroll,
                parameters=context,
                confidence=0.8,
                description=scroll_def.description
            )
        
        # Try module action matching
        module_action = self._match_module_action(intent)
        if module_action:
            module_name, action_name = module_action
            return ActionPlan(
                action_type=IntentType.MODULE_ACTION,
                target=f"{module_name}.{action_name}",
                parameters=context,
                confidence=0.7,
                description=f"Module action: {action_name}"
            )
        
        # Fallback to system command or error
        return ActionPlan(
            action_type=IntentType.SYSTEM_COMMAND,
            target="unknown",
            parameters=context,
            confidence=0.0,
            description=f"Unknown intent: {intent}",
            fallback_actions=["help", "list_scrolls"]
        )
    
    def _match_patterns(self, intent: str) -> Optional[str]:
        """Match intent against trigger patterns."""
        intent_lower = intent.lower()
        
        for scroll_name, patterns in self.trigger_patterns.items():
            if scroll_name in self.scroll_registry:
                for pattern in patterns:
                    if re.search(pattern, intent_lower):
                        return scroll_name
        
        return None
    
    def _match_module_action(self, intent: str) -> Optional[tuple]:
        """Match intent against module actions."""
        intent_lower = intent.lower()
        
        for module_name, actions in self.module_registry.items():
            for action_name in actions.keys():
                if action_name.lower() in intent_lower:
                    return (module_name, action_name)
        
        return None
    
    def execute_action(self, action_plan: ActionPlan) -> Dict[str, Any]:
        """
        Execute an action plan.
        
        Args:
            action_plan: The action plan to execute
            
        Returns:
            Result of the action execution
        """
        try:
            if action_plan.action_type == IntentType.SCROLL:
                if action_plan.target in self.scroll_registry:
                    scroll_def = self.scroll_registry[action_plan.target]
                    result = scroll_def.handler(action_plan.parameters)
                    return {
                        "success": True,
                        "result": result,
                        "action_type": "scroll",
                        "scroll_name": action_plan.target
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Scroll not found: {action_plan.target}"
                    }
            
            elif action_plan.action_type == IntentType.MODULE_ACTION:
                module_name, action_name = action_plan.target.split(".", 1)
                if module_name in self.module_registry and action_name in self.module_registry[module_name]:
                    handler = self.module_registry[module_name][action_name]
                    result = handler(action_plan.parameters)
                    return {
                        "success": True,
                        "result": result,
                        "action_type": "module_action",
                        "module": module_name,
                        "action": action_name
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Module action not found: {action_plan.target}"
                    }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown action type: {action_plan.action_type}"
                }
                
        except Exception as e:
            self.logger.error(f"Error executing action: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_scrolls(self, category: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        List all registered scrolls.
        
        Args:
            category: Optional category filter
            
        Returns:
            Dictionary of scroll information
        """
        scrolls = {}
        for name, scroll_def in self.scroll_registry.items():
            if category is None or scroll_def.category == category:
                scrolls[name] = {
                    "description": scroll_def.description,
                    "category": scroll_def.category,
                    "is_external": scroll_def.is_external,
                    "required_modules": scroll_def.required_modules,
                    "triggers": scroll_def.triggers
                }
        return scrolls
    
    def list_module_actions(self) -> Dict[str, List[str]]:
        """List all registered module actions."""
        actions = {}
        for module_name, module_actions in self.module_registry.items():
            actions[module_name] = list(module_actions.keys())
        return actions
    
    def get_scroll_info(self, scroll_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific scroll."""
        if scroll_name in self.scroll_registry:
            scroll_def = self.scroll_registry[scroll_name]
            return {
                "name": scroll_def.name,
                "description": scroll_def.description,
                "category": scroll_def.category,
                "is_external": scroll_def.is_external,
                "required_modules": scroll_def.required_modules,
                "triggers": scroll_def.triggers,
                "parameters": scroll_def.parameters
            }
        return None
    
    def add_trigger_pattern(self, scroll_name: str, pattern: str) -> None:
        """Add a trigger pattern for a scroll."""
        if scroll_name not in self.trigger_patterns:
            self.trigger_patterns[scroll_name] = []
        self.trigger_patterns[scroll_name].append(pattern)
    
    def remove_scroll(self, scroll_name: str) -> bool:
        """Remove a scroll from the registry."""
        if scroll_name in self.scroll_registry:
            del self.scroll_registry[scroll_name]
            self.logger.info(f"Removed scroll: {scroll_name}")
            return True
        return False
    
    def optimize_patterns(self) -> Dict[str, Any]:
        """Optimize trigger patterns for better matching."""
        self.logger.info("Optimizing trigger patterns")
        
        optimizations = {
            "patterns_optimized": len(self.trigger_patterns),
            "scrolls_registered": len(self.scroll_registry),
            "modules_registered": len(self.module_registry)
        }
        
        # Recompile regex patterns for better performance
        for scroll_name, patterns in self.trigger_patterns.items():
            try:
                # Validate patterns
                for pattern in patterns:
                    re.compile(pattern)
                optimizations[f"{scroll_name}_patterns"] = len(patterns)
            except re.error as e:
                self.logger.warning(f"Invalid pattern in {scroll_name}: {e}")
        
        self.logger.info(f"Pattern optimization complete: {optimizations}")
        return optimizations

# Global router instance
symbolic_router = SymbolicRouter()

def register_scroll(name: str, handler: Callable, description: str = "",
                   triggers: List[str] = None, required_modules: List[str] = None,
                   parameters: Dict[str, Any] = None, category: str = "general",
                   is_external: bool = False, protected: bool = False, 
                   founder_only: bool = False) -> None:
    """Register a scroll with the global symbolic router."""
    symbolic_router.register_scroll(
        name=name, handler=handler, description=description,
        triggers=triggers, required_modules=required_modules,
        parameters=parameters, category=category, is_external=is_external,
        protected=protected, founder_only=founder_only
    )

def route_intent(intent: str, context: Dict[str, Any] = None) -> ActionPlan:
    """Route an intent using the global router instance."""
    return symbolic_router.route_intent(intent, context)

def execute_action(action_plan: ActionPlan) -> Dict[str, Any]:
    """Execute an action plan using the global router instance."""
    return symbolic_router.execute_action(action_plan)

def list_scrolls(category: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """List scrolls using the global router instance."""
    return symbolic_router.list_scrolls(category) 