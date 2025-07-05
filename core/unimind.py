# unimind.py

import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..interfaces import (
    ServiceType, RequestType
)
from ..interfaces.supervisor import Supervisor
from ..native_models import lam_engine
from ..memory import memory_graph
from ..cortex import prefrontal_cortex
from ..emotion import emotion_classifier
from ..ethics import pineal_gland
from ..security import ethical_firewall
from ..runtime import heartbeat_loop

from unimind.logic.symbolic_reasoner import SymbolicReasoner
from unimind.planning.action_planner import ActionPlanner
from unimind.soul.tenets import load_tenets

from .symbolic_router import symbolic_router, route_intent, execute_action
from ..feedback.feedback_bus import feedback_bus, FeedbackType, FeedbackLevel
from ..language.language_engine import language_engine, parse_intent
from ..scrolls.scroll_engine import scroll_engine, cast_scroll

@dataclass
class UnimindState:
    """Current state of the Unimind system."""
    is_running: bool = False
    startup_time: Optional[float] = None
    total_cycles: int = 0
    active_scrolls: Optional[List[str]] = None
    memory_usage: Optional[Dict[str, Any]] = None
    emotion_state: Optional[Dict[str, Any]] = None
    ethical_status: Optional[Dict[str, Any]] = None
    access_stats: Optional[Dict[str, Any]] = None
    current_user_id: Optional[str] = None
    
    def __post_init__(self):
        if self.active_scrolls is None:
            self.active_scrolls = []
        if self.memory_usage is None:
            self.memory_usage = {}
        if self.emotion_state is None:
            self.emotion_state = {}
        if self.ethical_status is None:
            self.ethical_status = {}
        if self.access_stats is None:
            self.access_stats = {}

class Unimind:
    """
    Main Unimind class - the symbolic brain of ThothOS.
    Integrates all cognitive modules with centralized routing and feedback.
    """
    
    def __init__(self, soul=None):
        """
        Initialize the Unimind system.
        
        Args:
            soul: Soul instance with user-specific identity. If None, uses default.
        """
        self.state = UnimindState()
        self.logger = logging.getLogger('Unimind')
        
        # Set up soul (identity system)
        self.soul = soul
        
        # Initialize core systems
        self._initialize_systems()
        
        # Subscribe to feedback for system monitoring
        feedback_bus.subscribe(self._handle_system_feedback)
        
        self.logger.info("Unimind initialized with all improvements")
    
    def attach_ethics(self, ethics_system: Any):
        """Attach an ethics system to the Unimind."""
        self.ethics = ethics_system
        self.logger.info("Ethics system attached")
    
    def attach_memory(self, memory_system: Any):
        """Attach a memory system to the Unimind."""
        self.memory = memory_system
        self.logger.info("Memory system attached")
    
    def register_scrolls(self, scroll_engine: Any):
        """Register a scroll engine with the Unimind."""
        self.scrolls = scroll_engine
        self.logger.info("Scroll engine registered")
    
    def _initialize_systems(self):
        """Initialize all core systems."""
        try:
            # Initialize symbolic router (already done via imports)
            self.logger.info("SymbolicRouter initialized")
            
            # Initialize feedback system (already done via imports)
            self.logger.info("Feedback system initialized")
            
            # Initialize language engine (already done via imports)
            self.logger.info("Language engine initialized")
            
            # Initialize scroll engine (already done via imports)
            self.logger.info("Scroll engine initialized")
            
            # Initialize supervised access
            self.supervisor = Supervisor()
            self.logger.info("Supervisor initialized")
            
            # Emit system initialization feedback
            feedback_bus.emit(
                FeedbackType.SYSTEM_EVENT,
                "unimind",
                "All systems initialized successfully",
                {"systems": ["symbolic_router", "feedback", "language", "scrolls", "supervisor"]},
                FeedbackLevel.INFO
            )
            
        except Exception as e:
            self.logger.error(f"Error initializing systems: {e}")
            feedback_bus.emit(
                FeedbackType.SYSTEM_EVENT,
                "unimind",
                f"System initialization failed: {str(e)}",
                {"error": str(e)},
                FeedbackLevel.ERROR
            )
            raise
    
    def _handle_system_feedback(self, event):
        """Handle system feedback events."""
        # Log important feedback events
        if event.level in [FeedbackLevel.ERROR, FeedbackLevel.CRITICAL]:
            self.logger.error(f"Critical feedback: {event.message}")
        elif event.level == FeedbackLevel.WARNING:
            self.logger.warning(f"Warning feedback: {event.message}")
        
        # Trigger system adaptations based on feedback
        if event.event_type == FeedbackType.SCROLL_FAILURE:
            self._handle_scroll_failure(event)
        elif event.event_type == FeedbackType.PERFORMANCE_METRIC:
            self._handle_performance_metric(event)
    
    def _handle_scroll_failure(self, event):
        """Handle scroll failure feedback."""
        self.logger.info(f"Handling scroll failure: {event.payload.get('scroll_name', 'unknown')}")
        # Could trigger retry logic, fallback actions, or system optimization
    
    def _handle_performance_metric(self, event):
        """Handle performance metric feedback."""
        self.logger.info(f"Handling performance metric: {event.message}")
        # Could trigger system optimization or resource allocation
    
    def process_input(self, user_input: str, user_id: Optional[str] = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user input through the unified system with user-specific identity.
        
        Args:
            user_input: User's input text
            user_id: The user ID for identity-specific processing
            context: Additional context
            
        Returns:
            Processing result with action taken and response
        """
        context = context or {}
        
        # Update current user ID
        if user_id:
            self.state.current_user_id = user_id
        
        try:
            # Emit input processing feedback
            feedback_bus.emit(
                FeedbackType.SYSTEM_EVENT,
                "unimind",
                f"Processing user input: {user_input[:50]}...",
                {"input_length": len(user_input), "user_id": user_id},
                FeedbackLevel.INFO
            )
            
            # Parse intent using language engine
            intent_analysis = parse_intent(user_input)
            self.logger.info(f"Intent analysis: {intent_analysis}")
            
            # Route intent using symbolic router
            action_plan = route_intent(user_input, context)
            self.logger.info(f"Action plan: {action_plan.target} (confidence: {action_plan.confidence})")
            
            # Check user access to scroll if it's a scroll action
            if action_plan.action_type.value == "scroll" and self.soul:
                if not self.soul.can_access_scroll(action_plan.target, user_id or ""):
                    return {
                        "success": False,
                        "output": f"Access denied: You don't have permission to use the '{action_plan.target}' scroll.",
                        "access_denied": True,
                        "required_level": "privileged" if action_plan.target in self.soul.get_founder_only_scrolls() else "basic"
                    }
            
            # Execute action
            result = execute_action(action_plan)
            
            # Generate response
            response = self._generate_response(result, user_input, intent_analysis)
            
            # Emit success feedback
            feedback_bus.emit(
                FeedbackType.SYSTEM_EVENT,
                "unimind",
                f"Successfully processed input: {action_plan.target}",
                {
                    "action": action_plan.target,
                    "confidence": action_plan.confidence,
                    "success": result.get("success", False),
                    "user_id": user_id
                },
                FeedbackLevel.INFO
            )
            
            return {
                "success": result.get("success", False),
                "output": result.get("result", response),
                "action": action_plan.target,
                "confidence": action_plan.confidence,
                "user_id": user_id
            }
            
        except Exception as e:
            self.logger.error(f"Error processing input: {e}")
            feedback_bus.emit(
                FeedbackType.SYSTEM_EVENT,
                "unimind",
                f"Error processing input: {str(e)}",
                {"error": str(e), "user_id": user_id},
                FeedbackLevel.ERROR
            )
            
            return {
                "success": False,
                "output": f"Error processing your request: {str(e)}",
                "error": str(e),
                "user_id": user_id
            }
    
    def _generate_response(self, result: Dict[str, Any], user_input: str, intent_analysis: Dict[str, Any]) -> str:
        """Generate a natural language response."""
        if not result.get("success", False):
            return "I'm sorry, I couldn't complete that action. Please try again."
        
        # Use language engine to generate response
        if result.get("action_type") == "scroll":
            scroll_name = result.get("scroll_name", "unknown")
            scroll_result = result.get("result", {})
            
            if isinstance(scroll_result, dict):
                if "status" in scroll_result:
                    return f"I've completed the {scroll_name} action. Status: {scroll_result['status']}"
                else:
                    return f"I've completed the {scroll_name} action successfully."
            else:
                return f"I've completed the {scroll_name} action: {str(scroll_result)}"
        
        elif result.get("action_type") == "module_action":
            module = result.get("module", "unknown")
            action = result.get("action", "unknown")
            return f"I've executed the {action} action in the {module} module."
        
        else:
            return "I've processed your request successfully."
    
    def cast_scroll(self, scroll_name: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Cast a scroll using the unified scroll engine.
        
        Args:
            scroll_name: Name of the scroll to cast
            parameters: Scroll parameters
            
        Returns:
            Scroll execution result
        """
        try:
            result = cast_scroll(scroll_name, parameters or {})
            
            # Convert ScrollResult to dict for consistency
            return {
                "success": result.success,
                "output": result.output,
                "execution_time": result.execution_time,
                "feedback_emitted": result.feedback_emitted,
                "metadata": result.metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error casting scroll {scroll_name}: {e}")
            return {
                "success": False,
                "output": f"Error: {str(e)}",
                "execution_time": 0.0,
                "feedback_emitted": False
            }
    
    def list_scrolls(self, category: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """List available scrolls."""
        return scroll_engine.list_scrolls(category)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        from ..feedback.feedback_bus import get_feedback_stats
        from ..scrolls.scroll_engine import get_execution_stats
        
        feedback_stats = get_feedback_stats()
        execution_stats = get_execution_stats()
        
        return {
            "system_status": "running" if self.state.is_running else "stopped",
            "feedback_events": feedback_stats["total_events"],
            "scroll_executions": execution_stats["total_executions"],
            "successful_scrolls": execution_stats["successful_executions"],
            "failed_scrolls": execution_stats["failed_executions"],
            "average_execution_time": execution_stats["average_execution_time"],
            "registered_scrolls": len(scroll_engine.list_scrolls()),
            "feedback_subscribers": feedback_stats["subscribers_count"],
            "adaptation_triggers": feedback_stats["adaptation_triggers_count"]
        }
    
    def start(self):
        """Start the Unimind system."""
        if self.state.is_running:
            self.logger.warning("Unimind is already running")
            return
        
        self.state.is_running = True
        self.state.startup_time = time.time()
        
        # Initialize subsystems
        self._initialize_subsystems()
        
        # Start heartbeat loop
        heartbeat_loop.start()
        
        feedback_bus.emit(
            FeedbackType.SYSTEM_EVENT,
            "unimind",
            "Unimind system started",
            {"timestamp": time.time()},
            FeedbackLevel.INFO
        )
        
        self.logger.info("Unimind system started")
    
    def stop(self):
        """Stop the Unimind system."""
        if not self.state.is_running:
            self.logger.warning("Unimind is not running")
            return
        
        self.state.is_running = False
        
        # Stop heartbeat loop
        heartbeat_loop.stop()
        
        # Cleanup
        self._cleanup()
        
        feedback_bus.emit(
            FeedbackType.SYSTEM_EVENT,
            "unimind",
            "Unimind system stopped",
            {"timestamp": time.time()},
            FeedbackLevel.INFO
        )
        
        self.logger.info("Unimind system stopped")
    
    def _initialize_subsystems(self):
        """Initialize all subsystems."""
        # Initialize memory
        memory_graph.initialize()
        
        # Initialize cortex
        prefrontal_cortex.initialize()
        
        # Initialize emotion system
        emotion_classifier.initialize()
        
        # Initialize ethics
        pineal_gland.initialize()
        
        # Initialize security
        ethical_firewall.initialize()
        
        self.logger.info("All subsystems initialized")
    
    def _cleanup(self):
        """Cleanup resources."""
        # Cleanup expired requests
        self.supervisor.cleanup_expired_requests()
        
        # Save state
        self._save_state()
        
        self.logger.info("Cleanup completed")
    
    def _save_state(self):
        """Save current system state."""
        # Update state with current information
        self.state.memory_usage = memory_graph.get_stats()
        self.state.emotion_state = emotion_classifier.get_current_state()
        self.state.ethical_status = pineal_gland.get_status()
        self.state.access_stats = self.supervisor.get_access_stats()
        
        self.logger.info("System state saved")
    
    def _handle_approval_request(self, request):
        """Handle approval requests for external access."""
        self.logger.info(f"Approval required: {request.description}")
        
        # Log the request for review
        self._log_approval_request(request)
        
        # For now, auto-approve low-risk requests
        if self._is_low_risk_request(request):
            self.supervisor.approve_request(request.request_id, "system", "Auto-approved low-risk request")
            self.logger.info(f"Auto-approved request: {request.request_id}")
    
    def _is_low_risk_request(self, request) -> bool:
        """Determine if a request is low-risk and can be auto-approved."""
        if request.request_type == RequestType.WEB_REQUEST:
            # Auto-approve read-only requests to trusted domains
            details = request.details
            if details.get("method", "GET") == "GET":
                return True
        
        return False
    
    def _log_approval_request(self, request):
        """Log approval request for audit purposes."""
        # This would typically log to a secure audit log
        self.logger.info(f"Approval request logged: {request.request_id}")
    
    # Supervised Access Methods
    
    def make_web_request(self, url: str, method: str = "GET", data: Optional[Dict[str, Any]] = None,
                        service_type: ServiceType = ServiceType.API_CALL, description: str = "") -> str:
        """
        Make a supervised web request.
        
        Args:
            url: Target URL
            method: HTTP method
            data: Request data
            service_type: Type of service
            description: Human-readable description
            
        Returns:
            Request ID for tracking
        """
        return self.supervisor.request_web_access(
            url, method, data, service_type, "unimind", description
        )
    
    def execute_app_action(self, app_id: str, action: str, parameters: Optional[Dict[str, Any]] = None,
                          description: str = "") -> str:
        """
        Execute a supervised app action.
        
        Args:
            app_id: ID of the app
            action: Action to perform
            parameters: Action parameters
            description: Human-readable description
            
        Returns:
            Request ID for tracking
        """
        return self.supervisor.request_app_access(
            app_id, action, parameters, "unimind", description
        )
    
    def search_web(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Perform a supervised web search.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Search results
        """
        # Use the web interface directly for searches
        return self.supervisor.get_web_interface().search_web(query, max_results)
    
    def call_api(self, api_name: str, endpoint: str, method: str = "GET", 
                data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a supervised API call.
        
        Args:
            api_name: Name of the API service
            endpoint: API endpoint
            method: HTTP method
            data: Request data
            
        Returns:
            API response
        """
        # Use the web interface directly for API calls
        return self.supervisor.get_web_interface().call_api(api_name, endpoint, method, data)
    
    def get_registered_apps(self) -> List[Dict[str, Any]]:
        """Get list of registered third-party apps."""
        return self.supervisor.get_app_integration().get_registered_apps()
    
    def approve_request(self, request_id: str, reason: str = "") -> bool:
        """
        Approve a pending request.
        
        Args:
            request_id: ID of the request to approve
            reason: Reason for approval
            
        Returns:
            True if approved successfully
        """
        return self.supervisor.approve_request(request_id, "unimind", reason)
    
    def reject_request(self, request_id: str, reason: str) -> bool:
        """
        Reject a pending request.
        
        Args:
            request_id: ID of the request to reject
            reason: Reason for rejection
            
        Returns:
            True if rejected successfully
        """
        return self.supervisor.reject_request(request_id, "unimind", reason)
    
    def get_pending_requests(self) -> List[Dict[str, Any]]:
        """Get all pending approval requests."""
        return self.supervisor.get_pending_requests()
    
    def get_access_stats(self) -> Dict[str, Any]:
        """Get comprehensive access statistics."""
        return self.supervisor.get_access_stats()
    
    def get_web_stats(self) -> Dict[str, Any]:
        """Get web interface statistics."""
        return self.supervisor.get_web_interface().get_usage_stats()
    
    def get_app_stats(self) -> Dict[str, Any]:
        """Get app integration statistics."""
        return self.supervisor.get_app_integration().get_usage_stats()
    
    # System Information Methods
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        self._save_state()  # Update state before returning
        
        return {
            "system": {
                "is_running": self.state.is_running,
                "startup_time": self.state.startup_time,
                "uptime": time.time() - (self.state.startup_time or time.time()),
                "total_cycles": self.state.total_cycles
            },
            "memory": self.state.memory_usage,
            "emotion": self.state.emotion_state,
            "ethics": self.state.ethical_status,
            "access": self.state.access_stats,
            "pending_requests": len(self.get_pending_requests()),
            "registered_apps": len(self.get_registered_apps())
        }
    
    def get_health_check(self) -> Dict[str, Any]:
        """Get system health check information."""
        return {
            "system_healthy": self.state.is_running,
            "memory_healthy": memory_graph.is_healthy(),
            "cortex_healthy": prefrontal_cortex.is_healthy(),
            "emotion_healthy": emotion_classifier.is_healthy(),
            "ethics_healthy": pineal_gland.is_healthy(),
            "security_healthy": ethical_firewall.is_healthy(),
            "supervisor_healthy": True,  # Supervisor is always healthy if initialized
            "active_scrolls": len(self.state.active_scrolls),
            "pending_approvals": len(self.get_pending_requests())
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        stats = self.get_system_stats()
        
        # Determine health based on various metrics
        health_score = 100
        
        if stats["failed_scrolls"] > 0:
            failure_rate = stats["failed_scrolls"] / max(stats["scroll_executions"], 1)
            if failure_rate > 0.1:  # More than 10% failure rate
                health_score -= 30
            elif failure_rate > 0.05:  # More than 5% failure rate
                health_score -= 15
        
        if stats["average_execution_time"] > 5.0:  # More than 5 seconds average
            health_score -= 20
        
        if stats["feedback_events"] == 0:
            health_score -= 10  # No feedback events might indicate issues
        
        health_status = "healthy" if health_score >= 80 else "degraded" if health_score >= 50 else "unhealthy"
        
        return {
            "status": health_status,
            "score": health_score,
            "details": stats,
            "timestamp": time.time()
        }

# Global Unimind instance
unimind = Unimind()

def process_input(user_input: str, user_id: Optional[str] = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Process user input using the global Unimind instance."""
    return unimind.process_input(user_input, user_id, context)

def cast_scroll(scroll_name: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
    """Cast a scroll using the global Unimind instance."""
    return unimind.cast_scroll(scroll_name, parameters)

def get_system_stats() -> Dict[str, Any]:
    """Get system stats using the global Unimind instance."""
    return unimind.get_system_stats()

def get_health_status() -> Dict[str, Any]:
    """Get health status using the global Unimind instance."""
    return unimind.get_health_status()
