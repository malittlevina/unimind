"""
app_integration.py – Third-party app integration for Unimind daemon.
Provides supervised access to external applications and services.
"""

import os
import json
import time
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

class AppType(Enum):
    """Types of third-party applications."""
    EMAIL_CLIENT = "email_client"
    CALENDAR_APP = "calendar_app"
    FILE_MANAGER = "file_manager"
    TEXT_EDITOR = "text_editor"
    BROWSER = "browser"
    TERMINAL = "terminal"
    MEDIA_PLAYER = "media_player"
    CHAT_APP = "chat_app"
    PRODUCTIVITY = "productivity"
    DEVELOPMENT = "development"

class PermissionLevel(Enum):
    """Permission levels for app access."""
    READ_ONLY = "read_only"
    LIMITED_WRITE = "limited_write"
    FULL_ACCESS = "full_access"
    BLOCKED = "blocked"

@dataclass
class AppConfig:
    """Configuration for a third-party app."""
    name: str
    app_type: AppType
    executable_path: str
    permission_level: PermissionLevel
    allowed_actions: List[str]
    blocked_actions: List[str]
    timeout_seconds: int
    requires_auth: bool
    auth_method: Optional[str]

@dataclass
class AppRequest:
    """Represents a request to interact with a third-party app."""
    app_name: str
    action: str
    parameters: Dict[str, Any]
    user_id: str
    timestamp: float
    request_id: str

@dataclass
class AppResponse:
    """Represents a response from a third-party app."""
    success: bool
    output: str
    error_message: Optional[str]
    execution_time: float
    is_safe: bool
    warnings: List[str]

class AppIntegration:
    """
    Supervised third-party app integration for Unimind daemon.
    Provides safe, controlled access to external applications.
    """
    
    def __init__(self):
        """Initialize the app integration system."""
        self.registered_apps = {}
        self.app_configs = {}
        self.request_history = []
        self.active_sessions = {}
        self.safety_filters = []
        
        # Initialize safety filters
        self._initialize_safety_filters()
        
        # Register common apps
        self._register_common_apps()
    
    def _initialize_safety_filters(self):
        """Initialize safety filters for app interactions."""
        self.safety_filters = [
            self._filter_dangerous_commands,
            self._filter_system_access,
            self._filter_network_access,
            self._filter_file_access
        ]
    
    def _register_common_apps(self):
        """Register common third-party applications."""
        common_apps = {
            "email": {
                "name": "Email Client",
                "app_type": AppType.EMAIL_CLIENT,
                "executable_path": "/usr/bin/mail",
                "permission_level": PermissionLevel.LIMITED_WRITE,
                "allowed_actions": ["send", "read", "list"],
                "blocked_actions": ["delete", "forward", "reply"],
                "timeout_seconds": 30,
                "requires_auth": True,
                "auth_method": "oauth2"
            },
            "calendar": {
                "name": "Calendar App",
                "app_type": AppType.CALENDAR_APP,
                "executable_path": "/usr/bin/cal",
                "permission_level": PermissionLevel.READ_ONLY,
                "allowed_actions": ["view", "list"],
                "blocked_actions": ["create", "modify", "delete"],
                "timeout_seconds": 10,
                "requires_auth": False,
                "auth_method": None
            },
            "file_manager": {
                "name": "File Manager",
                "app_type": AppType.FILE_MANAGER,
                "executable_path": "/usr/bin/ls",
                "permission_level": PermissionLevel.READ_ONLY,
                "allowed_actions": ["list", "view"],
                "blocked_actions": ["delete", "move", "copy"],
                "timeout_seconds": 15,
                "requires_auth": False,
                "auth_method": None
            },
            "text_editor": {
                "name": "Text Editor",
                "app_type": AppType.TEXT_EDITOR,
                "executable_path": "/usr/bin/nano",
                "permission_level": PermissionLevel.LIMITED_WRITE,
                "allowed_actions": ["open", "read", "edit"],
                "blocked_actions": ["save", "delete"],
                "timeout_seconds": 60,
                "requires_auth": False,
                "auth_method": None
            }
        }
        
        for app_id, config in common_apps.items():
            self.register_app(app_id, AppConfig(**config))
    
    def register_app(self, app_id: str, config: AppConfig):
        """
        Register a third-party application.
        
        Args:
            app_id: Unique identifier for the app
            config: App configuration
        """
        self.app_configs[app_id] = config
        logging.info(f"Registered app: {app_id} ({config.name})")
    
    def execute_app_action(self, app_id: str, action: str, parameters: Dict[str, Any] = None, 
                          user_id: str = "system") -> AppResponse:
        """
        Execute an action on a registered third-party app.
        
        Args:
            app_id: ID of the app to interact with
            action: Action to perform
            parameters: Action parameters
            user_id: ID of the requesting user
            
        Returns:
            AppResponse with results and safety information
        """
        if app_id not in self.app_configs:
            return AppResponse(
                success=False,
                output="",
                error_message=f"App '{app_id}' not registered",
                execution_time=0.0,
                is_safe=False,
                warnings=["Unknown app"]
            )
        
        config = self.app_configs[app_id]
        parameters = parameters or {}
        
        # Generate request ID
        request_id = self._generate_request_id(app_id, action, user_id)
        
        # Create request object
        request = AppRequest(
            app_name=app_id,
            action=action,
            parameters=parameters,
            user_id=user_id,
            timestamp=time.time(),
            request_id=request_id
        )
        
        # Safety checks
        safety_check = self._perform_safety_checks(request, config)
        if not safety_check["allowed"]:
            return AppResponse(
                success=False,
                output="",
                error_message=f"Action blocked: {safety_check['reason']}",
                execution_time=0.0,
                is_safe=False,
                warnings=safety_check["warnings"]
            )
        
        # Check permissions
        if not self._check_permissions(action, config):
            return AppResponse(
                success=False,
                output="",
                error_message=f"Action '{action}' not allowed for app '{app_id}'",
                execution_time=0.0,
                is_safe=False,
                warnings=["Permission denied"]
            )
        
        # Execute the action
        try:
            start_time = time.time()
            output, error = self._execute_action(config, action, parameters)
            execution_time = time.time() - start_time
            
            # Check if execution was successful
            success = error is None
            
            # Check response safety
            is_safe, warnings = self._check_response_safety(output, error)
            
            # Store in history
            self.request_history.append({
                "request": request,
                "execution_time": execution_time,
                "success": success,
                "timestamp": time.time()
            })
            
            return AppResponse(
                success=success,
                output=output or "",
                error_message=error,
                execution_time=execution_time,
                is_safe=is_safe,
                warnings=warnings
            )
            
        except Exception as e:
            logging.error(f"App action failed: {e}")
            return AppResponse(
                success=False,
                output="",
                error_message=str(e),
                execution_time=0.0,
                is_safe=False,
                warnings=[f"Execution failed: {str(e)}"]
            )
    
    def _generate_request_id(self, app_id: str, action: str, user_id: str) -> str:
        """Generate a unique request ID."""
        import hashlib
        content = f"{app_id}:{action}:{user_id}:{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def _perform_safety_checks(self, request: AppRequest, config: AppConfig) -> Dict[str, Any]:
        """Perform comprehensive safety checks on the app request."""
        warnings = []
        
        # Check if app is blocked
        if config.permission_level == PermissionLevel.BLOCKED:
            return {"allowed": False, "reason": "App is blocked", "warnings": warnings}
        
        # Apply safety filters
        for filter_func in self.safety_filters:
            filter_result = filter_func(request, config)
            if not filter_result["allowed"]:
                return {"allowed": False, "reason": filter_result["reason"], "warnings": warnings}
            warnings.extend(filter_result["warnings"])
        
        return {"allowed": True, "reason": None, "warnings": warnings}
    
    def _filter_dangerous_commands(self, request: AppRequest, config: AppConfig) -> Dict[str, Any]:
        """Filter out dangerous commands."""
        dangerous_actions = [
            "rm", "del", "format", "shutdown", "reboot", "kill", "terminate",
            "delete", "remove", "destroy", "wipe", "clear"
        ]
        
        action_lower = request.action.lower()
        for dangerous in dangerous_actions:
            if dangerous in action_lower:
                return {
                    "allowed": False,
                    "reason": f"Dangerous action detected: {dangerous}",
                    "warnings": []
                }
        
        return {"allowed": True, "reason": None, "warnings": []}
    
    def _filter_system_access(self, request: AppRequest, config: AppConfig) -> Dict[str, Any]:
        """Filter out system-level access attempts."""
        system_actions = [
            "sudo", "root", "admin", "system", "kernel", "driver"
        ]
        
        action_lower = request.action.lower()
        for system_action in system_actions:
            if system_action in action_lower:
                return {
                    "allowed": False,
                    "reason": f"System access attempt detected: {system_action}",
                    "warnings": []
                }
        
        return {"allowed": True, "reason": None, "warnings": []}
    
    def _filter_network_access(self, request: AppRequest, config: AppConfig) -> Dict[str, Any]:
        """Filter out network access attempts."""
        network_actions = [
            "curl", "wget", "ftp", "ssh", "telnet", "netcat", "nc"
        ]
        
        action_lower = request.action.lower()
        for network_action in network_actions:
            if network_action in action_lower:
                return {
                    "allowed": False,
                    "reason": f"Network access attempt detected: {network_action}",
                    "warnings": []
                }
        
        return {"allowed": True, "reason": None, "warnings": []}
    
    def _filter_file_access(self, request: AppRequest, config: AppConfig) -> Dict[str, Any]:
        """Filter out sensitive file access attempts."""
        sensitive_paths = [
            "/etc/passwd", "/etc/shadow", "/root", "/home",
            "/var/log", "/proc", "/sys", "/dev"
        ]
        
        # Check parameters for file paths
        for param_value in request.parameters.values():
            if isinstance(param_value, str):
                for sensitive_path in sensitive_paths:
                    if sensitive_path in param_value:
                        return {
                            "allowed": False,
                            "reason": f"Sensitive file access attempt: {sensitive_path}",
                            "warnings": []
                        }
        
        return {"allowed": True, "reason": None, "warnings": []}
    
    def _check_permissions(self, action: str, config: AppConfig) -> bool:
        """Check if the action is allowed for the app."""
        # Check if action is explicitly blocked
        if action in config.blocked_actions:
            return False
        
        # Check if action is explicitly allowed
        if action in config.allowed_actions:
            return True
        
        # Check permission level
        if config.permission_level == PermissionLevel.READ_ONLY:
            return action in ["read", "view", "list", "show"]
        elif config.permission_level == PermissionLevel.LIMITED_WRITE:
            return action in ["read", "view", "list", "show", "edit", "modify"]
        elif config.permission_level == PermissionLevel.FULL_ACCESS:
            return True
        
        return False
    
    def _execute_action(self, config: AppConfig, action: str, parameters: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """Execute the actual app action."""
        try:
            # Build command based on app type and action
            if config.app_type == AppType.EMAIL_CLIENT:
                return self._execute_email_action(config, action, parameters)
            elif config.app_type == AppType.CALENDAR_APP:
                return self._execute_calendar_action(config, action, parameters)
            elif config.app_type == AppType.FILE_MANAGER:
                return self._execute_file_action(config, action, parameters)
            elif config.app_type == AppType.TEXT_EDITOR:
                return self._execute_text_editor_action(config, action, parameters)
            else:
                # Generic execution
                return self._execute_generic_action(config, action, parameters)
                
        except subprocess.TimeoutExpired:
            return None, f"Action timed out after {config.timeout_seconds} seconds"
        except Exception as e:
            return None, str(e)
    
    def _execute_email_action(self, config: AppConfig, action: str, parameters: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """Execute email client actions."""
        if action == "list":
            # Simulate listing emails
            return "Email list retrieved successfully", None
        elif action == "read":
            # Simulate reading an email
            email_id = parameters.get("id", "unknown")
            return f"Email {email_id} content retrieved", None
        elif action == "send":
            # Simulate sending an email
            return "Email sent successfully", None
        else:
            return None, f"Unknown email action: {action}"
    
    def _execute_calendar_action(self, config: AppConfig, action: str, parameters: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """Execute calendar app actions."""
        if action == "view":
            # Simulate viewing calendar
            return "Calendar view displayed", None
        elif action == "list":
            # Simulate listing events
            return "Calendar events listed", None
        else:
            return None, f"Unknown calendar action: {action}"
    
    def _execute_file_action(self, config: AppConfig, action: str, parameters: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """Execute file manager actions."""
        if action == "list":
            path = parameters.get("path", ".")
            try:
                result = subprocess.run(
                    [config.executable_path, path],
                    capture_output=True,
                    text=True,
                    timeout=config.timeout_seconds
                )
                if result.returncode == 0:
                    return result.stdout, None
                else:
                    return None, result.stderr
            except Exception as e:
                return None, str(e)
        else:
            return None, f"Unknown file action: {action}"
    
    def _execute_text_editor_action(self, config: AppConfig, action: str, parameters: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """Execute text editor actions."""
        if action == "open":
            file_path = parameters.get("file", "")
            if not file_path:
                return None, "No file specified"
            
            # Simulate opening a file
            return f"File {file_path} opened in editor", None
        elif action == "read":
            file_path = parameters.get("file", "")
            if not file_path:
                return None, "No file specified"
            
            # Simulate reading file content
            return f"File {file_path} content read", None
        else:
            return None, f"Unknown text editor action: {action}"
    
    def _execute_generic_action(self, config: AppConfig, action: str, parameters: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """Execute generic app actions."""
        # This would handle generic app execution
        # For now, return a placeholder
        return f"Action '{action}' executed on {config.name}", None
    
    def _check_response_safety(self, output: Optional[str], error: Optional[str]) -> Tuple[bool, List[str]]:
        """Check if the app response is safe."""
        warnings = []
        
        if error:
            warnings.append(f"Error occurred: {error}")
        
        if output and len(output) > 10000:  # 10KB limit
            warnings.append("Output too large")
        
        # Check for sensitive information in output
        sensitive_patterns = [
            r'password.*:', r'secret.*:', r'key.*:', r'token.*:',
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
        ]
        
        import re
        if output:
            for pattern in sensitive_patterns:
                if re.search(pattern, output, re.IGNORECASE):
                    warnings.append("Sensitive information detected in output")
                    break
        
        is_safe = len(warnings) == 0
        return is_safe, warnings
    
    def get_registered_apps(self) -> List[Dict[str, Any]]:
        """Get list of registered apps."""
        apps = []
        for app_id, config in self.app_configs.items():
            apps.append({
                "id": app_id,
                "name": config.name,
                "type": config.app_type.value,
                "permission_level": config.permission_level.value,
                "allowed_actions": config.allowed_actions,
                "blocked_actions": config.blocked_actions
            })
        return apps
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for monitoring."""
        return {
            "total_requests": len(self.request_history),
            "successful_requests": sum(1 for record in self.request_history if record["success"]),
            "failed_requests": sum(1 for record in self.request_history if not record["success"]),
            "average_execution_time": self._calculate_average_execution_time(),
            "requests_by_app": self._count_requests_by_app()
        }
    
    def _calculate_average_execution_time(self) -> float:
        """Calculate average execution time."""
        if not self.request_history:
            return 0.0
        total_time = sum(record["execution_time"] for record in self.request_history)
        return total_time / len(self.request_history)
    
    def _count_requests_by_app(self) -> Dict[str, int]:
        """Count requests by app."""
        counts = {}
        for record in self.request_history:
            app_name = record["request"].app_name
            counts[app_name] = counts.get(app_name, 0) + 1
        return counts

# Module-level instance
app_integration = AppIntegration()

def execute_app_action(app_id: str, action: str, parameters: Dict[str, Any] = None, 
                      user_id: str = "system") -> AppResponse:
    """Execute an app action using the module-level instance."""
    return app_integration.execute_app_action(app_id, action, parameters, user_id)

def get_registered_apps() -> List[Dict[str, Any]]:
    """Get registered apps using the module-level instance."""
    return app_integration.get_registered_apps() 