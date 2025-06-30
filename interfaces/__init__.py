"""
interfaces/__init__.py – Supervised access interfaces for Unimind daemon.
Provides safe, controlled access to external services and applications.
"""

from .web_interface import (
    WebInterface, ServiceType, PermissionLevel as WebPermissionLevel,
    WebRequest, WebResponse, web_interface,
    make_supervised_request, search_web_supervised, call_api_supervised
)

from .app_integration import (
    AppIntegration, AppType, PermissionLevel as AppPermissionLevel,
    AppConfig, AppRequest, AppResponse, app_integration,
    execute_app_action, get_registered_apps
)

from .supervisor import (
    Supervisor, ApprovalStatus, RequestType,
    ApprovalRequest, AccessLog, supervisor,
    request_web_access, request_app_access,
    approve_request, get_pending_requests, get_access_stats
)

from .system_control import SystemControl

def get_app_stats():
    """Get app integration statistics using the module-level instance."""
    return app_integration.get_usage_stats()

__all__ = [
    # Web Interface
    'WebInterface', 'ServiceType', 'WebPermissionLevel',
    'WebRequest', 'WebResponse', 'web_interface',
    'make_supervised_request', 'search_web_supervised', 'call_api_supervised',
    
    # App Integration
    'AppIntegration', 'AppType', 'AppPermissionLevel',
    'AppConfig', 'AppRequest', 'AppResponse', 'app_integration',
    'execute_app_action', 'get_registered_apps', 'get_app_stats',
    
    # Supervisor
    'Supervisor', 'ApprovalStatus', 'RequestType',
    'ApprovalRequest', 'AccessLog', 'supervisor',
    'request_web_access', 'request_app_access',
    'approve_request', 'get_pending_requests', 'get_access_stats',
    
    # System Control
    'SystemControl'
] 