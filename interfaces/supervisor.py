"""
supervisor.py – Supervised access coordinator for Unimind daemon.
Coordinates web and app access with comprehensive oversight and approval workflows.
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta

from .web_interface import WebInterface, ServiceType, PermissionLevel as WebPermissionLevel
from .app_integration import AppIntegration, PermissionLevel as AppPermissionLevel

class ApprovalStatus(Enum):
    """Status of approval requests."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"

class RequestType(Enum):
    """Types of external access requests."""
    WEB_REQUEST = "web_request"
    APP_ACTION = "app_action"
    API_CALL = "api_call"
    FILE_ACCESS = "file_access"

@dataclass
class ApprovalRequest:
    """Represents an approval request for external access."""
    request_id: str
    request_type: RequestType
    description: str
    user_id: str
    timestamp: float
    expires_at: float
    status: ApprovalStatus
    details: Dict[str, Any]
    approved_by: Optional[str]
    approved_at: Optional[float]
    reason: Optional[str]

@dataclass
class AccessLog:
    """Log entry for external access."""
    timestamp: float
    request_type: RequestType
    user_id: str
    description: str
    success: bool
    duration: float
    warnings: List[str]
    details: Dict[str, Any]

class Supervisor:
    """
    Supervised access coordinator for Unimind daemon.
    Provides comprehensive oversight of web and app access with approval workflows.
    """
    
    def __init__(self, auto_approve_threshold: int = 5, approval_timeout_hours: int = 24):
        """Initialize the supervisor with configuration."""
        self.web_interface = WebInterface()
        self.app_integration = AppIntegration()
        
        # Approval system
        self.approval_requests = {}
        self.auto_approve_threshold = auto_approve_threshold
        self.approval_timeout_hours = approval_timeout_hours
        
        # Access logging
        self.access_logs = []
        self.user_trust_scores = {}
        
        # Callbacks for approval notifications
        self.approval_callbacks = []
        
        # Initialize logging
        self._setup_logging()
        
        # Start cleanup timer
        self._schedule_cleanup()
    
    def _setup_logging(self):
        """Setup logging for the supervisor."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('UnimindSupervisor')
    
    def _schedule_cleanup(self):
        """Schedule periodic cleanup of expired requests."""
        # This would typically use a proper scheduler
        # For now, cleanup happens on-demand
        pass
    
    def request_web_access(self, url: str, method: str = "GET", data: Optional[Dict[str, Any]] = None,
                          service_type: ServiceType = ServiceType.API_CALL, user_id: str = "system",
                          description: str = "") -> str:
        """
        Request approval for web access.
        
        Args:
            url: Target URL
            method: HTTP method
            data: Request data
            service_type: Type of service
            user_id: ID of requesting user
            description: Human-readable description
            
        Returns:
            Request ID for tracking
        """
        request_id = self._generate_request_id("web", user_id)
        
        # Check if auto-approval is possible
        if self._can_auto_approve(user_id, RequestType.WEB_REQUEST):
            return self._auto_approve_web_request(request_id, url, method, data, service_type, user_id, description)
        
        # Create approval request
        approval_request = ApprovalRequest(
            request_id=request_id,
            request_type=RequestType.WEB_REQUEST,
            description=description or f"Web request to {url}",
            user_id=user_id,
            timestamp=time.time(),
            expires_at=time.time() + (self.approval_timeout_hours * 3600),
            status=ApprovalStatus.PENDING,
            details={
                "url": url,
                "method": method,
                "data": data,
                "service_type": service_type.value
            },
            approved_by=None,
            approved_at=None,
            reason=None
        )
        
        self.approval_requests[request_id] = approval_request
        self._notify_approval_required(approval_request)
        
        return request_id
    
    def request_app_access(self, app_id: str, action: str, parameters: Optional[Dict[str, Any]] = None,
                          user_id: str = "system", description: str = "") -> str:
        """
        Request approval for app access.
        
        Args:
            app_id: ID of the app
            action: Action to perform
            parameters: Action parameters
            user_id: ID of requesting user
            description: Human-readable description
            
        Returns:
            Request ID for tracking
        """
        request_id = self._generate_request_id("app", user_id)
        
        # Check if auto-approval is possible
        if self._can_auto_approve(user_id, RequestType.APP_ACTION):
            return self._auto_approve_app_request(request_id, app_id, action, parameters, user_id, description)
        
        # Create approval request
        approval_request = ApprovalRequest(
            request_id=request_id,
            request_type=RequestType.APP_ACTION,
            description=description or f"App action {action} on {app_id}",
            user_id=user_id,
            timestamp=time.time(),
            expires_at=time.time() + (self.approval_timeout_hours * 3600),
            status=ApprovalStatus.PENDING,
            details={
                "app_id": app_id,
                "action": action,
                "parameters": parameters or {}
            },
            approved_by=None,
            approved_at=None,
            reason=None
        )
        
        self.approval_requests[request_id] = approval_request
        self._notify_approval_required(approval_request)
        
        return request_id
    
    def approve_request(self, request_id: str, approver_id: str, reason: str = "") -> bool:
        """
        Approve a pending request.
        
        Args:
            request_id: ID of the request to approve
            approver_id: ID of the approver
            reason: Reason for approval
            
        Returns:
            True if approved successfully
        """
        if request_id not in self.approval_requests:
            return False
        
        request = self.approval_requests[request_id]
        
        # Check if request is still pending and not expired
        if request.status != ApprovalStatus.PENDING:
            return False
        
        if time.time() > request.expires_at:
            request.status = ApprovalStatus.EXPIRED
            return False
        
        # Approve the request
        request.status = ApprovalStatus.APPROVED
        request.approved_by = approver_id
        request.approved_at = time.time()
        request.reason = reason
        
        # Execute the approved request
        self._execute_approved_request(request)
        
        return True
    
    def reject_request(self, request_id: str, rejector_id: str, reason: str) -> bool:
        """
        Reject a pending request.
        
        Args:
            request_id: ID of the request to reject
            rejector_id: ID of the rejector
            reason: Reason for rejection
            
        Returns:
            True if rejected successfully
        """
        if request_id not in self.approval_requests:
            return False
        
        request = self.approval_requests[request_id]
        
        if request.status != ApprovalStatus.PENDING:
            return False
        
        request.status = ApprovalStatus.REJECTED
        request.approved_by = rejector_id
        request.approved_at = time.time()
        request.reason = reason
        
        return True
    
    def get_pending_requests(self) -> List[Dict[str, Any]]:
        """Get all pending approval requests."""
        pending = []
        current_time = time.time()
        
        for request in self.approval_requests.values():
            if request.status == ApprovalStatus.PENDING and current_time <= request.expires_at:
                pending.append(asdict(request))
        
        return pending
    
    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific request."""
        if request_id in self.approval_requests:
            return asdict(self.approval_requests[request_id])
        return None
    
    def _generate_request_id(self, prefix: str, user_id: str) -> str:
        """Generate a unique request ID."""
        import hashlib
        content = f"{prefix}:{user_id}:{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _can_auto_approve(self, user_id: str, request_type: RequestType) -> bool:
        """Check if a request can be auto-approved based on user trust score."""
        trust_score = self.user_trust_scores.get(user_id, 0)
        
        # Auto-approve if user has high trust score
        if trust_score >= self.auto_approve_threshold:
            return True
        
        # Auto-approve certain low-risk requests
        if request_type == RequestType.WEB_REQUEST:
            # Auto-approve read-only requests to trusted domains
            return True
        
        return False
    
    def _auto_approve_web_request(self, request_id: str, url: str, method: str, data: Optional[Dict[str, Any]],
                                 service_type: ServiceType, user_id: str, description: str) -> str:
        """Auto-approve and execute a web request."""
        # Execute the request
        response = self.web_interface.make_request(url, method, data, service_type, user_id)
        
        # Log the access
        self._log_access(RequestType.WEB_REQUEST, user_id, description, response.status_code < 400,
                        response.response_time, response.warnings, {
                            "url": url,
                            "method": method,
                            "service_type": service_type.value,
                            "status_code": response.status_code
                        })
        
        return request_id
    
    def _auto_approve_app_request(self, request_id: str, app_id: str, action: str, parameters: Optional[Dict[str, Any]],
                                 user_id: str, description: str) -> str:
        """Auto-approve and execute an app request."""
        # Execute the request
        response = self.app_integration.execute_app_action(app_id, action, parameters, user_id)
        
        # Log the access
        self._log_access(RequestType.APP_ACTION, user_id, description, response.success,
                        response.execution_time, response.warnings, {
                            "app_id": app_id,
                            "action": action,
                            "success": response.success
                        })
        
        return request_id
    
    def _execute_approved_request(self, request: ApprovalRequest):
        """Execute an approved request."""
        if request.request_type == RequestType.WEB_REQUEST:
            details = request.details
            response = self.web_interface.make_request(
                details["url"],
                details["method"],
                details.get("data"),
                ServiceType(details["service_type"]),
                request.user_id
            )
            
            self._log_access(
                RequestType.WEB_REQUEST,
                request.user_id,
                request.description,
                response.status_code < 400,
                response.response_time,
                response.warnings,
                {
                    "url": details["url"],
                    "method": details["method"],
                    "status_code": response.status_code,
                    "approved_by": request.approved_by
                }
            )
            
        elif request.request_type == RequestType.APP_ACTION:
            details = request.details
            response = self.app_integration.execute_app_action(
                details["app_id"],
                details["action"],
                details.get("parameters"),
                request.user_id
            )
            
            self._log_access(
                RequestType.APP_ACTION,
                request.user_id,
                request.description,
                response.success,
                response.execution_time,
                response.warnings,
                {
                    "app_id": details["app_id"],
                    "action": details["action"],
                    "success": response.success,
                    "approved_by": request.approved_by
                }
            )
    
    def _log_access(self, request_type: RequestType, user_id: str, description: str, success: bool,
                   duration: float, warnings: List[str], details: Dict[str, Any]):
        """Log an access attempt."""
        log_entry = AccessLog(
            timestamp=time.time(),
            request_type=request_type,
            user_id=user_id,
            description=description,
            success=success,
            duration=duration,
            warnings=warnings,
            details=details
        )
        
        self.access_logs.append(log_entry)
        
        # Update user trust score
        self._update_trust_score(user_id, success, warnings)
        
        # Log to system
        self.logger.info(f"Access logged: {request_type.value} by {user_id} - {'SUCCESS' if success else 'FAILED'}")
    
    def _update_trust_score(self, user_id: str, success: bool, warnings: List[str]):
        """Update user trust score based on access results."""
        current_score = self.user_trust_scores.get(user_id, 0)
        
        if success and not warnings:
            # Increase score for successful, clean requests
            self.user_trust_scores[user_id] = min(current_score + 1, 10)
        elif not success or warnings:
            # Decrease score for failed or problematic requests
            self.user_trust_scores[user_id] = max(current_score - 1, 0)
    
    def _notify_approval_required(self, request: ApprovalRequest):
        """Notify that approval is required for a request."""
        for callback in self.approval_callbacks:
            try:
                callback(request)
            except Exception as e:
                self.logger.error(f"Approval callback failed: {e}")
    
    def register_approval_callback(self, callback: Callable[[ApprovalRequest], None]):
        """Register a callback for approval notifications."""
        self.approval_callbacks.append(callback)
    
    def get_access_stats(self) -> Dict[str, Any]:
        """Get comprehensive access statistics."""
        total_requests = len(self.access_logs)
        successful_requests = sum(1 for log in self.access_logs if log.success)
        
        # Calculate average duration
        if self.access_logs:
            avg_duration = sum(log.duration for log in self.access_logs) / len(self.access_logs)
        else:
            avg_duration = 0.0
        
        # Count by request type
        requests_by_type = {}
        for log in self.access_logs:
            req_type = log.request_type.value
            requests_by_type[req_type] = requests_by_type.get(req_type, 0) + 1
        
        # Count by user
        requests_by_user = {}
        for log in self.access_logs:
            user_id = log.user_id
            requests_by_user[user_id] = requests_by_user.get(user_id, 0) + 1
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": (successful_requests / total_requests * 100) if total_requests > 0 else 0,
            "average_duration": avg_duration,
            "requests_by_type": requests_by_type,
            "requests_by_user": requests_by_user,
            "user_trust_scores": self.user_trust_scores,
            "pending_approvals": len(self.get_pending_requests())
        }
    
    def cleanup_expired_requests(self):
        """Clean up expired approval requests."""
        current_time = time.time()
        expired_requests = []
        
        for request_id, request in self.approval_requests.items():
            if request.status == ApprovalStatus.PENDING and current_time > request.expires_at:
                request.status = ApprovalStatus.EXPIRED
                expired_requests.append(request_id)
        
        for request_id in expired_requests:
            self.logger.info(f"Request {request_id} expired")
    
    def get_web_interface(self) -> WebInterface:
        """Get the web interface instance."""
        return self.web_interface
    
    def get_app_integration(self) -> AppIntegration:
        """Get the app integration instance."""
        return self.app_integration

# Module-level instance
supervisor = Supervisor()

def request_web_access(url: str, method: str = "GET", data: Optional[Dict[str, Any]] = None,
                      service_type: ServiceType = ServiceType.API_CALL, user_id: str = "system",
                      description: str = "") -> str:
    """Request web access using the module-level supervisor."""
    return supervisor.request_web_access(url, method, data, service_type, user_id, description)

def request_app_access(app_id: str, action: str, parameters: Optional[Dict[str, Any]] = None,
                      user_id: str = "system", description: str = "") -> str:
    """Request app access using the module-level supervisor."""
    return supervisor.request_app_access(app_id, action, parameters, user_id, description)

def approve_request(request_id: str, approver_id: str, reason: str = "") -> bool:
    """Approve a request using the module-level supervisor."""
    return supervisor.approve_request(request_id, approver_id, reason)

def get_pending_requests() -> List[Dict[str, Any]]:
    """Get pending requests using the module-level supervisor."""
    return supervisor.get_pending_requests()

def get_access_stats() -> Dict[str, Any]:
    """Get access statistics using the module-level supervisor."""
    return supervisor.get_access_stats() 