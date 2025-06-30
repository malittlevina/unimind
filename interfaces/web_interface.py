"""
web_interface.py – Supervised internet and third-party app integration for Unimind.
Provides safe, controlled access to external services with ethical oversight.
"""

import requests
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse
import logging

class ServiceType(Enum):
    """Types of external services."""
    WEB_SEARCH = "web_search"
    API_CALL = "api_call"
    FILE_DOWNLOAD = "file_download"
    EMAIL = "email"
    CALENDAR = "calendar"
    WEATHER = "weather"
    NEWS = "news"
    TRANSLATION = "translation"
    MAPS = "maps"

class PermissionLevel(Enum):
    """Permission levels for external access."""
    RESTRICTED = "restricted"    # Only whitelisted domains
    SUPERVISED = "supervised"    # Requires approval
    OPEN = "open"               # Full access (dangerous)
    BLOCKED = "blocked"         # No access

@dataclass
class WebRequest:
    """Represents a web request with metadata."""
    url: str
    method: str
    headers: Dict[str, str]
    data: Optional[Dict[str, Any]]
    service_type: ServiceType
    timestamp: float
    user_id: str
    request_id: str

@dataclass
class WebResponse:
    """Represents a web response with metadata."""
    status_code: int
    content: str
    headers: Dict[str, str]
    response_time: float
    size_bytes: int
    is_safe: bool
    warnings: List[str]

class WebInterface:
    """
    Supervised web interface for Unimind daemon.
    Provides safe, controlled access to external services with ethical oversight.
    """
    
    def __init__(self, permission_level: PermissionLevel = PermissionLevel.SUPERVISED):
        """Initialize the web interface with safety controls."""
        self.permission_level = permission_level
        self.session = requests.Session()
        self.request_history = []
        self.rate_limits = {}
        self.blocked_domains = set()
        self.whitelisted_domains = set()
        self.api_keys = {}
        self.ethical_filters = []
        
        # Configure session for safety
        self.session.headers.update({
            'User-Agent': 'Unimind-Daemon/1.0 (Supervised-AI)',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        
        # Rate limiting configuration
        self.rate_limit_config = {
            ServiceType.WEB_SEARCH: {"requests_per_minute": 10, "requests_per_hour": 100},
            ServiceType.API_CALL: {"requests_per_minute": 30, "requests_per_hour": 300},
            ServiceType.FILE_DOWNLOAD: {"requests_per_minute": 5, "requests_per_hour": 50},
            ServiceType.EMAIL: {"requests_per_minute": 5, "requests_per_hour": 50},
            ServiceType.CALENDAR: {"requests_per_minute": 10, "requests_per_hour": 100},
            ServiceType.WEATHER: {"requests_per_minute": 20, "requests_per_hour": 200},
            ServiceType.NEWS: {"requests_per_minute": 15, "requests_per_hour": 150},
            ServiceType.TRANSLATION: {"requests_per_minute": 25, "requests_per_hour": 250},
            ServiceType.MAPS: {"requests_per_minute": 10, "requests_per_hour": 100},
        }
        
        # Initialize safety settings
        self._initialize_safety_settings()
        
    def _initialize_safety_settings(self):
        """Initialize safety settings and filters."""
        # Blocked domains (malicious, inappropriate, etc.)
        self.blocked_domains.update([
            "malware.example.com",
            "phishing.example.com",
            "inappropriate.example.com"
        ])
        
        # Whitelisted domains (trusted services)
        self.whitelisted_domains.update([
            "api.openai.com",
            "api.github.com",
            "api.weather.gov",
            "api.nasa.gov",
            "api.nytimes.com",
            "translate.googleapis.com",
            "maps.googleapis.com",
            "calendar.googleapis.com",
            "gmail.googleapis.com"
        ])
        
        # Ethical filters
        self.ethical_filters = [
            self._filter_harmful_content,
            self._filter_personal_data,
            self._filter_inappropriate_requests
        ]
    
    def make_request(self, url: str, method: str = "GET", data: Optional[Dict[str, Any]] = None, 
                    service_type: ServiceType = ServiceType.API_CALL, user_id: str = "system") -> WebResponse:
        """
        Make a supervised web request with safety checks.
        
        Args:
            url: Target URL
            method: HTTP method
            data: Request data
            service_type: Type of service being accessed
            user_id: ID of the requesting user/system
            
        Returns:
            WebResponse with results and safety information
        """
        # Generate request ID
        request_id = self._generate_request_id(url, method, user_id)
        
        # Create request object
        request = WebRequest(
            url=url,
            method=method,
            headers=self.session.headers.copy(),
            data=data,
            service_type=service_type,
            timestamp=time.time(),
            user_id=user_id,
            request_id=request_id
        )
        
        # Safety checks
        safety_check = self._perform_safety_checks(request)
        if not safety_check["allowed"]:
            return WebResponse(
                status_code=403,
                content=json.dumps({"error": "Request blocked by safety filters", "reason": safety_check["reason"]}),
                headers={},
                response_time=0.0,
                size_bytes=0,
                is_safe=False,
                warnings=safety_check["warnings"]
            )
        
        # Rate limiting check
        if not self._check_rate_limit(service_type):
            return WebResponse(
                status_code=429,
                content=json.dumps({"error": "Rate limit exceeded"}),
                headers={},
                response_time=0.0,
                size_bytes=0,
                is_safe=False,
                warnings=["Rate limit exceeded"]
            )
        
        # Make the actual request
        try:
            start_time = time.time()
            response = self.session.request(method, url, json=data, timeout=30)
            response_time = time.time() - start_time
            
            # Process response
            content = response.text
            size_bytes = len(content.encode('utf-8'))
            
            # Check response safety
            is_safe, warnings = self._check_response_safety(response)
            
            # Store in history
            self.request_history.append({
                "request": request,
                "response_time": response_time,
                "status_code": response.status_code,
                "size_bytes": size_bytes,
                "timestamp": time.time()
            })
            
            return WebResponse(
                status_code=response.status_code,
                content=content,
                headers=dict(response.headers),
                response_time=response_time,
                size_bytes=size_bytes,
                is_safe=is_safe,
                warnings=warnings
            )
            
        except Exception as e:
            logging.error(f"Web request failed: {e}")
            return WebResponse(
                status_code=500,
                content=json.dumps({"error": str(e)}),
                headers={},
                response_time=0.0,
                size_bytes=0,
                is_safe=False,
                warnings=[f"Request failed: {str(e)}"]
            )
    
    def _generate_request_id(self, url: str, method: str, user_id: str) -> str:
        """Generate a unique request ID."""
        content = f"{url}:{method}:{user_id}:{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def _perform_safety_checks(self, request: WebRequest) -> Dict[str, Any]:
        """Perform comprehensive safety checks on the request."""
        warnings = []
        reason = None
        
        # Check permission level
        if self.permission_level == PermissionLevel.BLOCKED:
            return {"allowed": False, "reason": "Web access is blocked", "warnings": warnings}
        
        # Parse URL
        try:
            parsed_url = urlparse(request.url)
            domain = parsed_url.netloc.lower()
        except Exception:
            return {"allowed": False, "reason": "Invalid URL", "warnings": warnings}
        
        # Check blocked domains
        if domain in self.blocked_domains:
            return {"allowed": False, "reason": "Domain is blocked", "warnings": warnings}
        
        # Check whitelist for restricted mode
        if self.permission_level == PermissionLevel.RESTRICTED:
            if domain not in self.whitelisted_domains:
                return {"allowed": False, "reason": "Domain not in whitelist", "warnings": warnings}
        
        # Apply ethical filters
        for filter_func in self.ethical_filters:
            filter_result = filter_func(request)
            if not filter_result["allowed"]:
                return {"allowed": False, "reason": filter_result["reason"], "warnings": warnings}
            warnings.extend(filter_result["warnings"])
        
        return {"allowed": True, "reason": None, "warnings": warnings}
    
    def _filter_harmful_content(self, request: WebRequest) -> Dict[str, Any]:
        """Filter out potentially harmful content requests."""
        harmful_keywords = [
            "hack", "exploit", "malware", "virus", "phishing", "spam",
            "illegal", "unauthorized", "bypass", "crack", "pirate"
        ]
        
        url_lower = request.url.lower()
        if request.data:
            data_str = json.dumps(request.data).lower()
        else:
            data_str = ""
        
        for keyword in harmful_keywords:
            if keyword in url_lower or keyword in data_str:
                return {
                    "allowed": False,
                    "reason": f"Potentially harmful content detected: {keyword}",
                    "warnings": []
                }
        
        return {"allowed": True, "reason": None, "warnings": []}
    
    def _filter_personal_data(self, request: WebRequest) -> Dict[str, Any]:
        """Filter out requests that might contain personal data."""
        personal_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'  # Phone
        ]
        
        import re
        content = request.url + (json.dumps(request.data) if request.data else "")
        
        for pattern in personal_patterns:
            if re.search(pattern, content):
                return {
                    "allowed": False,
                    "reason": "Personal data detected in request",
                    "warnings": []
                }
        
        return {"allowed": True, "reason": None, "warnings": []}
    
    def _filter_inappropriate_requests(self, request: WebRequest) -> Dict[str, Any]:
        """Filter out inappropriate requests."""
        inappropriate_keywords = [
            "adult", "porn", "explicit", "inappropriate", "offensive"
        ]
        
        url_lower = request.url.lower()
        for keyword in inappropriate_keywords:
            if keyword in url_lower:
                return {
                    "allowed": False,
                    "reason": f"Inappropriate content detected: {keyword}",
                    "warnings": []
                }
        
        return {"allowed": True, "reason": None, "warnings": []}
    
    def _check_rate_limit(self, service_type: ServiceType) -> bool:
        """Check if request is within rate limits."""
        current_time = time.time()
        config = self.rate_limit_config[service_type]
        
        # Initialize rate limit tracking
        if service_type not in self.rate_limits:
            self.rate_limits[service_type] = {"requests": [], "last_reset": current_time}
        
        # Clean old requests
        minute_ago = current_time - 60
        hour_ago = current_time - 3600
        
        requests = self.rate_limits[service_type]["requests"]
        requests = [req for req in requests if req > minute_ago]
        
        # Check limits
        if len(requests) >= config["requests_per_minute"]:
            return False
        
        # Check hourly limit (simplified)
        hourly_requests = [req for req in requests if req > hour_ago]
        if len(hourly_requests) >= config["requests_per_hour"]:
            return False
        
        # Add current request
        requests.append(current_time)
        self.rate_limits[service_type]["requests"] = requests
        
        return True
    
    def _check_response_safety(self, response) -> Tuple[bool, List[str]]:
        """Check if response is safe to return."""
        warnings = []
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if 'application/json' not in content_type and 'text/' not in content_type:
            warnings.append("Unexpected content type")
        
        # Check response size
        if len(response.content) > 10 * 1024 * 1024:  # 10MB limit
            warnings.append("Response too large")
        
        # Check for error status codes
        if response.status_code >= 400:
            warnings.append(f"HTTP error: {response.status_code}")
        
        is_safe = len(warnings) == 0
        return is_safe, warnings
    
    def search_web(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Perform a supervised web search.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Search results with safety information
        """
        # This would integrate with a search API (Google, Bing, etc.)
        # For now, return a placeholder
        return {
            "query": query,
            "results": [],
            "safety": "supervised",
            "timestamp": time.time()
        }
    
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
            API response with safety information
        """
        # This would handle different API integrations
        # For now, return a placeholder
        return {
            "api": api_name,
            "endpoint": endpoint,
            "response": {},
            "safety": "supervised",
            "timestamp": time.time()
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for monitoring."""
        return {
            "total_requests": len(self.request_history),
            "requests_by_service": self._count_requests_by_service(),
            "average_response_time": self._calculate_average_response_time(),
            "blocked_requests": self._count_blocked_requests(),
            "rate_limit_hits": self._count_rate_limit_hits()
        }
    
    def _count_requests_by_service(self) -> Dict[str, int]:
        """Count requests by service type."""
        counts = {}
        for record in self.request_history:
            service_type = record["request"].service_type.value
            counts[service_type] = counts.get(service_type, 0) + 1
        return counts
    
    def _calculate_average_response_time(self) -> float:
        """Calculate average response time."""
        if not self.request_history:
            return 0.0
        total_time = sum(record["response_time"] for record in self.request_history)
        return total_time / len(self.request_history)
    
    def _count_blocked_requests(self) -> int:
        """Count blocked requests (placeholder)."""
        return 0
    
    def _count_rate_limit_hits(self) -> int:
        """Count rate limit hits (placeholder)."""
        return 0

# Module-level instance
web_interface = WebInterface()

def make_supervised_request(url: str, method: str = "GET", data: Optional[Dict[str, Any]] = None, 
                           service_type: ServiceType = ServiceType.API_CALL) -> WebResponse:
    """Make a supervised web request using the module-level instance."""
    return web_interface.make_request(url, method, data, service_type)

def search_web_supervised(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Perform a supervised web search using the module-level instance."""
    return web_interface.search_web(query, max_results)

def call_api_supervised(api_name: str, endpoint: str, method: str = "GET", 
                       data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Make a supervised API call using the module-level instance."""
    return web_interface.call_api(api_name, endpoint, method, data) 