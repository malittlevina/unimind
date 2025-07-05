"""
ThothOS System Integration

System-level integration between UniMind and ThothOS components.
Provides process coordination, resource management, and system services.
"""

import asyncio
import logging
import time
import json
import os
import signal
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
from datetime import datetime, timedelta
import hashlib

# System integration
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class SystemComponent(Enum):
    """ThothOS system components."""
    KERNEL = "kernel"
    INIT_SYSTEM = "init_system"
    FILE_SYSTEM = "file_system"
    NETWORK_STACK = "network_stack"
    DEVICE_MANAGER = "device_manager"
    SECURITY_MODULE = "security_module"
    SCHEDULER = "scheduler"
    MEMORY_MANAGER = "memory_manager"


class IntegrationLevel(Enum):
    """Integration levels with ThothOS."""
    MINIMAL = "minimal"
    BASIC = "basic"
    FULL = "full"
    DEEP = "deep"


@dataclass
class SystemService:
    """System service definition."""
    service_id: str
    name: str
    component: SystemComponent
    status: str  # "running", "stopped", "error"
    priority: int
    dependencies: List[str]
    config: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceAllocation:
    """Resource allocation for UniMind."""
    cpu_cores: int
    memory_mb: int
    disk_gb: int
    network_bandwidth: int
    gpu_memory: Optional[int] = None
    priority: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemEvent:
    """System event for coordination."""
    event_id: str
    event_type: str  # "startup", "shutdown", "error", "resource_change"
    component: SystemComponent
    timestamp: datetime
    data: Dict[str, Any]
    severity: str  # "info", "warning", "error", "critical"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ThothOSSystemIntegration:
    """
    System-level integration between UniMind and ThothOS.
    
    Provides process coordination, resource management, and system services.
    """
    
    def __init__(self):
        """Initialize the ThothOS system integration."""
        self.logger = logging.getLogger('ThothOSSystemIntegration')
        
        # Integration state
        self.integration_level = IntegrationLevel.BASIC
        self.is_registered = False
        self.unimind_system = None
        
        # System services
        self.system_services: Dict[str, SystemService] = {}
        self.active_services: Dict[str, SystemService] = {}
        
        # Resource management
        self.resource_allocation: Optional[ResourceAllocation] = None
        self.resource_monitoring = True
        
        # Event handling
        self.event_handlers: Dict[str, callable] = {}
        self.system_events: List[SystemEvent] = []
        
        # Performance metrics
        self.metrics = {
            'total_events': 0,
            'services_managed': 0,
            'resource_allocations': 0,
            'system_calls': 0,
            'integration_uptime': 0.0
        }
        
        # Threading
        self.lock = threading.RLock()
        self.start_time = datetime.now()
        
        # Check system capabilities
        self.psutil_available = PSUTIL_AVAILABLE
        
        # Initialize system services
        self._initialize_system_services()
        
        self.logger.info("ThothOS system integration initialized")
    
    def _initialize_system_services(self):
        """Initialize system services."""
        self.system_services = {
            'unimind_core': SystemService(
                service_id='unimind_core',
                name='UniMind Core System',
                component=SystemComponent.KERNEL,
                status='stopped',
                priority=10,
                dependencies=[],
                config={'auto_start': True, 'restart_on_failure': True}
            ),
            'unimind_memory': SystemService(
                service_id='unimind_memory',
                name='UniMind Memory Management',
                component=SystemComponent.MEMORY_MANAGER,
                status='stopped',
                priority=9,
                dependencies=['unimind_core'],
                config={'memory_limit': '2GB', 'swap_enabled': True}
            ),
            'unimind_reasoning': SystemService(
                service_id='unimind_reasoning',
                name='UniMind Reasoning Engine',
                component=SystemComponent.KERNEL,
                status='stopped',
                priority=8,
                dependencies=['unimind_core'],
                config={'max_threads': 4, 'timeout': 30}
            ),
            'unimind_perception': SystemService(
                service_id='unimind_perception',
                name='UniMind Perception System',
                component=SystemComponent.DEVICE_MANAGER,
                status='stopped',
                priority=7,
                dependencies=['unimind_core'],
                config={'sensor_timeout': 5, 'buffer_size': 1024}
            ),
            'unimind_emotion': SystemService(
                service_id='unimind_emotion',
                name='UniMind Emotion Engine',
                component=SystemComponent.KERNEL,
                status='stopped',
                priority=6,
                dependencies=['unimind_core'],
                config={'emotion_cache_size': 1000, 'update_interval': 1}
            ),
            'unimind_planning': SystemService(
                service_id='unimind_planning',
                name='UniMind Planning System',
                component=SystemComponent.SCHEDULER,
                status='stopped',
                priority=5,
                dependencies=['unimind_core'],
                config={'plan_horizon': 24, 'replan_interval': 5}
            ),
            'unimind_advanced': SystemService(
                service_id='unimind_advanced',
                name='UniMind Advanced Features',
                component=SystemComponent.KERNEL,
                status='stopped',
                priority=4,
                dependencies=['unimind_core', 'unimind_memory'],
                config={'feature_enabled': True, 'resource_limit': '1GB'}
            ),
            'unimind_rag': SystemService(
                service_id='unimind_rag',
                name='UniMind RAG System',
                component=SystemComponent.FILE_SYSTEM,
                status='stopped',
                priority=3,
                dependencies=['unimind_core', 'unimind_memory'],
                config={'knowledge_base_path': '/unimind/knowledge', 'index_update_interval': 3600}
            ),
            'unimind_web': SystemService(
                service_id='unimind_web',
                name='UniMind Web Interface',
                component=SystemComponent.NETWORK_STACK,
                status='stopped',
                priority=2,
                dependencies=['unimind_core'],
                config={'port': 8080, 'host': 'localhost', 'max_connections': 100}
            )
        }
    
    async def initialize(self):
        """Initialize the system integration."""
        self.logger.info("Initializing ThothOS system integration...")
        
        try:
            # Check ThothOS system availability
            await self._check_thothos_availability()
            
            # Initialize resource monitoring
            await self._initialize_resource_monitoring()
            
            # Register event handlers
            await self._register_event_handlers()
            
            # Start system monitoring
            await self._start_system_monitoring()
            
            self.logger.info("ThothOS system integration initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system integration: {e}")
            raise
    
    async def _check_thothos_availability(self):
        """Check ThothOS system availability."""
        self.logger.info("Checking ThothOS system availability...")
        
        # Check for ThothOS-specific files and directories
        thothos_paths = [
            Path("/thothos"),
            Path("/etc/thothos"),
            Path("/var/log/thothos"),
            Path("/usr/local/thothos")
        ]
        
        available_paths = []
        for path in thothos_paths:
            if path.exists():
                available_paths.append(str(path))
        
        if available_paths:
            self.logger.info(f"ThothOS paths found: {available_paths}")
            self.integration_level = IntegrationLevel.FULL
        else:
            self.logger.info("ThothOS paths not found - using simulation mode")
            self.integration_level = IntegrationLevel.BASIC
        
        # Check system capabilities
        if self.psutil_available:
            self.logger.info("System monitoring capabilities available")
        else:
            self.logger.warning("System monitoring capabilities limited")
    
    async def _initialize_resource_monitoring(self):
        """Initialize resource monitoring."""
        self.logger.info("Initializing resource monitoring...")
        
        if self.psutil_available:
            # Set up resource monitoring
            self.resource_monitoring = True
            
            # Initialize resource allocation
            await self._allocate_resources()
        
        self.logger.info("Resource monitoring initialized")
    
    async def _allocate_resources(self):
        """Allocate system resources for UniMind."""
        if not self.psutil_available:
            # Simulate resource allocation
            self.resource_allocation = ResourceAllocation(
                cpu_cores=2,
                memory_mb=2048,
                disk_gb=10,
                network_bandwidth=100,
                priority=5
            )
            return
        
        try:
            # Get system information
            cpu_count = psutil.cpu_count()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Calculate resource allocation (20% of available resources)
            allocated_cores = max(1, cpu_count // 5)
            allocated_memory = max(512, int(memory.total * 0.2 / (1024 * 1024)))  # MB
            allocated_disk = max(1, int(disk.free / (1024 * 1024 * 1024) * 0.1))  # GB
            
            self.resource_allocation = ResourceAllocation(
                cpu_cores=allocated_cores,
                memory_mb=allocated_memory,
                disk_gb=allocated_disk,
                network_bandwidth=100,  # Mbps
                priority=5
            )
            
            with self.lock:
                self.metrics['resource_allocations'] += 1
            
            self.logger.info(f"Resource allocation: {allocated_cores} cores, {allocated_memory}MB RAM, {allocated_disk}GB disk")
            
        except Exception as e:
            self.logger.error(f"Error allocating resources: {e}")
            # Fallback to minimal allocation
            self.resource_allocation = ResourceAllocation(
                cpu_cores=1,
                memory_mb=512,
                disk_gb=1,
                network_bandwidth=10,
                priority=5
            )
    
    async def _register_event_handlers(self):
        """Register system event handlers."""
        self.logger.info("Registering event handlers...")
        
        # Register handlers for different event types
        self.event_handlers = {
            'startup': self._handle_startup_event,
            'shutdown': self._handle_shutdown_event,
            'error': self._handle_error_event,
            'resource_change': self._handle_resource_change_event,
            'service_status_change': self._handle_service_status_change_event
        }
        
        self.logger.info("Event handlers registered")
    
    async def _start_system_monitoring(self):
        """Start system monitoring."""
        self.logger.info("Starting system monitoring...")
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_system_resources())
        asyncio.create_task(self._monitor_service_health())
        asyncio.create_task(self._monitor_system_events())
        
        self.logger.info("System monitoring started")
    
    async def register_unimind_system(self, unimind_system):
        """Register UniMind system with ThothOS."""
        self.logger.info("Registering UniMind system with ThothOS...")
        
        self.unimind_system = unimind_system
        self.is_registered = True
        
        # Create system event
        event = SystemEvent(
            event_id=f"unimind_registration_{int(time.time())}",
            event_type='startup',
            component=SystemComponent.KERNEL,
            timestamp=datetime.now(),
            data={'system_name': 'UniMind', 'version': '1.0.0'},
            severity='info'
        )
        
        await self._process_system_event(event)
        
        self.logger.info("UniMind system registered with ThothOS")
    
    async def start_service(self, service_id: str) -> bool:
        """Start a system service."""
        if service_id not in self.system_services:
            raise ValueError(f"Service {service_id} not found")
        
        service = self.system_services[service_id]
        
        # Check dependencies
        for dep_id in service.dependencies:
            if dep_id not in self.active_services:
                self.logger.warning(f"Service {service_id} depends on {dep_id} which is not active")
                return False
        
        try:
            # Start service
            service.status = 'running'
            self.active_services[service_id] = service
            
            # Create event
            event = SystemEvent(
                event_id=f"service_start_{service_id}_{int(time.time())}",
                event_type='service_status_change',
                component=service.component,
                timestamp=datetime.now(),
                data={'service_id': service_id, 'status': 'running'},
                severity='info'
            )
            
            await self._process_system_event(event)
            
            with self.lock:
                self.metrics['services_managed'] += 1
            
            self.logger.info(f"Service {service_id} started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start service {service_id}: {e}")
            service.status = 'error'
            return False
    
    async def stop_service(self, service_id: str) -> bool:
        """Stop a system service."""
        if service_id not in self.system_services:
            raise ValueError(f"Service {service_id} not found")
        
        service = self.system_services[service_id]
        
        try:
            # Stop service
            service.status = 'stopped'
            if service_id in self.active_services:
                del self.active_services[service_id]
            
            # Create event
            event = SystemEvent(
                event_id=f"service_stop_{service_id}_{int(time.time())}",
                event_type='service_status_change',
                component=service.component,
                timestamp=datetime.now(),
                data={'service_id': service_id, 'status': 'stopped'},
                severity='info'
            )
            
            await self._process_system_event(event)
            
            self.logger.info(f"Service {service_id} stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop service {service_id}: {e}")
            return False
    
    async def get_service_status(self, service_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a system service."""
        if service_id not in self.system_services:
            return None
        
        service = self.system_services[service_id]
        
        return {
            'service_id': service.service_id,
            'name': service.name,
            'component': service.component.value,
            'status': service.status,
            'priority': service.priority,
            'dependencies': service.dependencies,
            'config': service.config,
            'is_active': service_id in self.active_services
        }
    
    async def list_services(self) -> List[Dict[str, Any]]:
        """List all system services."""
        services = []
        
        for service in self.system_services.values():
            services.append({
                'service_id': service.service_id,
                'name': service.name,
                'component': service.component.value,
                'status': service.status,
                'priority': service.priority,
                'is_active': service.service_id in self.active_services
            })
        
        return services
    
    async def _monitor_system_resources(self):
        """Monitor system resources."""
        while self.resource_monitoring:
            try:
                if self.psutil_available and self.resource_allocation:
                    # Check resource usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    
                    # Check if usage exceeds allocation
                    if cpu_percent > 80 or memory.percent > 80:
                        event = SystemEvent(
                            event_id=f"resource_warning_{int(time.time())}",
                            event_type='resource_change',
                            component=SystemComponent.KERNEL,
                            timestamp=datetime.now(),
                            data={
                                'cpu_percent': cpu_percent,
                                'memory_percent': memory.percent,
                                'allocation': {
                                    'cpu_cores': self.resource_allocation.cpu_cores,
                                    'memory_mb': self.resource_allocation.memory_mb
                                }
                            },
                            severity='warning'
                        )
                        
                        await self._process_system_event(event)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring system resources: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _monitor_service_health(self):
        """Monitor service health."""
        while self.resource_monitoring:
            try:
                for service_id, service in self.active_services.items():
                    # Simple health check - could be enhanced with actual service checks
                    if service.status == 'running':
                        # Simulate health check
                        health_status = 'healthy'  # In real implementation, check actual service health
                        
                        if health_status != 'healthy':
                            event = SystemEvent(
                                event_id=f"service_health_{service_id}_{int(time.time())}",
                                event_type='error',
                                component=service.component,
                                timestamp=datetime.now(),
                                data={'service_id': service_id, 'health_status': health_status},
                                severity='warning'
                            )
                            
                            await self._process_system_event(event)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error monitoring service health: {e}")
                await asyncio.sleep(120)  # Wait longer on error
    
    async def _monitor_system_events(self):
        """Monitor system events."""
        while self.resource_monitoring:
            try:
                # Process any pending events
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error monitoring system events: {e}")
                await asyncio.sleep(5)
    
    async def _process_system_event(self, event: SystemEvent):
        """Process a system event."""
        try:
            # Add event to history
            with self.lock:
                self.system_events.append(event)
                self.metrics['total_events'] += 1
                
                # Keep only last 1000 events
                if len(self.system_events) > 1000:
                    self.system_events = self.system_events[-1000:]
            
            # Call appropriate handler
            handler = self.event_handlers.get(event.event_type)
            if handler:
                await handler(event)
            
            # Log event
            self.logger.info(f"System event: {event.event_type} - {event.severity}")
            
        except Exception as e:
            self.logger.error(f"Error processing system event: {e}")
    
    async def _handle_startup_event(self, event: SystemEvent):
        """Handle startup events."""
        if event.data.get('system_name') == 'UniMind':
            # Start core UniMind services
            await self.start_service('unimind_core')
            await self.start_service('unimind_memory')
            await self.start_service('unimind_reasoning')
    
    async def _handle_shutdown_event(self, event: SystemEvent):
        """Handle shutdown events."""
        # Stop all active services
        for service_id in list(self.active_services.keys()):
            await self.stop_service(service_id)
    
    async def _handle_error_event(self, event: SystemEvent):
        """Handle error events."""
        # Log error and potentially restart services
        self.logger.error(f"System error: {event.data}")
        
        # If it's a service error, try to restart
        if 'service_id' in event.data:
            service_id = event.data['service_id']
            service = self.system_services.get(service_id)
            if service and service.config.get('restart_on_failure', False):
                await self.stop_service(service_id)
                await asyncio.sleep(5)  # Wait before restart
                await self.start_service(service_id)
    
    async def _handle_resource_change_event(self, event: SystemEvent):
        """Handle resource change events."""
        # Adjust resource allocation if needed
        if event.severity == 'warning':
            self.logger.warning(f"Resource warning: {event.data}")
            
            # Could implement resource scaling here
            # For now, just log the warning
    
    async def _handle_service_status_change_event(self, event: SystemEvent):
        """Handle service status change events."""
        service_id = event.data.get('service_id')
        status = event.data.get('status')
        
        self.logger.info(f"Service {service_id} status changed to {status}")
        
        # Update service status
        if service_id in self.system_services:
            self.system_services[service_id].status = status
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system integration status."""
        # Update uptime
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        with self.lock:
            return {
                'integration_level': self.integration_level.value,
                'is_registered': self.is_registered,
                'unimind_system_available': self.unimind_system is not None,
                'metrics': self.metrics.copy(),
                'uptime_seconds': uptime,
                'active_services_count': len(self.active_services),
                'total_services_count': len(self.system_services),
                'recent_events_count': len(self.system_events),
                'resource_allocation': {
                    'cpu_cores': self.resource_allocation.cpu_cores if self.resource_allocation else 0,
                    'memory_mb': self.resource_allocation.memory_mb if self.resource_allocation else 0,
                    'disk_gb': self.resource_allocation.disk_gb if self.resource_allocation else 0,
                    'network_bandwidth': self.resource_allocation.network_bandwidth if self.resource_allocation else 0
                } if self.resource_allocation else None,
                'psutil_available': self.psutil_available
            }
    
    async def shutdown(self):
        """Shutdown the system integration."""
        self.logger.info("Shutting down ThothOS system integration...")
        
        # Stop resource monitoring
        self.resource_monitoring = False
        
        # Stop all services
        for service_id in list(self.active_services.keys()):
            await self.stop_service(service_id)
        
        # Clear data
        with self.lock:
            self.system_services.clear()
            self.active_services.clear()
            self.system_events.clear()
            self.event_handlers.clear()
        
        self.is_registered = False
        self.unimind_system = None
        
        self.logger.info("ThothOS system integration shutdown complete")


# Global instance
thothos_system_integration = ThothOSSystemIntegration() 