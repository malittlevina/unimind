"""
Autonomous Agents System

Advanced autonomous agents and multi-agent coordination for UniMind.
Provides agent orchestration, task decomposition, and inter-agent communication.
"""

import asyncio
import logging
import time
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
import uuid
from datetime import datetime
from queue import Queue, PriorityQueue


class AgentType(Enum):
    """Types of autonomous agents."""
    TASK_AGENT = "task_agent"
    COORDINATOR_AGENT = "coordinator_agent"
    SPECIALIST_AGENT = "specialist_agent"
    MONITOR_AGENT = "monitor_agent"
    RESOURCE_AGENT = "resource_agent"
    COMMUNICATION_AGENT = "communication_agent"


class TaskStatus(Enum):
    """Status of agent tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class MessageType(Enum):
    """Types of inter-agent messages."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_RESPONSE = "resource_response"
    COORDINATION = "coordination"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


@dataclass
class AgentCapability:
    """Capability of an agent."""
    name: str
    description: str
    proficiency: float  # 0.0 to 1.0
    resources_required: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTask:
    """Task for an agent to execute."""
    task_id: str
    task_type: str
    description: str
    priority: int
    agent_id: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    resources_required: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMessage:
    """Message between agents."""
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: MessageType
    content: Any
    priority: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """State of an agent."""
    agent_id: str
    agent_type: AgentType
    status: str  # "active", "idle", "busy", "error"
    current_task: Optional[str] = None
    capabilities: List[AgentCapability] = field(default_factory=list)
    resources: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutonomousAgent:
    """Base class for autonomous agents."""
    
    def __init__(self, agent_id: str, agent_type: AgentType, capabilities: List[AgentCapability]):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.logger = logging.getLogger(f'Agent_{agent_id}')
        
        # Agent state
        self.state = AgentState(
            agent_id=agent_id,
            agent_type=agent_type,
            status="idle",
            capabilities=capabilities
        )
        
        # Message handling
        self.message_queue = Queue()
        self.task_queue = PriorityQueue()
        
        # Threading
        self.running = False
        self.lock = threading.RLock()
        
        self.logger.info(f"Agent {agent_id} initialized")
    
    async def start(self):
        """Start the agent."""
        with self.lock:
            if not self.running:
                self.running = True
                self.state.status = "active"
                asyncio.create_task(self._message_loop())
                asyncio.create_task(self._task_loop())
                self.logger.info(f"Agent {self.agent_id} started")
    
    async def stop(self):
        """Stop the agent."""
        with self.lock:
            self.running = False
            self.state.status = "stopped"
            self.logger.info(f"Agent {self.agent_id} stopped")
    
    async def _message_loop(self):
        """Main message processing loop."""
        while self.running:
            try:
                if not self.message_queue.empty():
                    message = self.message_queue.get_nowait()
                    await self._process_message(message)
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Message loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def _task_loop(self):
        """Main task processing loop."""
        while self.running:
            try:
                if not self.task_queue.empty():
                    priority, task = self.task_queue.get_nowait()
                    await self._execute_task(task)
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Task loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_message(self, message: AgentMessage):
        """Process incoming message."""
        self.logger.debug(f"Processing message: {message.message_type.value}")
        
        if message.message_type == MessageType.TASK_REQUEST:
            await self._handle_task_request(message)
        elif message.message_type == MessageType.STATUS_UPDATE:
            await self._handle_status_update(message)
        elif message.message_type == MessageType.RESOURCE_REQUEST:
            await self._handle_resource_request(message)
        elif message.message_type == MessageType.COORDINATION:
            await self._handle_coordination(message)
        else:
            self.logger.warning(f"Unknown message type: {message.message_type}")
    
    async def _handle_task_request(self, message: AgentMessage):
        """Handle task request message."""
        task_data = message.content
        task = AgentTask(
            task_id=task_data.get('task_id'),
            task_type=task_data.get('task_type'),
            description=task_data.get('description'),
            priority=task_data.get('priority', 0),
            dependencies=task_data.get('dependencies', []),
            resources_required=task_data.get('resources_required', []),
            parameters=task_data.get('parameters', {})
        )
        
        # Check if agent can handle this task
        if self._can_handle_task(task):
            self.task_queue.put((task.priority, task))
            self.state.status = "busy"
        else:
            # Send rejection message
            response = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.TASK_RESPONSE,
                content={'task_id': task.task_id, 'accepted': False, 'reason': 'Cannot handle task'}
            )
            # In a real system, this would be sent through the message bus
    
    async def _handle_status_update(self, message: AgentMessage):
        """Handle status update message."""
        # Update agent state based on status update
        pass
    
    async def _handle_resource_request(self, message: AgentMessage):
        """Handle resource request message."""
        # Check if agent has requested resources
        requested_resources = message.content.get('resources', [])
        available_resources = []
        
        for resource in requested_resources:
            if resource in self.state.resources:
                available_resources.append(resource)
        
        response = AgentMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type=MessageType.RESOURCE_RESPONSE,
            content={'requested_resources': requested_resources, 'available_resources': available_resources}
        )
        # In a real system, this would be sent through the message bus
    
    async def _handle_coordination(self, message: AgentMessage):
        """Handle coordination message."""
        # Handle coordination requests
        pass
    
    def _can_handle_task(self, task: AgentTask) -> bool:
        """Check if agent can handle a task."""
        # Check if agent has required capabilities
        required_capabilities = task.parameters.get('required_capabilities', [])
        
        for capability_name in required_capabilities:
            if not any(cap.name == capability_name for cap in self.capabilities):
                return False
        
        # Check if agent has required resources
        for resource in task.resources_required:
            if resource not in self.state.resources:
                return False
        
        return True
    
    async def _execute_task(self, task: AgentTask):
        """Execute a task."""
        self.logger.info(f"Executing task: {task.task_id}")
        
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        self.state.current_task = task.task_id
        self.state.status = "busy"
        
        try:
            # Execute task based on type
            if task.task_type == "computation":
                result = await self._execute_computation_task(task)
            elif task.task_type == "data_processing":
                result = await self._execute_data_processing_task(task)
            elif task.task_type == "communication":
                result = await self._execute_communication_task(task)
            else:
                result = await self._execute_generic_task(task)
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            # Update performance metrics
            execution_time = (task.completed_at - task.started_at).total_seconds()
            self.state.performance_metrics['avg_execution_time'] = (
                (self.state.performance_metrics.get('avg_execution_time', 0) + execution_time) / 2
            )
            
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            self.logger.error(f"Task execution failed: {e}")
        
        finally:
            self.state.current_task = None
            self.state.status = "idle"
    
    async def _execute_computation_task(self, task: AgentTask) -> Any:
        """Execute computation task."""
        # Simulate computation
        await asyncio.sleep(0.1)
        return f"Computation result for {task.description}"
    
    async def _execute_data_processing_task(self, task: AgentTask) -> Any:
        """Execute data processing task."""
        # Simulate data processing
        await asyncio.sleep(0.1)
        return f"Data processing result for {task.description}"
    
    async def _execute_communication_task(self, task: AgentTask) -> Any:
        """Execute communication task."""
        # Simulate communication
        await asyncio.sleep(0.1)
        return f"Communication result for {task.description}"
    
    async def _execute_generic_task(self, task: AgentTask) -> Any:
        """Execute generic task."""
        # Simulate generic task execution
        await asyncio.sleep(0.1)
        return f"Generic task result for {task.description}"
    
    def send_message(self, message: AgentMessage):
        """Send message to another agent."""
        # In a real system, this would go through a message bus
        self.logger.debug(f"Sending message: {message.message_type.value} to {message.receiver_id}")
    
    def get_state(self) -> AgentState:
        """Get current agent state."""
        return self.state


class TaskAgent(AutonomousAgent):
    """Agent specialized in task execution."""
    
    def __init__(self, agent_id: str, capabilities: List[AgentCapability]):
        super().__init__(agent_id, AgentType.TASK_AGENT, capabilities)
    
    async def _execute_task(self, task: AgentTask):
        """Execute task with task-specific logic."""
        await super()._execute_task(task)


class CoordinatorAgent(AutonomousAgent):
    """Agent specialized in coordination."""
    
    def __init__(self, agent_id: str, capabilities: List[AgentCapability]):
        super().__init__(agent_id, AgentType.COORDINATOR_AGENT, capabilities)
        self.subordinate_agents: List[str] = []
    
    def add_subordinate(self, agent_id: str):
        """Add subordinate agent."""
        if agent_id not in self.subordinate_agents:
            self.subordinate_agents.append(agent_id)
    
    async def coordinate_tasks(self, tasks: List[AgentTask]) -> Dict[str, str]:
        """Coordinate tasks among subordinate agents."""
        task_assignments = {}
        
        for task in tasks:
            # Find best agent for task
            best_agent = self._find_best_agent_for_task(task)
            if best_agent:
                task_assignments[task.task_id] = best_agent
        
        return task_assignments
    
    def _find_best_agent_for_task(self, task: AgentTask) -> Optional[str]:
        """Find best agent for a task."""
        # Simple heuristic: return first available agent
        # In a real system, this would use more sophisticated matching
        return self.subordinate_agents[0] if self.subordinate_agents else None


class AutonomousAgentsSystem:
    """
    System for managing autonomous agents and multi-agent coordination.
    
    Provides agent orchestration, task decomposition, and inter-agent communication.
    """
    
    def __init__(self):
        """Initialize the autonomous agents system."""
        self.logger = logging.getLogger('AutonomousAgentsSystem')
        
        # Agent registry
        self.agents: Dict[str, AutonomousAgent] = {}
        self.agent_states: Dict[str, AgentState] = {}
        
        # Task management
        self.tasks: Dict[str, AgentTask] = {}
        self.task_queue: PriorityQueue = PriorityQueue()
        
        # Message bus
        self.message_bus: Queue = Queue()
        
        # Performance metrics
        self.metrics = {
            'total_agents': 0,
            'active_agents': 0,
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'avg_task_completion_time': 0.0,
            'total_messages': 0
        }
        
        # Threading
        self.lock = threading.RLock()
        self.running = False
        
        # Initialize system
        self._initialize_system()
        
        self.logger.info("Autonomous agents system initialized")
    
    def _initialize_system(self):
        """Initialize the agent system."""
        # Create coordinator agent
        coordinator_capabilities = [
            AgentCapability("task_coordination", "Coordinate tasks among agents", 0.9),
            AgentCapability("resource_management", "Manage system resources", 0.8),
            AgentCapability("performance_monitoring", "Monitor agent performance", 0.7)
        ]
        
        coordinator = CoordinatorAgent("coordinator_001", coordinator_capabilities)
        self.register_agent(coordinator)
    
    def register_agent(self, agent: AutonomousAgent) -> str:
        """Register an agent in the system."""
        with self.lock:
            self.agents[agent.agent_id] = agent
            self.agent_states[agent.agent_id] = agent.get_state()
            self.metrics['total_agents'] += 1
            
            # Add to coordinator if it's a task agent
            if agent.agent_type == AgentType.TASK_AGENT:
                coordinator = self.agents.get("coordinator_001")
                if coordinator and isinstance(coordinator, CoordinatorAgent):
                    coordinator.add_subordinate(agent.agent_id)
            
            self.logger.info(f"Registered agent: {agent.agent_id}")
            return agent.agent_id
    
    async def start_system(self):
        """Start the autonomous agents system."""
        with self.lock:
            if not self.running:
                self.running = True
                
                # Start all agents
                for agent in self.agents.values():
                    await agent.start()
                
                # Start system loops
                asyncio.create_task(self._system_monitor_loop())
                asyncio.create_task(self._message_bus_loop())
                
                self.logger.info("Autonomous agents system started")
    
    async def stop_system(self):
        """Stop the autonomous agents system."""
        with self.lock:
            self.running = False
            
            # Stop all agents
            for agent in self.agents.values():
                await agent.stop()
            
            self.logger.info("Autonomous agents system stopped")
    
    async def _system_monitor_loop(self):
        """Monitor system health and performance."""
        while self.running:
            try:
                # Update agent states
                for agent_id, agent in self.agents.items():
                    self.agent_states[agent_id] = agent.get_state()
                
                # Update metrics
                active_agents = sum(1 for state in self.agent_states.values() 
                                  if state.status == "active")
                self.metrics['active_agents'] = active_agents
                
                # Check for failed tasks
                failed_tasks = sum(1 for task in self.tasks.values() 
                                 if task.status == TaskStatus.FAILED)
                self.metrics['failed_tasks'] = failed_tasks
                
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"System monitor error: {e}")
                await asyncio.sleep(10.0)
    
    async def _message_bus_loop(self):
        """Process messages in the message bus."""
        while self.running:
            try:
                if not self.message_bus.empty():
                    message = self.message_bus.get_nowait()
                    await self._route_message(message)
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Message bus error: {e}")
                await asyncio.sleep(1.0)
    
    async def _route_message(self, message: AgentMessage):
        """Route message to appropriate agent."""
        if message.receiver_id in self.agents:
            agent = self.agents[message.receiver_id]
            agent.message_queue.put(message)
            self.metrics['total_messages'] += 1
        else:
            self.logger.warning(f"Unknown receiver: {message.receiver_id}")
    
    async def submit_task(self, task: AgentTask) -> str:
        """Submit a task to the system."""
        with self.lock:
            self.tasks[task.task_id] = task
            self.task_queue.put((task.priority, task))
            self.metrics['total_tasks'] += 1
            
            # Assign task to appropriate agent
            await self._assign_task(task)
            
            self.logger.info(f"Submitted task: {task.task_id}")
            return task.task_id
    
    async def _assign_task(self, task: AgentTask):
        """Assign task to appropriate agent."""
        # Find best agent for task
        best_agent = None
        best_score = 0.0
        
        for agent_id, agent in self.agents.items():
            if agent.agent_type == AgentType.TASK_AGENT:
                score = self._calculate_agent_fitness(agent, task)
                if score > best_score:
                    best_score = score
                    best_agent = agent
        
        if best_agent:
            task.agent_id = best_agent.agent_id
            best_agent.task_queue.put((task.priority, task))
            self.logger.info(f"Assigned task {task.task_id} to agent {best_agent.agent_id}")
        else:
            self.logger.warning(f"No suitable agent found for task {task.task_id}")
    
    def _calculate_agent_fitness(self, agent: AutonomousAgent, task: AgentTask) -> float:
        """Calculate how well an agent fits a task."""
        fitness = 0.0
        
        # Check capability match
        required_capabilities = task.parameters.get('required_capabilities', [])
        for capability_name in required_capabilities:
            for capability in agent.capabilities:
                if capability.name == capability_name:
                    fitness += capability.proficiency
        
        # Check resource availability
        for resource in task.resources_required:
            if resource in agent.state.resources:
                fitness += 0.5
        
        # Check current workload
        if agent.state.status == "idle":
            fitness += 1.0
        elif agent.state.status == "busy":
            fitness += 0.3
        
        return fitness
    
    async def decompose_task(self, complex_task: AgentTask) -> List[AgentTask]:
        """Decompose a complex task into simpler subtasks."""
        subtasks = []
        
        # Simple task decomposition based on task type
        if complex_task.task_type == "data_analysis":
            subtasks = [
                AgentTask(
                    task_id=f"{complex_task.task_id}_data_collection",
                    task_type="data_collection",
                    description="Collect required data",
                    priority=complex_task.priority,
                    dependencies=[],
                    parameters={'data_sources': complex_task.parameters.get('data_sources', [])}
                ),
                AgentTask(
                    task_id=f"{complex_task.task_id}_data_processing",
                    task_type="data_processing",
                    description="Process collected data",
                    priority=complex_task.priority,
                    dependencies=[f"{complex_task.task_id}_data_collection"],
                    parameters={'processing_method': complex_task.parameters.get('processing_method', 'default')}
                ),
                AgentTask(
                    task_id=f"{complex_task.task_id}_analysis",
                    task_type="analysis",
                    description="Analyze processed data",
                    priority=complex_task.priority,
                    dependencies=[f"{complex_task.task_id}_data_processing"],
                    parameters={'analysis_type': complex_task.parameters.get('analysis_type', 'general')}
                )
            ]
        elif complex_task.task_type == "computation":
            # Decompose computation into parallel subtasks
            num_subtasks = complex_task.parameters.get('num_subtasks', 3)
            for i in range(num_subtasks):
                subtasks.append(AgentTask(
                    task_id=f"{complex_task.task_id}_subtask_{i}",
                    task_type="computation",
                    description=f"Computation subtask {i}",
                    priority=complex_task.priority,
                    dependencies=[],
                    parameters={'subtask_id': i, 'total_subtasks': num_subtasks}
                ))
        else:
            # Generic decomposition
            subtasks = [complex_task]
        
        return subtasks
    
    def get_agent_status(self, agent_id: str) -> Optional[AgentState]:
        """Get status of a specific agent."""
        return self.agent_states.get(agent_id)
    
    def get_task_status(self, task_id: str) -> Optional[AgentTask]:
        """Get status of a specific task."""
        return self.tasks.get(task_id)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents in the system."""
        with self.lock:
            return [
                {
                    'agent_id': agent_id,
                    'agent_type': state.agent_type.value,
                    'status': state.status,
                    'current_task': state.current_task,
                    'capabilities': [cap.name for cap in state.capabilities]
                }
                for agent_id, state in self.agent_states.items()
            ]
    
    def list_tasks(self) -> List[Dict[str, Any]]:
        """List all tasks in the system."""
        with self.lock:
            return [
                {
                    'task_id': task.task_id,
                    'task_type': task.task_type,
                    'status': task.status.value,
                    'agent_id': task.agent_id,
                    'priority': task.priority,
                    'created_at': task.created_at.isoformat()
                }
                for task in self.tasks.values()
            ]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        with self.lock:
            return {
                'metrics': self.metrics.copy(),
                'total_agents': len(self.agents),
                'active_agents': self.metrics['active_agents'],
                'total_tasks': len(self.tasks),
                'pending_tasks': sum(1 for task in self.tasks.values() 
                                   if task.status == TaskStatus.PENDING),
                'running_tasks': sum(1 for task in self.tasks.values() 
                                   if task.status == TaskStatus.RUNNING),
                'completed_tasks': sum(1 for task in self.tasks.values() 
                                     if task.status == TaskStatus.COMPLETED),
                'failed_tasks': self.metrics['failed_tasks']
            }


# Global instance
autonomous_agents_system = AutonomousAgentsSystem() 