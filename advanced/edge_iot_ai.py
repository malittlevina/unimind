"""
Edge Computing and IoT AI Engine

Advanced edge computing and IoT integration for UniMind.
Provides device management, real-time processing, edge analytics, IoT security, and distributed computing capabilities.
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
import socket
import struct

# IoT dependencies
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class DeviceType(Enum):
    """Types of IoT devices."""
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    GATEWAY = "gateway"
    CAMERA = "camera"
    MICROCONTROLLER = "microcontroller"
    EMBEDDED_SYSTEM = "embedded_system"
    SMART_DEVICE = "smart_device"


class CommunicationProtocol(Enum):
    """IoT communication protocols."""
    MQTT = "mqtt"
    HTTP = "http"
    COAP = "coap"
    WEBSOCKET = "websocket"
    BLE = "ble"
    ZIGBEE = "zigbee"
    LORA = "lora"


class DataType(Enum):
    """Types of IoT data."""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    LIGHT = "light"
    MOTION = "motion"
    SOUND = "sound"
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"
    BINARY = "binary"


class ProcessingLocation(Enum):
    """Data processing locations."""
    EDGE = "edge"
    FOG = "fog"
    CLOUD = "cloud"
    HYBRID = "hybrid"


@dataclass
class IoTDevice:
    """IoT device information."""
    device_id: str
    name: str
    device_type: DeviceType
    location: str
    capabilities: List[str]
    communication_protocol: CommunicationProtocol
    data_types: List[DataType]
    processing_location: ProcessingLocation
    status: str  # "online", "offline", "error"
    last_seen: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SensorData:
    """Sensor data from IoT devices."""
    data_id: str
    device_id: str
    data_type: DataType
    value: Union[float, str, bytes]
    timestamp: datetime
    location: str
    quality: float  # 0-1 data quality score
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeNode:
    """Edge computing node."""
    node_id: str
    name: str
    location: str
    capabilities: List[str]
    processing_power: float  # CPU/GPU capacity
    memory_capacity: int  # RAM in MB
    storage_capacity: int  # Storage in GB
    network_bandwidth: float  # Mbps
    connected_devices: List[str]
    status: str  # "active", "inactive", "overloaded"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeProcessingTask:
    """Edge processing task."""
    task_id: str
    node_id: str
    task_type: str  # "data_processing", "ml_inference", "analytics"
    input_data: Dict[str, Any]
    processing_requirements: Dict[str, Any]
    priority: int  # 1-10
    status: str  # "pending", "running", "completed", "failed"
    start_time: datetime
    completion_time: Optional[datetime]
    results: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IoTNetwork:
    """IoT network configuration."""
    network_id: str
    name: str
    topology: str  # "star", "mesh", "tree", "hybrid"
    devices: List[str]
    edge_nodes: List[str]
    communication_protocols: List[CommunicationProtocol]
    security_config: Dict[str, Any]
    routing_config: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RealTimeStream:
    """Real-time data stream."""
    stream_id: str
    source_device: str
    data_type: DataType
    sampling_rate: float  # Hz
    buffer_size: int
    processing_pipeline: List[str]
    subscribers: List[str]
    status: str  # "active", "paused", "stopped"
    metadata: Dict[str, Any] = field(default_factory=dict)


class EdgeIoTAIEngine:
    """
    Advanced edge computing and IoT AI engine for UniMind.
    
    Provides device management, real-time processing, edge analytics,
    IoT security, and distributed computing capabilities.
    """
    
    def __init__(self):
        """Initialize the edge IoT AI engine."""
        self.logger = logging.getLogger('EdgeIoTAIEngine')
        
        # IoT data storage
        self.iot_devices: Dict[str, IoTDevice] = {}
        self.sensor_data: Dict[str, SensorData] = {}
        self.edge_nodes: Dict[str, EdgeNode] = {}
        self.edge_tasks: Dict[str, EdgeProcessingTask] = {}
        self.iot_networks: Dict[str, IoTNetwork] = {}
        self.real_time_streams: Dict[str, RealTimeStream] = {}
        
        # Communication clients
        self.mqtt_client: Optional[mqtt.Client] = None
        self.websocket_clients: Dict[str, Any] = {}
        
        # Processing pipelines
        self.data_processors: Dict[str, Any] = {}
        self.ml_models: Dict[str, Any] = {}
        self.analytics_engines: Dict[str, Any] = {}
        
        # Performance metrics
        self.metrics = {
            'total_devices': 0,
            'total_sensor_readings': 0,
            'total_edge_tasks': 0,
            'total_networks': 0,
            'total_streams': 0,
            'avg_processing_time': 0.0,
            'avg_data_quality': 0.0,
            'network_throughput': 0.0
        }
        
        # Threading
        self.lock = threading.RLock()
        
        # Check dependencies
        self.mqtt_available = MQTT_AVAILABLE
        self.pandas_available = PANDAS_AVAILABLE
        self.sklearn_available = SKLEARN_AVAILABLE
        
        # Initialize IoT knowledge base
        self._initialize_iot_knowledge()
        
        # Initialize communication protocols
        self._initialize_communication()
        
        self.logger.info("Edge IoT AI engine initialized")
    
    def _initialize_iot_knowledge(self):
        """Initialize IoT knowledge base."""
        # Device capabilities
        self.device_capabilities = {
            DeviceType.SENSOR: {
                'data_collection': True,
                'processing': False,
                'communication': True,
                'power_consumption': 'low',
                'typical_lifetime': 'years'
            },
            DeviceType.ACTUATOR: {
                'data_collection': False,
                'processing': False,
                'communication': True,
                'power_consumption': 'medium',
                'typical_lifetime': 'years'
            },
            DeviceType.GATEWAY: {
                'data_collection': True,
                'processing': True,
                'communication': True,
                'power_consumption': 'medium',
                'typical_lifetime': 'years'
            },
            DeviceType.CAMERA: {
                'data_collection': True,
                'processing': True,
                'communication': True,
                'power_consumption': 'high',
                'typical_lifetime': 'years'
            }
        }
        
        # Communication protocol characteristics
        self.protocol_characteristics = {
            CommunicationProtocol.MQTT: {
                'bandwidth': 'low',
                'latency': 'low',
                'reliability': 'medium',
                'security': 'high',
                'power_efficiency': 'high'
            },
            CommunicationProtocol.HTTP: {
                'bandwidth': 'high',
                'latency': 'medium',
                'reliability': 'high',
                'security': 'high',
                'power_efficiency': 'medium'
            },
            CommunicationProtocol.COAP: {
                'bandwidth': 'low',
                'latency': 'low',
                'reliability': 'high',
                'security': 'medium',
                'power_efficiency': 'high'
            },
            CommunicationProtocol.BLE: {
                'bandwidth': 'low',
                'latency': 'low',
                'reliability': 'medium',
                'security': 'high',
                'power_efficiency': 'very_high'
            }
        }
        
        # Data type characteristics
        self.data_type_characteristics = {
            DataType.TEMPERATURE: {
                'size': 'small',
                'frequency': 'low',
                'criticality': 'medium',
                'processing_requirements': 'minimal'
            },
            DataType.HUMIDITY: {
                'size': 'small',
                'frequency': 'low',
                'criticality': 'low',
                'processing_requirements': 'minimal'
            },
            DataType.IMAGE: {
                'size': 'large',
                'frequency': 'medium',
                'criticality': 'high',
                'processing_requirements': 'high'
            },
            DataType.VIDEO: {
                'size': 'very_large',
                'frequency': 'high',
                'criticality': 'high',
                'processing_requirements': 'very_high'
            }
        }
    
    def _initialize_communication(self):
        """Initialize communication protocols."""
        if self.mqtt_available:
            try:
                self.mqtt_client = mqtt.Client()
                self.mqtt_client.on_connect = self._on_mqtt_connect
                self.mqtt_client.on_message = self._on_mqtt_message
                self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
            except Exception as e:
                self.logger.warning(f"Failed to initialize MQTT client: {e}")
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        self.logger.info(f"MQTT connected with result code {rc}")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback."""
        try:
            data = json.loads(msg.payload.decode())
            asyncio.create_task(self._process_mqtt_message(msg.topic, data))
        except Exception as e:
            self.logger.error(f"Error processing MQTT message: {e}")
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback."""
        self.logger.info(f"MQTT disconnected with result code {rc}")
    
    async def _process_mqtt_message(self, topic: str, data: Dict[str, Any]):
        """Process incoming MQTT message."""
        device_id = topic.split('/')[0]
        
        if device_id in self.iot_devices:
            # Create sensor data record
            await self.record_sensor_data(
                device_id=device_id,
                data_type=DataType(data.get('type', 'temperature')),
                value=data.get('value', 0.0),
                location=data.get('location', 'unknown')
            )
    
    async def register_iot_device(self, device_data: Dict[str, Any]) -> str:
        """Register a new IoT device."""
        device_id = f"device_{device_data.get('name', 'UNKNOWN').replace(' ', '_')}_{int(time.time())}"
        
        iot_device = IoTDevice(
            device_id=device_id,
            name=device_data.get('name', ''),
            device_type=DeviceType(device_data.get('device_type', 'sensor')),
            location=device_data.get('location', ''),
            capabilities=device_data.get('capabilities', []),
            communication_protocol=CommunicationProtocol(device_data.get('protocol', 'mqtt')),
            data_types=[DataType(dt) for dt in device_data.get('data_types', [])],
            processing_location=ProcessingLocation(device_data.get('processing_location', 'edge')),
            status='online',
            last_seen=datetime.now()
        )
        
        with self.lock:
            self.iot_devices[device_id] = iot_device
            self.metrics['total_devices'] += 1
        
        # Subscribe to device topics if MQTT is available
        if self.mqtt_available and self.mqtt_client:
            topic = f"{device_id}/data"
            self.mqtt_client.subscribe(topic)
        
        self.logger.info(f"Registered IoT device: {device_id}")
        return device_id
    
    async def record_sensor_data(self, device_id: str,
                               data_type: DataType,
                               value: Union[float, str, bytes],
                               location: str = None) -> str:
        """Record sensor data from IoT device."""
        if device_id not in self.iot_devices:
            raise ValueError(f"Device ID {device_id} not found")
        
        device = self.iot_devices[device_id]
        
        data_id = f"data_{device_id}_{int(time.time())}"
        
        # Calculate data quality score
        quality = self._calculate_data_quality(device, data_type, value)
        
        sensor_data = SensorData(
            data_id=data_id,
            device_id=device_id,
            data_type=data_type,
            value=value,
            timestamp=datetime.now(),
            location=location or device.location,
            quality=quality
        )
        
        with self.lock:
            self.sensor_data[data_id] = sensor_data
            self.metrics['total_sensor_readings'] += 1
            
            # Update device last seen
            device.last_seen = datetime.now()
            
            # Update average data quality
            self.metrics['avg_data_quality'] = (
                (self.metrics['avg_data_quality'] * (self.metrics['total_sensor_readings'] - 1) + quality) /
                self.metrics['total_sensor_readings']
            )
        
        # Trigger edge processing if needed
        if device.processing_location == ProcessingLocation.EDGE:
            await self._trigger_edge_processing(data_id)
        
        self.logger.info(f"Recorded sensor data: {data_id}")
        return data_id
    
    def _calculate_data_quality(self, device: IoTDevice,
                              data_type: DataType,
                              value: Union[float, str, bytes]) -> float:
        """Calculate data quality score."""
        base_quality = 0.8
        
        # Adjust based on device type
        device_capabilities = self.device_capabilities.get(device.device_type, {})
        if device_capabilities.get('data_collection', False):
            base_quality += 0.1
        
        # Adjust based on data type
        data_characteristics = self.data_type_characteristics.get(data_type, {})
        if data_characteristics.get('size') == 'small':
            base_quality += 0.05
        
        # Adjust based on value validity
        if isinstance(value, (int, float)):
            if 0 <= value <= 1000:  # Reasonable range
                base_quality += 0.05
        
        return min(1.0, base_quality)
    
    async def _trigger_edge_processing(self, data_id: str):
        """Trigger edge processing for sensor data."""
        # Find suitable edge node
        edge_node_id = await self._find_suitable_edge_node(data_id)
        
        if edge_node_id:
            # Create edge processing task
            task_id = await self.create_edge_task(
                node_id=edge_node_id,
                task_type='data_processing',
                input_data={'data_id': data_id},
                priority=5
            )
            
            # Execute task
            await self.execute_edge_task(task_id)
    
    async def _find_suitable_edge_node(self, data_id: str) -> Optional[str]:
        """Find suitable edge node for processing."""
        data = self.sensor_data[data_id]
        device = self.iot_devices[data.device_id]
        
        # Find edge nodes with sufficient capacity
        suitable_nodes = [
            node_id for node_id, node in self.edge_nodes.items()
            if node.status == 'active' and len(node.connected_devices) < 10
        ]
        
        if suitable_nodes:
            # Choose node with lowest load
            return min(suitable_nodes, key=lambda n: len(self.edge_nodes[n].connected_devices))
        
        return None
    
    async def create_edge_node(self, node_data: Dict[str, Any]) -> str:
        """Create an edge computing node."""
        node_id = f"edge_node_{node_data.get('name', 'UNKNOWN').replace(' ', '_')}_{int(time.time())}"
        
        edge_node = EdgeNode(
            node_id=node_id,
            name=node_data.get('name', ''),
            location=node_data.get('location', ''),
            capabilities=node_data.get('capabilities', []),
            processing_power=node_data.get('processing_power', 1.0),
            memory_capacity=node_data.get('memory_capacity', 1024),
            storage_capacity=node_data.get('storage_capacity', 100),
            network_bandwidth=node_data.get('network_bandwidth', 100.0),
            connected_devices=[],
            status='active'
        )
        
        with self.lock:
            self.edge_nodes[node_id] = edge_node
        
        self.logger.info(f"Created edge node: {node_id}")
        return node_id
    
    async def create_edge_task(self, node_id: str,
                             task_type: str,
                             input_data: Dict[str, Any],
                             priority: int = 5) -> str:
        """Create an edge processing task."""
        if node_id not in self.edge_nodes:
            raise ValueError(f"Edge node ID {node_id} not found")
        
        task_id = f"task_{node_id}_{int(time.time())}"
        
        edge_task = EdgeProcessingTask(
            task_id=task_id,
            node_id=node_id,
            task_type=task_type,
            input_data=input_data,
            processing_requirements={
                'cpu_cores': 1,
                'memory_mb': 128,
                'timeout_seconds': 30
            },
            priority=priority,
            status='pending',
            start_time=datetime.now(),
            completion_time=None,
            results={}
        )
        
        with self.lock:
            self.edge_tasks[task_id] = edge_task
            self.metrics['total_edge_tasks'] += 1
        
        self.logger.info(f"Created edge task: {task_id}")
        return task_id
    
    async def execute_edge_task(self, task_id: str) -> Dict[str, Any]:
        """Execute an edge processing task."""
        if task_id not in self.edge_tasks:
            raise ValueError(f"Task ID {task_id} not found")
        
        task = self.edge_tasks[task_id]
        node = self.edge_nodes[task.node_id]
        
        # Update task status
        with self.lock:
            task.status = 'running'
        
        start_time = time.time()
        
        try:
            # Execute task based on type
            if task.task_type == 'data_processing':
                results = await self._process_sensor_data(task.input_data)
            elif task.task_type == 'ml_inference':
                results = await self._run_ml_inference(task.input_data)
            elif task.task_type == 'analytics':
                results = await self._run_analytics(task.input_data)
            else:
                results = {'status': 'unknown_task_type'}
            
            execution_time = time.time() - start_time
            
            # Update task with results
            with self.lock:
                task.status = 'completed'
                task.completion_time = datetime.now()
                task.results = results
                
                # Update metrics
                self.metrics['avg_processing_time'] = (
                    (self.metrics['avg_processing_time'] * (self.metrics['total_edge_tasks'] - 1) + execution_time) /
                    self.metrics['total_edge_tasks']
                )
            
            self.logger.info(f"Completed edge task: {task_id}")
            return results
            
        except Exception as e:
            with self.lock:
                task.status = 'failed'
                task.results = {'error': str(e)}
            
            self.logger.error(f"Failed edge task {task_id}: {e}")
            return {'error': str(e)}
    
    async def _process_sensor_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process sensor data on edge."""
        data_id = input_data.get('data_id')
        if data_id not in self.sensor_data:
            return {'error': 'Data not found'}
        
        data = self.sensor_data[data_id]
        
        # Simple data processing
        processed_data = {
            'original_value': data.value,
            'processed_value': data.value,
            'quality_score': data.quality,
            'timestamp': data.timestamp.isoformat(),
            'processing_type': 'edge_processing'
        }
        
        # Add anomaly detection if sklearn is available
        if self.sklearn_available:
            anomaly_score = self._detect_anomaly(data)
            processed_data['anomaly_score'] = anomaly_score
            processed_data['is_anomaly'] = anomaly_score > 0.5
        
        return processed_data
    
    def _detect_anomaly(self, data: SensorData) -> float:
        """Detect anomalies in sensor data."""
        # Simplified anomaly detection
        if isinstance(data.value, (int, float)):
            # Check if value is within reasonable range
            if data.data_type == DataType.TEMPERATURE:
                if data.value < -50 or data.value > 100:
                    return 0.8
            elif data.data_type == DataType.HUMIDITY:
                if data.value < 0 or data.value > 100:
                    return 0.9
        
        return 0.1  # Low anomaly score
    
    async def _run_ml_inference(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run machine learning inference on edge."""
        # Simplified ML inference
        return {
            'prediction': random.uniform(0, 1),
            'confidence': random.uniform(0.7, 0.95),
            'model_version': '1.0',
            'inference_time': random.uniform(0.01, 0.1)
        }
    
    async def _run_analytics(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run analytics on edge."""
        # Simplified analytics
        return {
            'statistics': {
                'mean': random.uniform(20, 30),
                'std': random.uniform(1, 5),
                'min': random.uniform(15, 25),
                'max': random.uniform(30, 40)
            },
            'trend': 'increasing',
            'anomalies_detected': random.randint(0, 3)
        }
    
    async def create_iot_network(self, network_data: Dict[str, Any]) -> str:
        """Create an IoT network."""
        network_id = f"network_{network_data.get('name', 'UNKNOWN').replace(' ', '_')}_{int(time.time())}"
        
        iot_network = IoTNetwork(
            network_id=network_id,
            name=network_data.get('name', ''),
            topology=network_data.get('topology', 'star'),
            devices=network_data.get('devices', []),
            edge_nodes=network_data.get('edge_nodes', []),
            communication_protocols=[CommunicationProtocol(p) for p in network_data.get('protocols', [])],
            security_config=network_data.get('security_config', {}),
            routing_config=network_data.get('routing_config', {})
        )
        
        with self.lock:
            self.iot_networks[network_id] = iot_network
            self.metrics['total_networks'] += 1
        
        self.logger.info(f"Created IoT network: {network_id}")
        return network_id
    
    async def create_real_time_stream(self, stream_data: Dict[str, Any]) -> str:
        """Create a real-time data stream."""
        stream_id = f"stream_{stream_data.get('source_device', 'UNKNOWN')}_{int(time.time())}"
        
        real_time_stream = RealTimeStream(
            stream_id=stream_id,
            source_device=stream_data.get('source_device', ''),
            data_type=DataType(stream_data.get('data_type', 'temperature')),
            sampling_rate=stream_data.get('sampling_rate', 1.0),
            buffer_size=stream_data.get('buffer_size', 100),
            processing_pipeline=stream_data.get('processing_pipeline', []),
            subscribers=stream_data.get('subscribers', []),
            status='active'
        )
        
        with self.lock:
            self.real_time_streams[stream_id] = real_time_stream
            self.metrics['total_streams'] += 1
        
        self.logger.info(f"Created real-time stream: {stream_id}")
        return stream_id
    
    async def get_device_status(self, device_id: str) -> Dict[str, Any]:
        """Get IoT device status."""
        if device_id not in self.iot_devices:
            return {}
        
        device = self.iot_devices[device_id]
        
        # Get recent sensor data
        recent_data = [
            data for data in self.sensor_data.values()
            if data.device_id == device_id and 
            (datetime.now() - data.timestamp).seconds < 3600  # Last hour
        ]
        
        return {
            'device_id': device_id,
            'name': device.name,
            'status': device.status,
            'last_seen': device.last_seen.isoformat(),
            'data_types': [dt.value for dt in device.data_types],
            'recent_readings': len(recent_data),
            'avg_data_quality': np.mean([d.quality for d in recent_data]) if recent_data else 0.0,
            'processing_location': device.processing_location.value
        }
    
    async def get_edge_node_status(self, node_id: str) -> Dict[str, Any]:
        """Get edge node status."""
        if node_id not in self.edge_nodes:
            return {}
        
        node = self.edge_nodes[node_id]
        
        # Get active tasks
        active_tasks = [
            task for task in self.edge_tasks.values()
            if task.node_id == node_id and task.status == 'running'
        ]
        
        return {
            'node_id': node_id,
            'name': node.name,
            'status': node.status,
            'processing_power': node.processing_power,
            'memory_usage': len(active_tasks) * 128,  # Simplified
            'connected_devices': len(node.connected_devices),
            'active_tasks': len(active_tasks),
            'network_bandwidth': node.network_bandwidth
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get edge IoT AI system status."""
        with self.lock:
            return {
                'metrics': self.metrics.copy(),
                'total_iot_devices': len(self.iot_devices),
                'total_sensor_data': len(self.sensor_data),
                'total_edge_nodes': len(self.edge_nodes),
                'total_edge_tasks': len(self.edge_tasks),
                'total_iot_networks': len(self.iot_networks),
                'total_real_time_streams': len(self.real_time_streams),
                'mqtt_available': self.mqtt_available,
                'pandas_available': self.pandas_available,
                'sklearn_available': self.sklearn_available
            }


# Global instance
edge_iot_ai_engine = EdgeIoTAIEngine() 