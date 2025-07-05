"""
web_interface.py - Comprehensive web interface for UniMind AI daemon system.
Provides real-time monitoring, interactive scroll management, and persona switching.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
from pathlib import Path

# Web framework imports
try:
    from flask import Flask, render_template, jsonify, request, session, redirect, url_for
    from flask_socketio import SocketIO, emit
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# UniMind imports
from ..soul.identity import Soul
from ..soul.soul_loader import soul_loader, list_available_profiles
from ..native_models.lam_engine import lam_engine
from ..memory.unified_memory import unified_memory, MemoryType
from ..scrolls.scroll_engine import scroll_engine
from ..performance.monitoring import performance_monitor
from ..emotion.emotional_intelligence import emotional_intelligence_engine

@dataclass
class SystemStatus:
    """Current system status information."""
    is_running: bool = False
    startup_time: Optional[float] = None
    total_cycles: int = 0
    memory_usage: Dict[str, Any] = None
    cpu_usage: float = 0.0
    active_scrolls: List[str] = None
    current_persona: str = "default"
    emotional_state: Dict[str, Any] = None
    phase_status: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.memory_usage is None:
            self.memory_usage = {}
        if self.active_scrolls is None:
            self.active_scrolls = []
        if self.emotional_state is None:
            self.emotional_state = {}
        if self.phase_status is None:
            self.phase_status = {}

@dataclass
class ScrollInfo:
    """Information about a scroll."""
    name: str
    description: str
    category: str
    access_level: str
    is_active: bool = False
    last_used: Optional[float] = None
    usage_count: int = 0

@dataclass
class PersonaInfo:
    """Information about a persona."""
    name: str
    description: str
    personality: str
    access_level: str
    is_active: bool = False

class WebInterface:
    """
    Comprehensive web interface for UniMind AI daemon system.
    Provides real-time monitoring, interactive scroll management, and persona switching.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
        """Initialize the web interface."""
        self.logger = logging.getLogger('WebInterface')
        self.host = host
        self.port = port
        self.debug = debug
        
        # System state
        self.system_status = SystemStatus()
        self.current_soul: Optional[Soul] = None
        self.available_scrolls: List[ScrollInfo] = []
        self.available_personas: List[PersonaInfo] = []
        
        # Web framework
        if FLASK_AVAILABLE:
            self.app = Flask(__name__)
            self.app.config['SECRET_KEY'] = 'unimind-secret-key-2024'
            CORS(self.app)
            self.socketio = SocketIO(self.app, cors_allowed_origins="*")
            self._setup_routes()
            self._setup_socketio_events()
        else:
            self.app = None
            self.socketio = None
            self.logger.warning("Flask not available - web interface disabled")
        
        # Background tasks
        self.monitoring_thread = None
        self.is_monitoring = False
        
        self.logger.info(f"Web interface initialized on {host}:{port}")
    
    def _setup_routes(self):
        """Set up Flask routes."""
        if not self.app:
            return
        
        @self.app.route('/')
        def index():
            """Main dashboard page."""
            return render_template('dashboard.html')
        
        @self.app.route('/api/status')
        def get_status():
            """Get current system status."""
            return jsonify(self._get_system_status())
        
        @self.app.route('/api/scrolls')
        def get_scrolls():
            """Get available scrolls."""
            return jsonify(self._get_scrolls_info())
        
        @self.app.route('/api/personas')
        def get_personas():
            """Get available personas."""
            return jsonify(self._get_personas_info())
        
        @self.app.route('/api/execute_scroll', methods=['POST'])
        def execute_scroll():
            """Execute a scroll."""
            data = request.get_json()
            scroll_name = data.get('scroll_name')
            parameters = data.get('parameters', {})
            
            result = self._execute_scroll(scroll_name, parameters)
            return jsonify(result)
        
        @self.app.route('/api/switch_persona', methods=['POST'])
        def switch_persona():
            """Switch to a different persona."""
            data = request.get_json()
            persona_name = data.get('persona_name')
            
            result = self._switch_persona(persona_name)
            return jsonify(result)
        
        @self.app.route('/api/memory')
        def get_memory():
            """Get memory information."""
            return jsonify(self._get_memory_info())
        
        @self.app.route('/api/performance')
        def get_performance():
            """Get performance metrics."""
            return jsonify(self._get_performance_metrics())
        
        @self.app.route('/api/rag/status')
        def get_rag_status():
            """Get RAG system status."""
            return jsonify(self._get_rag_status())
        
        @self.app.route('/api/rag/search', methods=['POST'])
        def rag_search():
            """Search knowledge using RAG system."""
            data = request.get_json()
            query = data.get('query', '')
            method = data.get('method', 'hybrid')
            max_results = data.get('max_results', 10)
            
            result = self._rag_search(query, method, max_results)
            return jsonify(result)
        
        @self.app.route('/api/rag/verify', methods=['POST'])
        def rag_verify():
            """Verify information using RAG system."""
            data = request.get_json()
            claim = data.get('claim', '')
            
            result = self._rag_verify(claim)
            return jsonify(result)
        
        @self.app.route('/api/rag/add_knowledge', methods=['POST'])
        def rag_add_knowledge():
            """Add knowledge to RAG system."""
            data = request.get_json()
            content = data.get('content', '')
            source_type = data.get('source_type', 'user_input')
            source_id = data.get('source_id', 'web_interface')
            confidence = data.get('confidence', 0.8)
            tags = data.get('tags', [])
            
            result = self._rag_add_knowledge(content, source_type, source_id, confidence, tags)
            return jsonify(result)
    
    def _setup_socketio_events(self):
        """Set up SocketIO events for real-time communication."""
        if not self.socketio:
            return
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            self.logger.info("Client connected to web interface")
            emit('status_update', self._get_system_status())
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            self.logger.info("Client disconnected from web interface")
        
        @self.socketio.on('request_status')
        def handle_status_request():
            """Handle status request."""
            emit('status_update', self._get_system_status())
    
    def start(self):
        """Start the web interface."""
        if not self.app:
            self.logger.error("Web interface not available - Flask not installed")
            return False
        
        try:
            # Start monitoring thread
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            # Start Flask app
            self.socketio.run(self.app, host=self.host, port=self.port, debug=self.debug)
            return True
        except Exception as e:
            self.logger.error(f"Failed to start web interface: {e}")
            return False
    
    def stop(self):
        """Stop the web interface."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                # Update system status
                self._update_system_status()
                
                # Emit status update to connected clients
                if self.socketio:
                    self.socketio.emit('status_update', self._get_system_status())
                
                time.sleep(2)  # Update every 2 seconds
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def _update_system_status(self):
        """Update system status information."""
        try:
            # Get memory usage
            memory_info = unified_memory.get_memory_stats()
            self.system_status.memory_usage = {
                'total_memories': memory_info.get('total_memories', 0),
                'active_contexts': memory_info.get('active_contexts', 0),
                'memory_size_mb': memory_info.get('memory_size_mb', 0)
            }
            
            # Get performance metrics
            if hasattr(performance_monitor, 'get_current_metrics'):
                perf_metrics = performance_monitor.get_current_metrics()
                self.system_status.cpu_usage = perf_metrics.get('cpu_usage', 0.0)
            
            # Get emotional state
            if hasattr(emotional_intelligence_engine, 'get_current_emotional_state'):
                emotional_state = emotional_intelligence_engine.get_current_emotional_state()
                self.system_status.emotional_state = {
                    'primary_emotion': emotional_state.primary_emotion.value if hasattr(emotional_state, 'primary_emotion') else 'neutral',
                    'intensity': emotional_state.intensity if hasattr(emotional_state, 'intensity') else 0.0
                }
            
            # Update phase status
            self.system_status.phase_status = {
                'phase1': True,  # Core Enhancement
                'phase2': True,  # Intelligence Parity
                'phase3': True,  # Performance Optimization
                'phase4': True   # Advanced Capabilities
            }
            
        except Exception as e:
            self.logger.error(f"Error updating system status: {e}")
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return asdict(self.system_status)
    
    def _get_scrolls_info(self) -> List[Dict[str, Any]]:
        """Get information about available scrolls."""
        scrolls = []
        try:
            # Get available scrolls from scroll engine
            if hasattr(scroll_engine, 'get_available_scrolls'):
                available_scrolls = scroll_engine.get_available_scrolls()
                for scroll_name in available_scrolls:
                    scroll_info = ScrollInfo(
                        name=scroll_name,
                        description=f"Scroll: {scroll_name}",
                        category="general",
                        access_level="basic",
                        is_active=scroll_name in self.system_status.active_scrolls
                    )
                    scrolls.append(asdict(scroll_info))
        except Exception as e:
            self.logger.error(f"Error getting scrolls info: {e}")
        
        return scrolls
    
    def _get_personas_info(self) -> List[Dict[str, Any]]:
        """Get information about available personas."""
        personas = []
        try:
            # Get available profiles
            available_profiles = list_available_profiles()
            for profile_name in available_profiles:
                profile_info = soul_loader.get_profile_info(profile_name)
                persona_info = PersonaInfo(
                    name=profile_name,
                    description=profile_info.get('description', 'Unknown persona'),
                    personality=profile_info.get('personality', 'Unknown'),
                    access_level=profile_info.get('access_level', 'basic'),
                    is_active=profile_name == self.system_status.current_persona
                )
                personas.append(asdict(persona_info))
        except Exception as e:
            self.logger.error(f"Error getting personas info: {e}")
        
        return personas
    
    def _execute_scroll(self, scroll_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a scroll."""
        try:
            if hasattr(scroll_engine, 'execute_scroll'):
                result = scroll_engine.execute_scroll(scroll_name, parameters)
                return {
                    'success': True,
                    'result': result,
                    'message': f'Scroll {scroll_name} executed successfully'
                }
            else:
                return {
                    'success': False,
                    'message': 'Scroll engine not available'
                }
        except Exception as e:
            self.logger.error(f"Error executing scroll {scroll_name}: {e}")
            return {
                'success': False,
                'message': f'Error executing scroll: {str(e)}'
            }
    
    def _switch_persona(self, persona_name: str) -> Dict[str, Any]:
        """Switch to a different persona."""
        try:
            # Load the new soul profile
            new_soul = soul_loader.load_soul(persona_name)
            if new_soul:
                self.current_soul = new_soul
                self.system_status.current_persona = persona_name
                return {
                    'success': True,
                    'message': f'Switched to persona: {persona_name}'
                }
            else:
                return {
                    'success': False,
                    'message': f'Persona {persona_name} not found'
                }
        except Exception as e:
            self.logger.error(f"Error switching persona: {e}")
            return {
                'success': False,
                'message': f'Error switching persona: {str(e)}'
            }
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information."""
        try:
            memory_stats = unified_memory.get_memory_stats()
            return {
                'total_memories': memory_stats.get('total_memories', 0),
                'active_contexts': memory_stats.get('active_contexts', 0),
                'memory_size_mb': memory_stats.get('memory_size_mb', 0),
                'recent_memories': unified_memory.get_memory(
                    'default', MemoryType.CONVERSATION, limit=10
                )
            }
        except Exception as e:
            self.logger.error(f"Error getting memory info: {e}")
            return {}
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        try:
            if hasattr(performance_monitor, 'get_current_metrics'):
                return performance_monitor.get_current_metrics()
            else:
                return {
                    'cpu_usage': 0.0,
                    'memory_usage': 0.0,
                    'response_time': 0.0
                }
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def _get_rag_status(self) -> Dict[str, Any]:
        """Get RAG system status."""
        try:
            # Import RAG system
            from ..advanced.rag_system import rag_system
            status = asyncio.run(rag_system.get_system_status())
            return status
        except Exception as e:
            self.logger.error(f"Error getting RAG status: {e}")
            return {
                'total_chunks': 0,
                'embedding_model_available': False,
                'web_search_enabled': False,
                'wikipedia_enabled': False,
                'verification_sources': 0,
                'error': str(e)
            }
    
    def _rag_search(self, query: str, method: str, max_results: int) -> Dict[str, Any]:
        """Search knowledge using RAG system."""
        try:
            from ..advanced.rag_system import rag_system, RetrievalMethod
            
            # Map method string to enum
            method_map = {
                'semantic': RetrievalMethod.SEMANTIC_SIMILARITY,
                'keyword': RetrievalMethod.KEYWORD_MATCHING,
                'vector': RetrievalMethod.VECTOR_SIMILARITY,
                'hybrid': RetrievalMethod.HYBRID,
                'contextual': RetrievalMethod.CONTEXTUAL
            }
            
            retrieval_method = method_map.get(method, RetrievalMethod.HYBRID)
            
            # Perform search
            result = asyncio.run(rag_system.retrieve_knowledge(
                query=query,
                method=retrieval_method,
                max_results=max_results
            ))
            
            return {
                'success': True,
                'query': query,
                'method': method,
                'total_results': result.total_results,
                'confidence': result.confidence,
                'retrieval_time': result.retrieval_time,
                'chunks': [
                    {
                        'content': chunk.content,
                        'source_id': chunk.source_id,
                        'confidence': chunk.confidence,
                        'verification_level': chunk.verification_level.value,
                        'tags': chunk.tags
                    }
                    for chunk in result.retrieved_chunks
                ]
            }
        except Exception as e:
            self.logger.error(f"Error in RAG search: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'method': method
            }
    
    def _rag_verify(self, claim: str) -> Dict[str, Any]:
        """Verify information using RAG system."""
        try:
            from ..advanced.rag_system import rag_system
            
            # Perform verification
            result = asyncio.run(rag_system.verify_information(claim))
            
            return {
                'success': True,
                'claim': claim,
                'verified': result['verified'],
                'confidence': result['confidence'],
                'verification_level': result['verification_level'],
                'analysis': result['analysis'],
                'supporting_evidence': result['supporting_evidence'],
                'contradicting_evidence': result['contradicting_evidence'],
                'sources': result['sources']
            }
        except Exception as e:
            self.logger.error(f"Error in RAG verification: {e}")
            return {
                'success': False,
                'error': str(e),
                'claim': claim
            }
    
    def _rag_add_knowledge(self, content: str, source_type: str, source_id: str, 
                          confidence: float, tags: List[str]) -> Dict[str, Any]:
        """Add knowledge to RAG system."""
        try:
            from ..advanced.rag_system import rag_system, KnowledgeSourceType
            
            # Map source type string to enum
            source_type_map = {
                'memory': KnowledgeSourceType.MEMORY,
                'document': KnowledgeSourceType.DOCUMENT,
                'web': KnowledgeSourceType.WEB,
                'wikipedia': KnowledgeSourceType.WIKIPEDIA,
                'database': KnowledgeSourceType.DATABASE,
                'api': KnowledgeSourceType.API,
                'user_input': KnowledgeSourceType.USER_INPUT,
                'verified_fact': KnowledgeSourceType.VERIFIED_FACT
            }
            
            knowledge_source_type = source_type_map.get(source_type, KnowledgeSourceType.USER_INPUT)
            
            # Add knowledge
            chunk_id = asyncio.run(rag_system.add_knowledge(
                content=content,
                source_type=knowledge_source_type,
                source_id=source_id,
                confidence=confidence,
                tags=tags
            ))
            
            return {
                'success': True,
                'chunk_id': chunk_id,
                'content': content,
                'source_type': source_type,
                'source_id': source_id,
                'confidence': confidence,
                'tags': tags
            }
        except Exception as e:
            self.logger.error(f"Error adding knowledge to RAG: {e}")
            return {
                'success': False,
                'error': str(e),
                'content': content,
                'source_type': source_type,
                'source_id': source_id
            }

# Global web interface instance
web_interface = WebInterface() 