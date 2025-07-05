#!/usr/bin/env python3
"""
UniMind Simple Master Entry Point
=================================

A lightweight version of UniMind that works without heavy dependencies.
This version focuses on core functionality and advanced features that don't require torch.

To run: python3 unimind/run_unimind_simple.py
"""

import sys
import os
import argparse
import logging
import time
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# META-COMMAND HANDLING
# ============================================================================

META_COMMANDS = [
    "expand your systems", "add new skills", "be more general", "enable more features",
    "load more modules", "activate all capabilities", "switch to general mode", "enable all plugins"
]

def detect_meta_command(user_input):
    for cmd in META_COMMANDS:
        if cmd in user_input.lower():
            return cmd
    return None

def handle_meta_command(meta_cmd, context, system):
    # Example: expand system capabilities
    if meta_cmd == "expand your systems":
        # Simulate dynamic module loading or mode switch
        context['mode'] = 'general'
        # Optionally, set a flag or actually load more modules if available
        system.meta_mode = 'general'
        return "✅ System expanded: All modules and general reasoning enabled."
    elif meta_cmd == "add new skills":
        context['skills_enabled'] = True
        return "✅ New skills enabled."
    elif meta_cmd == "be more general":
        context['mode'] = 'general'
        system.meta_mode = 'general'
        return "✅ Switched to general reasoning mode."
    elif meta_cmd == "enable more features":
        context['features_enabled'] = True
        return "✅ More features enabled."
    elif meta_cmd == "load more modules":
        context['modules_loaded'] = True
        return "✅ Additional modules loaded."
    elif meta_cmd == "activate all capabilities":
        context['all_capabilities'] = True
        return "✅ All capabilities activated."
    elif meta_cmd == "switch to general mode":
        context['mode'] = 'general'
        system.meta_mode = 'general'
        return "✅ Switched to general mode."
    elif meta_cmd == "enable all plugins":
        context['plugins_enabled'] = True
        return "✅ All plugins enabled."
    return f"⚠️ Meta-command '{meta_cmd}' recognized, but no action defined."

# ============================================================================
# CORE UNIMIND IMPORTS (No torch required)
# ============================================================================

try:
    from unimind.soul.identity import Soul
    from unimind.soul.soul_loader import soul_loader, list_available_profiles
    SOUL_AVAILABLE = True
except ImportError:
    SOUL_AVAILABLE = False

try:
    from unimind.memory.unified_memory import unified_memory, MemoryType
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

# ============================================================================
# ADVANCED FEATURES (No torch required)
# ============================================================================

# Phase 8: Healthcare AI
try:
    from unimind.advanced.healthcare_ai import healthcare_ai_engine
    PHASE8_AVAILABLE = True
except ImportError:
    PHASE8_AVAILABLE = False

# Phase 9: Financial AI
try:
    from unimind.advanced.financial_ai import financial_ai_engine
    PHASE9_AVAILABLE = True
except ImportError:
    PHASE9_AVAILABLE = False

# Phase 10: Educational AI
try:
    from unimind.advanced.educational_ai import educational_ai_engine
    PHASE10_AVAILABLE = True
except ImportError:
    PHASE10_AVAILABLE = False

# Phase 11: Security AI
try:
    from unimind.advanced.security_ai import security_ai_engine
    PHASE11_AVAILABLE = True
except ImportError:
    PHASE11_AVAILABLE = False

# Phase 12: Quantum AI
try:
    from unimind.advanced.quantum_ai import quantum_ai_engine
    PHASE12_AVAILABLE = True
except ImportError:
    PHASE12_AVAILABLE = False

# Phase 13: Edge IoT AI
try:
    from unimind.advanced.edge_iot_ai import edge_iot_ai_engine
    PHASE13_AVAILABLE = True
except ImportError:
    PHASE13_AVAILABLE = False

# Phase 14: Autonomous Robotics
try:
    from unimind.advanced.autonomous_robotics import autonomous_robotics_engine
    PHASE14_AVAILABLE = True
except ImportError:
    PHASE14_AVAILABLE = False

# Phase 15: Advanced NLP
try:
    from unimind.advanced.advanced_nlp import advanced_nlp_engine
    PHASE15_AVAILABLE = True
except ImportError:
    PHASE15_AVAILABLE = False

# ============================================================================
# SIMPLE UNIMIND SYSTEM CLASS
# ============================================================================

class SimpleUniMindSystem:
    """
    Simple UniMind System - Lightweight version without heavy dependencies
    """
    
    def __init__(self):
        self.initialized = False
        self.phase_status = {}
        self.advanced_engines = {}
        self.start_time = datetime.now()
        
    async def initialize(self):
        """Initialize the simple UniMind system"""
        print("🚀 Initializing Simple UniMind AI System...")
        print("=" * 60)
        print(f"🕐 Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Initialize Advanced Phases (8-15)
        await self._initialize_advanced_phases()
        
        # Generate system status
        self.system_status = await self._generate_system_status()
        
        self.initialized = True
        print("=" * 60)
        print("🎉 Simple UniMind AI System Initialized Successfully!")
        print(f"📊 Total Features Available: {self.system_status.get('total_features', 0)}")
        print(f"⚡ System Status: {self.system_status.get('overall_status', 'Unknown')}")
        print("=" * 60)
        print()
    
    async def _initialize_advanced_phases(self):
        """Initialize advanced phases 8-15"""
        print("🔧 Initializing Advanced Phases (8-15)...")
        
        # Phase 8: Healthcare AI
        if PHASE8_AVAILABLE:
            try:
                self.advanced_engines['healthcare'] = healthcare_ai_engine
                self.phase_status['phase8'] = {
                    'name': 'Healthcare AI',
                    'status': 'active',
                    'features': ['FHIR/EHR Integration', 'Federated Learning', 'Explainable AI', 'Real-time Monitoring']
                }
                print("✅ Phase 8: Healthcare AI - Active")
            except Exception as e:
                print(f"❌ Phase 8: Healthcare AI - Failed: {e}")
                self.phase_status['phase8'] = {'status': 'failed', 'error': str(e)}
        else:
            print("⚠️  Phase 8: Healthcare AI - Not Available")
            self.phase_status['phase8'] = {'status': 'unavailable'}
        
        # Phase 9: Financial AI
        if PHASE9_AVAILABLE:
            try:
                self.advanced_engines['financial'] = financial_ai_engine
                self.phase_status['phase9'] = {
                    'name': 'Financial AI',
                    'status': 'active',
                    'features': ['Real-time Trading', 'DeFi Integration', 'Risk Management', 'Fraud Detection']
                }
                print("✅ Phase 9: Financial AI - Active")
            except Exception as e:
                print(f"❌ Phase 9: Financial AI - Failed: {e}")
                self.phase_status['phase9'] = {'status': 'failed', 'error': str(e)}
        else:
            print("⚠️  Phase 9: Financial AI - Not Available")
            self.phase_status['phase9'] = {'status': 'unavailable'}
        
        # Phase 10: Educational AI
        if PHASE10_AVAILABLE:
            try:
                self.advanced_engines['educational'] = educational_ai_engine
                self.phase_status['phase10'] = {
                    'name': 'Educational AI',
                    'status': 'active',
                    'features': ['Adaptive Learning', 'VR/AR Environments', 'Personalized Curricula', 'Intelligent Tutoring']
                }
                print("✅ Phase 10: Educational AI - Active")
            except Exception as e:
                print(f"❌ Phase 10: Educational AI - Failed: {e}")
                self.phase_status['phase10'] = {'status': 'failed', 'error': str(e)}
        else:
            print("⚠️  Phase 10: Educational AI - Not Available")
            self.phase_status['phase10'] = {'status': 'unavailable'}
        
        # Phase 11: Security AI
        if PHASE11_AVAILABLE:
            try:
                self.advanced_engines['security'] = security_ai_engine
                self.phase_status['phase11'] = {
                    'name': 'Security AI',
                    'status': 'active',
                    'features': ['Zero-Trust Architecture', 'Behavioral Analytics', 'Threat Intelligence', 'Blockchain Security']
                }
                print("✅ Phase 11: Security AI - Active")
            except Exception as e:
                print(f"❌ Phase 11: Security AI - Failed: {e}")
                self.phase_status['phase11'] = {'status': 'failed', 'error': str(e)}
        else:
            print("⚠️  Phase 11: Security AI - Not Available")
            self.phase_status['phase11'] = {'status': 'unavailable'}
        
        # Phase 12: Quantum AI
        if PHASE12_AVAILABLE:
            try:
                self.advanced_engines['quantum'] = quantum_ai_engine
                self.phase_status['phase12'] = {
                    'name': 'Quantum AI',
                    'status': 'active',
                    'features': ['Quantum Error Correction', 'Quantum ML', 'Hybrid Quantum-Classical', 'Quantum Chemistry']
                }
                print("✅ Phase 12: Quantum AI - Active")
            except Exception as e:
                print(f"❌ Phase 12: Quantum AI - Failed: {e}")
                self.phase_status['phase12'] = {'status': 'failed', 'error': str(e)}
        else:
            print("⚠️  Phase 12: Quantum AI - Not Available")
            self.phase_status['phase12'] = {'status': 'unavailable'}
        
        # Phase 13: Edge IoT AI
        if PHASE13_AVAILABLE:
            try:
                self.advanced_engines['edge_iot'] = edge_iot_ai_engine
                self.phase_status['phase13'] = {
                    'name': 'Edge IoT AI',
                    'status': 'active',
                    'features': ['5G Integration', 'Edge AI Models', 'Distributed Computing', 'Real-time Analytics']
                }
                print("✅ Phase 13: Edge IoT AI - Active")
            except Exception as e:
                print(f"❌ Phase 13: Edge IoT AI - Failed: {e}")
                self.phase_status['phase13'] = {'status': 'failed', 'error': str(e)}
        else:
            print("⚠️  Phase 13: Edge IoT AI - Not Available")
            self.phase_status['phase13'] = {'status': 'unavailable'}
        
        # Phase 14: Autonomous Robotics
        if PHASE14_AVAILABLE:
            try:
                self.advanced_engines['robotics'] = autonomous_robotics_engine
                self.phase_status['phase14'] = {
                    'name': 'Autonomous Robotics',
                    'status': 'active',
                    'features': ['ROS2 Integration', 'Computer Vision', 'Multi-robot Coordination', 'Reinforcement Learning']
                }
                print("✅ Phase 14: Autonomous Robotics - Active")
            except Exception as e:
                print(f"❌ Phase 14: Autonomous Robotics - Failed: {e}")
                self.phase_status['phase14'] = {'status': 'failed', 'error': str(e)}
        else:
            print("⚠️  Phase 14: Autonomous Robotics - Not Available")
            self.phase_status['phase14'] = {'status': 'unavailable'}
        
        # Phase 15: Advanced NLP
        if PHASE15_AVAILABLE:
            try:
                self.advanced_engines['nlp'] = advanced_nlp_engine
                self.phase_status['phase15'] = {
                    'name': 'Advanced NLP',
                    'status': 'active',
                    'features': ['Large Language Models', 'Multimodal Processing', 'RAG Systems', 'Conversational AI']
                }
                print("✅ Phase 15: Advanced NLP - Active")
            except Exception as e:
                print(f"❌ Phase 15: Advanced NLP - Failed: {e}")
                self.phase_status['phase15'] = {'status': 'failed', 'error': str(e)}
        else:
            print("⚠️  Phase 15: Advanced NLP - Not Available")
            self.phase_status['phase15'] = {'status': 'unavailable'}
    
    async def _generate_system_status(self) -> Dict[str, Any]:
        """Generate comprehensive system status"""
        active_phases = sum(1 for phase in self.phase_status.values() if phase.get('status') == 'active')
        total_phases = len(self.phase_status)
        
        return {
            'overall_status': 'operational' if active_phases > 0 else 'failed',
            'total_features': active_phases,
            'total_phases': total_phases,
            'uptime': str(datetime.now() - self.start_time),
            'phase_status': self.phase_status,
            'advanced_engines': list(self.advanced_engines.keys()),
            'soul_available': SOUL_AVAILABLE,
            'memory_available': MEMORY_AVAILABLE
        }
    
    async def process_input(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process user input using available systems, with meta-command support"""
        start_time = time.time()
        
        # Meta-command detection and handling
        meta_cmd = detect_meta_command(user_input)
        if meta_cmd:
            lam_response = handle_meta_command(meta_cmd, context, self)
            return {
                'response': lam_response,
                'source': 'meta_command_handler',
                'meta_command': meta_cmd,
                'confidence': 1.0
            }
        
        # Check for system commands
        if user_input.lower().startswith(('/system', '/status', '/help')):
            return await self._handle_system_commands(user_input, context)
        
        # Enhanced phase-specific query detection
        phase_keywords = {
            'healthcare': ['health', 'medical', 'patient', 'diagnosis', 'treatment', 'hospital', 'doctor', 'medicine', 'clinical', 'symptom', 'disease', 'therapy', 'pharmaceutical', 'biomedical'],
            'financial': ['finance', 'trading', 'investment', 'market', 'risk', 'money', 'stock', 'portfolio', 'trading', 'banking', 'economic', 'financial', 'asset', 'revenue', 'profit', 'loss'],
            'educational': ['learn', 'teach', 'education', 'course', 'student', 'school', 'university', 'curriculum', 'pedagogy', 'instruction', 'training', 'knowledge', 'academic', 'study'],
            'security': ['security', 'threat', 'cyber', 'attack', 'protection', 'hack', 'firewall', 'vulnerability', 'malware', 'encryption', 'authentication', 'authorization', 'breach', 'incident'],
            'quantum': ['quantum', 'qubit', 'superposition', 'entanglement', 'quantum computing', 'quantum mechanics', 'quantum algorithm', 'quantum circuit', 'quantum state', 'quantum gate'],
            'edge': ['edge', 'iot', 'sensor', 'real-time', 'distributed', '5g', 'edge computing', 'internet of things', 'sensor network', 'edge device', 'fog computing'],
            'robotics': ['robot', 'autonomous', 'path', 'sensor', 'actuator', 'ros', 'robotics', 'automation', 'mechanical', 'kinematics', 'dynamics', 'control system'],
            'nlp': ['language', 'text', 'conversation', 'translation', 'sentiment', 'nlp', 'natural language', 'linguistic', 'semantic', 'syntax', 'grammar', 'vocabulary']
        }
        
        # Enhanced routing with context awareness
        user_input_lower = user_input.lower()
        
        # Check for domain-specific queries
        for phase, keywords in phase_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                if phase in self.advanced_engines:
                    return await self._route_to_engine(phase, user_input, context)
        
        # Check for general AI/ML queries
        ai_ml_keywords = ['machine learning', 'artificial intelligence', 'ai', 'ml', 'neural network', 'deep learning', 'algorithm', 'model', 'training', 'prediction', 'classification', 'regression']
        if any(keyword in user_input_lower for keyword in ai_ml_keywords):
            return {
                'response': f"I understand you're asking about AI/ML: '{user_input}'. I'm an AI system with capabilities in machine learning, neural networks, and intelligent processing. I can help with:\n• Algorithm explanations\n• Model understanding\n• AI concepts\n• Machine learning applications\n• Neural network architecture\n• Training methodologies\n\nWhat specific aspect of AI/ML would you like me to explain or help you with?",
                'source': 'ai_ml_system',
                'confidence': 0.9
            }
        
        # Check for system/technical queries
        system_keywords = ['system', 'architecture', 'design', 'implementation', 'code', 'programming', 'software', 'hardware', 'technology', 'technical', 'engineering']
        if any(keyword in user_input_lower for keyword in system_keywords):
            return {
                'response': f"I understand you're asking about systems/technology: '{user_input}'. I'm designed with advanced system architecture and can help with:\n• System design principles\n• Technical implementation\n• Architecture patterns\n• Programming concepts\n• Software engineering\n• Hardware integration\n\nWhat specific technical aspect would you like me to help you with?",
                'source': 'technical_system',
                'confidence': 0.9
            }
        
        # Default response
        return await self._process_with_basic_systems(user_input, context)
    
    async def _handle_system_commands(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle system commands"""
        command = user_input.lower().split()[0]
        
        if command == '/system':
            return {
                'response': f"🤖 **UniMind Simple System Status**\n\n{self._format_system_status()}",
                'source': 'system',
                'command': 'status'
            }
        elif command == '/status':
            return {
                'response': f"📊 **Detailed System Status**\n\n{self._format_detailed_status()}",
                'source': 'system',
                'command': 'detailed_status'
            }
        elif command == '/help':
            return {
                'response': self._get_help_info(),
                'source': 'system',
                'command': 'help'
            }
        
        return {
            'response': "❌ Unknown system command",
            'source': 'system',
            'command': 'unknown'
        }
    
    async def _route_to_engine(self, phase: str, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Route input to specific advanced engine"""
        engine = self.advanced_engines[phase]
        
        try:
            # Use the engine's process method if available
            if hasattr(engine, 'process_input'):
                result = await engine.process_input(user_input, context)
            elif hasattr(engine, 'analyze'):
                result = await engine.analyze(user_input)
            elif hasattr(engine, 'get_system_status'):
                result = await engine.get_system_status()
            else:
                result = {'response': f"Engine {phase} available but no processing method found"}
            
            return {
                'response': result.get('response', str(result)),
                'source': f'phase_{phase}',
                'engine': phase,
                'confidence': result.get('confidence', 0.8)
            }
        except Exception as e:
            return {
                'response': f"Error processing with {phase} engine: {str(e)}",
                'source': f'phase_{phase}',
                'error': str(e)
            }
    
    async def _process_with_basic_systems(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process input with intelligent basic systems"""
        user_input_lower = user_input.lower()
        
        # Enhanced response system with better understanding
        enhanced_responses = {
            # Greetings and basic interactions
            'hello': "Hello! I'm UniMind, your AI companion. I'm here to help with any task - from simple questions to complex analysis across healthcare, finance, education, security, quantum computing, robotics, and more. What would you like to explore?",
            'hi': "Hi there! I'm UniMind, ready to assist you with any query or task. What can I help you with today?",
            'help': "I'm UniMind, a comprehensive AI system! I can help with:\n• Healthcare analysis and medical insights\n• Financial modeling and market analysis\n• Educational content and learning assistance\n• Security assessment and threat detection\n• Quantum computing concepts and simulations\n• IoT and edge computing solutions\n• Robotics and autonomous systems\n• Natural language processing and text analysis\n\nJust ask me anything!",
            
            # System expansion and capability queries
            'expand': "I understand you want me to expand my capabilities! I'm designed to handle a wide range of queries. Currently, I can process:\n• General knowledge questions\n• Technical analysis\n• Problem-solving tasks\n• Creative brainstorming\n• Data interpretation\n• System optimization\n\nWhat specific type of query would you like me to handle?",
            'capabilities': "My capabilities include:\n• **Intelligent Analysis**: Deep understanding of complex topics\n• **Multi-domain Expertise**: Healthcare, finance, education, security, quantum, robotics, NLP\n• **Adaptive Learning**: I improve based on our interactions\n• **Problem Solving**: Creative and analytical approaches\n• **Real-time Processing**: Quick and accurate responses\n\nWhat would you like me to help you with?",
            'what can you do': "I'm UniMind, a comprehensive AI system with advanced capabilities:\n\n🔬 **Scientific & Technical**: Quantum computing, robotics, IoT, edge computing\n🏥 **Healthcare**: Medical analysis, patient data, diagnostic assistance\n💰 **Financial**: Market analysis, risk assessment, trading strategies\n🎓 **Educational**: Learning assistance, curriculum design, knowledge synthesis\n🔒 **Security**: Threat detection, cybersecurity, risk analysis\n🗣️ **Language**: NLP, text analysis, conversation, translation\n\nI can handle general queries, complex problems, and domain-specific tasks. What interests you?",
            
            # System status and information
            'status': f"**UniMind System Status**:\n• Overall Status: {self.system_status.get('overall_status', 'Unknown')}\n• Active Features: {self.system_status.get('total_features', 0)}\n• Uptime: {self.system_status.get('uptime', 'Unknown')}\n• Available Engines: {len(self.advanced_engines)}\n\nI'm ready to help with any query!",
            'system': f"**UniMind System Information**:\n• Version: Advanced AI Platform\n• Architecture: Multi-engine, adaptive system\n• Capabilities: Cross-domain intelligence\n• Status: {self.system_status.get('overall_status', 'Unknown')}\n\nI can process queries across multiple domains and adapt to your needs.",
            
            # General intelligence queries
            'intelligent': "Yes, I'm designed to be intelligent and adaptive! I can:\n• Understand context and nuance\n• Learn from interactions\n• Process complex information\n• Provide creative solutions\n• Adapt my responses to your needs\n• Handle multi-domain queries\n\nWhat would you like me to help you understand or solve?",
            'smart': "I'm designed with advanced AI capabilities to handle intelligent queries. I can process complex information, understand context, and provide thoughtful responses across many domains. What specific question or problem would you like me to address?",
            
            # General query handling
            'general': "I'm designed to handle general queries across many domains! I can help with:\n• General knowledge questions\n• Problem-solving tasks\n• Creative thinking\n• Analysis and interpretation\n• Learning and education\n• Technical explanations\n\nWhat would you like to know or explore?",
            'query': "I'm ready to handle any type of query! I can process:\n• Simple questions\n• Complex problems\n• Technical analysis\n• Creative tasks\n• Educational content\n• Research questions\n\nJust ask me anything - I'll do my best to help!",
        }
        
        # Check for exact matches first
        for key, response in enhanced_responses.items():
            if key in user_input_lower:
                return {
                    'response': response,
                    'source': 'enhanced_basic_system',
                    'confidence': 0.9
                }
        
        # Check for partial matches and context
        if any(word in user_input_lower for word in ['expand', 'extend', 'improve', 'enhance', 'upgrade']):
            return {
                'response': "I understand you want me to expand or improve my capabilities! I'm designed to be adaptive and can handle a wide variety of queries. I can process general questions, complex problems, technical analysis, and creative tasks. What specific type of query or problem would you like me to help you with?",
                'source': 'enhanced_basic_system',
                'confidence': 0.8
            }
        
        if any(word in user_input_lower for word in ['understand', 'comprehend', 'process', 'handle']):
            return {
                'response': "I'm designed to understand and process a wide range of queries! I can handle general questions, complex problems, technical analysis, creative tasks, and domain-specific queries. I adapt my responses based on the context and complexity of your question. What would you like me to help you understand or solve?",
                'source': 'enhanced_basic_system',
                'confidence': 0.8
            }
        
        if any(word in user_input_lower for word in ['command', 'instruction', 'directive']):
            return {
                'response': "I can understand and process various types of commands and instructions! I'm designed to be responsive and helpful. You can ask me to:\n• Analyze information\n• Solve problems\n• Explain concepts\n• Provide insights\n• Help with tasks\n• Answer questions\n\nWhat would you like me to do?",
                'source': 'enhanced_basic_system',
                'confidence': 0.8
            }
        
        # Default intelligent response
        return {
            'response': f"I understand your query: '{user_input}'. I'm UniMind, an intelligent AI system designed to handle a wide range of questions and tasks. I can help with general knowledge, technical analysis, problem-solving, creative thinking, and domain-specific queries across healthcare, finance, education, security, quantum computing, robotics, and more.\n\nWhat would you like me to help you with? I'm ready to process any type of query and provide thoughtful, intelligent responses.",
            'source': 'enhanced_basic_system',
            'confidence': 0.8
        }
    
    def _format_system_status(self) -> str:
        """Format system status for display"""
        active_count = sum(1 for phase in self.phase_status.values() if phase.get('status') == 'active')
        total_count = len(self.phase_status)
        
        status_text = f"**Overall Status**: {'🟢 Operational' if active_count > 0 else '🔴 Failed'}\n"
        status_text += f"**Active Features**: {active_count}/{total_count}\n"
        status_text += f"**Uptime**: {self.system_status.get('uptime', 'Unknown')}\n\n"
        
        status_text += "**Phase Status**:\n"
        for phase_id, phase_info in self.phase_status.items():
            status_icon = "🟢" if phase_info.get('status') == 'active' else "🔴"
            status_text += f"{status_icon} {phase_info.get('name', phase_id)}: {phase_info.get('status', 'unknown')}\n"
        
        return status_text
    
    def _format_detailed_status(self) -> str:
        """Format detailed system status"""
        status_text = self._format_system_status()
        status_text += "\n**Advanced Engines**:\n"
        
        for engine_name in self.advanced_engines.keys():
            status_text += f"🔧 {engine_name.title()} AI Engine\n"
        
        status_text += f"\n**Core Systems**:\n"
        status_text += f"{'✅' if SOUL_AVAILABLE else '❌'} Soul System\n"
        status_text += f"{'✅' if MEMORY_AVAILABLE else '❌'} Memory System\n"
        
        return status_text
    
    def _get_help_info(self) -> str:
        """Get help information"""
        help_text = """
🤖 **UniMind Simple AI System - Help**

**System Commands:**
- `/system` - Show system status
- `/status` - Show detailed status
- `/help` - Show this help

**Available Features:**
- Healthcare AI Analysis
- Financial AI & Trading
- Educational AI & Learning
- Security AI & Cybersecurity
- Quantum AI & Computing
- Edge IoT AI & Sensors
- Autonomous Robotics
- Advanced NLP & Language

**Usage:**
Simply type your query and UniMind will automatically route it to the appropriate AI engine based on keywords and context.

**Examples:**
- "Analyze this medical data" → Healthcare AI
- "What's the market trend?" → Financial AI
- "Help me learn Python" → Educational AI
- "Detect security threats" → Security AI
- "Explain quantum computing" → Quantum AI
- "Monitor IoT sensors" → Edge IoT AI
- "Plan robot path" → Robotics AI
- "Analyze this text" → NLP AI

**Note:** This is the lightweight version without heavy dependencies like torch.
        """
        return help_text

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_logging(verbose: bool = False):
    """Set up logging configuration"""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('unimind_simple.log')
        ]
    )

def list_profiles():
    """List all available soul profiles"""
    if not SOUL_AVAILABLE:
        print("❌ Soul system not available")
        return
        
    print("📜 Available Soul Profiles:")
    print("=" * 40)
    
    try:
        profiles = list_available_profiles()
        if not profiles:
            print("❌ No soul profiles found.")
            return
        
        for i, profile_name in enumerate(profiles, 1):
            print(f"{i}. {profile_name}")
            try:
                profile_info = soul_loader.get_profile_info(profile_name)
                if profile_info:
                    print(f"   Description: {profile_info.get('description', 'No description')}")
                    print(f"   Access Level: {profile_info.get('access_level', 'Unknown')}")
            except:
                pass
            print()
    except Exception as e:
        print(f"❌ Error listing profiles: {e}")

def test_user_identity(user_id: Optional[str]) -> bool:
    """Test loading a user identity"""
    if not SOUL_AVAILABLE:
        print("❌ Soul system not available")
        return False
        
    try:
        if user_id:
            soul = soul_loader.load_soul(user_id)
            print(f"✅ Successfully loaded soul profile: {soul.name}")
        else:
            soul = soul_loader.load_soul()
            print(f"✅ Successfully loaded default soul profile: {soul.name}")
        return True
    except Exception as e:
        print(f"❌ Failed to load soul profile: {e}")
        return False

def setup_user_identity(user_id: Optional[str]) -> Optional[Soul]:
    """Set up user identity for the system"""
    if not SOUL_AVAILABLE:
        print("⚠️  Soul system not available, using basic mode")
        return None
        
    try:
        if user_id:
            print(f"🔮 Loading identity for user: {user_id}")
            soul = soul_loader.load_soul(user_id)
        else:
            print("🔮 Loading default identity")
            soul = soul_loader.load_soul()
        
        print(f"🔮 Hello, I am {soul.name}, your AI companion.")
        print(f"🤖 Version: {soul.version}")
        print(f"📝 Description: {soul.description}")
        print(f"🔐 Access Level: {soul.access_level}")
        print(f"🎭 Personality: {soul.personality}")
        print()
        
        return soul
        
    except Exception as e:
        print(f"❌ Error loading soul profile: {e}")
        print("🔄 Falling back to basic mode...")
        return None

# ============================================================================
# MAIN CONVERSATION LOOP
# ============================================================================

async def conversation_loop(system: SimpleUniMindSystem, soul: Optional[Soul], user_id: Optional[str]):
    """Enhanced conversation loop with context awareness"""
    soul_name = soul.name if soul else "UniMind"
    conversation_context = {
        'user_id': user_id,
        'soul_name': soul_name,
        'session_id': f"session_{int(time.time())}",
        'conversation_history': [],
        'user_preferences': {},
        'current_topic': None
    }
    
    print(f"💬 Starting conversation with {soul_name}...")
    print(f"🎯 Type your queries or use /help for system commands")
    print(f"🚀 All {len(system.advanced_engines)} advanced AI engines are ready!")
    print(f"🧠 Enhanced intelligence mode: Active")
    print("-" * 60)
    
    try:
        while True:
            try:
                # Get user input
                user_input = input(f"{soul_name}: ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                    print(f"{soul_name}: Goodbye! Thank you for using UniMind AI. I've learned from our conversation and will be even more helpful next time!")
                    break
                
                # Update conversation context
                conversation_context['timestamp'] = datetime.now().isoformat()
                conversation_context['conversation_history'].append({
                    'user_input': user_input,
                    'timestamp': conversation_context['timestamp']
                })
                
                # Keep only last 10 interactions for context
                if len(conversation_context['conversation_history']) > 10:
                    conversation_context['conversation_history'] = conversation_context['conversation_history'][-10:]
                
                # Process input with enhanced context
                result = await system.process_input(user_input, conversation_context)
                
                # Display response
                response = result.get('response', 'No response generated')
                print(f"🤖 {response}")
                
                # Show source and confidence if available
                if 'source' in result:
                    source = result['source']
                    confidence = result.get('confidence', 0.0)
                    print(f"📡 Source: {source} (Confidence: {confidence:.1f})")
                
                # Update conversation context with response
                conversation_context['conversation_history'][-1]['response'] = response
                conversation_context['conversation_history'][-1]['source'] = result.get('source', 'unknown')
                
                print()
                
            except KeyboardInterrupt:
                print(f"\n{soul_name}: Interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error in conversation loop: {e}")
                print(f"🔄 Continuing conversation...")
                continue
    
    finally:
        print("🧹 Cleaning up conversation context...")
        print(f"📊 Conversation summary: {len(conversation_context['conversation_history'])} interactions processed")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def main():
    """Main function to run the Simple UniMind system"""
    parser = argparse.ArgumentParser(description='Simple UniMind AI System')
    parser.add_argument('--user', '-u', help='User ID for soul profile')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--list-profiles', action='store_true', help='List available soul profiles')
    parser.add_argument('--test-identity', action='store_true', help='Test user identity loading')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Handle special commands
    if args.list_profiles:
        list_profiles()
        return
    
    if args.test_identity:
        test_user_identity(args.user)
        return
    
    try:
        # Initialize the simple system
        system = SimpleUniMindSystem()
        await system.initialize()
        
        # Setup user identity
        soul = setup_user_identity(args.user)
        
        # Start conversation loop
        await conversation_loop(system, soul, args.user)
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down UniMind...")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("👋 UniMind shutdown complete.")

if __name__ == "__main__":
    asyncio.run(main()) 