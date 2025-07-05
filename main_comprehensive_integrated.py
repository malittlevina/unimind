#!/usr/bin/env python3
"""
main_comprehensive_integrated.py - Comprehensive integrated main entry point for Unimind AI system.
Integrates all phases including advanced features from phases 8-20.
"""

import sys
import os
import argparse
import logging
import time
import asyncio
import threading
from typing import Optional, Dict, Any, List
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Core UniMind imports
from unimind.soul.identity import Soul
from unimind.soul.soul_loader import soul_loader, list_available_profiles
from unimind.native_models.lam_engine import lam_engine
from unimind.memory.unified_memory import unified_memory, MemoryType
from unimind.native_models.lam_engine import context_aware_llm

# RAG System imports
try:
    from advanced_features.rag_system import RAGSystem
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# ThothOS Integration imports
try:
    from advanced_features.thothos_integration import ThothOSKernelBridge, ThothOSSystemIntegration
    THOTHOS_AVAILABLE = True
except ImportError:
    THOTHOS_AVAILABLE = False

# Advanced Features imports (Phases 8-20)
try:
    # Phase 8: Healthcare AI
    from advanced_features.phase_8_healthcare_ai import HealthcareAIEngine
    PHASE8_AVAILABLE = True
except ImportError:
    PHASE8_AVAILABLE = False

try:
    # Phase 9: Financial AI
    from advanced_features.phase_9_financial_ai import FinancialAIEngine
    PHASE9_AVAILABLE = True
except ImportError:
    PHASE9_AVAILABLE = False

try:
    # Phase 10: Educational AI
    from advanced_features.phase_10_educational_ai import EducationalAIEngine
    PHASE10_AVAILABLE = True
except ImportError:
    PHASE10_AVAILABLE = False

try:
    # Phase 11: Security AI
    from advanced_features.phase_11_security_ai import SecurityAIEngine
    PHASE11_AVAILABLE = True
except ImportError:
    PHASE11_AVAILABLE = False

try:
    # Phase 12: Quantum Computing
    from advanced_features.phase_12_quantum_computing import QuantumComputingEngine
    PHASE12_AVAILABLE = True
except ImportError:
    PHASE12_AVAILABLE = False

try:
    # Phase 13: Edge Computing
    from advanced_features.phase_13_edge_computing import EdgeComputingEngine
    PHASE13_AVAILABLE = True
except ImportError:
    PHASE13_AVAILABLE = False

try:
    # Phase 14: Autonomous Systems
    from advanced_features.phase_14_autonomous_systems import AutonomousSystemsEngine
    PHASE14_AVAILABLE = True
except ImportError:
    PHASE14_AVAILABLE = False

try:
    # Phase 15: Advanced NLP
    from advanced_features.phase_15_advanced_nlp import AdvancedNLPEngine
    PHASE15_AVAILABLE = True
except ImportError:
    PHASE15_AVAILABLE = False

try:
    # Phase 16: Computer Vision
    from advanced_features.phase_16_computer_vision import ComputerVisionEngine
    PHASE16_AVAILABLE = True
except ImportError:
    PHASE16_AVAILABLE = False

try:
    # Phase 17: Blockchain AI
    from advanced_features.phase_17_blockchain_ai import BlockchainEngine, FederatedLearningEngine
    PHASE17_AVAILABLE = True
except ImportError:
    PHASE17_AVAILABLE = False

try:
    # Phase 18: Robotics
    from advanced_features.phase_18_robotics import RobotController
    PHASE18_AVAILABLE = True
except ImportError:
    PHASE18_AVAILABLE = False

try:
    # Phase 19: Quantum ML
    from advanced_features.phase_19_quantum_ml import QuantumSimulator
    PHASE19_AVAILABLE = True
except ImportError:
    PHASE19_AVAILABLE = False

try:
    # Phase 20: Advanced NLP V2
    from advanced_features.phase_20_advanced_nlp import AdvancedNLPEngine as AdvancedNLPEngineV2
    PHASE20_AVAILABLE = True
except ImportError:
    PHASE20_AVAILABLE = False

# Web Interface imports
try:
    from unimind.daemon_web.web_interface import WebInterface
    WEB_INTERFACE_AVAILABLE = True
except ImportError:
    WEB_INTERFACE_AVAILABLE = False

class ComprehensiveUniMindSystem:
    """Comprehensive UniMind system integrating all advanced features"""
    
    def __init__(self):
        self.initialized = False
        self.advanced_engines = {}
        self.rag_system = None
        self.thothos_bridge = None
        self.web_interface = None
        self.system_status = {}
        
    async def initialize(self):
        """Initialize the comprehensive UniMind system"""
        print("🚀 Initializing Comprehensive UniMind System...")
        print("=" * 80)
        
        # Initialize RAG System
        if RAG_AVAILABLE:
            try:
                self.rag_system = RAGSystem()
                await self.rag_system.initialize()
                self.advanced_engines['rag'] = self.rag_system
                print("✅ RAG System - Initialized")
            except Exception as e:
                print(f"❌ RAG System - Failed: {e}")
        
        # Initialize ThothOS Integration
        if THOTHOS_AVAILABLE:
            try:
                self.thothos_bridge = ThothOSKernelBridge()
                self.thothos_bridge.initialize()
                self.advanced_engines['thothos'] = self.thothos_bridge
                print("✅ ThothOS Integration - Initialized")
            except Exception as e:
                print(f"❌ ThothOS Integration - Failed: {e}")
        
        # Initialize Advanced Features (Phases 8-20)
        await self._initialize_advanced_features()
        
        # Initialize Web Interface
        if WEB_INTERFACE_AVAILABLE:
            try:
                self.web_interface = WebInterface()
                await self.web_interface.initialize()
                print("✅ Web Interface - Initialized")
            except Exception as e:
                print(f"❌ Web Interface - Failed: {e}")
        
        self.initialized = True
        print("=" * 80)
        print("🎉 Comprehensive UniMind System Initialized Successfully!")
        print()
    
    async def _initialize_advanced_features(self):
        """Initialize all advanced features from phases 8-20"""
        print("🔧 Initializing Advanced Features (Phases 8-20)...")
        
        # Phase 8: Healthcare AI
        if PHASE8_AVAILABLE:
            try:
                healthcare_engine = HealthcareAIEngine()
                healthcare_engine.initialize()
                self.advanced_engines['healthcare'] = healthcare_engine
                print("✅ Phase 8: Healthcare AI - Initialized")
            except Exception as e:
                print(f"❌ Phase 8: Healthcare AI - Failed: {e}")
        
        # Phase 9: Financial AI
        if PHASE9_AVAILABLE:
            try:
                financial_engine = FinancialAIEngine()
                financial_engine.initialize()
                self.advanced_engines['financial'] = financial_engine
                print("✅ Phase 9: Financial AI - Initialized")
            except Exception as e:
                print(f"❌ Phase 9: Financial AI - Failed: {e}")
        
        # Phase 10: Educational AI
        if PHASE10_AVAILABLE:
            try:
                educational_engine = EducationalAIEngine()
                educational_engine.initialize()
                self.advanced_engines['educational'] = educational_engine
                print("✅ Phase 10: Educational AI - Initialized")
            except Exception as e:
                print(f"❌ Phase 10: Educational AI - Failed: {e}")
        
        # Phase 11: Security AI
        if PHASE11_AVAILABLE:
            try:
                security_engine = SecurityAIEngine()
                security_engine.initialize()
                self.advanced_engines['security'] = security_engine
                print("✅ Phase 11: Security AI - Initialized")
            except Exception as e:
                print(f"❌ Phase 11: Security AI - Failed: {e}")
        
        # Phase 12: Quantum Computing
        if PHASE12_AVAILABLE:
            try:
                quantum_engine = QuantumComputingEngine()
                quantum_engine.initialize()
                self.advanced_engines['quantum'] = quantum_engine
                print("✅ Phase 12: Quantum Computing - Initialized")
            except Exception as e:
                print(f"❌ Phase 12: Quantum Computing - Failed: {e}")
        
        # Phase 13: Edge Computing
        if PHASE13_AVAILABLE:
            try:
                edge_engine = EdgeComputingEngine()
                edge_engine.initialize()
                self.advanced_engines['edge'] = edge_engine
                print("✅ Phase 13: Edge Computing - Initialized")
            except Exception as e:
                print(f"❌ Phase 13: Edge Computing - Failed: {e}")
        
        # Phase 14: Autonomous Systems
        if PHASE14_AVAILABLE:
            try:
                autonomous_engine = AutonomousSystemsEngine()
                autonomous_engine.initialize()
                self.advanced_engines['autonomous'] = autonomous_engine
                print("✅ Phase 14: Autonomous Systems - Initialized")
            except Exception as e:
                print(f"❌ Phase 14: Autonomous Systems - Failed: {e}")
        
        # Phase 15: Advanced NLP
        if PHASE15_AVAILABLE:
            try:
                nlp_engine = AdvancedNLPEngine()
                nlp_engine.initialize()
                self.advanced_engines['nlp'] = nlp_engine
                print("✅ Phase 15: Advanced NLP - Initialized")
            except Exception as e:
                print(f"❌ Phase 15: Advanced NLP - Failed: {e}")
        
        # Phase 16: Computer Vision
        if PHASE16_AVAILABLE:
            try:
                vision_engine = ComputerVisionEngine()
                vision_engine.initialize()
                self.advanced_engines['vision'] = vision_engine
                print("✅ Phase 16: Computer Vision - Initialized")
            except Exception as e:
                print(f"❌ Phase 16: Computer Vision - Failed: {e}")
        
        # Phase 17: Blockchain AI
        if PHASE17_AVAILABLE:
            try:
                blockchain_engine = BlockchainEngine()
                blockchain_engine.initialize()
                federated_engine = FederatedLearningEngine()
                federated_engine.initialize()
                self.advanced_engines['blockchain'] = blockchain_engine
                self.advanced_engines['federated'] = federated_engine
                print("✅ Phase 17: Blockchain AI - Initialized")
            except Exception as e:
                print(f"❌ Phase 17: Blockchain AI - Failed: {e}")
        
        # Phase 18: Robotics
        if PHASE18_AVAILABLE:
            try:
                robot_controller = RobotController("mobile_robot", "robot_001")
                robot_controller.initialize()
                self.advanced_engines['robotics'] = robot_controller
                print("✅ Phase 18: Robotics - Initialized")
            except Exception as e:
                print(f"❌ Phase 18: Robotics - Failed: {e}")
        
        # Phase 19: Quantum ML
        if PHASE19_AVAILABLE:
            try:
                quantum_sim = QuantumSimulator(num_qubits=8)
                quantum_sim.initialize()
                self.advanced_engines['quantum_ml'] = quantum_sim
                print("✅ Phase 19: Quantum ML - Initialized")
            except Exception as e:
                print(f"❌ Phase 19: Quantum ML - Failed: {e}")
        
        # Phase 20: Advanced NLP V2
        if PHASE20_AVAILABLE:
            try:
                nlp_v2_engine = AdvancedNLPEngineV2()
                nlp_v2_engine.initialize()
                self.advanced_engines['nlp_v2'] = nlp_v2_engine
                print("✅ Phase 20: Advanced NLP V2 - Initialized")
            except Exception as e:
                print(f"❌ Phase 20: Advanced NLP V2 - Failed: {e}")
    
    async def process_input(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process user input using all available systems"""
        if not self.initialized:
            raise RuntimeError("Comprehensive UniMind System not initialized")
        
        start_time = time.time()
        
        # Check for specific commands
        if user_input.lower().startswith('/'):
            return await self._handle_system_commands(user_input, context)
        
        # Try RAG system first for information retrieval
        if self.rag_system:
            try:
                rag_result = await self.rag_system.retrieve_and_generate(user_input)
                if rag_result and rag_result.get('confidence', 0) > 0.7:
                    return {
                        'response': rag_result['response'],
                        'source': 'rag_system',
                        'confidence': rag_result['confidence'],
                        'response_time': time.time() - start_time
                    }
            except Exception as e:
                logging.warning(f"RAG processing failed: {e}")
        
        # Try advanced NLP for language understanding
        if 'nlp_v2' in self.advanced_engines:
            try:
                nlp_engine = self.advanced_engines['nlp_v2']
                document = nlp_engine.analyze_document(user_input)
                if document.sentences:
                    sentiment = document.sentences[0].sentiment.value
                    return {
                        'response': f"I understand your message. Sentiment: {sentiment}. How can I help you further?",
                        'source': 'advanced_nlp',
                        'confidence': 0.8,
                        'response_time': time.time() - start_time
                    }
            except Exception as e:
                logging.warning(f"Advanced NLP processing failed: {e}")
        
        # Fallback to core UniMind processing
        try:
            understanding_result = context_aware_llm.route_with_understanding(
                user_input, context.get('memory_context_id')
            )
            if understanding_result and "enhanced_prompt" in understanding_result:
                return {
                    'response': understanding_result["enhanced_prompt"],
                    'source': 'core_unimind',
                    'confidence': 0.6,
                    'response_time': time.time() - start_time
                }
        except Exception as e:
            logging.warning(f"Core UniMind processing failed: {e}")
        
        # Final fallback
        return {
            'response': "I'm here to help! What would you like to know or do?",
            'source': 'fallback',
            'confidence': 0.5,
            'response_time': time.time() - start_time
        }
    
    async def _handle_system_commands(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle system-specific commands"""
        command = user_input.lower().strip()
        
        if command == '/status':
            return await self._get_system_status()
        elif command == '/healthcare':
            return await self._handle_healthcare_query(user_input, context)
        elif command == '/financial':
            return await self._handle_financial_query(user_input, context)
        elif command == '/educational':
            return await self._handle_educational_query(user_input, context)
        elif command == '/security':
            return await self._handle_security_query(user_input, context)
        elif command == '/quantum':
            return await self._handle_quantum_query(user_input, context)
        elif command == '/vision':
            return await self._handle_vision_query(user_input, context)
        elif command == '/blockchain':
            return await self._handle_blockchain_query(user_input, context)
        elif command == '/robotics':
            return await self._handle_robotics_query(user_input, context)
        elif command == '/help':
            return self._get_help_info()
        else:
            return {
                'response': f"Unknown command: {command}. Type /help for available commands.",
                'source': 'system',
                'confidence': 1.0,
                'response_time': 0.0
            }
    
    async def _get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'system': 'Comprehensive UniMind AI System',
            'initialized': self.initialized,
            'available_engines': list(self.advanced_engines.keys()),
            'engine_count': len(self.advanced_engines),
            'rag_available': RAG_AVAILABLE,
            'thothos_available': THOTHOS_AVAILABLE,
            'web_interface_available': WEB_INTERFACE_AVAILABLE,
            'phases_available': {
                'phase8_healthcare': PHASE8_AVAILABLE,
                'phase9_financial': PHASE9_AVAILABLE,
                'phase10_educational': PHASE10_AVAILABLE,
                'phase11_security': PHASE11_AVAILABLE,
                'phase12_quantum': PHASE12_AVAILABLE,
                'phase13_edge': PHASE13_AVAILABLE,
                'phase14_autonomous': PHASE14_AVAILABLE,
                'phase15_nlp': PHASE15_AVAILABLE,
                'phase16_vision': PHASE16_AVAILABLE,
                'phase17_blockchain': PHASE17_AVAILABLE,
                'phase18_robotics': PHASE18_AVAILABLE,
                'phase19_quantum_ml': PHASE19_AVAILABLE,
                'phase20_nlp_v2': PHASE20_AVAILABLE
            }
        }
        
        return {
            'response': f"System Status:\n{json.dumps(status, indent=2)}",
            'source': 'system',
            'confidence': 1.0,
            'response_time': 0.0,
            'data': status
        }
    
    def _get_help_info(self) -> Dict[str, Any]:
        """Get help information"""
        help_text = """
Available Commands:
/status - Show system status
/healthcare - Healthcare AI features
/financial - Financial AI features
/educational - Educational AI features
/security - Security AI features
/quantum - Quantum computing features
/vision - Computer vision features
/blockchain - Blockchain AI features
/robotics - Robotics features
/help - Show this help

Advanced Features Available:
- RAG System for information retrieval
- ThothOS Integration for system-level operations
- Healthcare AI for medical analysis
- Financial AI for trading and analysis
- Educational AI for learning systems
- Security AI for threat detection
- Quantum Computing for advanced algorithms
- Edge Computing for distributed processing
- Autonomous Systems for robotics
- Advanced NLP for language understanding
- Computer Vision for image analysis
- Blockchain AI for decentralized systems
- Quantum ML for quantum machine learning
        """
        
        return {
            'response': help_text,
            'source': 'system',
            'confidence': 1.0,
            'response_time': 0.0
        }
    
    async def _handle_healthcare_query(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle healthcare-related queries"""
        if 'healthcare' not in self.advanced_engines:
            return {
                'response': "Healthcare AI engine not available.",
                'source': 'system',
                'confidence': 1.0,
                'response_time': 0.0
            }
        
        engine = self.advanced_engines['healthcare']
        status = engine.get_system_status()
        
        return {
            'response': f"Healthcare AI Status:\n{json.dumps(status, indent=2)}",
            'source': 'healthcare_ai',
            'confidence': 0.9,
            'response_time': 0.0
        }
    
    async def _handle_financial_query(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle financial-related queries"""
        if 'financial' not in self.advanced_engines:
            return {
                'response': "Financial AI engine not available.",
                'source': 'system',
                'confidence': 1.0,
                'response_time': 0.0
            }
        
        engine = self.advanced_engines['financial']
        status = engine.get_system_status()
        
        return {
            'response': f"Financial AI Status:\n{json.dumps(status, indent=2)}",
            'source': 'financial_ai',
            'confidence': 0.9,
            'response_time': 0.0
        }
    
    async def _handle_educational_query(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle educational-related queries"""
        if 'educational' not in self.advanced_engines:
            return {
                'response': "Educational AI engine not available.",
                'source': 'system',
                'confidence': 1.0,
                'response_time': 0.0
            }
        
        engine = self.advanced_engines['educational']
        status = engine.get_system_status()
        
        return {
            'response': f"Educational AI Status:\n{json.dumps(status, indent=2)}",
            'source': 'educational_ai',
            'confidence': 0.9,
            'response_time': 0.0
        }
    
    async def _handle_security_query(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle security-related queries"""
        if 'security' not in self.advanced_engines:
            return {
                'response': "Security AI engine not available.",
                'source': 'system',
                'confidence': 1.0,
                'response_time': 0.0
            }
        
        engine = self.advanced_engines['security']
        status = engine.get_system_status()
        
        return {
            'response': f"Security AI Status:\n{json.dumps(status, indent=2)}",
            'source': 'security_ai',
            'confidence': 0.9,
            'response_time': 0.0
        }
    
    async def _handle_quantum_query(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle quantum computing queries"""
        if 'quantum' not in self.advanced_engines:
            return {
                'response': "Quantum computing engine not available.",
                'source': 'system',
                'confidence': 1.0,
                'response_time': 0.0
            }
        
        engine = self.advanced_engines['quantum']
        status = engine.get_system_status()
        
        return {
            'response': f"Quantum Computing Status:\n{json.dumps(status, indent=2)}",
            'source': 'quantum_computing',
            'confidence': 0.9,
            'response_time': 0.0
        }
    
    async def _handle_vision_query(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle computer vision queries"""
        if 'vision' not in self.advanced_engines:
            return {
                'response': "Computer vision engine not available.",
                'source': 'system',
                'confidence': 1.0,
                'response_time': 0.0
            }
        
        engine = self.advanced_engines['vision']
        status = engine.get_system_status()
        
        return {
            'response': f"Computer Vision Status:\n{json.dumps(status, indent=2)}",
            'source': 'computer_vision',
            'confidence': 0.9,
            'response_time': 0.0
        }
    
    async def _handle_blockchain_query(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle blockchain queries"""
        if 'blockchain' not in self.advanced_engines:
            return {
                'response': "Blockchain AI engine not available.",
                'source': 'system',
                'confidence': 1.0,
                'response_time': 0.0
            }
        
        engine = self.advanced_engines['blockchain']
        status = engine.get_blockchain_info()
        
        return {
            'response': f"Blockchain AI Status:\n{json.dumps(status, indent=2)}",
            'source': 'blockchain_ai',
            'confidence': 0.9,
            'response_time': 0.0
        }
    
    async def _handle_robotics_query(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle robotics queries"""
        if 'robotics' not in self.advanced_engines:
            return {
                'response': "Robotics engine not available.",
                'source': 'system',
                'confidence': 1.0,
                'response_time': 0.0
            }
        
        engine = self.advanced_engines['robotics']
        status = engine.get_system_status()
        
        return {
            'response': f"Robotics Status:\n{json.dumps(status, indent=2)}",
            'source': 'robotics',
            'confidence': 0.9,
            'response_time': 0.0
        }
    
    async def start_web_interface(self):
        """Start the web interface"""
        if self.web_interface:
            try:
                await self.web_interface.start()
                print("🌐 Web interface started")
            except Exception as e:
                print(f"❌ Failed to start web interface: {e}")
    
    async def cleanup(self):
        """Cleanup all systems"""
        print("🧹 Cleaning up Comprehensive UniMind System...")
        
        # Cleanup advanced engines
        for name, engine in self.advanced_engines.items():
            try:
                if hasattr(engine, 'cleanup'):
                    engine.cleanup()
                print(f"✅ Cleaned up {name}")
            except Exception as e:
                print(f"❌ Failed to cleanup {name}: {e}")
        
        # Cleanup web interface
        if self.web_interface:
            try:
                await self.web_interface.cleanup()
                print("✅ Cleaned up web interface")
            except Exception as e:
                print(f"❌ Failed to cleanup web interface: {e}")
        
        print("🎉 Cleanup completed!")

def setup_logging(verbose: bool = False):
    """Set up logging configuration"""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('unimind_comprehensive.log')
        ]
    )

async def conversation_loop(system: ComprehensiveUniMindSystem, soul: Soul, user_id: Optional[str]):
    """Main conversation loop"""
    session_id = f"session_{int(time.time())}"
    memory_context_id = unified_memory.create_context(session_id, "comprehensive_conversation")
    
    # Store initial context
    unified_memory.add_memory(
        memory_context_id,
        MemoryType.CONTEXT,
        {
            "user_identity": user_id or "default",
            "daemon_identity": soul.name,
            "session_start": time.time(),
            "system_type": "comprehensive"
        },
        tags=["session_start", "comprehensive"]
    )
    
    print(f"💭 Comprehensive UniMind Conversation started.")
    print(f"💡 Type /help for available commands.")
    print(f"👤 Type 'whoami' to see current identity.")
    print(f"📊 Type /status to see system status.")
    print()
    
    while True:
        try:
            user_input = input("> ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() == 'whoami':
                print(f"🔮 You are interacting with {soul.name}")
                print(f"📝 {soul.description}")
                continue
            
            # Process input
            context = {
                "user_id": user_id or "default",
                "memory_context_id": memory_context_id,
                "session_id": session_id
            }
            
            result = await system.process_input(user_input, context)
            
            # Display response
            print(f"[{result['source'].upper()}] {result['response']}")
            if result['response_time'] > 0:
                print(f"   ⚡ Response time: {result['response_time']:.3f}s")
            
            # Store in memory
            unified_memory.add_memory(
                memory_context_id,
                MemoryType.CONVERSATION,
                {
                    "user_input": user_input,
                    "system_response": result['response'],
                    "source": result['source'],
                    "confidence": result['confidence'],
                    "response_time": result['response_time'],
                    "timestamp": time.time()
                },
                tags=["conversation", result['source']]
            )
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Comprehensive UniMind AI System")
    parser.add_argument("--user", "-u", type=str, help="User ID to load specific daemon identity")
    parser.add_argument("--list-profiles", "-l", action="store_true", help="List all available soul profiles")
    parser.add_argument("--test", "-t", action="store_true", help="Test user identity loading")
    parser.add_argument("--web", "-w", action="store_true", help="Start web interface")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    print("🔮 Comprehensive UniMind AI System")
    print("🧠 All Advanced Features Integrated (Phases 8-20)")
    print("=" * 80)
    
    # Handle list profiles command
    if args.list_profiles:
        list_profiles()
        return
    
    # Handle test command
    if args.test:
        success = test_user_identity(args.user)
        if success:
            print("\n✅ User identity test completed successfully!")
        else:
            print("\n❌ User identity test failed!")
        return
    
    # Initialize comprehensive system
    system = ComprehensiveUniMindSystem()
    await system.initialize()
    
    # Set up user identity
    soul = setup_user_identity(args.user)
    
    # Start web interface if requested
    if args.web:
        await system.start_web_interface()
    
    # Start conversation loop
    await conversation_loop(system, soul, args.user)
    
    # Cleanup
    await system.cleanup()

def test_user_identity(user_id: Optional[str]) -> bool:
    """Test loading a user identity"""
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

def setup_user_identity(user_id: Optional[str]) -> Soul:
    """Set up user identity"""
    try:
        if user_id:
            print(f"🔮 Loading daemon identity for user: {user_id}")
            soul = soul_loader.load_soul(user_id)
        else:
            print("🔮 Loading daemon identity for user: default")
            soul = soul_loader.load_soul()
        
        print(f"🔮 Hello, I am {soul.name}, your comprehensive AI companion.")
        print(f"🤖 Daemon: {soul.name} v{soul.version}")
        print(f"📝 Description: {soul.description}")
        print()
        
        return soul
        
    except Exception as e:
        print(f"❌ Error loading soul profile: {e}")
        print("🔄 Falling back to default soul profile...")
        from unimind.soul.identity import Soul
        return Soul(user_id=None)

if __name__ == "__main__":
    asyncio.run(main()) 