#!/usr/bin/env python3
"""
main.py - Unified main entry point for Unimind AI daemon system.
Integrates all phases: Core Enhancement, Intelligence Parity, Performance Optimization, and Advanced Capabilities.
Handles conversation loop, scroll execution, system coordination, and advanced features.
"""

import sys
import os
import argparse
import logging
import time
import asyncio
from typing import Optional, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unimind.soul.identity import Soul
from unimind.soul.soul_loader import soul_loader, list_available_profiles
from unimind.native_models.lam_engine import lam_engine
from unimind.memory.unified_memory import unified_memory, MemoryType
from unimind.native_models.lam_engine import context_aware_llm

# Phase 1: Core Enhancement Components
try:
    from unimind.core.memory_reasoning_integration import process_query
    from unimind.memory.hierarchical_memory import hierarchical_memory
    from unimind.native_models.sophisticated_reasoning import sophisticated_reasoning_engine as enhanced_reasoning_engine
    PHASE1_AVAILABLE = True
except ImportError:
    PHASE1_AVAILABLE = False

# Phase 2: Intelligence Parity Components
try:
    from unimind.context.advanced_context_engine import AdvancedContextEngine
    from unimind.language.enhanced_language_processor import EnhancedLanguageProcessor
    from unimind.native_models.sophisticated_reasoning_engine import SophisticatedReasoningEngine
    PHASE2_AVAILABLE = True
except ImportError:
    PHASE2_AVAILABLE = False

# Phase 3: Performance Optimization Components
try:
    from unimind.performance.performance_optimizer import performance_optimizer, OptimizationLevel
    from unimind.performance.cache_manager import cache_manager
    from unimind.performance.memory_optimizer import memory_optimizer
    from unimind.performance.response_optimizer import response_optimizer
    from unimind.performance.monitoring import performance_monitor
    PHASE3_AVAILABLE = True
except ImportError:
    PHASE3_AVAILABLE = False

# Phase 4: Advanced Capabilities Components
try:
    from unimind.advanced.creative_engine import creative_engine
    from unimind.advanced.knowledge_synthesis import knowledge_synthesis_engine
    from unimind.advanced.adaptive_learning import adaptive_learning_engine
    from unimind.advanced.meta_cognition import meta_cognition_engine
    from unimind.advanced.rag_system import rag_system
    from unimind.emotion.emotional_intelligence import emotional_intelligence_engine
    PHASE4_AVAILABLE = True
except ImportError:
    PHASE4_AVAILABLE = False

def setup_logging(verbose: bool = False):
    """Set up basic logging."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('unimind.log')
        ]
    )

async def initialize_all_phases():
    """Initialize all available phases."""
    print("🚀 Initializing Unimind Phases...")
    print("="*60)
    
    # Phase 1: Core Enhancement
    if PHASE1_AVAILABLE:
        print("✅ Phase 1: Core Enhancement - Available")
        print("   • Hierarchical Memory System")
        print("   • Enhanced Reasoning Engine")
        print("   • Memory-Reasoning Integration")
    else:
        print("⚠️  Phase 1: Core Enhancement - Not Available")
    
    # Phase 2: Intelligence Parity
    if PHASE2_AVAILABLE:
        print("✅ Phase 2: Intelligence Parity - Available")
        print("   • Advanced Context Engine")
        print("   • Enhanced Language Processor")
        print("   • Sophisticated Reasoning Engine")
    else:
        print("⚠️  Phase 2: Intelligence Parity - Not Available")
    
    # Phase 3: Performance Optimization
    if PHASE3_AVAILABLE:
        print("✅ Phase 3: Performance Optimization - Available")
        print("   • Performance Optimizer")
        print("   • Cache Manager")
        print("   • Memory Optimizer")
        print("   • Response Optimizer")
        print("   • Performance Monitor")
        
        # Start performance optimization
        try:
            await performance_optimizer.start()
            print("   • Performance optimization started")
        except Exception as e:
            print(f"   • Performance optimization failed: {e}")
    else:
        print("⚠️  Phase 3: Performance Optimization - Not Available")
    
    # Phase 4: Advanced Capabilities
    if PHASE4_AVAILABLE:
        print("✅ Phase 4: Advanced Capabilities - Available")
        print("   • Creative Engine")
        print("   • Knowledge Synthesis Engine")
        print("   • Adaptive Learning Engine")
        print("   • Meta-Cognition Engine")
        print("   • RAG System")
        print("   • Emotional Intelligence Engine")
        
        # Initialize advanced capabilities
        try:
            # Initialize creative engine
            creative_insights = creative_engine.get_creative_insights()
            print(f"   • Creative engine: {creative_insights.get('total_ideas', 0)} ideas generated")
            
            # Initialize knowledge synthesis
            synthesis_insights = knowledge_synthesis_engine.get_synthesis_insights()
            print(f"   • Knowledge synthesis: {synthesis_insights.get('total_syntheses', 0)} syntheses created")
            
            # Initialize adaptive learning
            learning_insights = adaptive_learning_engine.get_learning_insights()
            print(f"   • Adaptive learning: {learning_insights.get('total_learning_events', 0)} events processed")
            
            # Initialize meta-cognition
            meta_insights = meta_cognition_engine.get_meta_cognition_insights()
            print(f"   • Meta-cognition: {meta_insights.get('total_events', 0)} events monitored")
            
            # Initialize RAG system
            rag_status = await rag_system.get_system_status()
            print(f"   • RAG system: {rag_status.get('total_chunks', 0)} knowledge chunks available")
            
            # Initialize emotional intelligence
            emotional_insights = emotional_intelligence_engine.get_emotional_insights()
            print(f"   • Emotional intelligence: {emotional_insights.get('total_events', 0)} emotions analyzed")
            
            print("   • Advanced capabilities initialized")
        except Exception as e:
            print(f"   • Advanced capabilities initialization failed: {e}")
    else:
        print("⚠️  Phase 4: Advanced Capabilities - Not Available")
    
    print("="*60)
    print()

def list_profiles():
    """List all available soul profiles."""
    print("📜 Available Soul Profiles:")
    print("=" * 40)
    
    profiles = list_available_profiles()
    if not profiles:
        print("❌ No soul profiles found.")
        return
    
    for i, profile_name in enumerate(profiles, 1):
        print(f"{i}. {profile_name}")
        # Try to get more info about the profile
        try:
            profile_info = soul_loader.get_profile_info(profile_name)
            if profile_info:
                print(f"   Description: {profile_info.get('description', 'No description')}")
                print(f"   Access Level: {profile_info.get('access_level', 'Unknown')}")
        except:
            pass
        print()

def test_user_identity(user_id: Optional[str]) -> bool:
    """Test loading a user identity."""
    try:
        if user_id:
            soul = soul_loader.load_soul(user_id)
            print(f"✅ Successfully loaded soul profile: {soul.name}")
            print(f"   Access Level: {soul.access_level}")
            print(f"   Description: {soul.description}")
        else:
            soul = soul_loader.load_soul()
            print(f"✅ Successfully loaded default soul profile: {soul.name}")
            print(f"   Access Level: {soul.access_level}")
            print(f"   Description: {soul.description}")
        return True
    except Exception as e:
        print(f"❌ Failed to load soul profile: {e}")
        return False

def setup_user_identity(user_id: Optional[str]) -> Soul:
    """Set up user identity for the daemon."""
    try:
        if user_id:
            print(f"🔮 Loading daemon identity for user: {user_id}")
            soul = soul_loader.load_soul(user_id)
        else:
            print("🔮 Loading daemon identity for user: default")
            soul = soul_loader.load_soul()
        
        # Display soul information
        print(f"🔮 Hello, I am {soul.name}, your companion daemon. {soul.personality}")
        print()
        print(f"🤖 Daemon: {soul.name} v{soul.version}")
        print(f"📝 Description: {soul.description}")
        print(f"🔐 Access Level: {soul.access_level}")
        print(f"🎭 Personality: {soul.personality}")
        print(f"💬 Greeting: {soul.greeting}")
        print()
        
        return soul
        
    except Exception as e:
        print(f"❌ Error loading soul profile: {e}")
        print("🔄 Falling back to default soul profile...")
        return soul_loader.load_soul()

def is_complex_query(user_input: str) -> bool:
    """Determine if a query is complex enough to use advanced phases."""
    complex_keywords = [
        "analyze", "evaluate", "explain", "compare", "contrast", "synthesize",
        "create", "design", "develop", "implement", "optimize", "debug",
        "understand", "comprehend", "interpret", "reason", "logic",
        "complex", "advanced", "sophisticated", "intelligent", "smart",
        "creative", "innovative", "learn", "adapt", "emotion", "feel"
    ]
    
    user_input_lower = user_input.lower()
    return any(keyword in user_input_lower for keyword in complex_keywords)

async def process_with_phases(user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Process user input using all available phases."""
    start_time = time.time()
    
    # Phase 3: Performance Optimization - Check cache first
    if PHASE3_AVAILABLE:
        try:
            cached_result = await performance_optimizer.optimize_query_response(user_input, context)
            if cached_result['source'] == 'cache':
                return {
                    'response': cached_result['response'],
                    'source': 'cache',
                    'phase_used': 'phase3',
                    'response_time': cached_result['response_time'],
                    'optimization_applied': True
                }
        except Exception as e:
            logging.warning(f"Cache optimization failed: {e}")
    
    # Phase 4: Advanced Capabilities - For complex queries
    if PHASE4_AVAILABLE and is_complex_query(user_input):
        try:
            # Emotional intelligence analysis
            emotional_state = emotional_intelligence_engine.analyze_emotional_state(user_input, context)
            
            # Adaptive learning from interaction
            learning_result = adaptive_learning_engine.learn_from_interaction(
                user_input, "Processing complex query", True, 0.8, context
            )
            
            # Creative problem solving if needed
            if any(word in user_input.lower() for word in ["create", "design", "innovate", "creative"]):
                creative_ideas = creative_engine.generate_creative_ideas(
                    user_input, creative_engine.CreativeDomain.TECHNICAL, 
                    creative_engine.CreativeMode.DIVERGENT, 3
                )
                context['creative_ideas'] = [idea.title for idea in creative_ideas]
            
            # Knowledge synthesis if needed
            if any(word in user_input.lower() for word in ["synthesize", "combine", "integrate", "knowledge"]):
                synthesis = knowledge_synthesis_engine.synthesize_knowledge(user_input)
                context['synthesized_knowledge'] = synthesis.content
            
            # Meta-cognition monitoring
            meta_cognition_engine.monitor_cognitive_process(
                meta_cognition_engine.CognitiveProcess.PROBLEM_SOLVING,
                0.8, 0.9, 0.7, 0.85, {"cpu": 0.4, "memory": 0.3}
            )
            
        except Exception as e:
            logging.warning(f"Advanced capabilities processing failed: {e}")
    
    # Phase 1: Core Enhancement - Memory and reasoning
    if PHASE1_AVAILABLE:
        try:
            memory_result = process_query(user_input, context.get('user_id', 'default'))
            context['memory_context'] = memory_result
        except Exception as e:
            logging.warning(f"Memory reasoning failed: {e}")
    
    # Phase 2: Intelligence Parity - Advanced context and language
    if PHASE2_AVAILABLE:
        try:
            # Advanced context processing
            context_engine = AdvancedContextEngine()
            enhanced_context = context_engine.process_context(user_input, context)
            context.update(enhanced_context)
        except Exception as e:
            logging.warning(f"Advanced context processing failed: {e}")
    
    # LAM Engine - Core processing
    try:
        lam_result = lam_engine.route_with_understanding(user_input, context.get('memory_context_id'))
        response = lam_result.get('enhanced_prompt', lam_result.get('response', 'I understand your request.'))
    except Exception as e:
        logging.warning(f"LAM engine processing failed: {e}")
        response = "I'm processing your request. Please give me a moment."
    
    # RAG System - Enhance response with verified information
    if PHASE4_AVAILABLE:
        try:
            # Check if this is an information-seeking query
            info_keywords = ["what", "how", "why", "when", "where", "who", "which", "explain", "tell me", "find", "search", "verify", "check"]
            is_info_query = any(keyword in user_input.lower() for keyword in info_keywords)
            
            if is_info_query:
                # Generate RAG-enhanced response
                rag_response = await rag_system.generate_rag_response(
                    query=user_input,
                    base_response=response,
                    retrieval_method=rag_system.RetrievalMethod.HYBRID,
                    enhance_response=True
                )
                
                # Use enhanced response if confidence is good
                if rag_response.confidence > 0.5:
                    response = rag_response.enhanced_response
                    context['rag_verification'] = rag_response.verification_summary
                    context['rag_sources'] = rag_response.sources_used
                    context['rag_confidence'] = rag_response.confidence
                    
                    # Add retrieved knowledge to memory for future use
                    for chunk in rag_response.retrieved_context[:3]:  # Top 3 chunks
                        await rag_system.add_knowledge(
                            content=chunk.content,
                            source_type=chunk.source_type,
                            source_id=chunk.source_id,
                            confidence=chunk.confidence,
                            tags=chunk.tags
                        )
                
                logging.info(f"RAG enhancement applied with confidence {rag_response.confidence:.2f}")
            
        except Exception as e:
            logging.warning(f"RAG processing failed: {e}")
    
    # Phase 3: Performance Optimization - Optimize response
    if PHASE3_AVAILABLE:
        try:
            optimized_response = await response_optimizer.optimize_response(response, context)
            response = optimized_response['optimized_response']
        except Exception as e:
            logging.warning(f"Response optimization failed: {e}")
    
    response_time = time.time() - start_time
    
    return {
        'response': response,
        'source': 'lam_engine',
        'phase_used': 'all_phases',
        'response_time': response_time,
        'context': context,
        'optimization_applied': PHASE3_AVAILABLE
    }

async def conversation_loop(soul: Soul, user_id: Optional[str]):
    """Main conversation loop with context-aware understanding and advanced features."""
    
    # Create memory context for this session
    session_id = f"session_{int(time.time())}"
    memory_context_id = unified_memory.create_context(session_id, "main_conversation")
    
    # Store initial context
    unified_memory.add_memory(
        memory_context_id,
        MemoryType.CONTEXT,
        {
            "user_identity": user_id or "default",
            "daemon_identity": soul.name,
            "session_start": time.time(),
            "phases_available": {
                "phase1": PHASE1_AVAILABLE,
                "phase2": PHASE2_AVAILABLE,
                "phase3": PHASE3_AVAILABLE,
                "phase4": PHASE4_AVAILABLE
            }
        },
        tags=["session_start"]
    )
    
    print(f"💭 Conversation started. Type 'exit' or 'quit' to end.")
    print(f"💡 Type 'help' for available commands.")
    print(f"👤 Type 'whoami' to see current identity.")
    print(f"📜 Type 'describe_soul' to see detailed soul information.")
    print(f"🧠 Type 'phases' to see available capabilities.")
    print(f"🎭 Type 'emotion' to test emotional intelligence.")
    print(f"🎨 Type 'creative' to test creative engine.")
    print(f"📚 Type 'synthesize' to test knowledge synthesis.")
    print(f"🤔 Type 'meta' to test meta-cognition.")
    print()
    
    while True:
        try:
            # Get user input
            user_input = input("> ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() == 'help':
                print("\n📖 Available Commands:")
                print("  help          - Show this help")
                print("  whoami        - Show current identity")
                print("  describe_soul - Show detailed soul information")
                print("  phases        - Show available capabilities")
                print("  emotion       - Test emotional intelligence")
                print("  creative      - Test creative engine")
                print("  synthesize    - Test knowledge synthesis")
                print("  rag           - Test RAG system")
                print("  verify        - Test information verification")
                print("  meta          - Test meta-cognition")
                print("  exit/quit     - End conversation")
                print()
                continue
            
            if user_input.lower() == 'whoami':
                print(f"\n👤 Current Identity: {soul.name}")
                print(f"   Access Level: {soul.access_level}")
                print(f"   Description: {soul.description}")
                print()
                continue
            
            if user_input.lower() == 'describe_soul':
                print(f"\n📜 Soul Profile: {soul.name}")
                print(f"   Version: {soul.version}")
                print(f"   Access Level: {soul.access_level}")
                print(f"   Description: {soul.description}")
                print(f"   Personality: {soul.personality}")
                print(f"   Greeting: {soul.greeting}")
                print()
                continue
            
            if user_input.lower() == 'phases':
                print("\n🧠 Available Capabilities:")
                print(f"   Phase 1 (Core Enhancement): {'✅' if PHASE1_AVAILABLE else '❌'}")
                print(f"   Phase 2 (Intelligence Parity): {'✅' if PHASE2_AVAILABLE else '❌'}")
                print(f"   Phase 3 (Performance Optimization): {'✅' if PHASE3_AVAILABLE else '❌'}")
                print(f"   Phase 4 (Advanced Capabilities): {'✅' if PHASE4_AVAILABLE else '❌'}")
                print()
                continue
            
            # Test advanced capabilities
            if user_input.lower() == 'emotion' and PHASE4_AVAILABLE:
                print("\n🎭 Testing Emotional Intelligence...")
                emotional_state = emotional_intelligence_engine.analyze_emotional_state(
                    "I'm feeling excited about this new project!", {}
                )
                response = emotional_intelligence_engine.generate_emotional_response(
                    emotional_state, "I'm feeling excited about this new project!", {}
                )
                print(f"   Detected Emotion: {emotional_state.primary_emotion.value}")
                print(f"   Intensity: {emotional_state.intensity:.2f}")
                print(f"   Response: {response.response_text}")
                print()
                continue
            
            if user_input.lower() == 'creative' and PHASE4_AVAILABLE:
                print("\n🎨 Testing Creative Engine...")
                ideas = creative_engine.generate_creative_ideas(
                    "Design a new user interface", creative_engine.CreativeDomain.TECHNICAL, 
                    creative_engine.CreativeMode.DIVERGENT, 3
                )
                print("   Generated Ideas:")
                for i, idea in enumerate(ideas, 1):
                    print(f"   {i}. {idea.title}")
                print()
                continue
            
            if user_input.lower() == 'synthesize' and PHASE4_AVAILABLE:
                print("\n📚 Testing Knowledge Synthesis...")
                # Add some test knowledge
                chunk1 = knowledge_synthesis_engine.add_knowledge_chunk(
                    "User interfaces should be intuitive and easy to use.",
                    knowledge_synthesis_engine.KnowledgeType.CONCEPTUAL,
                    "design_principles", 0.9
                )
                chunk2 = knowledge_synthesis_engine.add_knowledge_chunk(
                    "Modern UI design emphasizes minimalism and clarity.",
                    knowledge_synthesis_engine.KnowledgeType.FACTUAL,
                    "design_trends", 0.8
                )
                synthesis = knowledge_synthesis_engine.synthesize_knowledge(
                    "UI design principles", knowledge_synthesis_engine.SynthesisMethod.INTEGRATION
                )
                print(f"   Synthesized: {synthesis.content[:100]}...")
                print()
                continue
            
            if user_input.lower() == 'meta' and PHASE4_AVAILABLE:
                print("\n🤔 Testing Meta-Cognition...")
                meta_cognition_engine.monitor_cognitive_process(
                    meta_cognition_engine.CognitiveProcess.PROBLEM_SOLVING,
                    0.8, 0.9, 0.7, 0.85, {"cpu": 0.4, "memory": 0.3}
                )
                insights = meta_cognition_engine.get_meta_cognition_insights()
                print(f"   Total Events: {insights['total_events']}")
                print(f"   Cognitive States: {len(insights['cognitive_states'])}")
                print()
                continue
            
            if user_input.lower() == 'rag' and PHASE4_AVAILABLE:
                print("\n🔍 Testing RAG System...")
                # Add some test knowledge
                await rag_system.add_knowledge(
                    content="Python is a high-level programming language known for its simplicity and readability.",
                    source_type=rag_system.KnowledgeSourceType.VERIFIED_FACT,
                    source_id="python_docs",
                    confidence=0.95,
                    tags=["programming", "python", "language"]
                )
                await rag_system.add_knowledge(
                    content="Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
                    source_type=rag_system.KnowledgeSourceType.VERIFIED_FACT,
                    source_id="ai_textbook",
                    confidence=0.92,
                    tags=["ai", "machine_learning", "technology"]
                )
                
                # Test retrieval
                retrieval_result = await rag_system.retrieve_knowledge("What is Python?")
                print(f"   Retrieved {len(retrieval_result.retrieved_chunks)} chunks")
                print(f"   Confidence: {retrieval_result.confidence:.2f}")
                
                # Test RAG response generation
                rag_response = await rag_system.generate_rag_response(
                    query="What is Python?",
                    base_response="Python is a programming language.",
                    enhance_response=True
                )
                print(f"   Enhanced Response: {rag_response.enhanced_response[:100]}...")
                print(f"   RAG Confidence: {rag_response.confidence:.2f}")
                print()
                continue
            
            if user_input.lower() == 'verify' and PHASE4_AVAILABLE:
                print("\n✅ Testing Information Verification...")
                # Test verification
                verification_result = await rag_system.verify_information(
                    "Python is a programming language"
                )
                print(f"   Claim: {verification_result['claim']}")
                print(f"   Verified: {verification_result['verified']}")
                print(f"   Confidence: {verification_result['confidence']:.2f}")
                print(f"   Level: {verification_result['verification_level']}")
                print(f"   Analysis: {verification_result['analysis']}")
                print()
                continue
            
            # Process regular user input
            print(f"\n🤖 {soul.name}: ", end="", flush=True)
            
            # Create context for processing
            context = {
                'user_id': user_id or 'default',
                'session_id': session_id,
                'memory_context_id': memory_context_id,
                'daemon_identity': soul.name,
                'timestamp': time.time()
            }
            
            # Process with all phases
            result = await process_with_phases(user_input, context)
            
            # Print response
            print(result['response'])
            
            # Store interaction in memory
            unified_memory.add_memory(
                memory_context_id,
                MemoryType.INTERACTION,
                {
                    "user_input": user_input,
                    "daemon_response": result['response'],
                    "response_time": result['response_time'],
                    "phase_used": result['phase_used'],
                    "optimization_applied": result.get('optimization_applied', False)
                },
                tags=["conversation", "interaction"]
            )
            
            # Phase 4: Adaptive learning from this interaction
            if PHASE4_AVAILABLE:
                try:
                    adaptive_learning_engine.learn_from_interaction(
                        user_input, result['response'], True, 0.8, context
                    )
                except Exception as e:
                    logging.debug(f"Adaptive learning failed: {e}")
            
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logging.error(f"Conversation loop error: {e}")

async def main():
    """Main entry point for the Unimind AI daemon."""
    parser = argparse.ArgumentParser(description="Unimind AI Daemon")
    parser.add_argument("--user", "-u", help="User ID for soul profile")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--list-profiles", "-l", action="store_true", help="List available soul profiles")
    parser.add_argument("--test-identity", "-t", help="Test loading a specific soul profile")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # List profiles if requested
    if args.list_profiles:
        list_profiles()
        return
    
    # Test identity if requested
    if args.test_identity:
        success = test_user_identity(args.test_identity)
        sys.exit(0 if success else 1)
    
    try:
        # Initialize all phases
        await initialize_all_phases()
        
        # Setup user identity
        soul = setup_user_identity(args.user)
        
        # Start conversation loop
        await conversation_loop(soul, args.user)
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logging.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 