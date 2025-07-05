#!/usr/bin/env python3
"""
main_integrated.py - Integrated main entry point for Unimind AI daemon system.
Integrates Phase 1 (Core Enhancement), Phase 2 (Intelligence Parity), and Phase 3 (Performance Optimization).
"""

import sys
import os
import argparse
import logging
import time
import asyncio
from typing import Optional, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unimind.soul.identity import Soul
from unimind.soul.soul_loader import soul_loader, list_available_profiles
from unimind.native_models.lam_engine import lam_engine
from unimind.memory.unified_memory import unified_memory, MemoryType
# Context-aware functionality is now integrated into LAM engine
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
    
    for i, profile in enumerate(profiles, 1):
        print(f"{i}. {profile['name']} ({profile['access_level']})")
        print(f"   Description: {profile['description']}")
        print(f"   File: {profile['file']}")
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
            soul = soul_loader.load_soul()  # This returns a Soul object
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
            soul = soul_loader.load_soul()  # This returns a Soul object
        
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
        from unimind.soul.identity import Soul
        return Soul(user_id=None)

def is_complex_query(user_input: str) -> bool:
    """Determine if a query is complex enough to use advanced phases."""
    complex_keywords = [
        "analyze", "evaluate", "explain", "compare", "contrast", "synthesize",
        "create", "design", "develop", "implement", "optimize", "debug",
        "understand", "comprehend", "interpret", "reason", "logic",
        "complex", "advanced", "sophisticated", "intelligent", "smart"
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
            logging.warning(f"Phase 3 cache check failed: {e}")
    
    # Phase 2: Intelligence Parity - Use advanced intelligence
    if PHASE2_AVAILABLE and is_complex_query(user_input):
        try:
            # Use Phase 2 components for complex queries
            from unimind.context.advanced_context_engine import process_input_context
            from unimind.language.enhanced_language_processor import understand_language
            from unimind.native_models.sophisticated_reasoning_engine import chain_of_thought_reasoning
            
            # Process with advanced context
            context_snapshot = process_input_context(user_input, context.get('user_id', 'default'))
            
            # Enhanced language understanding
            understanding = understand_language(user_input, context_snapshot)
            
            # Sophisticated reasoning
            reasoning_chain = chain_of_thought_reasoning(user_input, context_snapshot)
            
            response = f"🧠 **Advanced Analysis**: {reasoning_chain.final_answer}"
            
            return {
                'response': response,
                'source': 'phase2_intelligence',
                'phase_used': 'phase2',
                'response_time': time.time() - start_time,
                'optimization_applied': True,
                'confidence': understanding.confidence
            }
        except Exception as e:
            logging.warning(f"Phase 2 processing failed: {e}")
    
    # Phase 1: Core Enhancement - Use enhanced memory and reasoning
    if PHASE1_AVAILABLE:
        try:
            result = process_query(user_input, context.get('user_id', 'default'))
            if result.success:
                return {
                    'response': result.primary_result,
                    'source': 'phase1_enhancement',
                    'phase_used': 'phase1',
                    'response_time': time.time() - start_time,
                    'optimization_applied': True
                }
        except Exception as e:
            logging.warning(f"Phase 1 processing failed: {e}")
    
    # Fallback: Use context-aware LLM
    try:
        understanding_result = context_aware_llm.route_with_understanding(user_input, context.get('memory_context_id'))
        if understanding_result and "enhanced_prompt" in understanding_result:
            enhanced_prompt = understanding_result["enhanced_prompt"]
            return {
                'response': enhanced_prompt,
                'source': 'context_aware_llm',
                'phase_used': 'fallback',
                'response_time': time.time() - start_time,
                'optimization_applied': False
            }
    except Exception as e:
        logging.warning(f"Context-aware LLM failed: {e}")
    
    # Final fallback
    return {
        'response': "I'm not sure I understood that. Let me try to help you with general assistance.",
        'source': 'fallback',
        'phase_used': 'none',
        'response_time': time.time() - start_time,
        'optimization_applied': False
    }

async def conversation_loop(soul: Soul, user_id: Optional[str]):
    """Main conversation loop with all phases integrated."""
    
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
            "session_start": time.time()
        },
        tags=["session_start"]
    )
    
    print(f"💭 Conversation started. Type 'exit' or 'quit' to end.")
    print(f"💡 Type 'help' for available commands.")
    print(f"👤 Type 'whoami' to see current identity.")
    print(f"📜 Type 'describe_soul' to see detailed soul information.")
    print(f"⚡ Type 'performance' to see performance status.")
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
            if user_input.lower() == 'performance' and PHASE3_AVAILABLE:
                status = await performance_optimizer.get_optimization_status()
                print(f"⚡ Performance Status:")
                print(f"   • CPU Usage: {status['current_metrics']['cpu_usage']:.1f}%")
                print(f"   • Memory Usage: {status['current_metrics']['memory_usage']:.1f}%")
                print(f"   • Response Time: {status['current_metrics']['avg_response_time']:.3f}s")
                print(f"   • Cache Hit Rate: {status['cache_status']['hit_rate']:.1f}%")
                continue
            
            # Store user input in memory
            unified_memory.add_memory(
                memory_context_id,
                MemoryType.CONVERSATION,
                {"user_input": user_input, "timestamp": time.time()},
                tags=["user_input"]
            )
            
            # Process with all phases
            context = {
                "user_id": user_id or "default",
                "memory_context_id": memory_context_id,
                "session_id": session_id
            }
            
            result = await process_with_phases(user_input, context)
            
            # Display response with phase information
            if result['source'] == 'cache':
                print(f"⚡ {result['response']}")
            elif result['phase_used'] == 'phase2':
                print(f"🧠 {result['response']}")
            elif result['phase_used'] == 'phase1':
                print(f"💡 {result['response']}")
            else:
                print(f"🤖 {result['response']}")
            
            # Show performance info if optimization was applied
            if result['optimization_applied']:
                print(f"   ⚡ Response time: {result['response_time']:.3f}s")
            
            # Record performance metrics
            if PHASE3_AVAILABLE:
                await performance_monitor.record_response_time(result['response_time'])
            
            # Store response in memory
            unified_memory.add_memory(
                memory_context_id,
                MemoryType.CONVERSATION,
                {
                    "system_response": result['response'],
                    "phase_used": result['phase_used'],
                    "response_time": result['response_time'],
                    "timestamp": time.time()
                },
                tags=["system_response"]
            )
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            if PHASE3_AVAILABLE:
                await performance_monitor.record_error(str(e))
            continue

async def main():
    """Main entry point for the integrated Unimind daemon."""
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Unimind AI Daemon - Integrated Phases 1, 2, and 3")
    parser.add_argument("--user", "-u", type=str, help="User ID to load specific daemon identity")
    parser.add_argument("--list-profiles", "-l", action="store_true", help="List all available soul profiles")
    parser.add_argument("--test", "-t", action="store_true", help="Test user identity loading")
    parser.add_argument("--dev-mode", "-d", action="store_true", help="Run in development mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    print("🔮 Unimind AI Daemon - Integrated Phases 1, 2, and 3")
    print("🧠 Core Enhancement | 🧠 Intelligence Parity | ⚡ Performance Optimization")
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
    
    # Initialize all phases
    await initialize_all_phases()
    
    # Set up user identity
    soul = setup_user_identity(args.user)
    
    # Start conversation loop
    await conversation_loop(soul, args.user)

if __name__ == "__main__":
    asyncio.run(main()) 