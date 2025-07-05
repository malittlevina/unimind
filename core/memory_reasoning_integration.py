"""
memory_reasoning_integration.py - Memory and reasoning integration for Phase 1.
Integrates hierarchical memory with sophisticated reasoning capabilities.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import threading

from ..memory.hierarchical_memory import (
    HierarchicalMemory, MemoryItem, MemoryQuery, MemoryType, MemoryLevel,
    store_memory, retrieve_memories, consolidate_memories, get_memory_stats
)
from ..native_models.sophisticated_reasoning import (
    EnhancedReasoningEngine, ReasoningMode, ReasoningChain,
    reason as reason_query, get_reasoning_stats
)

class IntegrationMode(Enum):
    """Integration modes for memory and reasoning."""
    MEMORY_ONLY = "memory_only"           # Use only memory system
    REASONING_ONLY = "reasoning_only"     # Use only reasoning engine
    INTEGRATED = "integrated"             # Use both systems together
    ADAPTIVE = "adaptive"                 # Automatically choose best approach

@dataclass
class IntegrationContext:
    """Context for memory-reasoning integration."""
    session_id: str
    user_id: str
    query_type: str
    complexity_estimate: float
    memory_requirements: List[str]
    reasoning_requirements: List[str]
    integration_mode: IntegrationMode
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IntegrationResult:
    """Result of memory-reasoning integration."""
    success: bool
    primary_result: Any
    memory_contributions: List[Dict[str, Any]]
    reasoning_contributions: List[Dict[str, Any]]
    integration_mode_used: IntegrationMode
    execution_time: float
    confidence: float
    insights: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class MemoryReasoningIntegration:
    """
    Integration layer for hierarchical memory and enhanced reasoning engine.
    Provides unified interface and intelligent coordination between systems.
    """
    
    def __init__(self):
        """Initialize the memory-reasoning integration layer."""
        self.logger = logging.getLogger('MemoryReasoningIntegration')
        
        # Core systems
        self.hierarchical_memory = HierarchicalMemory()
        self.enhanced_reasoning = EnhancedReasoningEngine()
        
        # Integration configuration
        self.default_integration_mode = IntegrationMode.ADAPTIVE
        self.memory_threshold = 0.6  # Confidence threshold for memory-based responses
        self.reasoning_threshold = 0.7  # Confidence threshold for reasoning-based responses
        self.complexity_threshold = 0.5  # Complexity threshold for reasoning
        
        # Session management
        self.active_sessions: Dict[str, IntegrationContext] = {}
        
        # Performance tracking
        self.performance_stats = {
            "memory_queries": 0,
            "reasoning_queries": 0,
            "integrated_queries": 0,
            "average_execution_time": 0.0,
            "success_rate": 0.0
        }
        
        # Threading
        self.lock = threading.RLock()
        
        self.logger.info("Memory-reasoning integration layer initialized")
    
    def process_query(self, query: str, user_id: str = "default",
                     integration_mode: Optional[IntegrationMode] = None,
                     context: Dict[str, Any] = None) -> IntegrationResult:
        """
        Process a query using integrated memory and reasoning systems.
        
        Args:
            query: The query to process
            user_id: User identifier
            integration_mode: Specific integration mode to use
            context: Additional context information
            
        Returns:
            Integration result with combined memory and reasoning contributions
        """
        start_time = time.time()
        
        # Create integration context
        session_id = self._generate_session_id(query, user_id)
        integration_context = self._create_integration_context(
            session_id, user_id, query, integration_mode, context
        )
        
        # Store context
        self.active_sessions[session_id] = integration_context
        
        try:
            # Determine integration mode
            mode = integration_mode or self._determine_integration_mode(query, context)
            integration_context.integration_mode = mode
            
            # Process based on mode
            if mode == IntegrationMode.MEMORY_ONLY:
                result = self._process_memory_only(query, integration_context)
            elif mode == IntegrationMode.REASONING_ONLY:
                result = self._process_reasoning_only(query, integration_context)
            elif mode == IntegrationMode.INTEGRATED:
                result = self._process_integrated(query, integration_context)
            else:  # ADAPTIVE
                result = self._process_adaptive(query, integration_context)
            
            # Update performance stats
            self._update_performance_stats(result, time.time() - start_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            return self._create_error_result(query, str(e), time.time() - start_time)
        
        finally:
            # Clean up session
            self.active_sessions.pop(session_id, None)
    
    def _create_integration_context(self, session_id: str, user_id: str, query: str,
                                  integration_mode: Optional[IntegrationMode],
                                  context: Dict[str, Any] = None) -> IntegrationContext:
        """Create integration context for the query."""
        # Analyze query characteristics
        query_type = self._classify_query_type(query)
        complexity = self._estimate_complexity(query)
        memory_reqs = self._identify_memory_requirements(query)
        reasoning_reqs = self._identify_reasoning_requirements(query)
        
        return IntegrationContext(
            session_id=session_id,
            user_id=user_id,
            query_type=query_type,
            complexity_estimate=complexity,
            memory_requirements=memory_reqs,
            reasoning_requirements=reasoning_reqs,
            integration_mode=integration_mode or self.default_integration_mode,
            metadata=context or {}
        )
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["remember", "recall", "what", "when", "where", "who"]):
            return "information_retrieval"
        elif any(word in query_lower for word in ["explain", "why", "how", "analyze", "evaluate"]):
            return "analysis"
        elif any(word in query_lower for word in ["create", "generate", "design", "invent", "solve"]):
            return "creation"
        elif any(word in query_lower for word in ["compare", "contrast", "similar", "different"]):
            return "comparison"
        elif any(word in query_lower for word in ["predict", "forecast", "future", "will"]):
            return "prediction"
        else:
            return "general"
    
    def _estimate_complexity(self, query: str) -> float:
        """Estimate the complexity of a query."""
        # Simple complexity estimation based on query characteristics
        complexity = 0.0
        
        # Length factor
        complexity += min(0.3, len(query.split()) / 50)
        
        # Keyword complexity
        complex_keywords = ["analyze", "evaluate", "synthesize", "create", "design", "solve", "explain"]
        complexity += sum(0.1 for keyword in complex_keywords if keyword in query.lower())
        
        # Question complexity
        if "?" in query:
            complexity += 0.1
        
        # Multi-part queries
        if any(connector in query.lower() for connector in ["and", "or", "but", "however", "although"]):
            complexity += 0.2
        
        return min(1.0, complexity)
    
    def _identify_memory_requirements(self, query: str) -> List[str]:
        """Identify memory requirements for the query."""
        requirements = []
        
        # Check for factual information needs
        if any(word in query.lower() for word in ["fact", "information", "data", "knowledge"]):
            requirements.append("factual_knowledge")
        
        # Check for procedural knowledge needs
        if any(word in query.lower() for word in ["how", "procedure", "process", "method"]):
            requirements.append("procedural_knowledge")
        
        # Check for conceptual knowledge needs
        if any(word in query.lower() for word in ["concept", "theory", "principle", "understanding"]):
            requirements.append("conceptual_knowledge")
        
        # Check for experiential knowledge needs
        if any(word in query.lower() for word in ["experience", "example", "case", "instance"]):
            requirements.append("experiential_knowledge")
        
        return requirements
    
    def _identify_reasoning_requirements(self, query: str) -> List[str]:
        """Identify reasoning requirements for the query."""
        requirements = []
        
        # Check for deductive reasoning needs
        if any(word in query.lower() for word in ["therefore", "thus", "consequently", "logically"]):
            requirements.append("deductive_reasoning")
        
        # Check for inductive reasoning needs
        if any(word in query.lower() for word in ["pattern", "trend", "generalize", "usually"]):
            requirements.append("inductive_reasoning")
        
        # Check for abductive reasoning needs
        if any(word in query.lower() for word in ["explain", "why", "cause", "reason"]):
            requirements.append("abductive_reasoning")
        
        # Check for creative reasoning needs
        if any(word in query.lower() for word in ["create", "invent", "design", "novel"]):
            requirements.append("creative_reasoning")
        
        # Check for critical reasoning needs
        if any(word in query.lower() for word in ["evaluate", "assess", "criticize", "analyze"]):
            requirements.append("critical_reasoning")
        
        return requirements
    
    def _determine_integration_mode(self, query: str, context: Dict[str, Any] = None) -> IntegrationMode:
        """Determine the best integration mode for the query."""
        # Check if context specifies a mode
        if context and "integration_mode" in context:
            try:
                return IntegrationMode(context["integration_mode"])
            except ValueError:
                pass
        
        # Analyze query characteristics
        complexity = self._estimate_complexity(query)
        memory_reqs = self._identify_memory_requirements(query)
        reasoning_reqs = self._identify_reasoning_requirements(query)
        
        # Simple decision logic
        if complexity < 0.3 and len(memory_reqs) > 0 and len(reasoning_reqs) == 0:
            return IntegrationMode.MEMORY_ONLY
        elif complexity > 0.7 and len(reasoning_reqs) > 0:
            return IntegrationMode.REASONING_ONLY
        elif len(memory_reqs) > 0 and len(reasoning_reqs) > 0:
            return IntegrationMode.INTEGRATED
        else:
            return IntegrationMode.ADAPTIVE
    
    def _process_memory_only(self, query: str, context: IntegrationContext) -> IntegrationResult:
        """Process query using only memory system."""
        self.logger.debug(f"Processing query with memory-only mode: {query}")
        
        # Create memory query
        memory_query = MemoryQuery(
            query=query,
            memory_types=[MemoryType.FACT, MemoryType.CONCEPT, MemoryType.PROCEDURE],
            memory_levels=[MemoryLevel.WORKING, MemoryLevel.LONG_TERM],
            limit=10
        )
        
        # Retrieve memories
        memories = self.hierarchical_memory.retrieve(memory_query)
        
        # Process results
        if memories:
            primary_result = self._synthesize_memory_results(memories, query)
            confidence = self._calculate_memory_confidence(memories)
        else:
            primary_result = {"message": "No relevant memories found", "status": "no_data"}
            confidence = 0.0
        
        return IntegrationResult(
            success=confidence > 0.3,
            primary_result=primary_result,
            memory_contributions=[{"type": "retrieval", "memories": len(memories), "confidence": confidence}],
            reasoning_contributions=[],
            integration_mode_used=IntegrationMode.MEMORY_ONLY,
            execution_time=0.0,  # Will be set by caller
            confidence=confidence,
            insights=["Used memory-only processing", f"Retrieved {len(memories)} memories"],
            metadata={"memory_count": len(memories)}
        )
    
    def _process_reasoning_only(self, query: str, context: IntegrationContext) -> IntegrationResult:
        """Process query using only reasoning engine."""
        self.logger.debug(f"Processing query with reasoning-only mode: {query}")
        
        # Determine reasoning mode
        reasoning_mode = self._map_requirements_to_reasoning_mode(context.reasoning_requirements)
        
        # Perform reasoning
        reasoning_chain = self.enhanced_reasoning.reason(query, reasoning_mode, context.metadata)
        
        # Process results
        if reasoning_chain.final_result:
            primary_result = reasoning_chain.final_result
            confidence = reasoning_chain.overall_confidence
        else:
            primary_result = {"message": "Reasoning failed", "status": "failed"}
            confidence = 0.0
        
        return IntegrationResult(
            success=confidence > 0.3,
            primary_result=primary_result,
            memory_contributions=[],
            reasoning_contributions=[{
                "type": "reasoning_chain",
                "mode": reasoning_chain.reasoning_mode.value,
                "steps": len(reasoning_chain.steps),
                "confidence": confidence
            }],
            integration_mode_used=IntegrationMode.REASONING_ONLY,
            execution_time=0.0,  # Will be set by caller
            confidence=confidence,
            insights=[
                f"Used {reasoning_chain.reasoning_mode.value} reasoning",
                f"Completed {len(reasoning_chain.steps)} reasoning steps"
            ],
            metadata={
                "reasoning_mode": reasoning_chain.reasoning_mode.value,
                "steps_count": len(reasoning_chain.steps)
            }
        )
    
    def _process_integrated(self, query: str, context: IntegrationContext) -> IntegrationResult:
        """Process query using both memory and reasoning systems."""
        self.logger.debug(f"Processing query with integrated mode: {query}")
        
        # Step 1: Retrieve relevant memories
        memory_query = MemoryQuery(
            query=query,
            memory_types=[MemoryType.FACT, MemoryType.CONCEPT, MemoryType.PROCEDURE, MemoryType.EXPERIENCE],
            memory_levels=[MemoryLevel.WORKING, MemoryLevel.LONG_TERM],
            limit=15
        )
        
        memories = self.hierarchical_memory.retrieve(memory_query)
        memory_confidence = self._calculate_memory_confidence(memories)
        
        # Step 2: Perform reasoning with memory context
        reasoning_mode = self._map_requirements_to_reasoning_mode(context.reasoning_requirements)
        
        # Enhance context with memory information
        enhanced_context = context.metadata.copy()
        enhanced_context["relevant_memories"] = [m.content for m in memories]
        enhanced_context["memory_confidence"] = memory_confidence
        
        reasoning_chain = self.enhanced_reasoning.reason(query, reasoning_mode, enhanced_context)
        reasoning_confidence = reasoning_chain.overall_confidence
        
        # Step 3: Synthesize results
        primary_result = self._synthesize_integrated_results(
            memories, reasoning_chain, query
        )
        
        # Calculate overall confidence
        overall_confidence = (memory_confidence + reasoning_confidence) / 2
        
        return IntegrationResult(
            success=overall_confidence > 0.4,
            primary_result=primary_result,
            memory_contributions=[{
                "type": "retrieval",
                "memories": len(memories),
                "confidence": memory_confidence
            }],
            reasoning_contributions=[{
                "type": "reasoning_chain",
                "mode": reasoning_chain.reasoning_mode.value,
                "steps": len(reasoning_chain.steps),
                "confidence": reasoning_confidence
            }],
            integration_mode_used=IntegrationMode.INTEGRATED,
            execution_time=0.0,  # Will be set by caller
            confidence=overall_confidence,
            insights=[
                f"Integrated {len(memories)} memories with {reasoning_chain.reasoning_mode.value} reasoning",
                f"Memory confidence: {memory_confidence:.2f}, Reasoning confidence: {reasoning_confidence:.2f}"
            ],
            metadata={
                "memory_count": len(memories),
                "reasoning_mode": reasoning_chain.reasoning_mode.value,
                "steps_count": len(reasoning_chain.steps)
            }
        )
    
    def _process_adaptive(self, query: str, context: IntegrationContext) -> IntegrationResult:
        """Process query using adaptive mode selection."""
        self.logger.debug(f"Processing query with adaptive mode: {query}")
        
        # Try memory first
        memory_result = self._process_memory_only(query, context)
        
        # If memory is sufficient, use it
        if memory_result.confidence > self.memory_threshold:
            return memory_result
        
        # Try reasoning
        reasoning_result = self._process_reasoning_only(query, context)
        
        # If reasoning is sufficient, use it
        if reasoning_result.confidence > self.reasoning_threshold:
            return reasoning_result
        
        # If neither is sufficient alone, try integrated
        if memory_result.confidence > 0.3 and reasoning_result.confidence > 0.3:
            return self._process_integrated(query, context)
        
        # Fallback to the better of the two
        if memory_result.confidence > reasoning_result.confidence:
            return memory_result
        else:
            return reasoning_result
    
    def _map_requirements_to_reasoning_mode(self, requirements: List[str]) -> ReasoningMode:
        """Map reasoning requirements to reasoning mode."""
        if "deductive_reasoning" in requirements:
            return ReasoningMode.DEDUCTIVE
        elif "inductive_reasoning" in requirements:
            return ReasoningMode.INDUCTIVE
        elif "abductive_reasoning" in requirements:
            return ReasoningMode.ABDUCTIVE
        elif "creative_reasoning" in requirements:
            return ReasoningMode.CREATIVE
        elif "critical_reasoning" in requirements:
            return ReasoningMode.CRITICAL
        else:
            return ReasoningMode.ABDUCTIVE  # Default
    
    def _synthesize_memory_results(self, memories: List[MemoryItem], query: str) -> Dict[str, Any]:
        """Synthesize memory results into a coherent response."""
        if not memories:
            return {"message": "No relevant memories found", "status": "no_data"}
        
        # Group memories by type
        facts = [m for m in memories if m.memory_type == MemoryType.FACT]
        concepts = [m for m in memories if m.memory_type == MemoryType.CONCEPT]
        procedures = [m for m in memories if m.memory_type == MemoryType.PROCEDURE]
        experiences = [m for m in memories if m.memory_type == MemoryType.EXPERIENCE]
        
        # Create synthesis
        synthesis = {
            "query": query,
            "status": "success",
            "memory_types": {
                "facts": len(facts),
                "concepts": len(concepts),
                "procedures": len(procedures),
                "experiences": len(experiences)
            },
            "key_information": [],
            "insights": []
        }
        
        # Extract key information from memories
        for memory in sorted(memories, key=lambda m: m.importance, reverse=True)[:5]:
            synthesis["key_information"].append({
                "content": memory.content,
                "type": memory.memory_type.value,
                "importance": memory.importance,
                "confidence": memory.confidence
            })
        
        # Generate insights
        if facts:
            synthesis["insights"].append(f"Found {len(facts)} relevant facts")
        if concepts:
            synthesis["insights"].append(f"Retrieved {len(concepts)} conceptual memories")
        if procedures:
            synthesis["insights"].append(f"Identified {len(procedures)} procedural memories")
        
        return synthesis
    
    def _synthesize_integrated_results(self, memories: List[MemoryItem],
                                     reasoning_chain: ReasoningChain,
                                     query: str) -> Dict[str, Any]:
        """Synthesize integrated memory and reasoning results."""
        synthesis = {
            "query": query,
            "status": "success",
            "approach": "integrated",
            "memory_contribution": {
                "memories_retrieved": len(memories),
                "memory_types": {
                    "facts": len([m for m in memories if m.memory_type == MemoryType.FACT]),
                    "concepts": len([m for m in memories if m.memory_type == MemoryType.CONCEPT]),
                    "procedures": len([m for m in memories if m.memory_type == MemoryType.PROCEDURE]),
                    "experiences": len([m for m in memories if m.memory_type == MemoryType.EXPERIENCE])
                }
            },
            "reasoning_contribution": {
                "mode": reasoning_chain.reasoning_mode.value,
                "steps_completed": len(reasoning_chain.steps),
                "main_conclusion": reasoning_chain.final_result.get("main_conclusion", "No conclusion"),
                "key_insights": reasoning_chain.final_result.get("key_insights", [])
            },
            "integrated_response": self._create_integrated_response(memories, reasoning_chain)
        }
        
        return synthesis
    
    def _create_integrated_response(self, memories: List[MemoryItem],
                                  reasoning_chain: ReasoningChain) -> str:
        """Create an integrated response combining memory and reasoning."""
        # Extract key memory information
        memory_summary = []
        for memory in sorted(memories, key=lambda m: m.importance, reverse=True)[:3]:
            memory_summary.append(f"- {memory.content}")
        
        # Extract reasoning conclusion
        reasoning_conclusion = reasoning_chain.final_result.get("main_conclusion", "")
        
        # Combine into integrated response
        response_parts = []
        
        if memory_summary:
            response_parts.append("Based on relevant memories:")
            response_parts.extend(memory_summary)
        
        if reasoning_conclusion:
            response_parts.append(f"\nThrough {reasoning_chain.reasoning_mode.value} reasoning:")
            response_parts.append(reasoning_conclusion)
        
        return "\n".join(response_parts) if response_parts else "No integrated response available"
    
    def _calculate_memory_confidence(self, memories: List[MemoryItem]) -> float:
        """Calculate confidence based on memory retrieval results."""
        if not memories:
            return 0.0
        
        # Calculate average importance and confidence
        avg_importance = sum(m.importance for m in memories) / len(memories)
        avg_confidence = sum(m.confidence for m in memories) / len(memories)
        
        # Factor in recency and relevance
        current_time = time.time()
        recency_factor = sum(1.0 / (current_time - m.last_accessed + 1) for m in memories) / len(memories)
        
        # Combine factors
        confidence = (avg_importance * 0.4 + avg_confidence * 0.4 + recency_factor * 0.2)
        
        return min(1.0, confidence)
    
    def _create_error_result(self, query: str, error: str, execution_time: float) -> IntegrationResult:
        """Create an error result."""
        return IntegrationResult(
            success=False,
            primary_result={"error": error, "status": "failed"},
            memory_contributions=[],
            reasoning_contributions=[],
            integration_mode_used=IntegrationMode.MEMORY_ONLY,
            execution_time=execution_time,
            confidence=0.0,
            insights=[f"Error occurred: {error}"],
            metadata={"error": error}
        )
    
    def _update_performance_stats(self, result: IntegrationResult, execution_time: float):
        """Update performance statistics."""
        with self.lock:
            if result.integration_mode_used == IntegrationMode.MEMORY_ONLY:
                self.performance_stats["memory_queries"] += 1
            elif result.integration_mode_used == IntegrationMode.REASONING_ONLY:
                self.performance_stats["reasoning_queries"] += 1
            else:
                self.performance_stats["integrated_queries"] += 1
            
            # Update average execution time
            total_queries = (self.performance_stats["memory_queries"] + 
                           self.performance_stats["reasoning_queries"] + 
                           self.performance_stats["integrated_queries"])
            
            if total_queries > 0:
                current_avg = self.performance_stats["average_execution_time"]
                self.performance_stats["average_execution_time"] = (
                    (current_avg * (total_queries - 1) + execution_time) / total_queries
                )
            
            # Update success rate
            if result.success:
                self.performance_stats["success_rate"] = (
                    (self.performance_stats["success_rate"] * (total_queries - 1) + 1) / total_queries
                )
            else:
                self.performance_stats["success_rate"] = (
                    self.performance_stats["success_rate"] * (total_queries - 1) / total_queries
                )
    
    def _generate_session_id(self, query: str, user_id: str) -> str:
        """Generate unique session ID."""
        import hashlib
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        user_hash = hashlib.md5(user_id.encode()).hexdigest()[:4]
        timestamp = int(time.time() * 1000) % 1000000
        return f"integration_{user_hash}_{query_hash}_{timestamp}"
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration layer statistics."""
        with self.lock:
            stats = {
                "performance": self.performance_stats.copy(),
                "active_sessions": len(self.active_sessions),
                "memory_stats": get_memory_stats(),
                "reasoning_stats": get_reasoning_stats()
            }
            
            return stats

# Global integration instance
memory_reasoning_integration = MemoryReasoningIntegration()

def process_query(query: str, user_id: str = "default",
                 integration_mode: Optional[IntegrationMode] = None,
                 context: Dict[str, Any] = None) -> IntegrationResult:
    """Process a query using the global memory-reasoning integration instance."""
    return memory_reasoning_integration.process_query(query, user_id, integration_mode, context)

def get_integration_stats() -> Dict[str, Any]:
    """Get integration statistics using the global instance."""
    return memory_reasoning_integration.get_integration_stats()

@dataclass
class QueryResult:
    """Result of a memory reasoning query."""
    success: bool
    primary_result: str
    secondary_results: list = None
    confidence: float = 0.0
    reasoning_chain: list = None
    memory_context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.secondary_results is None:
            self.secondary_results = []
        if self.reasoning_chain is None:
            self.reasoning_chain = []
        if self.memory_context is None:
            self.memory_context = {}

def integrate_memory_with_reasoning(memory_data: Dict[str, Any], reasoning_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Integrate memory data with reasoning context.
    
    Args:
        memory_data: Memory data to integrate
        reasoning_context: Reasoning context
        
    Returns:
        Integrated result
    """
    logger.info("Integrating memory with reasoning (stub)")
    
    # TODO: Implement actual integration logic
    return {
        "integrated_result": "stub_integration",
        "memory_used": memory_data,
        "reasoning_applied": reasoning_context
    }

def enhance_query_with_memory(user_input: str, memory_context: Dict[str, Any]) -> str:
    """
    Enhance a query using memory context.
    
    Args:
        user_input: Original user input
        memory_context: Memory context to use for enhancement
        
    Returns:
        Enhanced query
    """
    logger.info("Enhancing query with memory (stub)")
    
    # TODO: Implement actual enhancement logic
    return f"Enhanced: {user_input} (with memory context)"

# Module-level exports
__all__ = ['process_query', 'QueryResult', 'integrate_memory_with_reasoning', 'enhance_query_with_memory'] 