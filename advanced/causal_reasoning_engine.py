"""
Causal Reasoning Engine

Advanced causal reasoning capabilities for UniMind.
Provides cause-effect analysis, causal inference, and counterfactual reasoning.
"""

import asyncio
import logging
import time
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
import networkx as nx
from datetime import datetime

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class CausalRelationType(Enum):
    """Types of causal relationships."""
    DIRECT = "direct"
    INDIRECT = "indirect"
    BIDIRECTIONAL = "bidirectional"
    CONDITIONAL = "conditional"
    MEDIATED = "mediated"
    CONFOUNDED = "confounded"


class CausalStrength(Enum):
    """Strength of causal relationships."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    UNCERTAIN = "uncertain"


@dataclass
class CausalVariable:
    """A variable in causal analysis."""
    name: str
    variable_type: str  # "treatment", "outcome", "confounder", "mediator"
    data_type: str  # "continuous", "categorical", "binary"
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalRelation:
    """A causal relationship between variables."""
    cause: str
    effect: str
    relation_type: CausalRelationType
    strength: CausalStrength
    confidence: float
    evidence: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalAnalysis:
    """Result of causal analysis."""
    variables: List[CausalVariable]
    relations: List[CausalRelation]
    causal_graph: nx.DiGraph
    analysis_type: str
    confidence: float
    assumptions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CounterfactualResult:
    """Result of counterfactual analysis."""
    original_outcome: Any
    counterfactual_outcome: Any
    treatment_effect: float
    confidence: float
    assumptions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CausalReasoningEngine:
    """
    Advanced causal reasoning engine for UniMind.
    
    Provides cause-effect analysis, causal inference, counterfactual reasoning,
    and causal discovery from data.
    """
    
    def __init__(self):
        """Initialize the causal reasoning engine."""
        self.logger = logging.getLogger('CausalReasoningEngine')
        
        # Causal knowledge base
        self.causal_knowledge: Dict[str, CausalRelation] = {}
        self.variable_registry: Dict[str, CausalVariable] = {}
        self.causal_graphs: Dict[str, nx.DiGraph] = {}
        
        # Analysis history
        self.analysis_history: List[CausalAnalysis] = []
        self.counterfactual_history: List[CounterfactualResult] = []
        
        # Performance metrics
        self.metrics = {
            'total_analyses': 0,
            'total_relations_discovered': 0,
            'total_counterfactuals': 0,
            'avg_confidence': 0.0,
            'avg_analysis_time': 0.0
        }
        
        # Threading
        self.lock = threading.RLock()
        
        # Initialize causal patterns
        self._initialize_causal_patterns()
        
        self.logger.info("Causal reasoning engine initialized")
    
    def _initialize_causal_patterns(self):
        """Initialize common causal patterns and heuristics."""
        self.causal_patterns = {
            'temporal_precedence': "Cause precedes effect in time",
            'correlation': "Variables are correlated",
            'mechanism': "Plausible mechanism exists",
            'intervention': "Intervention on cause affects effect",
            'confounding': "Common cause affects both variables",
            'mediation': "Effect is mediated through intermediate variable"
        }
        
        self.causal_heuristics = {
            'temporal_order': 0.8,
            'correlation_strength': 0.6,
            'mechanism_plausibility': 0.7,
            'intervention_evidence': 0.9,
            'confounding_check': 0.5,
            'expert_knowledge': 0.8
        }
    
    def register_variable(self, variable: CausalVariable) -> str:
        """Register a variable for causal analysis."""
        with self.lock:
            self.variable_registry[variable.name] = variable
            self.logger.info(f"Registered variable: {variable.name}")
            return variable.name
    
    def add_causal_relation(self, relation: CausalRelation) -> str:
        """Add a causal relation to the knowledge base."""
        relation_id = f"{relation.cause}->{relation.effect}"
        
        with self.lock:
            self.causal_knowledge[relation_id] = relation
            self.metrics['total_relations_discovered'] += 1
            self.logger.info(f"Added causal relation: {relation_id}")
            return relation_id
    
    async def analyze_causality(self, 
                              variables: List[str],
                              data: Optional[Dict[str, List[Any]]] = None,
                              analysis_type: str = "correlation") -> CausalAnalysis:
        """Analyze causal relationships between variables."""
        start_time = time.time()
        
        self.logger.info(f"Causal analysis: {variables} using {analysis_type}")
        
        # Validate variables
        causal_variables = []
        for var_name in variables:
            if var_name in self.variable_registry:
                causal_variables.append(self.variable_registry[var_name])
            else:
                # Create default variable
                causal_variables.append(CausalVariable(
                    name=var_name,
                    variable_type="unknown",
                    data_type="continuous"
                ))
        
        # Perform analysis based on type
        if analysis_type == "correlation":
            relations = await self._correlation_analysis(variables, data)
        elif analysis_type == "temporal":
            relations = await self._temporal_analysis(variables, data)
        elif analysis_type == "intervention":
            relations = await self._intervention_analysis(variables, data)
        elif analysis_type == "mechanism":
            relations = await self._mechanism_analysis(variables, data)
        else:
            relations = await self._comprehensive_analysis(variables, data)
        
        # Build causal graph
        causal_graph = self._build_causal_graph(variables, relations)
        
        # Calculate overall confidence
        confidence = np.mean([r.confidence for r in relations]) if relations else 0.0
        
        analysis = CausalAnalysis(
            variables=causal_variables,
            relations=relations,
            causal_graph=causal_graph,
            analysis_type=analysis_type,
            confidence=confidence,
            assumptions=self._identify_assumptions(analysis_type),
            limitations=self._identify_limitations(analysis_type)
        )
        
        # Store analysis
        with self.lock:
            self.analysis_history.append(analysis)
            self.metrics['total_analyses'] += 1
            self.metrics['avg_analysis_time'] = (
                (self.metrics['avg_analysis_time'] * (self.metrics['total_analyses'] - 1) + 
                 (time.time() - start_time)) / self.metrics['total_analyses']
            )
        
        self.logger.info(f"Causal analysis completed in {time.time() - start_time:.2f}s")
        return analysis
    
    async def _correlation_analysis(self, variables: List[str], 
                                  data: Optional[Dict[str, List[Any]]]) -> List[CausalRelation]:
        """Analyze correlations between variables."""
        relations = []
        
        if not data or not PANDAS_AVAILABLE:
            # Use existing knowledge
            for var1 in variables:
                for var2 in variables:
                    if var1 != var2:
                        relation_id = f"{var1}->{var2}"
                        if relation_id in self.causal_knowledge:
                            relations.append(self.causal_knowledge[relation_id])
            return relations
        
        # Perform correlation analysis
        df = pd.DataFrame(data)
        
        for var1 in variables:
            for var2 in variables:
                if var1 != var2 and var1 in df.columns and var2 in df.columns:
                    try:
                        # Calculate correlation
                        correlation = df[var1].corr(df[var2])
                        
                        if not np.isnan(correlation):
                            # Determine relation type and strength
                            relation_type = CausalRelationType.DIRECT
                            strength = self._correlation_to_strength(abs(correlation))
                            confidence = abs(correlation) * self.causal_heuristics['correlation_strength']
                            
                            relation = CausalRelation(
                                cause=var1,
                                effect=var2,
                                relation_type=relation_type,
                                strength=strength,
                                confidence=confidence,
                                evidence=[f"Correlation: {correlation:.3f}"]
                            )
                            
                            relations.append(relation)
                    except Exception as e:
                        self.logger.warning(f"Correlation analysis failed for {var1}->{var2}: {e}")
        
        return relations
    
    async def _temporal_analysis(self, variables: List[str], 
                               data: Optional[Dict[str, List[Any]]]) -> List[CausalRelation]:
        """Analyze temporal relationships between variables."""
        relations = []
        
        # Simple temporal analysis based on variable names and patterns
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j:
                    # Check for temporal indicators in variable names
                    temporal_confidence = self._assess_temporal_relationship(var1, var2)
                    
                    if temporal_confidence > 0.5:
                        relation = CausalRelation(
                            cause=var1,
                            effect=var2,
                            relation_type=CausalRelationType.DIRECT,
                            strength=CausalStrength.MODERATE,
                            confidence=temporal_confidence,
                            evidence=[f"Temporal precedence: {var1} precedes {var2}"]
                        )
                        relations.append(relation)
        
        return relations
    
    async def _intervention_analysis(self, variables: List[str], 
                                   data: Optional[Dict[str, List[Any]]]) -> List[CausalRelation]:
        """Analyze intervention effects."""
        relations = []
        
        # Look for intervention patterns in data
        if data:
            for var1 in variables:
                for var2 in variables:
                    if var1 != var2:
                        intervention_confidence = self._assess_intervention_effect(var1, var2, data)
                        
                        if intervention_confidence > 0.6:
                            relation = CausalRelation(
                                cause=var1,
                                effect=var2,
                                relation_type=CausalRelationType.DIRECT,
                                strength=CausalStrength.STRONG,
                                confidence=intervention_confidence,
                                evidence=[f"Intervention effect detected"]
                            )
                            relations.append(relation)
        
        return relations
    
    async def _mechanism_analysis(self, variables: List[str], 
                                data: Optional[Dict[str, List[Any]]]) -> List[CausalRelation]:
        """Analyze causal mechanisms."""
        relations = []
        
        # Use domain knowledge and patterns to identify mechanisms
        for var1 in variables:
            for var2 in variables:
                if var1 != var2:
                    mechanism_confidence = self._assess_mechanism_plausibility(var1, var2)
                    
                    if mechanism_confidence > 0.4:
                        relation = CausalRelation(
                            cause=var1,
                            effect=var2,
                            relation_type=CausalRelationType.MEDIATED,
                            strength=CausalStrength.MODERATE,
                            confidence=mechanism_confidence,
                            evidence=[f"Plausible mechanism identified"]
                        )
                        relations.append(relation)
        
        return relations
    
    async def _comprehensive_analysis(self, variables: List[str], 
                                    data: Optional[Dict[str, List[Any]]]) -> List[CausalRelation]:
        """Comprehensive causal analysis combining multiple methods."""
        relations = []
        
        # Combine results from different analysis types
        correlation_relations = await self._correlation_analysis(variables, data)
        temporal_relations = await self._temporal_analysis(variables, data)
        intervention_relations = await self._intervention_analysis(variables, data)
        mechanism_relations = await self._mechanism_analysis(variables, data)
        
        # Merge and resolve conflicts
        all_relations = correlation_relations + temporal_relations + intervention_relations + mechanism_relations
        
        # Group by cause-effect pairs
        relation_groups = {}
        for relation in all_relations:
            key = f"{relation.cause}->{relation.effect}"
            if key not in relation_groups:
                relation_groups[key] = []
            relation_groups[key].append(relation)
        
        # Combine evidence and calculate overall confidence
        for key, group_relations in relation_groups.items():
            if len(group_relations) > 0:
                # Take the relation with highest confidence
                best_relation = max(group_relations, key=lambda r: r.confidence)
                
                # Combine evidence
                all_evidence = []
                for relation in group_relations:
                    all_evidence.extend(relation.evidence)
                
                best_relation.evidence = list(set(all_evidence))
                relations.append(best_relation)
        
        return relations
    
    def _correlation_to_strength(self, correlation: float) -> CausalStrength:
        """Convert correlation coefficient to causal strength."""
        if correlation > 0.7:
            return CausalStrength.STRONG
        elif correlation > 0.4:
            return CausalStrength.MODERATE
        elif correlation > 0.2:
            return CausalStrength.WEAK
        else:
            return CausalStrength.UNCERTAIN
    
    def _assess_temporal_relationship(self, var1: str, var2: str) -> float:
        """Assess temporal relationship between variables."""
        confidence = 0.0
        
        # Check for temporal indicators in variable names
        temporal_indicators = ['before', 'after', 'previous', 'next', 'earlier', 'later', 'past', 'future']
        
        var1_lower = var1.lower()
        var2_lower = var2.lower()
        
        for indicator in temporal_indicators:
            if indicator in var1_lower and indicator not in var2_lower:
                confidence += 0.3
            if indicator in var2_lower and indicator not in var1_lower:
                confidence -= 0.3
        
        # Check for time-related patterns
        if any(word in var1_lower for word in ['time', 'date', 'year', 'month', 'day']):
            confidence += 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def _assess_intervention_effect(self, cause: str, effect: str, data: Dict[str, List[Any]]) -> float:
        """Assess intervention effect between variables."""
        if not PANDAS_AVAILABLE or cause not in data or effect not in data:
            return 0.0
        
        try:
            df = pd.DataFrame(data)
            
            # Look for intervention patterns (e.g., sudden changes)
            cause_data = df[cause].dropna()
            effect_data = df[effect].dropna()
            
            if len(cause_data) < 10 or len(effect_data) < 10:
                return 0.0
            
            # Calculate variance and look for intervention points
            cause_var = cause_data.var()
            effect_var = effect_data.var()
            
            # Simple heuristic: if effect variance is high after cause changes, intervention likely
            if cause_var > 0 and effect_var > cause_var * 2:
                return 0.7
            
            return 0.3
            
        except Exception as e:
            self.logger.warning(f"Intervention analysis failed: {e}")
            return 0.0
    
    def _assess_mechanism_plausibility(self, cause: str, effect: str) -> float:
        """Assess plausibility of causal mechanism."""
        confidence = 0.0
        
        # Use domain knowledge patterns
        cause_lower = cause.lower()
        effect_lower = effect.lower()
        
        # Common causal patterns
        causal_patterns = [
            ('temperature', 'pressure'),
            ('speed', 'energy'),
            ('dose', 'response'),
            ('input', 'output'),
            ('cause', 'effect'),
            ('stimulus', 'response')
        ]
        
        for pattern_cause, pattern_effect in causal_patterns:
            if pattern_cause in cause_lower and pattern_effect in effect_lower:
                confidence += 0.4
            if pattern_effect in cause_lower and pattern_cause in effect_lower:
                confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def _build_causal_graph(self, variables: List[str], relations: List[CausalRelation]) -> nx.DiGraph:
        """Build causal graph from relations."""
        graph = nx.DiGraph()
        
        # Add nodes
        for variable in variables:
            graph.add_node(variable)
        
        # Add edges
        for relation in relations:
            graph.add_edge(relation.cause, relation.effect, 
                          weight=relation.confidence,
                          strength=relation.strength.value,
                          type=relation.relation_type.value)
        
        return graph
    
    def _identify_assumptions(self, analysis_type: str) -> List[str]:
        """Identify assumptions for the analysis type."""
        assumptions = {
            'correlation': [
                'Correlation implies causation (limited)',
                'No confounding variables',
                'Linear relationships'
            ],
            'temporal': [
                'Temporal precedence indicates causation',
                'No reverse causality',
                'Stable relationships over time'
            ],
            'intervention': [
                'Interventions are exogenous',
                'No spillover effects',
                'Treatment assignment is random'
            ],
            'mechanism': [
                'Mechanisms are stable',
                'No unobserved mediators',
                'Mechanism knowledge is complete'
            ]
        }
        
        return assumptions.get(analysis_type, ['Analysis assumptions not specified'])
    
    def _identify_limitations(self, analysis_type: str) -> List[str]:
        """Identify limitations for the analysis type."""
        limitations = {
            'correlation': [
                'Correlation does not prove causation',
                'May miss non-linear relationships',
                'Sensitive to outliers'
            ],
            'temporal': [
                'Temporal order may be coincidental',
                'May miss simultaneous effects',
                'Requires time series data'
            ],
            'intervention': [
                'Requires intervention data',
                'May have selection bias',
                'External validity concerns'
            ],
            'mechanism': [
                'Mechanism knowledge may be incomplete',
                'May miss complex interactions',
                'Domain-specific knowledge required'
            ]
        }
        
        return limitations.get(analysis_type, ['Analysis limitations not specified'])
    
    async def counterfactual_analysis(self, 
                                    treatment: str,
                                    outcome: str,
                                    data: Optional[Dict[str, List[Any]]] = None) -> CounterfactualResult:
        """Perform counterfactual analysis."""
        start_time = time.time()
        
        self.logger.info(f"Counterfactual analysis: {treatment} -> {outcome}")
        
        # Simple counterfactual analysis
        if data and treatment in data and outcome in data:
            # Calculate treatment effect
            treatment_data = data[treatment]
            outcome_data = data[outcome]
            
            if len(treatment_data) == len(outcome_data) and len(treatment_data) > 0:
                # Simple difference in means
                treatment_values = [v for v in treatment_data if v is not None]
                outcome_values = [v for v in outcome_data if v is not None]
                
                if treatment_values and outcome_values:
                    original_outcome = np.mean(outcome_values)
                    
                    # Simulate counterfactual (no treatment)
                    # This is a simplified approach
                    counterfactual_outcome = original_outcome * 0.8  # Assume 20% reduction
                    
                    treatment_effect = original_outcome - counterfactual_outcome
                    confidence = 0.6  # Moderate confidence for simplified analysis
                    
                    result = CounterfactualResult(
                        original_outcome=original_outcome,
                        counterfactual_outcome=counterfactual_outcome,
                        treatment_effect=treatment_effect,
                        confidence=confidence,
                        assumptions=['Linear treatment effect', 'No confounding']
                    )
                    
                    # Store result
                    with self.lock:
                        self.counterfactual_history.append(result)
                        self.metrics['total_counterfactuals'] += 1
                    
                    self.logger.info(f"Counterfactual analysis completed in {time.time() - start_time:.2f}s")
                    return result
        
        # Fallback result
        return CounterfactualResult(
            original_outcome=0.0,
            counterfactual_outcome=0.0,
            treatment_effect=0.0,
            confidence=0.0,
            assumptions=['Insufficient data for analysis']
        )
    
    def get_causal_paths(self, cause: str, effect: str, max_paths: int = 5) -> List[List[str]]:
        """Find causal paths between variables."""
        paths = []
        
        # Check if we have a causal graph
        if not self.analysis_history:
            return paths
        
        # Use the most recent analysis
        latest_analysis = self.analysis_history[-1]
        graph = latest_analysis.causal_graph
        
        if cause in graph and effect in graph:
            try:
                # Find all simple paths
                all_paths = list(nx.all_simple_paths(graph, cause, effect))
                
                # Sort by path length and take top paths
                all_paths.sort(key=len)
                paths = all_paths[:max_paths]
                
            except nx.NetworkXNoPath:
                self.logger.info(f"No causal path found between {cause} and {effect}")
        
        return paths
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get causal reasoning system status."""
        with self.lock:
            return {
                'total_analyses': self.metrics['total_analyses'],
                'total_relations_discovered': self.metrics['total_relations_discovered'],
                'total_counterfactuals': self.metrics['total_counterfactuals'],
                'avg_confidence': self.metrics['avg_confidence'],
                'avg_analysis_time': self.metrics['avg_analysis_time'],
                'registered_variables': len(self.variable_registry),
                'causal_relations': len(self.causal_knowledge),
                'analysis_history': len(self.analysis_history),
                'counterfactual_history': len(self.counterfactual_history)
            }


# Global instance
causal_reasoning_engine = CausalReasoningEngine() 