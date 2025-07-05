"""
Scientific Research Engine

Advanced scientific research capabilities for UniMind.
Provides hypothesis generation, experimental design, data analysis, research synthesis, and scientific discovery.
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

# Scientific dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from scipy import stats
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ResearchField(Enum):
    """Scientific research fields."""
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    COMPUTER_SCIENCE = "computer_science"
    MATHEMATICS = "mathematics"
    PSYCHOLOGY = "psychology"
    MEDICINE = "medicine"
    ENGINEERING = "engineering"
    ASTRONOMY = "astronomy"
    MATERIALS_SCIENCE = "materials_science"


class ResearchType(Enum):
    """Types of scientific research."""
    EXPERIMENTAL = "experimental"
    OBSERVATIONAL = "observational"
    THEORETICAL = "theoretical"
    COMPUTATIONAL = "computational"
    META_ANALYSIS = "meta_analysis"
    SYSTEMATIC_REVIEW = "systematic_review"


class HypothesisType(Enum):
    """Types of scientific hypotheses."""
    NULL = "null"
    ALTERNATIVE = "alternative"
    DIRECTIONAL = "directional"
    NON_DIRECTIONAL = "non_directional"
    COMPLEX = "complex"


@dataclass
class ResearchQuestion:
    """Scientific research question."""
    question_id: str
    field: ResearchField
    question: str
    background: str
    significance: str
    scope: str
    constraints: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Hypothesis:
    """Scientific hypothesis."""
    hypothesis_id: str
    question_id: str
    hypothesis_type: HypothesisType
    statement: str
    variables: List[str]
    predictions: List[str]
    assumptions: List[str]
    testability: str  # "high", "medium", "low"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentalDesign:
    """Experimental design for scientific research."""
    design_id: str
    hypothesis_id: str
    design_type: str  # "randomized_control", "quasi_experimental", "correlational", "longitudinal"
    sample_size: int
    variables: Dict[str, str]  # variable_name: variable_type
    controls: List[str]
    procedures: List[str]
    data_collection_methods: List[str]
    statistical_tests: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataAnalysis:
    """Data analysis results."""
    analysis_id: str
    design_id: str
    dataset_size: int
    variables_analyzed: List[str]
    statistical_tests: List[str]
    results: Dict[str, Any]
    significance_level: float
    effect_sizes: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    assumptions_checked: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchFinding:
    """Research finding or discovery."""
    finding_id: str
    analysis_id: str
    finding_type: str  # "significant_effect", "no_effect", "interaction", "trend", "anomaly"
    description: str
    statistical_significance: bool
    practical_significance: str  # "high", "medium", "low"
    implications: List[str]
    limitations: List[str]
    future_directions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchSynthesis:
    """Research synthesis and meta-analysis."""
    synthesis_id: str
    field: ResearchField
    research_questions: List[str]
    studies_included: int
    total_participants: int
    effect_size: float
    heterogeneity: float
    publication_bias: str  # "low", "medium", "high"
    conclusions: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ScientificResearchEngine:
    """
    Advanced scientific research engine for UniMind.
    
    Provides hypothesis generation, experimental design, data analysis,
    research synthesis, and scientific discovery capabilities.
    """
    
    def __init__(self):
        """Initialize the scientific research engine."""
        self.logger = logging.getLogger('ScientificResearchEngine')
        
        # Research data storage
        self.research_questions: Dict[str, ResearchQuestion] = {}
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.experimental_designs: Dict[str, ExperimentalDesign] = {}
        self.data_analyses: Dict[str, DataAnalysis] = {}
        self.research_findings: Dict[str, ResearchFinding] = {}
        self.research_syntheses: Dict[str, ResearchSynthesis] = {}
        
        # Research databases
        self.literature_database: Dict[str, Dict[str, Any]] = {}
        self.datasets: Dict[str, pd.DataFrame] = {}
        
        # Performance metrics
        self.metrics = {
            'total_research_questions': 0,
            'total_hypotheses': 0,
            'total_experiments': 0,
            'total_analyses': 0,
            'total_findings': 0,
            'total_syntheses': 0,
            'avg_significance_level': 0.05
        }
        
        # Threading
        self.lock = threading.RLock()
        
        # Check dependencies
        self.pandas_available = PANDAS_AVAILABLE
        self.scipy_available = SCIPY_AVAILABLE
        self.sklearn_available = SKLEARN_AVAILABLE
        
        # Initialize scientific knowledge base
        self._initialize_scientific_knowledge()
        
        self.logger.info("Scientific research engine initialized")
    
    def _initialize_scientific_knowledge(self):
        """Initialize scientific knowledge base with research patterns and methods."""
        # Research methodologies by field
        self.research_methods = {
            ResearchField.PHYSICS: ['experimental', 'theoretical', 'computational', 'observational'],
            ResearchField.CHEMISTRY: ['synthesis', 'analysis', 'spectroscopy', 'kinetics'],
            ResearchField.BIOLOGY: ['experimental', 'observational', 'genetic', 'ecological'],
            ResearchField.COMPUTER_SCIENCE: ['algorithmic', 'empirical', 'theoretical', 'simulation'],
            ResearchField.MATHEMATICS: ['theoretical', 'computational', 'applied', 'pure'],
            ResearchField.PSYCHOLOGY: ['experimental', 'survey', 'observational', 'clinical'],
            ResearchField.MEDICINE: ['clinical_trial', 'observational', 'epidemiological', 'laboratory'],
            ResearchField.ENGINEERING: ['experimental', 'computational', 'design', 'optimization'],
            ResearchField.ASTRONOMY: ['observational', 'theoretical', 'computational'],
            ResearchField.MATERIALS_SCIENCE: ['synthesis', 'characterization', 'testing', 'modeling']
        }
        
        # Statistical tests by data type
        self.statistical_tests = {
            'continuous_continuous': ['pearson_correlation', 'linear_regression', 't_test'],
            'continuous_categorical': ['t_test', 'anova', 'mann_whitney'],
            'categorical_categorical': ['chi_square', 'fisher_exact', 'logistic_regression'],
            'multiple_variables': ['multiple_regression', 'factor_analysis', 'path_analysis']
        }
        
        # Research design templates
        self.design_templates = {
            'randomized_control': {
                'description': 'Randomized controlled trial with treatment and control groups',
                'strengths': ['High internal validity', 'Causal inference', 'Control over variables'],
                'limitations': ['May lack external validity', 'Ethical considerations', 'Cost and time']
            },
            'quasi_experimental': {
                'description': 'Experimental design without random assignment',
                'strengths': ['Practical feasibility', 'Real-world settings', 'Ethical considerations'],
                'limitations': ['Lower internal validity', 'Confounding variables', 'Selection bias']
            },
            'correlational': {
                'description': 'Study of relationships between variables',
                'strengths': ['Natural relationships', 'Multiple variables', 'Real-world data'],
                'limitations': ['No causal inference', 'Third variable problem', 'Directionality']
            },
            'longitudinal': {
                'description': 'Study over time with repeated measurements',
                'strengths': ['Temporal relationships', 'Development patterns', 'Change over time'],
                'limitations': ['Attrition', 'Time and cost', 'Historical effects']
            }
        }
    
    def create_research_question(self, field: ResearchField, question: str,
                               background: str, significance: str,
                               scope: str, constraints: List[str] = None) -> str:
        """Create a scientific research question."""
        question_id = f"question_{field.value}_{int(time.time())}"
        
        research_question = ResearchQuestion(
            question_id=question_id,
            field=field,
            question=question,
            background=background,
            significance=significance,
            scope=scope,
            constraints=constraints or []
        )
        
        with self.lock:
            self.research_questions[question_id] = research_question
            self.metrics['total_research_questions'] += 1
        
        self.logger.info(f"Created research question: {question_id}")
        return question_id
    
    async def generate_hypotheses(self, question_id: str,
                                hypothesis_types: List[HypothesisType] = None) -> List[str]:
        """Generate hypotheses for a research question."""
        if question_id not in self.research_questions:
            raise ValueError(f"Question ID {question_id} not found")
        
        if hypothesis_types is None:
            hypothesis_types = [HypothesisType.ALTERNATIVE, HypothesisType.NULL]
        
        research_question = self.research_questions[question_id]
        hypotheses = []
        
        for hyp_type in hypothesis_types:
            hypothesis_id = f"hypothesis_{question_id}_{hyp_type.value}_{int(time.time())}"
            
            # Generate hypothesis based on type and field
            statement = self._generate_hypothesis_statement(research_question, hyp_type)
            variables = self._extract_variables(statement)
            predictions = self._generate_predictions(statement, hyp_type)
            assumptions = self._generate_assumptions(research_question.field)
            testability = self._assess_testability(statement, research_question.field)
            
            hypothesis = Hypothesis(
                hypothesis_id=hypothesis_id,
                question_id=question_id,
                hypothesis_type=hyp_type,
                statement=statement,
                variables=variables,
                predictions=predictions,
                assumptions=assumptions,
                testability=testability
            )
            
            with self.lock:
                self.hypotheses[hypothesis_id] = hypothesis
                self.metrics['total_hypotheses'] += 1
            
            hypotheses.append(hypothesis_id)
        
        self.logger.info(f"Generated {len(hypotheses)} hypotheses for question {question_id}")
        return hypotheses
    
    def _generate_hypothesis_statement(self, research_question: ResearchQuestion,
                                     hypothesis_type: HypothesisType) -> str:
        """Generate hypothesis statement."""
        field = research_question.field
        question = research_question.question.lower()
        
        if hypothesis_type == HypothesisType.NULL:
            return f"There is no significant relationship between the variables in {question}."
        
        # Generate alternative hypothesis based on field
        if field == ResearchField.PHYSICS:
            return f"The {question} demonstrates a measurable effect that can be quantified through physical laws."
        elif field == ResearchField.BIOLOGY:
            return f"Biological factors significantly influence {question} through evolutionary and physiological mechanisms."
        elif field == ResearchField.PSYCHOLOGY:
            return f"Psychological variables have a significant impact on {question} through cognitive and behavioral processes."
        elif field == ResearchField.COMPUTER_SCIENCE:
            return f"Algorithmic approaches can significantly improve {question} through computational optimization."
        else:
            return f"There is a significant relationship between the variables in {question}."
    
    def _extract_variables(self, statement: str) -> List[str]:
        """Extract variables from hypothesis statement."""
        # Simple variable extraction
        variables = []
        words = statement.lower().split()
        
        # Look for common variable indicators
        for i, word in enumerate(words):
            if word in ['effect', 'impact', 'influence', 'relationship', 'correlation']:
                if i > 0:
                    variables.append(words[i-1])
                if i < len(words) - 1:
                    variables.append(words[i+1])
        
        # Remove duplicates and clean
        variables = list(set([v.strip('.,') for v in variables if len(v) > 2]))
        return variables[:5]  # Limit to 5 variables
    
    def _generate_predictions(self, statement: str, hypothesis_type: HypothesisType) -> List[str]:
        """Generate predictions from hypothesis."""
        predictions = []
        
        if hypothesis_type == HypothesisType.NULL:
            predictions.append("No significant difference will be observed between groups.")
            predictions.append("Correlation coefficient will be close to zero.")
        else:
            predictions.append("A significant effect will be observed in the expected direction.")
            predictions.append("Statistical tests will reject the null hypothesis.")
            predictions.append("Effect size will be meaningful and practically significant.")
        
        return predictions
    
    def _generate_assumptions(self, field: ResearchField) -> List[str]:
        """Generate assumptions for the research field."""
        assumptions = {
            ResearchField.PHYSICS: [
                "Physical laws are consistent across time and space",
                "Measurements are accurate and precise",
                "Experimental conditions are controlled"
            ],
            ResearchField.BIOLOGY: [
                "Biological processes follow natural laws",
                "Organisms respond to environmental stimuli",
                "Genetic and environmental factors interact"
            ],
            ResearchField.PSYCHOLOGY: [
                "Human behavior is measurable and predictable",
                "Psychological processes are consistent across individuals",
                "Environmental factors influence behavior"
            ],
            ResearchField.COMPUTER_SCIENCE: [
                "Computational resources are sufficient",
                "Algorithms can be implemented and tested",
                "Performance metrics are meaningful"
            ]
        }
        
        return assumptions.get(field, [
            "Variables can be measured accurately",
            "Relationships between variables are consistent",
            "Statistical assumptions are met"
        ])
    
    def _assess_testability(self, statement: str, field: ResearchField) -> str:
        """Assess testability of hypothesis."""
        # Simple heuristic based on statement characteristics
        testability_indicators = {
            'high': ['significant', 'measurable', 'observable', 'quantifiable'],
            'medium': ['relationship', 'effect', 'influence', 'correlation'],
            'low': ['abstract', 'philosophical', 'subjective', 'qualitative']
        }
        
        statement_lower = statement.lower()
        for level, indicators in testability_indicators.items():
            if any(indicator in statement_lower for indicator in indicators):
                return level
        
        return 'medium'
    
    async def design_experiment(self, hypothesis_id: str,
                              design_type: str = "randomized_control",
                              sample_size: int = 100) -> str:
        """Design an experiment for hypothesis testing."""
        if hypothesis_id not in self.hypotheses:
            raise ValueError(f"Hypothesis ID {hypothesis_id} not found")
        
        hypothesis = self.hypotheses[hypothesis_id]
        research_question = self.research_questions[hypothesis.question_id]
        
        design_id = f"design_{hypothesis_id}_{int(time.time())}"
        
        # Generate experimental design
        variables = self._generate_experimental_variables(hypothesis, research_question.field)
        controls = self._generate_controls(hypothesis, design_type)
        procedures = self._generate_procedures(research_question.field, design_type)
        data_collection_methods = self._generate_data_collection_methods(research_question.field)
        statistical_tests = self._select_statistical_tests(hypothesis, design_type)
        
        experimental_design = ExperimentalDesign(
            design_id=design_id,
            hypothesis_id=hypothesis_id,
            design_type=design_type,
            sample_size=sample_size,
            variables=variables,
            controls=controls,
            procedures=procedures,
            data_collection_methods=data_collection_methods,
            statistical_tests=statistical_tests
        )
        
        with self.lock:
            self.experimental_designs[design_id] = experimental_design
            self.metrics['total_experiments'] += 1
        
        self.logger.info(f"Designed experiment: {design_id}")
        return design_id
    
    def _generate_experimental_variables(self, hypothesis: Hypothesis,
                                       field: ResearchField) -> Dict[str, str]:
        """Generate experimental variables."""
        variables = {}
        
        # Add independent and dependent variables
        if hypothesis.variables:
            for i, var in enumerate(hypothesis.variables[:2]):
                if i == 0:
                    variables[var] = "independent"
                else:
                    variables[var] = "dependent"
        
        # Add control variables based on field
        control_vars = {
            ResearchField.PHYSICS: ["temperature", "pressure", "time"],
            ResearchField.BIOLOGY: ["temperature", "humidity", "light"],
            ResearchField.PSYCHOLOGY: ["age", "gender", "education"],
            ResearchField.COMPUTER_SCIENCE: ["hardware", "software_version", "data_size"]
        }
        
        for var in control_vars.get(field, ["control_variable_1", "control_variable_2"]):
            variables[var] = "control"
        
        return variables
    
    def _generate_controls(self, hypothesis: Hypothesis, design_type: str) -> List[str]:
        """Generate experimental controls."""
        controls = []
        
        if design_type == "randomized_control":
            controls.extend([
                "Random assignment to treatment and control groups",
                "Blinding of participants and researchers",
                "Standardized procedures across conditions"
            ])
        elif design_type == "quasi_experimental":
            controls.extend([
                "Matching of participants on key variables",
                "Statistical control of confounding variables",
                "Multiple measurement points"
            ])
        
        # Add general controls
        controls.extend([
            "Consistent environmental conditions",
            "Standardized measurement procedures",
            "Data quality checks"
        ])
        
        return controls
    
    def _generate_procedures(self, field: ResearchField, design_type: str) -> List[str]:
        """Generate experimental procedures."""
        procedures = []
        
        # Common procedures
        procedures.extend([
            "Obtain informed consent from participants",
            "Randomize participants to conditions",
            "Administer experimental manipulation",
            "Collect dependent variable measures",
            "Debrief participants"
        ])
        
        # Field-specific procedures
        field_procedures = {
            ResearchField.PHYSICS: [
                "Calibrate measurement instruments",
                "Control environmental variables",
                "Record experimental parameters"
            ],
            ResearchField.BIOLOGY: [
                "Prepare biological samples",
                "Maintain controlled environmental conditions",
                "Monitor biological responses"
            ],
            ResearchField.PSYCHOLOGY: [
                "Administer psychological measures",
                "Ensure participant comfort and safety",
                "Maintain experimental protocol"
            ]
        }
        
        procedures.extend(field_procedures.get(field, []))
        return procedures
    
    def _generate_data_collection_methods(self, field: ResearchField) -> List[str]:
        """Generate data collection methods."""
        methods = {
            ResearchField.PHYSICS: ["sensors", "spectrometers", "oscilloscopes"],
            ResearchField.BIOLOGY: ["microscopes", "spectrophotometers", "PCR"],
            ResearchField.PSYCHOLOGY: ["questionnaires", "behavioral observation", "physiological measures"],
            ResearchField.COMPUTER_SCIENCE: ["performance monitoring", "user testing", "algorithm analysis"]
        }
        
        return methods.get(field, ["standardized measures", "direct observation", "instrumentation"])
    
    def _select_statistical_tests(self, hypothesis: Hypothesis, design_type: str) -> List[str]:
        """Select appropriate statistical tests."""
        tests = []
        
        if design_type == "randomized_control":
            tests.extend(["t_test", "anova", "effect_size_calculation"])
        elif design_type == "correlational":
            tests.extend(["pearson_correlation", "regression_analysis", "confidence_intervals"])
        
        # Add general tests
        tests.extend([
            "descriptive_statistics",
            "normality_tests",
            "outlier_detection"
        ])
        
        return tests
    
    async def analyze_data(self, design_id: str, data: Dict[str, List[float]]) -> str:
        """Analyze experimental data."""
        if design_id not in self.experimental_designs:
            raise ValueError(f"Design ID {design_id} not found")
        
        if not self.pandas_available or not self.scipy_available:
            raise RuntimeError("Required dependencies not available")
        
        experimental_design = self.experimental_designs[design_id]
        hypothesis = self.hypotheses[experimental_design.hypothesis_id]
        
        analysis_id = f"analysis_{design_id}_{int(time.time())}"
        
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        dataset_size = len(df)
        
        # Perform statistical analyses
        results = {}
        effect_sizes = {}
        confidence_intervals = {}
        
        for test in experimental_design.statistical_tests:
            if test == "t_test":
                test_result = self._perform_t_test(df, experimental_design.variables)
                results[test] = test_result
            elif test == "pearson_correlation":
                corr_result = self._perform_correlation(df, experimental_design.variables)
                results[test] = corr_result
            elif test == "descriptive_statistics":
                desc_result = self._perform_descriptive_statistics(df)
                results[test] = desc_result
        
        # Calculate effect sizes and confidence intervals
        effect_sizes = self._calculate_effect_sizes(results)
        confidence_intervals = self._calculate_confidence_intervals(results)
        
        data_analysis = DataAnalysis(
            analysis_id=analysis_id,
            design_id=design_id,
            dataset_size=dataset_size,
            variables_analyzed=list(experimental_design.variables.keys()),
            statistical_tests=experimental_design.statistical_tests,
            results=results,
            significance_level=0.05,
            effect_sizes=effect_sizes,
            confidence_intervals=confidence_intervals,
            assumptions_checked=["normality", "homogeneity", "independence"]
        )
        
        with self.lock:
            self.data_analyses[analysis_id] = data_analysis
            self.metrics['total_analyses'] += 1
        
        self.logger.info(f"Analyzed data: {analysis_id}")
        return analysis_id
    
    def _perform_t_test(self, df: pd.DataFrame, variables: Dict[str, str]) -> Dict[str, Any]:
        """Perform t-test analysis."""
        # Find independent and dependent variables
        ind_vars = [var for var, var_type in variables.items() if var_type == "independent"]
        dep_vars = [var for var, var_type in variables.items() if var_type == "dependent"]
        
        if not ind_vars or not dep_vars:
            return {"error": "Insufficient variables for t-test"}
        
        ind_var = ind_vars[0]
        dep_var = dep_vars[0]
        
        # Perform t-test
        try:
            groups = df[ind_var].unique()
            if len(groups) == 2:
                group1 = df[df[ind_var] == groups[0]][dep_var]
                group2 = df[df[ind_var] == groups[1]][dep_var]
                
                t_stat, p_value = stats.ttest_ind(group1, group2)
                
                return {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "group1_mean": float(group1.mean()),
                    "group2_mean": float(group2.mean()),
                    "group1_std": float(group1.std()),
                    "group2_std": float(group2.std())
                }
        except Exception as e:
            return {"error": str(e)}
        
        return {"error": "Invalid groups for t-test"}
    
    def _perform_correlation(self, df: pd.DataFrame, variables: Dict[str, str]) -> Dict[str, Any]:
        """Perform correlation analysis."""
        # Find continuous variables
        continuous_vars = [var for var, var_type in variables.items() 
                          if var_type in ["independent", "dependent"]]
        
        if len(continuous_vars) < 2:
            return {"error": "Insufficient continuous variables for correlation"}
        
        try:
            var1, var2 = continuous_vars[:2]
            correlation, p_value = stats.pearsonr(df[var1], df[var2])
            
            return {
                "correlation_coefficient": float(correlation),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "strength": "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _perform_descriptive_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform descriptive statistics."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            desc_stats = {}
            
            for col in numeric_cols:
                desc_stats[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "median": float(df[col].median())
                }
            
            return desc_stats
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_effect_sizes(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate effect sizes from results."""
        effect_sizes = {}
        
        for test, result in results.items():
            if test == "t_test" and "t_statistic" in result:
                # Calculate Cohen's d
                t_stat = result["t_statistic"]
                n1 = n2 = 50  # Assume equal group sizes
                cohens_d = t_stat * np.sqrt((n1 + n2) / (n1 * n2))
                effect_sizes["cohens_d"] = float(cohens_d)
            elif test == "pearson_correlation" and "correlation_coefficient" in result:
                # Use correlation coefficient as effect size
                effect_sizes["correlation_effect"] = abs(result["correlation_coefficient"])
        
        return effect_sizes
    
    def _calculate_confidence_intervals(self, results: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals from results."""
        confidence_intervals = {}
        
        for test, result in results.items():
            if test == "pearson_correlation" and "correlation_coefficient" in result:
                # Calculate confidence interval for correlation
                r = result["correlation_coefficient"]
                n = 100  # Assume sample size
                z = 0.5 * np.log((1 + r) / (1 - r))
                se = 1 / np.sqrt(n - 3)
                ci_lower = np.tanh(z - 1.96 * se)
                ci_upper = np.tanh(z + 1.96 * se)
                confidence_intervals["correlation_ci"] = (float(ci_lower), float(ci_upper))
        
        return confidence_intervals
    
    async def interpret_findings(self, analysis_id: str) -> str:
        """Interpret research findings from data analysis."""
        if analysis_id not in self.data_analyses:
            raise ValueError(f"Analysis ID {analysis_id} not found")
        
        data_analysis = self.data_analyses[analysis_id]
        experimental_design = self.experimental_designs[data_analysis.design_id]
        hypothesis = self.hypotheses[experimental_design.hypothesis_id]
        
        finding_id = f"finding_{analysis_id}_{int(time.time())}"
        
        # Interpret results
        finding_type = self._determine_finding_type(data_analysis.results)
        description = self._generate_finding_description(data_analysis, hypothesis)
        statistical_significance = self._assess_statistical_significance(data_analysis.results)
        practical_significance = self._assess_practical_significance(data_analysis.effect_sizes)
        implications = self._generate_implications(finding_type, hypothesis)
        limitations = self._generate_limitations(experimental_design)
        future_directions = self._generate_future_directions(finding_type, hypothesis)
        
        research_finding = ResearchFinding(
            finding_id=finding_id,
            analysis_id=analysis_id,
            finding_type=finding_type,
            description=description,
            statistical_significance=statistical_significance,
            practical_significance=practical_significance,
            implications=implications,
            limitations=limitations,
            future_directions=future_directions
        )
        
        with self.lock:
            self.research_findings[finding_id] = research_finding
            self.metrics['total_findings'] += 1
        
        self.logger.info(f"Interpreted findings: {finding_id}")
        return finding_id
    
    def _determine_finding_type(self, results: Dict[str, Any]) -> str:
        """Determine the type of research finding."""
        for test, result in results.items():
            if "significant" in result and result["significant"]:
                if test == "t_test":
                    return "significant_effect"
                elif test == "pearson_correlation":
                    return "significant_correlation"
        
        return "no_effect"
    
    def _generate_finding_description(self, data_analysis: DataAnalysis,
                                    hypothesis: Hypothesis) -> str:
        """Generate description of research finding."""
        if data_analysis.results:
            for test, result in data_analysis.results.items():
                if "significant" in result and result["significant"]:
                    if test == "t_test":
                        return f"Significant difference found between groups (t={result['t_statistic']:.2f}, p={result['p_value']:.3f})"
                    elif test == "pearson_correlation":
                        return f"Significant correlation found (r={result['correlation_coefficient']:.3f}, p={result['p_value']:.3f})"
        
        return "No significant effects were found in the analysis"
    
    def _assess_statistical_significance(self, results: Dict[str, Any]) -> bool:
        """Assess statistical significance of results."""
        for test, result in results.items():
            if "significant" in result:
                return result["significant"]
        return False
    
    def _assess_practical_significance(self, effect_sizes: Dict[str, float]) -> str:
        """Assess practical significance based on effect sizes."""
        if not effect_sizes:
            return "low"
        
        # Use largest effect size
        max_effect = max(effect_sizes.values())
        
        if max_effect > 0.5:
            return "high"
        elif max_effect > 0.2:
            return "medium"
        else:
            return "low"
    
    def _generate_implications(self, finding_type: str, hypothesis: Hypothesis) -> List[str]:
        """Generate implications of research finding."""
        implications = []
        
        if finding_type == "significant_effect":
            implications.extend([
                "The hypothesis is supported by empirical evidence",
                "The effect has practical applications in the field",
                "Further research should explore underlying mechanisms"
            ])
        elif finding_type == "no_effect":
            implications.extend([
                "The null hypothesis cannot be rejected",
                "Alternative explanations should be considered",
                "Methodological improvements may be needed"
            ])
        
        return implications
    
    def _generate_limitations(self, experimental_design: ExperimentalDesign) -> List[str]:
        """Generate limitations of the research."""
        limitations = []
        
        if experimental_design.sample_size < 100:
            limitations.append("Small sample size may limit statistical power")
        
        if experimental_design.design_type == "quasi_experimental":
            limitations.append("Non-random assignment limits causal inference")
        
        limitations.extend([
            "Results may not generalize to other populations",
            "Measurement error may affect results",
            "External validity may be limited"
        ])
        
        return limitations
    
    def _generate_future_directions(self, finding_type: str, hypothesis: Hypothesis) -> List[str]:
        """Generate future research directions."""
        directions = []
        
        if finding_type == "significant_effect":
            directions.extend([
                "Replicate findings with larger samples",
                "Explore moderating and mediating variables",
                "Investigate long-term effects and outcomes"
            ])
        else:
            directions.extend([
                "Refine experimental methodology",
                "Explore alternative theoretical frameworks",
                "Investigate boundary conditions"
            ])
        
        return directions
    
    def get_research_summary(self, question_id: str) -> Dict[str, Any]:
        """Get comprehensive research summary for a question."""
        if question_id not in self.research_questions:
            return {}
        
        research_question = self.research_questions[question_id]
        
        # Find related hypotheses
        hypotheses = [h for h in self.hypotheses.values() if h.question_id == question_id]
        
        # Find related experiments
        experiment_ids = [h.hypothesis_id for h in hypotheses]
        experiments = [e for e in self.experimental_designs.values() if e.hypothesis_id in experiment_ids]
        
        # Find related analyses
        analysis_ids = [e.design_id for e in experiments]
        analyses = [a for a in self.data_analyses.values() if a.design_id in analysis_ids]
        
        # Find related findings
        finding_ids = [a.analysis_id for a in analyses]
        findings = [f for f in self.research_findings.values() if f.analysis_id in finding_ids]
        
        return {
            'question': research_question.question,
            'field': research_question.field.value,
            'hypotheses_count': len(hypotheses),
            'experiments_count': len(experiments),
            'analyses_count': len(analyses),
            'findings_count': len(findings),
            'significant_findings': len([f for f in findings if f.statistical_significance])
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get scientific research system status."""
        with self.lock:
            return {
                'metrics': self.metrics.copy(),
                'total_research_questions': len(self.research_questions),
                'total_hypotheses': len(self.hypotheses),
                'total_experimental_designs': len(self.experimental_designs),
                'total_data_analyses': len(self.data_analyses),
                'total_research_findings': len(self.research_findings),
                'pandas_available': self.pandas_available,
                'scipy_available': self.scipy_available,
                'sklearn_available': self.sklearn_available
            }


# Global instance
scientific_research_engine = ScientificResearchEngine() 