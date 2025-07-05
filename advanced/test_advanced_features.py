"""
Test Advanced Features

Comprehensive test suite for UniMind advanced features.
Tests all implemented advanced capabilities and their integration.
"""

import asyncio
import logging
import time
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import advanced features
try:
    from .deep_learning_engine import deep_learning_engine, ModelConfig, ModelType, TaskType, LearningType
    DEEP_LEARNING_AVAILABLE = True
except ImportError as e:
    print(f"Deep Learning Engine not available: {e}")
    DEEP_LEARNING_AVAILABLE = False

try:
    from .enhanced_rag_system import enhanced_rag_system
    ENHANCED_RAG_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced RAG System not available: {e}")
    ENHANCED_RAG_AVAILABLE = False

try:
    from .causal_reasoning_engine import causal_reasoning_engine
    CAUSAL_REASONING_AVAILABLE = True
except ImportError as e:
    print(f"Causal Reasoning Engine not available: {e}")
    CAUSAL_REASONING_AVAILABLE = False

try:
    from .autonomous_agents import autonomous_agents_system, AutonomousAgent, AgentType, AgentCapability
    AUTONOMOUS_AGENTS_AVAILABLE = True
except ImportError as e:
    print(f"Autonomous Agents System not available: {e}")
    AUTONOMOUS_AGENTS_AVAILABLE = False

try:
    from .advanced_analytics import advanced_analytics_engine
    ADVANCED_ANALYTICS_AVAILABLE = True
except ImportError as e:
    print(f"Advanced Analytics Engine not available: {e}")
    ADVANCED_ANALYTICS_AVAILABLE = False

# Import base RAG system for comparison
try:
    from .rag_system import rag_system
    RAG_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"Base RAG System not available: {e}")
    RAG_SYSTEM_AVAILABLE = False


class AdvancedFeaturesTester:
    """Comprehensive tester for advanced UniMind features."""
    
    def __init__(self):
        """Initialize the tester."""
        self.logger = logging.getLogger('AdvancedFeaturesTester')
        self.logger.setLevel(logging.INFO)
        
        # Test results
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'total': 0
        }
        
        # Performance metrics
        self.performance_metrics = {}
        
        self.logger.info("Advanced Features Tester initialized")
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all advanced feature tests."""
        self.logger.info("Starting comprehensive advanced features test suite")
        
        start_time = time.time()
        
        # Test each advanced feature
        await self.test_deep_learning_engine()
        await self.test_enhanced_rag_system()
        await self.test_causal_reasoning_engine()
        await self.test_autonomous_agents()
        await self.test_advanced_analytics()
        await self.test_system_integration()
        
        total_time = time.time() - start_time
        
        # Generate test report
        report = self.generate_test_report(total_time)
        
        self.logger.info(f"Test suite completed in {total_time:.2f}s")
        return report
    
    async def test_deep_learning_engine(self):
        """Test deep learning engine functionality."""
        self.logger.info("Testing Deep Learning Engine")
        
        if not DEEP_LEARNING_AVAILABLE:
            self.logger.warning("Deep Learning Engine not available - skipping tests")
            self.test_results['skipped'] += 1
            return
        
        try:
            # Test model creation
            config = ModelConfig(
                model_type=ModelType.TRANSFORMER,
                task_type=TaskType.CLASSIFICATION,
                learning_type=LearningType.SUPERVISED,
                input_dim=100,
                output_dim=10,
                hidden_dims=[64, 32]
            )
            
            model_id = deep_learning_engine.create_model(config)
            self.logger.info(f"Created model: {model_id}")
            
            # Test model listing
            models = deep_learning_engine.list_models()
            self.logger.info(f"Found {len(models)} models")
            
            # Test system status
            status = deep_learning_engine.get_system_status()
            self.logger.info(f"Deep Learning Engine status: {status}")
            
            self.test_results['passed'] += 1
            
        except Exception as e:
            self.logger.error(f"Deep Learning Engine test failed: {e}")
            self.test_results['failed'] += 1
        
        self.test_results['total'] += 1
    
    async def test_enhanced_rag_system(self):
        """Test enhanced RAG system functionality."""
        self.logger.info("Testing Enhanced RAG System")
        
        if not ENHANCED_RAG_AVAILABLE:
            self.logger.warning("Enhanced RAG System not available - skipping tests")
            self.test_results['skipped'] += 1
            return
        
        try:
            # Test enhanced search
            query = "What is artificial intelligence?"
            search_results = await enhanced_rag_system.enhanced_search(
                query, 
                method="hybrid_search",
                max_results=5
            )
            
            self.logger.info(f"Enhanced search returned {len(search_results)} results")
            
            # Test enhanced generation
            if search_results:
                generation_result = await enhanced_rag_system.enhanced_generate(
                    query,
                    search_results,
                    method="hybrid_generation"
                )
                
                self.logger.info(f"Generated text: {generation_result.generated_text[:100]}...")
            
            # Test system status
            status = enhanced_rag_system.get_system_status()
            self.logger.info(f"Enhanced RAG System status: {status}")
            
            self.test_results['passed'] += 1
            
        except Exception as e:
            self.logger.error(f"Enhanced RAG System test failed: {e}")
            self.test_results['failed'] += 1
        
        self.test_results['total'] += 1
    
    async def test_causal_reasoning_engine(self):
        """Test causal reasoning engine functionality."""
        self.logger.info("Testing Causal Reasoning Engine")
        
        if not CAUSAL_REASONING_AVAILABLE:
            self.logger.warning("Causal Reasoning Engine not available - skipping tests")
            self.test_results['skipped'] += 1
            return
        
        try:
            # Test variable registration
            from .causal_reasoning_engine import CausalVariable, CausalRelation, CausalRelationType, CausalStrength
            
            variable = CausalVariable(
                name="temperature",
                variable_type="treatment",
                data_type="continuous",
                description="Temperature variable"
            )
            
            variable_id = causal_reasoning_engine.register_variable(variable)
            self.logger.info(f"Registered variable: {variable_id}")
            
            # Test causal analysis
            variables = ["temperature", "pressure"]
            analysis = await causal_reasoning_engine.analyze_causality(
                variables,
                analysis_type="correlation"
            )
            
            self.logger.info(f"Causal analysis completed with {len(analysis.relations)} relations")
            
            # Test system status
            status = causal_reasoning_engine.get_system_status()
            self.logger.info(f"Causal Reasoning Engine status: {status}")
            
            self.test_results['passed'] += 1
            
        except Exception as e:
            self.logger.error(f"Causal Reasoning Engine test failed: {e}")
            self.test_results['failed'] += 1
        
        self.test_results['total'] += 1
    
    async def test_autonomous_agents(self):
        """Test autonomous agents system functionality."""
        self.logger.info("Testing Autonomous Agents System")
        
        if not AUTONOMOUS_AGENTS_AVAILABLE:
            self.logger.warning("Autonomous Agents System not available - skipping tests")
            self.test_results['skipped'] += 1
            return
        
        try:
            # Test agent creation
            capabilities = [
                AgentCapability("data_processing", "Process data", 0.8),
                AgentCapability("computation", "Perform computations", 0.9)
            ]
            
            agent = AutonomousAgent("test_agent_001", AgentType.TASK_AGENT, capabilities)
            agent_id = autonomous_agents_system.register_agent(agent)
            
            self.logger.info(f"Registered agent: {agent_id}")
            
            # Test system start
            await autonomous_agents_system.start_system()
            
            # Test task submission
            from .autonomous_agents import AgentTask, TaskStatus
            
            task = AgentTask(
                task_id="test_task_001",
                task_type="computation",
                description="Test computation task",
                priority=1
            )
            
            task_id = await autonomous_agents_system.submit_task(task)
            self.logger.info(f"Submitted task: {task_id}")
            
            # Wait for task completion
            await asyncio.sleep(2)
            
            # Test system status
            status = autonomous_agents_system.get_system_status()
            self.logger.info(f"Autonomous Agents System status: {status}")
            
            # Stop system
            await autonomous_agents_system.stop_system()
            
            self.test_results['passed'] += 1
            
        except Exception as e:
            self.logger.error(f"Autonomous Agents System test failed: {e}")
            self.test_results['failed'] += 1
        
        self.test_results['total'] += 1
    
    async def test_advanced_analytics(self):
        """Test advanced analytics engine functionality."""
        self.logger.info("Testing Advanced Analytics Engine")
        
        if not ADVANCED_ANALYTICS_AVAILABLE:
            self.logger.warning("Advanced Analytics Engine not available - skipping tests")
            self.test_results['skipped'] += 1
            return
        
        try:
            # Test time series data addition
            from datetime import datetime, timedelta
            
            timestamps = [datetime.now() + timedelta(hours=i) for i in range(24)]
            values = [i + np.random.normal(0, 0.1) for i in range(24)]
            
            data_id = advanced_analytics_engine.add_time_series_data(
                "test_data",
                timestamps,
                values
            )
            
            self.logger.info(f"Added time series data: {data_id}")
            
            # Test prediction
            prediction_result = await advanced_analytics_engine.predict_values(
                data_id,
                prediction_horizon=5
            )
            
            self.logger.info(f"Prediction completed with {len(prediction_result.predicted_values)} predictions")
            
            # Test anomaly detection
            anomaly_result = await advanced_analytics_engine.detect_anomalies(data_id)
            
            self.logger.info(f"Anomaly detection found {len(anomaly_result.anomaly_indices)} anomalies")
            
            # Test trend analysis
            trend_result = await advanced_analytics_engine.analyze_trends(data_id)
            
            self.logger.info(f"Trend analysis: {trend_result.trend_direction} with strength {trend_result.trend_strength}")
            
            # Test system status
            status = advanced_analytics_engine.get_system_status()
            self.logger.info(f"Advanced Analytics Engine status: {status}")
            
            self.test_results['passed'] += 1
            
        except Exception as e:
            self.logger.error(f"Advanced Analytics Engine test failed: {e}")
            self.test_results['failed'] += 1
        
        self.test_results['total'] += 1
    
    async def test_system_integration(self):
        """Test integration between advanced features."""
        self.logger.info("Testing System Integration")
        
        try:
            # Test RAG with deep learning integration
            if ENHANCED_RAG_AVAILABLE and DEEP_LEARNING_AVAILABLE:
                query = "Explain machine learning concepts"
                
                # Enhanced search
                search_results = await enhanced_rag_system.enhanced_search(query)
                
                # Enhanced generation
                if search_results:
                    generation_result = await enhanced_rag_system.enhanced_generate(
                        query, search_results
                    )
                    
                    self.logger.info("RAG-Deep Learning integration successful")
            
            # Test analytics with causal reasoning
            if ADVANCED_ANALYTICS_AVAILABLE and CAUSAL_REASONING_AVAILABLE:
                # Add data and analyze
                timestamps = [datetime.now() + timedelta(hours=i) for i in range(10)]
                values = [i * 2 + np.random.normal(0, 0.1) for i in range(10)]
                
                data_id = advanced_analytics_engine.add_time_series_data(
                    "integration_test",
                    timestamps,
                    values
                )
                
                # Causal analysis
                analysis = await causal_reasoning_engine.analyze_causality(
                    ["time", "value"],
                    analysis_type="correlation"
                )
                
                self.logger.info("Analytics-Causal Reasoning integration successful")
            
            # Test agents with analytics
            if AUTONOMOUS_AGENTS_AVAILABLE and ADVANCED_ANALYTICS_AVAILABLE:
                # Create agent with analytics capability
                analytics_capability = AgentCapability(
                    "analytics",
                    "Perform data analysis",
                    0.8
                )
                
                agent = AutonomousAgent(
                    "analytics_agent",
                    AgentType.SPECIALIST_AGENT,
                    [analytics_capability]
                )
                
                autonomous_agents_system.register_agent(agent)
                
                self.logger.info("Agents-Analytics integration successful")
            
            self.test_results['passed'] += 1
            
        except Exception as e:
            self.logger.error(f"System Integration test failed: {e}")
            self.test_results['failed'] += 1
        
        self.test_results['total'] += 1
    
    def generate_test_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        report = {
            'test_summary': {
                'total_tests': self.test_results['total'],
                'passed': self.test_results['passed'],
                'failed': self.test_results['failed'],
                'skipped': self.test_results['skipped'],
                'success_rate': (self.test_results['passed'] / self.test_results['total'] * 100) if self.test_results['total'] > 0 else 0,
                'total_time': total_time
            },
            'feature_status': {
                'deep_learning_engine': DEEP_LEARNING_AVAILABLE,
                'enhanced_rag_system': ENHANCED_RAG_AVAILABLE,
                'causal_reasoning_engine': CAUSAL_REASONING_AVAILABLE,
                'autonomous_agents': AUTONOMOUS_AGENTS_AVAILABLE,
                'advanced_analytics': ADVANCED_ANALYTICS_AVAILABLE,
                'base_rag_system': RAG_SYSTEM_AVAILABLE
            },
            'performance_metrics': self.performance_metrics,
            'recommendations': self.generate_recommendations(),
            'timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if self.test_results['failed'] > 0:
            recommendations.append("Fix failed tests before proceeding")
        
        if not DEEP_LEARNING_AVAILABLE:
            recommendations.append("Install PyTorch and transformers for deep learning capabilities")
        
        if not ENHANCED_RAG_AVAILABLE:
            recommendations.append("Complete enhanced RAG system implementation")
        
        if not CAUSAL_REASONING_AVAILABLE:
            recommendations.append("Complete causal reasoning engine implementation")
        
        if not AUTONOMOUS_AGENTS_AVAILABLE:
            recommendations.append("Complete autonomous agents system implementation")
        
        if not ADVANCED_ANALYTICS_AVAILABLE:
            recommendations.append("Install pandas, scipy, and scikit-learn for analytics")
        
        if self.test_results['success_rate'] < 80:
            recommendations.append("Improve test coverage and fix failing components")
        
        if not recommendations:
            recommendations.append("All systems operational - proceed with deployment")
        
        return recommendations


async def main():
    """Main test function."""
    print("🚀 UniMind Advanced Features Test Suite")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    tester = AdvancedFeaturesTester()
    report = await tester.run_all_tests()
    
    # Print results
    print("\n📊 Test Results Summary")
    print("=" * 30)
    print(f"Total Tests: {report['test_summary']['total_tests']}")
    print(f"Passed: {report['test_summary']['passed']}")
    print(f"Failed: {report['test_summary']['failed']}")
    print(f"Skipped: {report['test_summary']['skipped']}")
    print(f"Success Rate: {report['test_summary']['success_rate']:.1f}%")
    print(f"Total Time: {report['test_summary']['total_time']:.2f}s")
    
    print("\n🔧 Feature Status")
    print("=" * 20)
    for feature, available in report['feature_status'].items():
        status = "✅ Available" if available else "❌ Not Available"
        print(f"{feature}: {status}")
    
    print("\n💡 Recommendations")
    print("=" * 20)
    for i, recommendation in enumerate(report['recommendations'], 1):
        print(f"{i}. {recommendation}")
    
    # Save report
    report_file = Path("test_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Exit with appropriate code
    if report['test_summary']['failed'] > 0:
        print("\n❌ Some tests failed - check the report for details")
        sys.exit(1)
    else:
        print("\n✅ All tests passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main()) 