# Persona 1 specialization logic
"""
Persona 01 – The Cognitive Strategist
Specializes in logical reasoning, planning, and verbal problem-solving.
Primary brain node affiliations: Prefrontal Cortex, Broca’s Area
"""

from unimind.cortex.prefrontal_cortex import analyze_logical_structure
from unimind.perception.brocas_area import interpret_syntax
from unimind.planning.action_planner import plan_steps
from unimind.logic.symbolic_reasoner import evaluate_statement

class CognitiveStrategist:
    def __init__(self, name="Strategos"):
        self.name = name
        self.persona_type = "Cognitive Strategist"
        self.skills = [
            "strategic planning",
            "syllogistic analysis",
            "verbal logic interpretation",
            "ethical evaluation",
            "goal-oriented synthesis"
        ]

    def analyze_statement(self, statement):
        syntax_tree = interpret_syntax(statement)
        logic_map = analyze_logical_structure(syntax_tree)
        result = evaluate_statement(logic_map)
        return result

    def plan_action(self, scenario):
        return plan_steps(scenario)

    def summary(self):
        return {
            "name": self.name,
            "type": self.persona_type,
            "skills": self.skills
        }