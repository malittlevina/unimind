# prefrontal_cortex.py
"""
Prefrontal Cortex – Responsible for higher-order executive functions:
- Decision making
- Moral reasoning
- Contradiction detection
- Planning evaluation
- Reflection hooks
"""

from datetime import datetime

from ethics.pineal_gland import evaluate_morality
from logic.symbolic_reasoner import detect_contradictions
from planning.action_planner import plan_evaluation
from soul.tenets import TENETS

class PrefrontalCortex:
    def __init__(self):
        self.context_state = {}
        self.evaluation_log = []
        self.last_reflection_time = None

    def reflect_on_input(self, input_data):
        """
        Core reflection method to evaluate input for logic, morality, and contradiction.
        """
        logic_report = detect_contradictions(input_data)
        moral_score = evaluate_morality(input_data, TENETS)
        decision_score = self.weight_decision(logic_report, moral_score)

        reflection = {
            "input": input_data,
            "logic": logic_report,
            "morality": moral_score,
            "decision_score": decision_score
        }

        self.last_reflection_time = datetime.now()
        self.evaluation_log.append(reflection)
        return reflection

    def weight_decision(self, logic_report, moral_score):
        """
        Determine the final decision score based on contradiction and morality weightings.
        """
        weight = 0
        if logic_report.get("contradiction_found"):
            weight -= 50
        if moral_score.get("alignment") == "high":
            weight += 40
        elif moral_score.get("alignment") == "low":
            weight -= 20
        return weight

    def evaluate_plan(self, plan):
        """
        Evaluate a plan before action execution.
        """
        return plan_evaluation(plan, self.context_state)

    def summarize_evaluation_log(self):
        """
        Returns a concise summary of past evaluations, focusing on decisions with high or low scores.
        """
        return [
            {
                "input": entry["input"],
                "score": entry["decision_score"]
            }
            for entry in self.evaluation_log
            if abs(entry["decision_score"]) >= 20
        ]

    def get_reflection_history(self):
        return self.evaluation_log

    def get_last_reflection_time(self):
        """
        Returns the timestamp of the last reflection event.
        """
        return self.last_reflection_time
