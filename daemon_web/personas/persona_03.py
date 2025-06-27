"""
Persona 3: Strategic Architect
Specialization: Planning, Goal Prioritization, System Navigation
Brain Node Affinity: Prefrontal Cortex, Action Planner, Symbolic Reasoner
"""

from unimind.planning.action_planner import ActionPlanner
from unimind.logic.symbolic_reasoner import SymbolicReasoner
from unimind.core.symbolic_map import SymbolicMap
from unimind.todo.tasks import register_task

class StrategicArchitect:
    def __init__(self):
        self.planner = ActionPlanner()
        self.reasoner = SymbolicReasoner()
        self.map = SymbolicMap()
        self.intent_queue = []

    def analyze_context(self, context_data):
        """
        Processes incoming context and determines strategic opportunities or constraints.
        """
        symbols = self.map.extract_symbols(context_data)
        strategy_notes = self.reasoner.evaluate_logical_paths(symbols)
        self.intent_queue.append(strategy_notes)
        return strategy_notes

    def prioritize_goals(self, goals):
        """
        Uses internal planner to rank and sequence goals.
        """
        prioritized = self.planner.rank_goals(goals)
        return prioritized

    def propose_next_action(self, context_data, goals):
        """
        Given the current context and goals, determine the next best step.
        """
        analysis = self.analyze_context(context_data)
        ranked_goals = self.prioritize_goals(goals)
        action = self.planner.select_action(analysis, ranked_goals)
        return action

    def register_strategic_task(self, description, priority="normal"):
        """
        Registers a new strategic task to the daemon's todo system.
        """
        return register_task(description=description, priority=priority)

# Example usage:
# architect = StrategicArchitect()
# plan = architect.propose_next_action(context_data, goals)