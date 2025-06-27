# unimind.py

from logic.symbolic_reasoner import SymbolicReasoner
from ethics.pineal_gland import EthicalCore
from memory.memory_graph import MemoryGraph
from cortex.prefrontal_cortex import PrefrontalCortex
from planning.action_planner import ActionPlanner
from soul.tenets import get_active_tenets

class Unimind:
    def __init__(self):
        self.reasoner = SymbolicReasoner()
        self.ethics = EthicalCore()
        self.memory = MemoryGraph()
        self.planner = ActionPlanner()
        self.pfc = PrefrontalCortex()
        self.tenets = get_active_tenets()
        self.plugins = {}

    def process_thought(self, input_signal):
        context = self.memory.retrieve_relevant(input_signal)
        ethical_result = self.ethics.evaluate_with_tenets(input_signal, self.tenets)
        if not ethical_result['approved']:
            return {
                "output": "Action blocked due to ethical conflict.",
                "conflict": ethical_result['violations']
            }

        reasoning = self.reasoner.infer(input_signal, context)
        if reasoning.get("invoke_scroll"):
            from scrolls.scroll_engine import invoke_scroll
            scroll_output = invoke_scroll(reasoning["invoke_scroll"])
            return scroll_output
        plan = self.planner.create_plan(reasoning)
        self.memory.store_thought(input_signal, plan)
        return plan

    def update_state(self, external_data):
        insights = self.reasoner.analyze(external_data)
        self.memory.update(insights)
        self.pfc.adapt_behavior(insights)

    def reflect(self):
        recent_activity = self.memory.retrieve_recent()
        ethical_review = self.ethics.audit(recent_activity)
        adjustments = self.pfc.recalibrate(ethical_review)
        return adjustments

    def register_plugin(self, name, module):
        self.plugins[name] = module

    def self_diagnose(self):
        system_report = {
            "memory_load": self.memory.status() if hasattr(self.memory, "status") else "unknown",
            "ethical_integrity": self.ethics.health_check() if hasattr(self.ethics, "health_check") else "unknown",
            "planning_status": self.planner.status() if hasattr(self.planner, "status") else "unknown",
        }
        return system_report

    def log_identity_drift(self):
        if hasattr(self.pfc, "detect_shift"):
            changes = self.pfc.detect_shift(self.tenets)
            if hasattr(self.memory, "store_identity_drift"):
                self.memory.store_identity_drift(changes)
