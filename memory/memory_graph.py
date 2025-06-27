from unimind.soul.tenets import evaluate_against_tenets
from unimind.logic.symbolic_reasoner import SymbolicReasoner
from unimind.ethics.pineal_gland import check_action_against_tenets

class Unimind:
    def __init__(self):
        self.reasoner = SymbolicReasoner()
        self.tenet_filter = evaluate_against_tenets
        self.ethical_checker = check_action_against_tenets

    def reflect_on_intent(self, intent_description: str) -> bool:
        evaluation = self.tenet_filter(intent_description)
        if evaluation["violates_tenets"]:
            print(f"[Unimind] Intent blocked: {evaluation['reason']}")
            return False
        print("[Unimind] Intent approved.")
        return True

    def validate_action(self, action_summary: str) -> bool:
        return self.ethical_checker(action_summary)
