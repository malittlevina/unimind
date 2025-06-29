import logging
logging.basicConfig(level=logging.INFO)
from unimind.soul.tenets import evaluate_against_tenets
from unimind.logic.symbolic_reasoner import SymbolicReasoner
from unimind.ethics.pineal_gland import EthicalGovernor

class Unimind:
    def __init__(self):
        self.reasoner = SymbolicReasoner()
        self.tenet_filter = evaluate_against_tenets
        self.ethical_checker = lambda action: EthicalGovernor().evaluate_action(action)

    def reflect_on_intent(self, intent_description: str) -> bool:
        evaluation = self.tenet_filter(intent_description)
        if evaluation["violates_tenets"]:
            logging.warning(f"[Unimind] Intent blocked: {evaluation['reason']}")
            return False
        logging.info("[Unimind] Intent approved.")
        return True

    def validate_action(self, action_summary: str) -> bool:
        result = self.ethical_checker(action_summary)
        logging.info(f"[Unimind] Action validation result: {result}")
        return result

    def summarize_memory_logic(self) -> str:
        return (
            f"Symbolic Reasoner: {self.reasoner.__class__.__name__}\n"
            f"Tenet Filter Active: {self.tenet_filter.__name__}\n"
            f"Ethical Checker Active: {self.ethical_checker.__name__}"
        )
