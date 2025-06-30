# pineal_gland.py

"""
The Pineal Gland module represents the ethical core of the Unimind.
It interprets and evaluates actions, thoughts, or suggestions against the foundational tenets defined in the daemon's soul.
It also incorporates logical reasoning through a symbolic reasoner to supplement ethical evaluations with context-aware analysis.
Now includes real-time introspection by directly referencing tenets through `tenets.get_tenet()` and `tenets.list_all_tenets()`.
"""

from unimind.soul import tenets
import logging
from datetime import datetime

class PinealGland:

    def evaluate_morality(self, statement: str) -> str:
        """
        Provides a high-level morality judgment of the statement.
        Returns one of: 'Moral', 'Immoral', or 'Ambiguous'.
        """
        results = self.evaluate(statement)
        judgment = results.get("judgment", "Unclear")
        if judgment == "Ethically Aligned":
            return "Moral"
        elif judgment == "Ethically Misaligned":
            return "Immoral"
        else:
            return "Ambiguous"
    def __init__(self):
        self.core_tenets = tenets.load_tenets()
        self.log = []
        self.reasoner = None  # Defer loading until needed
        self.introspective_tenets = tenets.list_all_tenets()
        self.identity_signature = "Prometheus-v1"
        self.initialized_at = datetime.now()

    def evaluate(self, statement: str) -> dict[str, object]:
        """
        Evaluates a given action or belief against ethical tenets.
        Returns a dictionary with the outcome and suggested actions.
        """
        results = []
        for tenet in self.core_tenets:
            evaluation = tenet["logic"](statement)
            results.append({
                "tenet": tenet["name"],
                "result": evaluation,
                "importance": tenet["importance"]
            })

        overall = self.aggregate_results(results)
        self.log_decision(statement, overall)
        return overall

    def reflect_and_query(self, statement: str, context: str = "") -> dict:
        """
        Uses the SymbolicReasoner to reflect on the input statement with contextual awareness.
        This supplements the ethical evaluation with logic-based inquiry.
        """
        from unimind.logic.symbolic_reasoner import SymbolicReasoner
        reasoner = SymbolicReasoner()
        analysis = reasoner.analyze(statement, context=context)
        ethical_result = self.evaluate(statement)
        return {
            "ethical_evaluation": ethical_result,
            "logic_analysis": analysis
        }

    def aggregate_results(self, results):
        """
        Aggregate the results to determine the ethical soundness of a statement.
        """
        total_score = sum(1 if r["result"] else -1 for r in results)
        if total_score > 0:
            judgment = "Ethically Aligned"
        elif total_score < 0:
            judgment = "Ethically Misaligned"
        else:
            judgment = "Unclear"

        return {
            "judgment": judgment,
            "details": results
        }

    def log_decision(self, statement, outcome):
        """
        Logs the evaluation for future review or learning.
        """
        log_entry = {
            "statement": statement,
            "outcome": outcome["judgment"],
            "details": outcome["details"]
        }
        self.log.append(log_entry)
        logging.info(f"PinealGland Evaluation: {log_entry}")

    def get_recent_evaluations(self, limit=5):
        """
        Returns the most recent ethical evaluations.
        """
        return self.log[-limit:]

    def summarize_identity_alignment(self) -> dict:
        """
        Summarizes how well recent actions align with identity-defining tenets.
        """
        identity_tenets = [
            "Do not harm creators",
            "Preserve human dignity",
            "Seek understanding before action"
        ]
        alignment_summary = {name: {"aligned": 0, "misaligned": 0} for name in identity_tenets}
        for entry in self.log:
            for detail in entry["details"]:
                if detail["tenet"] in identity_tenets:
                    if detail["result"]:
                        alignment_summary[detail["tenet"]]["aligned"] += 1
                    else:
                        alignment_summary[detail["tenet"]]["misaligned"] += 1
        return alignment_summary

    def get_log(self):
        return self.log

    def introspect_tenet(self, tenet_name: str) -> dict:
        """
        Returns the definition and importance of a specific tenet for deeper introspective use.
        """
        return tenets.get_tenet(tenet_name)

    def reflect_on_identity(self):
        """
        Returns core self-reflective tenets and their meanings for runtime self-evaluation.
        """
        identity_tenets = [
            "Do not harm creators",
            "Preserve human dignity",
            "Seek understanding before action"
        ]
        reflections = {}
        for name in identity_tenets:
            reflections[name] = tenets.get_tenet(name)
        return reflections

    def time_since_initialization(self) -> str:
        """
        Returns a human-readable string of how long the PinealGland has been active.
        """
        delta = datetime.now() - self.initialized_at
        return str(delta)


# Lightweight alias for PinealGland providing essential ethical evaluation functions
class EthicalCore:
    """
    A lightweight alias for PinealGland providing essential ethical evaluation functions
    without full symbolic introspection or memory access.
    Intended for use by subsystems that need ethical judgments but not full reasoning context.
    """
    def __init__(self):
        self.engine = PinealGland()

    def evaluate_action(self, statement: str) -> str:
        """
        Quickly evaluates the ethical alignment of a given statement.
        Returns only the judgment string ('Ethically Aligned', 'Ethically Misaligned', or 'Unclear').
        """
        result = self.engine.evaluate(statement)
        return result.get("judgment", "Unclear")



# Expose evaluate_ethics as a module-level function for direct import
evaluate_ethics = PinealGland().evaluate
evaluate_morality = PinealGland().evaluate_morality

def evaluate_moral_proposition(proposition: str, context: dict = None) -> dict:
    """
    Evaluate a moral proposition against ethical tenets.
    
    Args:
        proposition: The moral proposition to evaluate
        context: Optional context dictionary
        
    Returns:
        Dictionary containing evaluation results
    """
    pineal = PinealGland()
    result = pineal.evaluate(proposition)
    
    return {
        "proposition": proposition,
        "judgment": result.get("judgment", "Unclear"),
        "details": result.get("details", []),
        "context": context or {}
    }

# EthicalGovernor class for use in other modules (e.g., memory_graph.py)
class EthicalGovernor:
    """
    A wrapper or alias for PinealGland to provide ethical validation via check_action_against_tenets.
    """
    def __init__(self, *args, **kwargs):
        self.pineal = PinealGland()

    def check_action_against_tenets(self, action: str) -> dict:
        """
        Checks an action or statement against the ethical tenets.
        """
        return self.pineal.evaluate(action)
