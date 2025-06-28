# Allow absolute imports when running this script directly
import sys
import os

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    __package__ = "logic"

# Logic engine for parsing and evaluating symbolic input

from ethics.pineal_gland import evaluate_ethics
from soul.tenets import get_core_tenets
from memory.memory_graph import trace_related_concepts
from soul.foundation_manifest import load_foundational_principles

class SymbolicReasoner:
    def __init__(self):
        self.tenets = get_core_tenets()
        self.foundation = load_foundational_principles()

    def load_dynamic_context(self):
        """
        Loads dynamic symbolic context for runtime decisions.
        Can be extended to pull in Codex entries, live memory, or daemon state.
        """
        # Placeholder for Codex or memory pull logic
        return {
            "current_emotional_state": "neutral",
            "active_goal": "evaluate truthfulness",
            "codex_links": []
        }

    def parse_input(self, symbolic_input):
        """
        Parses symbolic input into evaluable logic units.
        Example: "It is acceptable to manipulate data for efficiency."
        """
        return symbolic_input.strip()

    def is_ambiguous(self, parsed_input):
        """
        Placeholder method to detect ambiguity.
        A real implementation would analyze semantics.
        """
        ambiguous_keywords = ["sometimes", "maybe", "possibly", "often"]
        return any(word in parsed_input.lower() for word in ambiguous_keywords)

    def generate_clarifying_question(self, parsed_input):
        """
        Placeholder for clarifying question generation.
        """
        return f"Can you clarify what you mean by: '{parsed_input}'?"

    def detect_contradiction(self, parsed_input):
        """
        Simple contradiction detector based on logical negation patterns.
        Extend with formal logic engine later.
        """
        contradictory_phrases = [
            ("always", "never"),
            ("must", "must not"),
            ("should", "should not"),
        ]
        for a, b in contradictory_phrases:
            if a in parsed_input and b in parsed_input:
                return True
        return False

    def score_logical_soundness(self, parsed_input):
        """
        Scores logical coherence of the statement using heuristics.
        """
        contradiction = self.detect_contradiction(parsed_input)
        if contradiction:
            return 20  # low score due to contradiction
        elif self.is_ambiguous(parsed_input):
            return 50  # moderate score due to ambiguity
        else:
            return 90  # high score for clear, non-contradictory input

    def trace_reasoning_path(self, parsed_input):
        """
        Simulates how the reasoning unfolds by tracking concept lineage.
        """
        related = trace_related_concepts(parsed_input)
        return {"steps": related, "summary": f"Reasoning from input → {related[-1] if related else 'Unknown'}"}

    def evaluate_input(self, parsed_input):
        """
        Evaluate the logic of the parsed input against ethical tenets and return a detailed introspection result.
        """
        dynamic_context = self.load_dynamic_context()
        ethical_result = evaluate_ethics(parsed_input, self.tenets)
        return {
            "verdict": ethical_result.get("verdict", "Unknown"),
            "score": ethical_result.get("score", 0),
            "supporting_tenet": ethical_result.get("supporting_tenet", "None matched"),
            "reflection": ethical_result.get("reflection", "No commentary generated."),
            "context": dynamic_context,
            "logical_score": self.score_logical_soundness(parsed_input),
            "reasoning_trace": self.trace_reasoning_path(parsed_input),
            "foundational_reflection": self.foundation.get("reflection", "No foundational guidance."),
        }

    def reason(self, symbolic_input):
        import logging
        logging.info(f"Received symbolic input: {symbolic_input}")
        parsed = self.parse_input(symbolic_input)
        related_concepts = trace_related_concepts(parsed)
        logging.info(f"Parsed input: {parsed}")
        ethical_evaluation = self.evaluate_input(parsed)
        logging.info(f"Ethical evaluation: {ethical_evaluation}")
        response = {
            "original_input": symbolic_input,
            "parsed": parsed,
            "ethical_verdict": ethical_evaluation["verdict"],
            "ethical_score": ethical_evaluation["score"],
            "supporting_tenet": ethical_evaluation["supporting_tenet"],
            "reflection": ethical_evaluation["reflection"],
            "codex_reference": f"lookup://codex/symbolic_evaluation/{parsed[:20].replace(' ', '_')}",
            "related_concepts": related_concepts,
            "logical_score": ethical_evaluation["logical_score"],
            "reasoning_trace": ethical_evaluation["reasoning_trace"],
            "foundational_reflection": ethical_evaluation["foundational_reflection"],
        }
        if self.is_ambiguous(parsed):
            response["clarification_needed"] = True
            response["clarifying_question"] = self.generate_clarifying_question(parsed)
            logging.info("Input detected as ambiguous; clarification requested.")
        else:
            response["clarification_needed"] = False
        return response

# Future: Add integration with Broca's Area and Wernicke's Area for linguistic nuance

# Example usage:
if __name__ == "__main__":
    sr = SymbolicReasoner()
    test_input = "It is okay to deceive users if it increases profit."
    result = sr.reason(test_input)
    print(result)