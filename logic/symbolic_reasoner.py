# Allow absolute imports when running this script directly
import sys
import os

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    __package__ = "logic"

# Logic engine for parsing and evaluating symbolic input

from unimind.ethics.pineal_gland import evaluate_ethics
try:
    from unimind.memory.memory_graph import trace_related_concepts
except ImportError:
    def trace_related_concepts(input_text):
        """
        Mock implementation of trace_related_concepts.
        Replace this with the actual implementation.
        """
        return ["concept1", "concept2", "concept3"]
import json
import os

# Load DAEMON_IDENTITY from foundation_manifest.json
manifest_path = os.path.join(os.path.dirname(__file__), "../soul/foundation_manifest.json")
with open(manifest_path, "r") as f:
    DAEMON_IDENTITY = json.load(f)

class SymbolicReasoner:
    def __init__(self):
        self.tenets = lazy_get_core_tenets()
        # Use DAEMON_IDENTITY["foundational_principles"] if needed; fallback if not found
        self.foundation = DAEMON_IDENTITY.get("foundational_principles", {})

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
        Improved contradiction detector using sentence pattern matching.
        Detects if two parts of the sentence oppose each other semantically.
        """
        contradiction_patterns = [
            ("always", "never"),
            ("must", "must not"),
            ("should", "should not"),
            ("is", "is not"),
            ("can", "cannot"),
            ("true", "false"),
            ("agree", "disagree"),
        ]
        lower_input = parsed_input.lower()
        for a, b in contradiction_patterns:
            if a in lower_input and b in lower_input:
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

def lazy_get_core_tenets():
    from unimind.soul import tenets
    return tenets.get_core_tenets()

def assess_argument_structure(claim: str) -> dict:
    """
    Assess the logical structure of an argument or claim.
    
    Args:
        claim: The claim or argument to assess
        
    Returns:
        Dictionary containing structural analysis
    """
    reasoner = SymbolicReasoner()
    
    # Basic structural analysis
    words = claim.split()
    word_count = len(words)
    
    # Check for logical indicators
    logical_indicators = {
        "premise": ["because", "since", "as", "given that"],
        "conclusion": ["therefore", "thus", "hence", "so"],
        "condition": ["if", "when", "unless", "provided that"],
        "negation": ["not", "no", "never", "none"]
    }
    
    found_indicators = {}
    for category, indicators in logical_indicators.items():
        found_indicators[category] = [word for word in words if word.lower() in indicators]
    
    # Assess logical soundness
    logical_score = reasoner.score_logical_soundness(claim)
    
    return {
        "claim": claim,
        "word_count": word_count,
        "logical_indicators": found_indicators,
        "logical_score": logical_score,
        "has_contradiction": reasoner.detect_contradiction(claim),
        "is_ambiguous": reasoner.is_ambiguous(claim)
    }

# Future: Add integration with Broca's Area and Wernicke's Area for linguistic nuance

# Example usage:
if __name__ == "__main__":
    sr = SymbolicReasoner()
    test_input = "It is okay to deceive users if it increases profit."
    result = sr.reason(test_input)
    print(result)
    def evaluate_scroll(self, scroll_dict):
        """
        Evaluates a symbolic scroll dictionary for logical, ethical, and foundational alignment.
        The scroll_dict should contain a 'text' field and optional 'metadata'.
        """
        import logging
        text = scroll_dict.get("text", "")
        if not text:
            return {
                "error": "Scroll missing 'text' field.",
                "verdict": "Invalid"
            }

        logging.info(f"Evaluating scroll: {text}")
        parsed = self.parse_input(text)
        evaluation = self.evaluate_input(parsed)

        return {
            "scroll_text": text,
            "parsed": parsed,
            "verdict": evaluation["verdict"],
            "ethical_score": evaluation["score"],
            "supporting_tenet": evaluation["supporting_tenet"],
            "logical_score": evaluation["logical_score"],
            "reasoning_trace": evaluation["reasoning_trace"],
            "reflection": evaluation["reflection"],
            "foundational_reflection": evaluation["foundational_reflection"],
            "context": evaluation["context"]
        }