from unimind.logic.symbolic_reasoner import SymbolicReasoner

class LogicEngine:
    def __init__(self):
        self.reasoner = SymbolicReasoner()

    def reflect_on_input(self, input_text):
        """
        Reflects on the input and detects contradictions using the symbolic reasoner.
        Returns a dictionary with analysis results.
        """
        contradictions_found = self.reasoner.detect_contradiction(input_text)
        return {
            "input": input_text,
            "contradiction": contradictions_found
        }