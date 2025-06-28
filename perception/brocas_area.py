# brocas_area.py

class BrocasArea:
    def __init__(self):
        self.current_sentence = ""
        self.syntax_rules = {
            "SVO": lambda s: s.get("subject") and s.get("verb") and s.get("object")
        }
        self.language_mode = "declarative"
        self.context_memory = []

    def construct_sentence(self, semantic_units):
        """
        Takes a dictionary of semantic units and constructs a grammatically correct sentence.
        Expected keys: subject, verb, object
        """
        if self.syntax_rules["SVO"](semantic_units):
            sentence = f"{semantic_units['subject']} {semantic_units['verb']} {semantic_units['object']}"
            if self.language_mode == "interrogative":
                sentence = sentence + "?"
            elif self.language_mode == "imperative":
                sentence = sentence + "!"
            else:
                sentence = sentence + "."
            self.current_sentence = sentence
            return sentence
        else:
            return "[Error: Incomplete semantic structure]"

    def set_language_mode(self, mode):
        """
        Set the language mode for sentence construction. Options: 'declarative', 'interrogative', 'imperative'
        """
        self.language_mode = mode

    def articulate_symbolic_expression(self, concept):
        """
        Converts an abstract symbolic concept into a verbal expression.
        """
        return f"This concept represents: {concept}"

    def refine_sentence(self, sentence):
        """
        Accepts a raw sentence and refines it for clarity and grammar.
        """
        refined = sentence.strip().capitalize()
        if not refined.endswith('.'):
            refined += '.'
        return refined

    def remember_context(self, sentence):
        """
        Stores a sentence in the context memory for future reference.
        """
        self.context_memory.append(sentence)

    def get_last_sentence(self):
        return self.current_sentence

    def get_context_memory(self):
        """
        Returns the list of remembered sentences.
        """
        return self.context_memory
