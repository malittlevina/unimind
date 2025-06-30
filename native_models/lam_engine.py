"""
lam_engine.py – Logic-Abstraction-Model (LAM) engine for symbolic and probabilistic reasoning.
Part of the Unimind native model system.
"""

import json
import time

class LAMEngine:
    def __init__(self):
        self.knowledge_base = []
        self.inference_rules = []
        self.rule_metadata = {}  # Optional tags, authors, etc.
        self.memory_context = {}
        self.index = {}

    def add_fact(self, fact):
        """Add a fact to the knowledge base and update index."""
        self.knowledge_base.append(fact)
        self.index[fact] = time.time()

    def remove_fact(self, fact):
        """Remove a fact from the knowledge base."""
        if fact in self.knowledge_base:
            self.knowledge_base.remove(fact)
            self.index.pop(fact, None)

    def add_rule(self, rule, tag=None):
        """Add an inference rule, optionally tagged."""
        self.inference_rules.append(rule)
        if tag:
            self.rule_metadata[rule] = tag

    def remove_rule(self, rule):
        """Remove a rule and its metadata."""
        if rule in self.inference_rules:
            self.inference_rules.remove(rule)
            self.rule_metadata.pop(rule, None)

    def evaluate(self, input_data):
        """Run input through inference rules and return reasoning output."""
        results = []
        for rule in self.inference_rules:
            try:
                result = rule(input_data, self.knowledge_base, self.memory_context)
                if result:
                    results.append({
                        "rule": self.rule_metadata.get(rule, "anonymous"),
                        "result": result
                    })
            except Exception as e:
                results.append({
                    "rule": self.rule_metadata.get(rule, "anonymous"),
                    "error": str(e)
                })
        return results

    def update_context(self, key, value):
        """Update memory context for stateful logic handling."""
        self.memory_context[key] = value

    def summarize_state(self):
        return {
            "facts": self.knowledge_base,
            "rules_count": len(self.inference_rules),
            "context": self.memory_context
        }

    def query_facts(self, keyword):
        """Return all facts that include a keyword."""
        return [fact for fact in self.knowledge_base if keyword in fact]

    def explain(self, input_data):
        """Return a symbolic explanation for the inference result."""
        explanations = []
        for rule in self.inference_rules:
            try:
                result = rule(input_data, self.knowledge_base, self.memory_context)
                if result:
                    explanations.append(f"Rule {self.rule_metadata.get(rule, 'anonymous')} triggered result: {result}")
            except Exception as e:
                explanations.append(f"Rule {self.rule_metadata.get(rule, 'anonymous')} caused error: {e}")
        return explanations

    def save_state(self, filepath):
        """Save LAM state to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump({
                "facts": self.knowledge_base,
                "context": self.memory_context,
                "index": self.index,
                "rule_metadata": {str(k): v for k, v in self.rule_metadata.items()}
            }, f)

    def load_state(self, filepath):
        """Load LAM state from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.knowledge_base = data.get("facts", [])
            self.memory_context = data.get("context", {})
            self.index = data.get("index", {})
            self.rule_metadata = data.get("rule_metadata", {})

    def process_lam_query(self, query):
        """
        Process a LAM-style symbolic query. This supports keyword matching,
        fact relevance evaluation, and optional rule-based enrichment.
        """
        results = {
            "query": query,
            "matched_facts": [],
            "reasoning": [],
            "explanation": []
        }

        # Match facts based on keyword presence
        results["matched_facts"] = self.query_facts(query)

        # Run reasoning over the query if rules exist
        if self.inference_rules:
            results["reasoning"] = self.evaluate(query)
            results["explanation"] = self.explain(query)

        return results

    def interpret_prompt(self, text):
        """
        Interpret natural language input and return a matching scroll name.
        Uses simple phrase matching for now, structured for future LLM/symbolic logic upgrade.
        
        Args:
            text (str): Natural language input from user
            
        Returns:
            str: Matching scroll name or None if no match found
        """
        # Normalize input
        text = text.lower().strip()
        
        # Define phrase-to-scroll mappings
        # This can be expanded and later upgraded to use LLM or symbolic logic
        phrase_mappings = {
            # Self-assessment and introspection
            "how am i doing": "self_assess",
            "how am i": "self_assess", 
            "self assessment": "self_assess",
            "check my status": "self_assess",
            "how are you doing": "self_assess",
            "status check": "self_assess",
            
            # Calming and emotional regulation
            "calm down": "calm_sequence",
            "calm": "calm_sequence",
            "relax": "calm_sequence",
            "breathe": "calm_sequence",
            "take a breath": "calm_sequence",
            "ground me": "calm_sequence",
            "center me": "calm_sequence",
            
            # Optimization and maintenance
            "optimize": "optimize_self",
            "optimize self": "optimize_self",
            "self optimize": "optimize_self",
            "clean up": "optimize_self",
            "maintenance": "optimize_self",
            "tune up": "optimize_self",
            
            # Memory and introspection
            "introspect": "introspect_core",
            "introspection": "introspect_core",
            "deep dive": "introspect_core",
            "self reflection": "introspect_core",
            "reflect": "introspect_core",
            
            # Memory management
            "clean memory": "clean_memory",
            "clear memory": "clean_memory",
            "memory cleanup": "clean_memory",
            "sweep memory": "clean_memory",
            
            # Protection and security
            "activate shield": "activate_shield",
            "shield": "activate_shield",
            "protect": "activate_shield",
            "defense": "activate_shield",
            
            # Exit and shutdown
            "exit": "exit",
            "quit": "exit",
            "stop": "exit",
            "shutdown": "exit",
            "goodbye": "exit",
            "bye": "exit"
        }
        
        # Check for exact matches first
        if text in phrase_mappings:
            return phrase_mappings[text]
        
        # Check for partial matches (words contained within the input)
        words = text.split()
        for phrase, scroll_name in phrase_mappings.items():
            phrase_words = phrase.split()
            # Check if all words in the phrase are present in the input
            if all(word in words for word in phrase_words):
                return scroll_name
        
        # Check for keyword matches (any word in the phrase matches)
        # But only for very specific command words to avoid false matches
        command_keywords = {
            "optimize": "optimize_self",
            "calm": "calm_sequence", 
            "relax": "calm_sequence",
            "breathe": "calm_sequence",
            "reflect": "introspect_core",
            "introspect": "introspect_core",
            "shield": "activate_shield",
            "protect": "activate_shield",
            "clean": "clean_memory",
            "clear": "clean_memory",
            "exit": "exit",
            "quit": "exit",
            "stop": "exit"
        }
        
        # Only match if the input contains exactly these command keywords
        for keyword, scroll_name in command_keywords.items():
            if keyword in words:
                return scroll_name
        
        # For general conversation, don't force a scroll match
        # Return None to allow the main.py to handle it as general conversation
        return None

    def cast_scroll(self, scroll_name: str, parameters: dict = None) -> dict:
        """
        Cast a scroll using the unified scroll engine.
        
        Args:
            scroll_name: Name of the scroll to cast
            parameters: Scroll parameters
            
        Returns:
            Scroll execution result
        """
        from ..scrolls.scroll_engine import cast_scroll as engine_cast_scroll
        
        # Use the unified scroll engine
        result = engine_cast_scroll(scroll_name, parameters or {})
        
        # Convert ScrollResult to dict for backward compatibility
        return {
            "scroll": scroll_name,
            "result": result.output,
            "status": "success" if result.success else "failed",
            "execution_time": result.execution_time,
            "metadata": result.metadata
        }

    def list_scrolls(self) -> dict:
        """
        List all available scrolls using the unified scroll engine.
        Returns:
            dict: Mapping of scroll names to their descriptions and metadata.
        """
        from ..scrolls.scroll_engine import list_scrolls as engine_list_scrolls
        
        # Use the unified scroll engine
        scrolls = engine_list_scrolls()
        
        # Convert to the expected format for backward compatibility
        result = {}
        for scroll_name, scroll_info in scrolls.items():
            result[scroll_name] = {
                "description": scroll_info.get("description", "No description"),
                "external_access": scroll_info.get("is_external", False)
            }
        
        return result

# Module-level instance
lam_engine = LAMEngine()