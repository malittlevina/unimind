

import random
import json

class PromptImprover:
    def __init__(self, prompt_library=None):
        self.prompt_library = prompt_library or []
        self.improvement_strategies = [
            self.rewrite_with_context,
            self.add_constraints,
            self.use_examples
        ]

    def improve_prompt(self, original_prompt):
        improved_prompt = original_prompt
        for strategy in self.improvement_strategies:
            improved_prompt = strategy(improved_prompt)
        return improved_prompt

    def rewrite_with_context(self, prompt):
        return f"Consider the context before answering: {prompt}"

    def add_constraints(self, prompt):
        return f"{prompt} [Respond in under 100 words]"

    def use_examples(self, prompt):
        return f"{prompt} For example: 'What is the capital of France?'"

    def evaluate_prompt(self, prompt):
        return random.uniform(0, 1)  # Placeholder for real scoring model

    def suggest_best(self, prompt_variants):
        scored = [(p, self.evaluate_prompt(p)) for p in prompt_variants]
        return max(scored, key=lambda x: x[1])[0]

if __name__ == "__main__":
    improver = PromptImprover()
    test_prompt = "Explain reinforcement learning."
    improved = improver.improve_prompt(test_prompt)
    print("Original:", test_prompt)
    print("Improved:", improved)