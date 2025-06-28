# text_to_code.py – model integration for converting natural language to source code

from models.native_models.codegen_model import CodeGenModel

class TextToCodeEngine:
    def __init__(self):
        self.model = CodeGenModel()

    def generate_code(self, prompt: str, language: str = "python") -> str:
        """
        Convert a text prompt into source code using the specified language.
        """
        code_output = self.model.infer(prompt=prompt, target_language=language)
        return code_output

# Example usage
if __name__ == "__main__":
    engine = TextToCodeEngine()
    prompt = "Create a Python function that calculates Fibonacci numbers."
    result = engine.generate_code(prompt)
    print("Generated Code:\n", result)