"""
text_to_code.py – Natural language to code generation engine for Unimind
Provides symbolic, rule-based, and LLM-backed code synthesis from user prompts.
"""

import ast
import textwrap
from typing import Optional, Dict, Any

class SimpleLLM:
    """
    Placeholder LLM class. Replace with a real LLM integration (OpenAI, HuggingFace, etc).
    """
    def __init__(self, model_name: str = "simple-echo-llm"):
        self.model_name = model_name

    def generate_code(self, prompt: str, language: str = "python") -> str:
        """
        Simulate LLM code generation. Replace this with a real API/model call.
        """
        # For now, just echo the prompt as a comment and a dummy function
        return f"""# LLM({self.model_name}) generated code for: {prompt}\ndef dummy_function():\n    pass\n"""

class TextToCodeEngine:
    def __init__(self, llm: Optional[Any] = None):
        """
        Initialize the engine. Optionally provide an LLM or API client for advanced code generation.
        """
        self.llm = llm
        self.language_keywords = {
            "python": ["def ", "import ", "print(", "lambda ", "#"],
            "javascript": ["function ", "console.log(", "let ", "const ", "//"],
            "bash": ["#!/bin/bash", "echo ", "$", "#"],
            "sql": ["SELECT ", "INSERT ", "UPDATE ", "DELETE ", ";"],
        }

    def detect_language(self, code: str) -> str:
        """
        Detect the programming language of a code snippet.
        """
        code = code.strip()
        for lang, keywords in self.language_keywords.items():
            if any(kw in code for kw in keywords):
                return lang
        return "unknown"

    def validate_python_syntax(self, code: str) -> bool:
        """
        Validate Python code syntax using ast parsing.
        """
        try:
            ast.parse(code)
            return True
        except Exception:
            return False

    def format_code(self, code: str, language: str = "python") -> str:
        """
        Format code for readability. For Python, uses textwrap for indentation.
        """
        if language == "python":
            return textwrap.dedent(code).strip()
        # Placeholder: Add more formatters for other languages
        return code.strip()

    def rule_based_generation(self, prompt: str) -> Optional[str]:
        """
        Simple rule-based text-to-code for common tasks.
        """
        prompt = prompt.lower().strip()
        # Example rules
        if "hello world" in prompt:
            if "python" in prompt:
                return 'print("Hello, world!")'
            elif "javascript" in prompt:
                return 'console.log("Hello, world!");'
            elif "bash" in prompt:
                return 'echo "Hello, world!"'
            else:
                return 'print("Hello, world!")'
        if "fibonacci" in prompt and "python" in prompt:
            return (
                "def fibonacci(n):\n"
                "    a, b = 0, 1\n"
                "    for _ in range(n):\n"
                "        yield a\n"
                "        a, b = b, a + b\n"
            )
        if "factorial" in prompt and "python" in prompt:
            return (
                "def factorial(n):\n"
                "    if n == 0: return 1\n"
                "    return n * factorial(n-1)\n"
            )
        if "reverse a string" in prompt and "python" in prompt:
            return (
                "def reverse_string(s):\n"
                "    return s[::-1]\n"
            )
        # Add more rules as needed
        return None

    def generate_code(self, prompt: str, language: str = "python") -> Dict[str, Any]:
        """
        Main entry: Generate code from a natural language prompt.
        Tries rule-based first, then LLM if available.
        Returns a dict with code, language, and metadata.
        """
        # Try rule-based
        code = self.rule_based_generation(prompt)
        used_llm = False
        if not code and self.llm:
            code = self.llm_generate_code(prompt, language)
            used_llm = True
        if not code:
            code = f"# Unable to generate code for: {prompt}"
        formatted = self.format_code(code, language)
        return {
            "code": formatted,
            "language": language,
            "used_llm": used_llm,
            "valid_syntax": self.validate_python_syntax(formatted) if language == "python" else None
        }

    def llm_generate_code(self, prompt: str, language: str = "python") -> Optional[str]:
        """
        Placeholder for LLM-based code generation. Replace with actual LLM call.
        """
        if self.llm:
            # Example: return self.llm.generate_code(prompt, language=language)
            return None  # Not implemented
        return None

    def explain_code(self, code: str, language: str = "python") -> str:
        """
        Provide a simple explanation of what the code does (rule-based placeholder).
        """
        if language == "python":
            if "print(" in code:
                return "Prints output to the console."
            if "def fibonacci" in code:
                return "Defines a function to generate Fibonacci numbers."
            if "def factorial" in code:
                return "Defines a function to compute the factorial of a number."
            if "[::-1]" in code:
                return "Reverses a string."
        return "No explanation available."

# Module-level engine instance for convenience
engine = TextToCodeEngine()

def text_to_code(prompt: str, language: str = "python") -> Dict[str, Any]:
    """
    Generate code from text using the module-level engine.
    """
    return engine.generate_code(prompt, language)
