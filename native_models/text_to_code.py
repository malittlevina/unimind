"""
text_to_code.py – Natural language to code generation engine for Unimind
Provides symbolic, rule-based, and LLM-backed code synthesis from user prompts.
"""

import ast
import textwrap
from typing import Optional, Dict, Any

# Import SOTA model loaders
try:
    from unimind.native_models.free_models.code.codellama_7b_loader import Codellama_7BLoader
except ImportError:
    Codellama_7BLoader = None
try:
    from unimind.native_models.free_models.code.deepseek_coder_7b_loader import Deepseek_Coder_7BLoader
except ImportError:
    Deepseek_Coder_7BLoader = None

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
    """
    Unified Text-to-Code Engine supporting rule-based, SimpleLLM, and SOTA model backends.
    Backends: 'rule-based', 'simplellm', 'codellama', 'deepseek'
    """
    def __init__(self, backend: str = "rule-based", quantization: str = "4bit"):
        self.backend = backend
        self.quantization = quantization
        self.llm = SimpleLLM()
        self.language_keywords = {
            "python": ["def ", "import ", "print(", "lambda ", "#"],
            "javascript": ["function ", "console.log(", "let ", "const ", "//"],
            "bash": ["#!/bin/bash", "echo ", "$", "#"],
            "sql": ["SELECT ", "INSERT ", "UPDATE ", "DELETE ", ";"],
        }
        # SOTA model instances
        self.codellama = None
        self.deepseek = None
        if backend == "codellama" and Codellama_7BLoader:
            self.codellama = Codellama_7BLoader()
            self.codellama.load_model(quantization=quantization)
        elif backend == "deepseek" and Deepseek_Coder_7BLoader:
            self.deepseek = Deepseek_Coder_7BLoader()
            self.deepseek.load_model(quantization=quantization)

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
        
        # Self-modification and identity update rules
        if any(keyword in prompt for keyword in ["update identity", "modify identity", "change identity", "edit soul"]):
            return self._generate_identity_update_code(prompt)
        
        if any(keyword in prompt for keyword in ["update version", "upgrade version", "bump version"]):
            return self._generate_version_update_code(prompt)
        
        if any(keyword in prompt for keyword in ["add trait", "modify trait", "change personality"]):
            return self._generate_trait_update_code(prompt)
        
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
    
    def _generate_identity_update_code(self, prompt: str) -> str:
        """Generate code for identity updates."""
        return '''import json
import os
from pathlib import Path

def update_daemon_identity(user_id: str, updates: dict):
    """
    Update the daemon identity for a specific user.
    
    Args:
        user_id: The user ID to update
        updates: Dictionary of identity fields to update
    """
    # Path to the soul profile
    profile_path = Path(f"unimind/soul/soul_profiles/{user_id}.json")
    
    if not profile_path.exists():
        print(f"Profile not found: {profile_path}")
        return False
    
    try:
        # Load current profile
        with open(profile_path, 'r') as f:
            profile_data = json.load(f)
        
        # Update daemon identity
        if "daemon_identity" in profile_data:
            profile_data["daemon_identity"].update(updates)
        else:
            profile_data["daemon_identity"] = updates
        
        # Save updated profile
        with open(profile_path, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        print(f"✅ Identity updated for user: {user_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating identity: {e}")
        return False

# Example usage:
# updates = {"version": "1.0.0", "description": "Updated description"}
# update_daemon_identity("malittlevina", updates)
'''
    
    def _generate_version_update_code(self, prompt: str) -> str:
        """Generate code for version updates."""
        return '''import json
import os
from pathlib import Path

def update_daemon_version(user_id: str, new_version: str):
    """
    Update the daemon version for a specific user.
    
    Args:
        user_id: The user ID to update
        new_version: New version string (e.g., "1.0.0")
    """
    profile_path = Path(f"unimind/soul/soul_profiles/{user_id}.json")
    
    if not profile_path.exists():
        print(f"Profile not found: {profile_path}")
        return False
    
    try:
        # Load current profile
        with open(profile_path, 'r') as f:
            profile_data = json.load(f)
        
        # Update version
        if "daemon_identity" in profile_data:
            profile_data["daemon_identity"]["version"] = new_version
        else:
            profile_data["daemon_identity"] = {"version": new_version}
        
        # Save updated profile
        with open(profile_path, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        print(f"✅ Version updated to {new_version} for user: {user_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating version: {e}")
        return False

# Example usage:
# update_daemon_version("malittlevina", "1.0.0")
'''
    
    def _generate_trait_update_code(self, prompt: str) -> str:
        """Generate code for trait updates."""
        return '''import json
import os
from pathlib import Path

def update_daemon_traits(user_id: str, new_traits: list):
    """
    Update the daemon personality traits for a specific user.
    
    Args:
        user_id: The user ID to update
        new_traits: List of new personality traits
    """
    profile_path = Path(f"unimind/soul/soul_profiles/{user_id}.json")
    
    if not profile_path.exists():
        print(f"Profile not found: {profile_path}")
        return False
    
    try:
        # Load current profile
        with open(profile_path, 'r') as f:
            profile_data = json.load(f)
        
        # Update personality traits
        if "daemon_identity" in profile_data:
            profile_data["daemon_identity"]["personality_traits"] = new_traits
        else:
            profile_data["daemon_identity"] = {"personality_traits": new_traits}
        
        # Save updated profile
        with open(profile_path, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        print(f"✅ Traits updated for user: {user_id}")
        print(f"New traits: {', '.join(new_traits)}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating traits: {e}")
        return False

# Example usage:
# new_traits = ["wise and knowledgeable", "innovative and creative", "protective and ethical"]
# update_daemon_traits("malittlevina", new_traits)
'''

    def generate_code(self, prompt: str, language: str = "python", backend: Optional[str] = None, max_length: int = 512) -> Dict[str, Any]:
        """
        Generate code from a natural language prompt using the selected backend.
        """
        backend = backend or self.backend
        code = None
        used_backend = backend
        error = None
        # SOTA: CodeLlama
        if backend == "codellama" and self.codellama:
            code = self.codellama.generate(prompt, max_length=max_length)
            if code is None:
                error = "CodeLlama generation failed."
        # SOTA: DeepSeek Coder
        elif backend == "deepseek" and self.deepseek:
            code = self.deepseek.generate(prompt, max_length=max_length)
            if code is None:
                error = "DeepSeek Coder generation failed."
        # Rule-based
        elif backend == "rule-based":
            code = self.rule_based_generation(prompt)
            if code is None:
                error = "No rule-based match."
        # SimpleLLM fallback
        else:
            code = self.llm.generate_code(prompt, language=language)
            used_backend = "simplellm"
            if code is None:
                error = "SimpleLLM generation failed."
        return {
            "code": code,
            "backend": used_backend,
            "error": error
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

# Export the engine instance with the expected name
text_to_code_engine = engine

def text_to_code(prompt: str, language: str = "python", backend: str = "rule-based", max_length: int = 512) -> Dict[str, Any]:
    """
    Module-level function for unified text-to-code generation.
    Args:
        prompt: Natural language prompt
        language: Target programming language
        backend: 'rule-based', 'simplellm', 'codellama', 'deepseek'
        max_length: Max tokens for SOTA models
    Returns:
        Dict with 'code', 'backend', and 'error' (if any)
    """
    engine = TextToCodeEngine(backend=backend)
    return engine.generate_code(prompt, language=language, backend=backend, max_length=max_length)
