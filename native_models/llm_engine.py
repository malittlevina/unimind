import os
import logging
from typing import Optional, List, Dict, Any, Callable

class LLMEngine:
    """
    Unified LLM engine for Unimind. Supports OpenAI, Ollama, HuggingFace, Replicate, and simulated models.
    Provides:
    - Synchronous and streaming inference
    - Prompt templating and system messages
    - Model listing and health checks
    - Function/tool calling for advanced LLMs
    """
    def __init__(self, default_model="gpt-3.5-turbo", temperature=0.7, max_tokens=256):
        self.default_model = default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.providers = ["openai", "ollama", "huggingface", "replicate", "simulate"]
        self.available_models = {
            "openai": ["gpt-3.5-turbo", "gpt-4"],
            "ollama": ["llama3", "mistral"],
            "huggingface": ["bigscience/bloom", "gpt2"],
            "replicate": ["replicate/llama2-7b"],
            "simulate": ["sim-llm"]
        }

    def run(self, prompt: str, model_name: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None, stream: bool = False, system_message: Optional[str] = None, template: Optional[str] = None, functions: Optional[List[Dict]] = None) -> Any:
        """
        Run LLM inference with optional streaming, system message, prompt templating, and function/tool calling.
        """
        model_name = model_name or self.default_model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        provider = self._detect_provider(model_name)
        prompt = self._apply_template(prompt, template)
        try:
            if provider == "openai":
                return self._run_openai(model_name, prompt, temperature, max_tokens, stream, system_message, functions)
            elif provider == "ollama":
                return self._run_ollama(prompt, model_name, stream)
            elif provider == "huggingface":
                return self._run_huggingface(prompt, model_name, stream)
            elif provider == "replicate":
                return self._run_replicate(prompt, model_name, stream)
            else:
                return self._simulate_response(model_name, prompt, temperature, max_tokens)
        except Exception as e:
            logging.error(f"LLMEngine error: {e}")
            return f"[{model_name}] Error: {str(e)}"

    def _detect_provider(self, model_name: str) -> str:
        """Infer provider from model name."""
        if model_name.startswith("gpt-") or model_name.startswith("openai"): return "openai"
        if model_name in self.available_models.get("ollama", []): return "ollama"
        if "/" in model_name and model_name.startswith("replicate/"): return "replicate"
        if "/" in model_name: return "huggingface"
        return "simulate"

    def _apply_template(self, prompt: str, template: Optional[str]) -> str:
        """Apply a prompt template if provided."""
        if template:
            return template.format(prompt=prompt)
        return prompt

    def _run_openai(self, model_name, prompt, temperature, max_tokens, stream, system_message, functions):
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        kwargs = dict(model=model_name, messages=messages, temperature=temperature, max_tokens=max_tokens)
        if functions:
            kwargs["functions"] = functions
        if stream:
            response = openai.ChatCompletion.create(**kwargs, stream=True)
            return self._stream_openai(response)
        else:
            response = openai.ChatCompletion.create(**kwargs)
            return response.choices[0].message["content"].strip()

    def _stream_openai(self, response):
        """Yield tokens from OpenAI streaming response."""
        for chunk in response:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                yield delta["content"]

    def _run_ollama(self, prompt, model_name, stream):
        import requests
        url = "http://localhost:11434/api/generate"
        payload = {"model": model_name, "prompt": prompt, "stream": stream}
        with requests.post(url, json=payload, stream=stream) as resp:
            if stream:
                for line in resp.iter_lines():
                    if line:
                        yield line.decode()
            else:
                return resp.json().get("response", f"[ollama] No response for {model_name}")

    def _run_huggingface(self, prompt, model_name, stream):
        import requests
        api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN', '')}"}
        payload = {"inputs": prompt}
        resp = requests.post(api_url, headers=headers, json=payload, stream=stream)
        if stream:
            for line in resp.iter_lines():
                if line:
                    yield line.decode()
        else:
            return resp.json()[0]["generated_text"] if resp.ok else f"[huggingface] Error: {resp.text}"

    def _run_replicate(self, prompt, model_name, stream):
        import requests
        # Example: model_name = "replicate/llama2-7b"
        model = model_name.split("/", 1)[-1]
        api_token = os.getenv("REPLICATE_API_TOKEN", "")
        url = f"https://api.replicate.com/v1/predictions"
        headers = {"Authorization": f"Token {api_token}", "Content-Type": "application/json"}
        payload = {"version": model, "input": {"prompt": prompt}}
        resp = requests.post(url, headers=headers, json=payload, stream=stream)
        if stream:
            for line in resp.iter_lines():
                if line:
                    yield line.decode()
        else:
            return resp.json().get("output", f"[replicate] Error: {resp.text}")

    def _simulate_response(self, model_name, prompt, temperature, max_tokens):
        return f"[{model_name}] Simulated response to: {prompt} (temp={temperature}, max_tokens={max_tokens})"

    def list_models(self, provider: Optional[str] = None) -> List[str]:
        """List available models for a provider or all providers."""
        if provider:
            return self.available_models.get(provider, [])
        models = []
        for mlist in self.available_models.values():
            models.extend(mlist)
        return models

    def check_model_health(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Check if a model is available and healthy (basic check).
        """
        model_name = model_name or self.default_model
        provider = self._detect_provider(model_name)
        # For real checks, ping the API or try a dummy inference
        try:
            if provider == "openai":
                import openai
                openai.api_key = os.getenv("OPENAI_API_KEY")
                openai.Model.retrieve(model_name)
                return {"model": model_name, "status": "ok"}
            elif provider == "ollama":
                import requests
                resp = requests.get("http://localhost:11434/api/tags")
                if model_name in resp.text:
                    return {"model": model_name, "status": "ok"}
            elif provider == "huggingface":
                # Just check if model is in the list
                if model_name in self.available_models["huggingface"]:
                    return {"model": model_name, "status": "ok"}
            elif provider == "replicate":
                # Just check if model is in the list
                if model_name in self.available_models["replicate"]:
                    return {"model": model_name, "status": "ok"}
        except Exception as e:
            return {"model": model_name, "status": "error", "error": str(e)}
        return {"model": model_name, "status": "unknown"}

    def call_function(self, function: Callable, *args, **kwargs) -> Any:
        """
        Call a function/tool as part of LLM tool-calling. (Stub for advanced LLMs)
        """
        try:
            return function(*args, **kwargs)
        except Exception as e:
            logging.error(f"Function call error: {e}")
            return {"error": str(e)}

# Singleton for other modules
llm_engine = LLMEngine()

def run_llm_inference(model_name, prompt, temperature=0.7, max_tokens=256, **kwargs):
    """
    Convenience function for running LLM inference.
    Supports extra kwargs: stream, system_message, template, functions, etc.
    """
    return llm_engine.run(prompt, model_name, temperature, max_tokens, **kwargs)