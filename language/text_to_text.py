# text_to_text.py model integration placeholder
# text_to_text.py
# Handles text-to-text transformation tasks using the native Unimind LLM engine

from native_models.llm_engine import run_llm_inference

def transform_text(prompt: str, temperature: float = 0.7, max_tokens: int = 256) -> str:
    """
    Transforms input text using the default Unimind LLM engine.

    Args:
        prompt (str): The input prompt to process.
        temperature (float): Sampling temperature for generation.
        max_tokens (int): Maximum number of tokens to generate.

    Returns:
        str: The generated output text.
    """
    try:
        result = run_llm_inference(
            model_name="unimind-llm",
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return result.strip()
    except Exception as e:
        return f"[Text Transformation Error] {e}"