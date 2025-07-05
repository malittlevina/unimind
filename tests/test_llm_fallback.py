def test_llm_fallback():
    prompt = "What is the capital of France?"
    try:
        from unimind.native_models.llm_engine import llm_engine
        print("Trying Ollama...")
        response = llm_engine.run_with_fallback(prompt, model_name="llama3")
        print("Response:", response)
    except Exception as e:
        print("LLM fallback test failed:", e)

if __name__ == "__main__":
    test_llm_fallback() 