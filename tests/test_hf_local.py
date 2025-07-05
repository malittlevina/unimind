from transformers import pipeline

def test_gpt2_local():
    generator = pipeline("text-generation", model="gpt2")
    prompt = "Once upon a time,"
    result = generator(prompt, max_length=50, num_return_sequences=1)
    print("[GPT-2]", result[0]['generated_text'])

if __name__ == "__main__":
    test_gpt2_local() 