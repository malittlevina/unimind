from transformers import pipeline

def test_dialogpt_local():
    chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")
    prompt = "Hello, how are you?"
    result = chatbot(prompt, max_length=50)
    print("[DialoGPT]", result[0]['generated_text'])

if __name__ == "__main__":
    test_dialogpt_local() 