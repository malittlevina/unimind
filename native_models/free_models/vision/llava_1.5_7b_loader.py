#!/usr/bin/env python3
"""
llava_1.5_7b Model Loader
Automatically generated loader for llava-hf/llava-1.5-7b-hf
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

class Llava_1_5_7BLoader:
    def __init__(self):
        self.model_id = "llava-hf/llava-1.5-7b-hf"
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self, quantization="4bit"):
        """Load the model with specified quantization"""
        try:
            # Configure quantization
            if quantization == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            else:
                bnb_config = None
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            
            print(f"✓ llava_1.5_7b loaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load llava_1.5_7b: {e}")
            return False
    
    def generate(self, prompt: str, max_length: int = 512):
        """Generate text from prompt"""
        if self.model is None:
            print("Model not loaded. Call load_model() first.")
            return None
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
            
        except Exception as e:
            print(f"Generation failed: {e}")
            return None

if __name__ == "__main__":
    # Test the loader
    loader = Llava_1_5_7BLoader()
    if loader.load_model():
        response = loader.generate("Hello, how are you?")
        print(f"Response: {response}")
