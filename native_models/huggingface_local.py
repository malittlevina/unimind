"""
huggingface_local.py – Local HuggingFace model integration for Unimind.
Runs models locally without requiring API keys or internet connection.
"""

import logging
import os
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

@dataclass
class LocalModelConfig:
    """Configuration for a local HuggingFace model."""
    model_name: str
    model_path: str
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = False

class HuggingFaceLocal:
    """
    Local HuggingFace model runner for Unimind.
    Downloads and runs models locally without API keys.
    """
    
    def __init__(self):
        """Initialize the local HuggingFace runner."""
        self.logger = logging.getLogger('HuggingFaceLocal')
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.pipelines: Dict[str, Any] = {}
        
        # Default models that work well locally
        self.default_models = {
            "gpt2": "gpt2",
            "gpt2-medium": "gpt2-medium", 
            "gpt2-large": "gpt2-large",
            "distilgpt2": "distilgpt2",
            "microsoft/DialoGPT-medium": "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large": "microsoft/DialoGPT-large",
            "EleutherAI/gpt-neo-125M": "EleutherAI/gpt-neo-125M",
            "EleutherAI/gpt-neo-1.3B": "EleutherAI/gpt-neo-1.3B",
            "facebook/opt-125m": "facebook/opt-125m",
            "facebook/opt-350m": "facebook/opt-350m",
            "microsoft/DialoGPT-small": "microsoft/DialoGPT-small"
        }
        
        # Model cache directory
        self.cache_dir = os.path.expanduser("~/.cache/huggingface/unimind")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.logger.info(f"HuggingFace Local initialized with cache dir: {self.cache_dir}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available local models."""
        return list(self.default_models.keys())
    
    def load_model(self, model_name: str, config: Optional[LocalModelConfig] = None) -> bool:
        """
        Load a HuggingFace model locally.
        
        Args:
            model_name: Name of the model to load
            config: Optional configuration for the model
            
        Returns:
            True if model loaded successfully
        """
        try:
            if model_name in self.models:
                self.logger.info(f"Model {model_name} already loaded")
                return True
            
            # Use default config if none provided
            if config is None:
                config = LocalModelConfig(
                    model_name=model_name,
                    model_path=model_name
                )
            
            self.logger.info(f"Loading model: {model_name}")
            
            # Determine device
            device = config.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.logger.info(f"Using device: {device}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                local_files_only=False  # Allow downloading if not cached
            )
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                "cache_dir": self.cache_dir,
                "local_files_only": False
            }
            
            # Add quantization if requested
            if config.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            elif config.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Move to device
            model = model.to(device)
            
            # Create pipeline for easier text generation
            text_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Store components
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            self.pipelines[model_name] = text_pipeline
            
            self.logger.info(f"Successfully loaded model: {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def generate_text(self, prompt: str, model_name: str = "gpt2", 
                     max_length: int = 100, temperature: float = 0.7,
                     top_p: float = 0.9, timeout: float = 30.0) -> Optional[str]:
        """
        Generate text using a local model.
        
        Args:
            prompt: Input prompt
            model_name: Name of the model to use
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            timeout: Timeout in seconds
            
        Returns:
            Generated text or None if failed
        """
        try:
            # Load model if not already loaded
            if model_name not in self.pipelines:
                if not self.load_model(model_name):
                    return None
            
            pipeline = self.pipelines[model_name]
            
            # Generate text
            start_time = time.time()
            
            outputs = pipeline(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=pipeline.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            generation_time = time.time() - start_time
            
            if outputs and len(outputs) > 0:
                generated_text = outputs[0]['generated_text']
                self.logger.info(f"Generated text in {generation_time:.2f}s")
                return generated_text.strip()
            else:
                self.logger.warning("No text generated")
                return None
                
        except Exception as e:
            self.logger.error(f"Text generation failed: {e}")
            return None
    
    def chat_generate(self, messages: List[Dict[str, str]], model_name: str = "microsoft/DialoGPT-medium",
                     max_length: int = 150, temperature: float = 0.7) -> Optional[str]:
        """
        Generate chat responses using a conversational model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model_name: Name of the conversational model
            max_length: Maximum length of response
            temperature: Sampling temperature
            
        Returns:
            Generated response or None if failed
        """
        try:
            # Load model if not already loaded
            if model_name not in self.pipelines:
                if not self.load_model(model_name):
                    return None
            
            # Build conversation context
            conversation = ""
            for message in messages:
                role = message.get('role', 'user')
                content = message.get('content', '')
                if role == 'user':
                    conversation += f"User: {content}\n"
                elif role == 'assistant':
                    conversation += f"Assistant: {content}\n"
            
            # Add prompt for next response
            conversation += "Assistant: "
            
            # Generate response
            response = self.generate_text(
                prompt=conversation,
                model_name=model_name,
                max_length=max_length,
                temperature=temperature
            )
            
            if response:
                # Clean up the response (remove the prompt part)
                if "Assistant: " in response:
                    response = response.split("Assistant: ")[-1]
                
                return response.strip()
            
            return None
            
        except Exception as e:
            self.logger.error(f"Chat generation failed: {e}")
            return None
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model to free memory."""
        try:
            if model_name in self.models:
                del self.models[model_name]
                del self.tokenizers[model_name]
                del self.pipelines[model_name]
                
                # Force garbage collection
                import gc
                gc.collect()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.logger.info(f"Unloaded model: {model_name}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to unload model {model_name}: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a loaded model."""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        info = {
            "model_name": model_name,
            "device": str(next(model.parameters()).device),
            "vocab_size": tokenizer.vocab_size,
            "max_length": tokenizer.model_max_length,
            "model_type": type(model).__name__,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        return info
    
    def optimize(self) -> Dict[str, Any]:
        """Optimize the local model system."""
        self.logger.info("Optimizing HuggingFace Local system")
        
        optimizations = {
            "models_loaded": len(self.models),
            "cache_dir": self.cache_dir,
            "available_models": len(self.default_models),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "cuda_available": torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            optimizations["cuda_memory_allocated"] = torch.cuda.memory_allocated()
            optimizations["cuda_memory_reserved"] = torch.cuda.memory_reserved()
        
        self.logger.info(f"HuggingFace Local optimization complete: {optimizations}")
        return optimizations

# Global instance
huggingface_local = HuggingFaceLocal()

def load_local_model(model_name: str, config: Optional[LocalModelConfig] = None) -> bool:
    """Load a local HuggingFace model using the global instance."""
    return huggingface_local.load_model(model_name, config)

def generate_local_text(prompt: str, model_name: str = "gpt2", **kwargs) -> Optional[str]:
    """Generate text using a local HuggingFace model."""
    return huggingface_local.generate_text(prompt, model_name, **kwargs)

def chat_local_generate(messages: List[Dict[str, str]], model_name: str = "microsoft/DialoGPT-medium", **kwargs) -> Optional[str]:
    """Generate chat responses using a local conversational model."""
    return huggingface_local.chat_generate(messages, model_name, **kwargs)

def get_available_local_models() -> List[str]:
    """Get list of available local models."""
    return huggingface_local.get_available_models()

def optimize_local_models() -> Dict[str, Any]:
    """Optimize the local model system."""
    return huggingface_local.optimize() 

# Export the engine instance with the expected name
huggingface_local_engine = huggingface_local 