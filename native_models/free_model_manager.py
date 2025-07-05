#!/usr/bin/env python3
"""
Unified Free Model Manager
Manages all deployed free models for the bridge strategy
"""

import json
import torch
from pathlib import Path
from typing import Dict, Optional, List
import logging

class FreeModelManager:
    def __init__(self):
        self.base_dir = Path("/Volumes/DeveloperDrive")
        self.config_dir = self.base_dir / "configs" / "free_models"
        self.models_dir = self.base_dir / "unimind" / "native_models" / "free_models"
        
        self.loaded_models = {}
        self.model_configs = {}
        
        # Load all configurations
        self.load_configurations()
    
    def load_configurations(self):
        """Load all model configurations"""
        if self.config_dir.exists():
            for config_file in self.config_dir.glob("*.json"):
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        model_name = config_file.stem
                        self.model_configs[model_name] = config
                except Exception as e:
                    print(f"Failed to load config {config_file}: {e}")
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available models by type"""
        models_by_type = {"llm": [], "vision": [], "code": []}
        
        for model_name, config in self.model_configs.items():
            model_type = config.get("model_type", "llm")
            if model_type in models_by_type:
                models_by_type[model_type].append(model_name)
        
        return models_by_type
    
    def load_model(self, model_name: str) -> bool:
        """Load a specific model"""
        if model_name not in self.model_configs:
            print(f"Model {model_name} not found in configurations")
            return False
        
        config = self.model_configs[model_name]
        model_type = config["model_type"]
        
        # Import and load the model
        try:
            loader_path = self.models_dir / model_type / f"{model_name}_loader.py"
            
            if loader_path.exists():
                import importlib.util
                spec = importlib.util.spec_from_file_location("loader", loader_path)
                loader_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(loader_module)
                
                # Get the loader class
                loader_class_name = model_name.replace('.', '_').replace('-', '_').title() + "Loader"
                loader_class = getattr(loader_module, loader_class_name)
                
                # Create and load the model
                loader = loader_class()
                if loader.load_model():
                    self.loaded_models[model_name] = loader
                    print(f"✓ Model {model_name} loaded successfully")
                    return True
                else:
                    print(f"✗ Failed to load model {model_name}")
                    return False
            else:
                print(f"Loader not found for {model_name}")
                return False
                
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            return False
    
    def generate(self, model_name: str, prompt: str, max_length: int = 512) -> Optional[str]:
        """Generate text using a loaded model"""
        if model_name not in self.loaded_models:
            print(f"Model {model_name} not loaded. Call load_model() first.")
            return None
        
        return self.loaded_models[model_name].generate(prompt, max_length)
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a model"""
        return self.model_configs.get(model_name)
    
    def benchmark_model(self, model_name: str) -> Dict:
        """Run basic benchmarks on a model"""
        if model_name not in self.loaded_models:
            print(f"Model {model_name} not loaded")
            return {}
        
        model = self.loaded_models[model_name]
        
        # Simple benchmark tests
        test_prompts = [
            "What is the capital of France?",
            "Write a Python function to calculate fibonacci numbers:",
            "Explain quantum computing in simple terms:"
        ]
        
        results = {
            "model_name": model_name,
            "tests": []
        }
        
        for i, prompt in enumerate(test_prompts):
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_time:
                start_time.record()
            
            response = model.generate(prompt, max_length=256)
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                duration = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
            else:
                duration = 0.0
            
            results["tests"].append({
                "prompt": prompt,
                "response": response,
                "duration": duration
            })
        
        return results

if __name__ == "__main__":
    # Test the manager
    manager = FreeModelManager()
    
    print("Available models:")
    models = manager.list_available_models()
    for model_type, model_list in models.items():
        print(f"  {model_type}: {model_list}")
    
    # Load and test first available model
    if models["llm"]:
        first_model = models["llm"][0]
        print(f"
Testing {first_model}...")
        
        if manager.load_model(first_model):
            response = manager.generate(first_model, "Hello, how are you?")
            print(f"Response: {response}")
            
            # Run benchmark
            benchmark = manager.benchmark_model(first_model)
            print(f"Benchmark results: {benchmark}")
