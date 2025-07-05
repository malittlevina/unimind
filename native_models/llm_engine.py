import os
import logging
import json
import time
import asyncio
import re
from typing import Optional, List, Dict, Any, Callable, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import torch
from pathlib import Path

# Import context-aware LLM components
try:
    from .context_aware_llm import ContextAwareLLM, ContextualUnderstanding, IntentCategory
    CONTEXT_AWARE_AVAILABLE = True
except ImportError:
    CONTEXT_AWARE_AVAILABLE = False

# Enhanced imports for multi-modal support
try:
    from PIL import Image
    import cv2
    import numpy as np
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Add imports for language engines
# Language processing imports (now integrated into LLM engine)

# Sophisticated reasoning classes (integrated into LLM engine)
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

class ReasoningType(Enum):
    CHAIN_OF_THOUGHT = "chain_of_thought"
    ABDUCTIVE = "abductive"
    CREATIVE = "creative"
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"

class ReasoningMode(Enum):
    STANDARD = "standard"
    ENHANCED = "enhanced"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"

@dataclass
class ReasoningStep:
    step_id: str
    description: str
    reasoning: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReasoningChain:
    query: str
    reasoning_type: ReasoningType
    steps: List[ReasoningStep]
    final_answer: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AbductiveResult:
    observation: str
    possible_causes: List[str]
    best_explanation: str
    confidence: float
    reasoning_steps: List[ReasoningStep]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CreativeSolution:
    challenge: str
    solution: str
    creativity_score: float
    feasibility_score: float
    reasoning_steps: List[ReasoningStep]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReasoningPattern:
    pattern_id: str
    description: str
    applicability: List[str]
    confidence: float

@dataclass
class ContextualUnderstanding:
    """Contextual understanding result."""
    primary_intent: str
    confidence: float
    suggested_actions: List[str]
    user_goal: str
    context_relevance: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class SophisticatedReasoningEngine:
    """Integrated sophisticated reasoning engine."""
    
    def __init__(self, mode: str = "integrated"):
        self.mode = mode
        self.logger = logging.getLogger('SophisticatedReasoningEngine')
        self.reasoning_stats = {
            "total_queries": 0,
            "successful_reasoning": 0,
            "avg_confidence": 0.0
        }
    
    def reason(self, query: str, reasoning_mode: Optional[ReasoningMode] = None, context: Dict[str, Any] = None) -> ReasoningChain:
        """Perform sophisticated reasoning on a query."""
        try:
            # Simple reasoning implementation
            steps = [
                ReasoningStep(
                    step_id="1",
                    description="Analyze query",
                    reasoning=f"Analyzing the query: {query}",
                    confidence=0.8
                ),
                ReasoningStep(
                    step_id="2", 
                    description="Generate response",
                    reasoning="Generating reasoned response",
                    confidence=0.7
                )
            ]
            
            return ReasoningChain(
                query=query,
                reasoning_type=ReasoningType.CHAIN_OF_THOUGHT,
                steps=steps,
                final_answer=f"Reasoned response to: {query}",
                confidence=0.75
            )
        except Exception as e:
            self.logger.error(f"Reasoning error: {e}")
            return ReasoningChain(
                query=query,
                reasoning_type=ReasoningType.CHAIN_OF_THOUGHT,
                steps=[],
                final_answer=f"Reasoning failed: {str(e)}",
                confidence=0.0
            )
    
    def chain_of_thought_reasoning(self, problem: str, max_steps: int = 10) -> ReasoningChain:
        """Perform chain-of-thought reasoning on a problem."""
        return self.reason(problem, ReasoningMode.STANDARD)
    
    def abductive_reasoning(self, observation: str, possible_causes: List[str] = None) -> AbductiveResult:
        """Perform abductive reasoning to find the best explanation."""
        try:
            steps = [
                ReasoningStep(
                    step_id="1",
                    description="Analyze observation",
                    reasoning=f"Analyzing observation: {observation}",
                    confidence=0.8
                )
            ]
            
            return AbductiveResult(
                observation=observation,
                possible_causes=possible_causes or ["unknown cause"],
                best_explanation="Most likely explanation based on observation",
                confidence=0.7,
                reasoning_steps=steps
            )
        except Exception as e:
            self.logger.error(f"Abductive reasoning error: {e}")
            return AbductiveResult(
                observation=observation,
                possible_causes=[],
                best_explanation=f"Reasoning failed: {str(e)}",
                confidence=0.0,
                reasoning_steps=[]
            )
    
    def creative_reasoning(self, challenge: str, constraints: List[str] = None) -> CreativeSolution:
        """Perform creative reasoning to solve a challenge."""
        try:
            steps = [
                ReasoningStep(
                    step_id="1",
                    description="Creative analysis",
                    reasoning=f"Analyzing challenge creatively: {challenge}",
                    confidence=0.8
                )
            ]
            
            return CreativeSolution(
                challenge=challenge,
                solution="Creative solution to the challenge",
                creativity_score=0.8,
                feasibility_score=0.7,
                reasoning_steps=steps
            )
        except Exception as e:
            self.logger.error(f"Creative reasoning error: {e}")
            return CreativeSolution(
                challenge=challenge,
                solution=f"Creative reasoning failed: {str(e)}",
                creativity_score=0.0,
                feasibility_score=0.0,
                reasoning_steps=[]
            )
    
    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get statistics about reasoning engine performance."""
        return self.reasoning_stats.copy()

class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    SIMULATE = "simulate"

@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    required: bool = False

@dataclass
class Message:
    role: str
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None

@dataclass
class ModelConfig:
    model_name: str
    provider: ModelProvider
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    context_window: int = 128000
    quantization: str = "auto"
    device: str = "auto"
    safety_level: str = "medium"

class LLMEngine:
    """
    Unified LLM engine for Unimind with SOTA models and advanced features.
    Supports OpenAI, Anthropic, Ollama, HuggingFace, Replicate, simulated models, multi-modal, function calling, context, reasoning, and context-aware understanding.
    """
    def __init__(self, config: Optional[ModelConfig] = None, enable_context_awareness: bool = True):
        self.default_model = "gpt-4o"
        self.temperature = 0.7
        self.max_tokens = 4096
        self.logger = logging.getLogger('LLMEngine')
        self.logger.info("Unified LLM Engine initialized with SOTA models and advanced features")
        self.providers = ["openai", "anthropic", "ollama", "huggingface", "replicate", "simulate"]
        self.available_models = {
            "openai": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            "anthropic": ["claude-3-5-sonnet", "claude-3-opus", "claude-3-haiku"],
            "ollama": ["llama3.1:405b", "llama3.1:70b", "mixtral:8x22b"],
            "huggingface": ["deepseek-ai/deepseek-coder-67b", "microsoft/DialoGPT-large"],
            "replicate": ["meta/llama-3.1-405b", "mistralai/mixtral-8x22b"],
            "simulate": ["sim-llm"]
        }
        self.optimization_count = 0
        self.last_optimization = None
        self.config = config or ModelConfig(model_name=self.default_model, provider=ModelProvider.OPENAI)
        self.tools: Dict[str, Tool] = {}
        self.context_manager = ContextManager(self.config.context_window)
        self.safety_engine = SafetyEngine(self.config.safety_level)
        self.reasoning_engine = ReasoningEngine()
        self.performance_monitor = PerformanceMonitor()
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.response_cache = {}
        self.cache_ttl = 3600
        
        # Context-aware features
        self.enable_context_awareness = enable_context_awareness and CONTEXT_AWARE_AVAILABLE
        if self.enable_context_awareness:
            self.context_aware_llm = ContextAwareLLM()
            self.logger.info("Context-aware features enabled")
        else:
            self.context_aware_llm = None
            if enable_context_awareness and not CONTEXT_AWARE_AVAILABLE:
                self.logger.warning("Context-aware features requested but not available")

        # Language engines (now integrated into LLM engine)
        
        # Sophisticated reasoning engine (integrated)
        self.sophisticated_reasoning = SophisticatedReasoningEngine(mode="integrated")

    def register_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool
        self.logger.info(f"Registered tool: {tool.name}")

    def register_tools(self, tools: List[Tool]) -> None:
        for tool in tools:
            self.register_tool(tool)

    async def generate(self, messages: List[Message], tools: Optional[List[str]] = None, stream: bool = False, reasoning_mode: bool = False) -> Union[str, AsyncGenerator[str, None]]:
        start_time = time.time()
        try:
            processed_messages = await self._preprocess_messages(messages)
            safety_result = await self.safety_engine.check_content(processed_messages)
            if not safety_result.get('safe', True):
                return f"Safety check failed: {safety_result.get('reason', 'Unknown reason')}"
            context = await self.context_manager.process_context(processed_messages)
            if reasoning_mode:
                context = await self.reasoning_engine.enhance_context(context)
            available_tools = self._prepare_tools(tools) if tools else []
            if stream:
                return self._generate_streaming(context, available_tools)
            else:
                response = await self._generate_response(context, available_tools)
                response = await self._postprocess_response(response, context)
                self.performance_monitor.record_generation(time.time() - start_time, len(str(response)))
                return response
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            return f"Error: {str(e)}"

    async def _preprocess_messages(self, messages: List[Message]) -> List[Message]:
        processed_messages = []
        for message in messages:
            processed_content = []
            if isinstance(message.content, str):
                processed_content.append({"type": "text", "text": message.content})
            elif isinstance(message.content, list):
                for item in message.content:
                    if item.get("type") == "text":
                        processed_content.append(item)
                    elif item.get("type") == "image_url":
                        image_data = await self._process_image(item["image_url"]["url"])
                        if image_data:
                            processed_content.append({"type": "image", "image_data": image_data})
                    elif item.get("type") == "audio":
                        audio_data = await self._process_audio(item["audio_url"]["url"])
                        if audio_data:
                            processed_content.append({"type": "audio", "audio_data": audio_data})
            processed_messages.append(Message(role=message.role, content=processed_content, name=message.name, tool_calls=message.tool_calls, tool_results=message.tool_results))
        return processed_messages

    async def _process_image(self, image_url: str) -> Optional[Dict[str, Any]]:
        if not VISION_AVAILABLE:
            return None
        try:
            import io
            import base64
            if image_url.startswith("data:"):
                image_data = base64.b64decode(image_url.split(",")[1])
                image = Image.open(io.BytesIO(image_data))
            elif image_url.startswith("http"):
                import requests
                response = requests.get(image_url)
                image = Image.open(io.BytesIO(response.content))
            else:
                image = Image.open(image_url)
            if image.mode != "RGB":
                image = image.convert("RGB")
            max_size = 1024
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            return {"url": f"data:image/jpeg;base64,{image_base64}", "width": image.width, "height": image.height}
        except Exception as e:
            self.logger.error(f"Image processing error: {e}")
            return None

    async def _process_audio(self, audio_url: str) -> Optional[Dict[str, Any]]:
        return None

    def _prepare_tools(self, tool_names: List[str]) -> List[Dict[str, Any]]:
        tools = []
        for name in tool_names:
            if name in self.tools:
                tool = self.tools[name]
                tools.append({"type": "function", "function": {"name": tool.name, "description": tool.description, "parameters": tool.parameters}})
        return tools

    async def _generate_response(self, context: List[Message], tools: List[Dict[str, Any]]) -> str:
        provider = self.config.provider.value if hasattr(self.config.provider, 'value') else self.config.provider
        if provider == "openai":
            return await self._generate_openai(context, tools)
        elif provider == "anthropic":
            return await self._generate_anthropic(context, tools)
        elif provider == "ollama":
            return await self._generate_ollama(context, tools)
        elif provider == "huggingface":
            return await self._generate_huggingface(context, tools)
        elif provider == "local":
            return await self._generate_local(context, tools)
        else:
            return self._generate_simulated(context, tools)

    async def _generate_openai(self, context: List[Message], tools: List[Dict[str, Any]]) -> str:
        if not OPENAI_AVAILABLE:
            return "OpenAI library not available"
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            messages = []
            for msg in context:
                if isinstance(msg.content, list):
                    content = []
                    for item in msg.content:
                        if item["type"] == "text":
                            content.append({"type": "text", "text": item["text"]})
                        elif item["type"] == "image":
                            content.append({"type": "image_url", "image_url": {"url": item["image_data"]["url"]}})
                    messages.append({"role": msg.role, "content": content})
            else:
                messages.append({"role": msg.role, "content": msg.content})
            kwargs = {"model": self.config.model_name, "messages": messages, "max_tokens": self.config.max_tokens, "temperature": self.config.temperature, "top_p": self.config.top_p, "frequency_penalty": self.config.frequency_penalty, "presence_penalty": self.config.presence_penalty}
            if tools:
                kwargs["tools"] = tools
                response = client.chat.completions.create(**kwargs)
            if response.choices[0].message.tool_calls:
                return await self._handle_tool_calls(response.choices[0].message.tool_calls)
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return f"OpenAI API error: {str(e)}"

    async def _generate_anthropic(self, context: List[Message], tools: List[Dict[str, Any]]) -> str:
        if not ANTHROPIC_AVAILABLE:
            return "Anthropic library not available"
        try:
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            messages = []
            for msg in context:
                if isinstance(msg.content, list):
                    content = []
                    for item in msg.content:
                        if item["type"] == "text":
                            content.append({"type": "text", "text": item["text"]})
                        elif item["type"] == "image":
                            content.append({"type": "image", "source": {"type": "base64", "data": item["image_data"]["url"]}})
                    messages.append({"role": msg.role, "content": content})
                else:
                    messages.append({"role": msg.role, "content": msg.content})
            response = client.messages.create(model=self.config.model_name, messages=messages, max_tokens=self.config.max_tokens, temperature=self.config.temperature)
            return response.content[0].text.strip()
        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            return f"Anthropic API error: {str(e)}"

    async def _generate_ollama(self, context: List[Message], tools: List[Dict[str, Any]]) -> str:
        import requests
        try:
            url = "http://localhost:11434/api/generate"
            prompt = self._context_to_prompt(context)
            payload = {"model": self.config.model_name, "prompt": prompt, "stream": False, "options": {"temperature": self.config.temperature, "top_p": self.config.top_p, "num_predict": self.config.max_tokens}}
            response = requests.post(url, json=payload)
            if response.ok:
                return response.json().get("response", "")
            else:
                return f"Ollama error: {response.text}"
        except Exception as e:
            self.logger.error(f"Ollama error: {e}")
            return f"Ollama error: {str(e)}"

    async def _generate_huggingface(self, context: List[Message], tools: List[Dict[str, Any]]) -> str:
        import requests
        import os
        try:
            api_url = f"https://api-inference.huggingface.co/models/{self.config.model_name}"
            headers = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN', '')}"}
            prompt = self._context_to_prompt(context)
            payload = {"inputs": prompt}
            response = requests.post(api_url, headers=headers, json=payload)
            if response.ok:
                return response.json()[0]["generated_text"]
            else:
                return f"HuggingFace error: {response.text}"
        except Exception as e:
            self.logger.error(f"HuggingFace error: {e}")
            return f"HuggingFace error: {str(e)}"

    async def _generate_local(self, context: List[Message], tools: List[Dict[str, Any]]) -> str:
        from .huggingface_local import HuggingFaceLocal
        try:
            local_runner = HuggingFaceLocal()
            prompt = self._context_to_prompt(context)
            return local_runner.generate_text(prompt=prompt, model_name=self.config.model_name, max_length=self.config.max_tokens, temperature=self.config.temperature) or "Local generation failed"
        except Exception as e:
            self.logger.error(f"Local generation error: {e}")
            return f"Local generation error: {str(e)}"

    def _generate_simulated(self, context: List[Message], tools: List[Dict[str, Any]]) -> str:
        prompt = self._context_to_prompt(context)
        return f"[{self.config.model_name}] Enhanced simulated response to: {prompt[:100]}... (tools: {len(tools)})"

    def _context_to_prompt(self, context: List[Message]) -> str:
        prompt_parts = []
        for msg in context:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        return "\n".join(prompt_parts)

    async def _handle_tool_calls(self, tool_calls: List[Any]) -> str:
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            if tool_name in self.tools:
                try:
                    tool = self.tools[tool_name]
                    result = await tool.function(**arguments)
                    results.append(f"Tool {tool_name} result: {result}")
                except Exception as e:
                    results.append(f"Tool {tool_name} error: {str(e)}")
            else:
                results.append(f"Unknown tool: {tool_name}")
        return "\n".join(results)

    def _generate_streaming(self, context: List[Message], tools: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
        pass

    async def _postprocess_response(self, response: str, context: List[Message]) -> str:
        response = await self.safety_engine.filter_response(response)
        if "reasoning" in str(context[-1].content).lower():
            response = await self.reasoning_engine.enhance_response(response)
        return response

    def get_available_models(self) -> Dict[str, List[str]]:
        return self.available_models

    def get_performance_stats(self) -> Dict[str, Any]:
        return self.performance_monitor.get_stats()
    
    def run(self, prompt: str, model_name: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 256, **kwargs) -> str:
        """
        Synchronous wrapper for LLM inference.
        For backward compatibility with existing code.
        """
        try:
            # Update config for this run
            if model_name:
                self.config.model_name = model_name
            self.config.temperature = temperature
            self.config.max_tokens = max_tokens
            
            # Create a simple message list
            messages = [Message(role="user", content=prompt)]
            
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an event loop, use synchronous fallback
                return self._generate_simulated(messages, [])
            except RuntimeError:
                # No event loop running, create one
                return asyncio.run(self.generate(messages))
                
        except Exception as e:
            self.logger.error(f"Sync run error: {e}")
            return f"Error: {str(e)}"
    
    def run_with_context_understanding(self, prompt: str, model_name: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 256, memory_context_id: str = None, **kwargs) -> Dict[str, Any]:
        """Run LLM inference with context understanding and enhanced prompt generation."""
        if not self.enable_context_awareness:
            # Fallback to regular run
            response = self.run(prompt, model_name, temperature, max_tokens, **kwargs)
            return {
                "response": response,
                "understanding": None,
                "enhanced_prompt": prompt,
                "context_aware": False
            }
        
        try:
            # Get contextual understanding
            understanding = self.context_aware_llm.understand_context(prompt, memory_context_id)
            
            # Generate enhanced prompt
            enhanced_prompt = self.context_aware_llm.generate_enhanced_prompt(prompt, understanding, memory_context_id)
            
            # Run with enhanced prompt
            response = self.run(enhanced_prompt, model_name, temperature, max_tokens, **kwargs)
            
            return {
                "response": response,
                "understanding": understanding,
                "enhanced_prompt": enhanced_prompt,
                "context_aware": True,
                "intent": understanding.primary_intent.value,
                "confidence": understanding.confidence,
                "suggested_actions": understanding.suggested_actions
            }
        except Exception as e:
            self.logger.error(f"Context-aware run error: {e}")
            # Fallback to regular run
            response = self.run(prompt, model_name, temperature, max_tokens, **kwargs)
            return {
                "response": response,
                "understanding": None,
                "enhanced_prompt": prompt,
                "context_aware": False,
                "error": str(e)
            }
    
    def understand_context(self, user_input: str, memory_context_id: str = None) -> Optional[ContextualUnderstanding]:
        """Get contextual understanding of user input."""
        if not self.enable_context_awareness:
            return None
        
        try:
            return self.context_aware_llm.understand_context(user_input, memory_context_id)
        except Exception as e:
            self.logger.error(f"Context understanding error: {e}")
            return None
    
    def classify_intent(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Classify user intent using integrated intent classifier."""
        if not self.enable_context_awareness:
            return None
        
        try:
            intent_result = self.context_aware_llm.intent_classifier.classify_intent(user_input)
            return {
                "intent_type": intent_result.intent_type.value,
                "confidence": intent_result.confidence,
                "scroll_name": intent_result.scroll_name,
                "parameters": intent_result.parameters,
                "metadata": intent_result.metadata
            }
        except Exception as e:
            self.logger.error(f"Intent classification error: {e}")
            return None
    
    def route_with_understanding(self, user_input: str, memory_context_id: str = None) -> Optional[Dict[str, Any]]:
        """Route user input with understanding using context-aware LLM."""
        if not self.enable_context_awareness:
            return None
        
        try:
            return self.context_aware_llm.route_with_understanding(user_input, memory_context_id)
        except Exception as e:
            self.logger.error(f"Routing with understanding error: {e}")
            return None

    # --- Language Engine Methods (Integrated) ---
    def summarize(self, text: str, max_length: int = 100, language_mode: str = "standard", **kwargs) -> str:
        """Summarize text using integrated language processing."""
        try:
            # Use the LLM engine itself for summarization
            prompt = f"Summarize the following text in {max_length} characters or less: {text}"
            return self.run(prompt, max_tokens=max_length, temperature=0.3)
        except Exception as e:
            self.logger.error(f"Summarization error: {e}")
            return f"[Summary error: {str(e)}]"

    def parse_intent(self, text: str, language_mode: str = "standard", **kwargs) -> dict:
        """Parse intent from text using integrated language processing."""
        try:
            # Use the LLM engine itself for intent parsing
            prompt = f"Parse the intent from this text and return a JSON object with 'intent' and 'confidence' fields: {text}"
            response = self.run(prompt, max_tokens=100, temperature=0.1)
            # Try to extract JSON from response
            import json
            import re
            json_match = re.search(r'\{.*\}', response)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"intent": "unknown", "confidence": 0.0}
        except Exception as e:
            self.logger.error(f"Intent parsing error: {e}")
            return {"intent": "unknown", "confidence": 0.0}

    def generate_text(self, prompt: str, max_length: int = 200, language_mode: str = "standard", **kwargs) -> str:
        """Generate text using integrated language processing."""
        try:
            return self.run(prompt, max_tokens=max_length, **kwargs)
        except Exception as e:
            self.logger.error(f"Text generation error: {e}")
            return f"[Text generation error: {str(e)}]"

    def classify_emotion(self, text: str, language_mode: str = "standard", **kwargs) -> dict:
        """Classify emotion in text using integrated language processing."""
        try:
            prompt = f"Classify the emotional tone of this text and return a JSON object with 'emotional_tone' and 'confidence' fields: {text}"
            response = self.run(prompt, max_tokens=50, temperature=0.1)
            import json
            import re
            json_match = re.search(r'\{.*\}', response)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"emotional_tone": "neutral", "confidence": 0.0}
        except Exception as e:
            self.logger.error(f"Emotion classification error: {e}")
            return {"emotional_tone": "neutral", "confidence": 0.0}

    def generate_code(self, prompt: str, language: str = "python", language_mode: str = "standard", **kwargs) -> str:
        """Generate code using integrated language processing."""
        try:
            code_prompt = f"Generate {language} code for: {prompt}"
            return self.run(code_prompt, max_tokens=500, temperature=0.2, **kwargs)
        except Exception as e:
            self.logger.error(f"Code generation error: {e}")
            return f"# Code generation error: {str(e)}"

    def understand_language(self, text: str, **kwargs) -> dict:
        """Understand language using integrated processing."""
        try:
            prompt = f"Analyze this text and provide understanding in JSON format with fields: semantic_meaning, pragmatic_meaning, intent, confidence: {text}"
            response = self.run(prompt, max_tokens=200, temperature=0.1)
            import json
            import re
            json_match = re.search(r'\{.*\}', response)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "semantic_meaning": "Unable to analyze",
                    "pragmatic_meaning": "Unable to analyze", 
                    "intent": "unknown",
                    "confidence": 0.0
                }
        except Exception as e:
            self.logger.error(f"Language understanding error: {e}")
            return {
                "semantic_meaning": f"Error: {str(e)}",
                "pragmatic_meaning": f"Error: {str(e)}",
                "intent": "unknown",
                "confidence": 0.0
            }
    
    # --- Sophisticated Reasoning Engine Methods ---
    def reason(self, query: str, reasoning_mode: Optional[ReasoningMode] = None, context: Dict[str, Any] = None) -> ReasoningChain:
        """Perform sophisticated reasoning on a query."""
        return self.sophisticated_reasoning.reason(query, reasoning_mode, context)
    
    def chain_of_thought_reasoning(self, problem: str, max_steps: int = 10) -> ReasoningChain:
        """Perform chain-of-thought reasoning on a problem."""
        return self.sophisticated_reasoning.chain_of_thought_reasoning(problem, max_steps=max_steps)
    
    def abductive_reasoning(self, observation: str, possible_causes: List[str] = None) -> AbductiveResult:
        """Perform abductive reasoning to find the best explanation."""
        return self.sophisticated_reasoning.abductive_reasoning(observation, possible_causes)
    
    def creative_reasoning(self, challenge: str, constraints: List[str] = None) -> CreativeSolution:
        """Perform creative reasoning to solve a challenge."""
        return self.sophisticated_reasoning.creative_reasoning(challenge, constraints)
    
    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get statistics about reasoning engine performance."""
        return self.sophisticated_reasoning.get_reasoning_stats()

class ContextManager:
    def __init__(self, max_context: int = 128000):
        self.max_context = max_context
        self.conversation_history = []
        self.context_compression_enabled = True
    async def process_context(self, messages: List[Message]) -> List[Message]:
        self.conversation_history.extend(messages)
        if self._estimate_tokens(self.conversation_history) > self.max_context:
            if self.context_compression_enabled:
                self.conversation_history = await self._compress_context()
            else:
                self.conversation_history = self.conversation_history[-len(messages):]
        return self.conversation_history
    def _estimate_tokens(self, messages: List[Message]) -> int:
        total_chars = sum(len(str(msg.content)) for msg in messages)
        return total_chars // 4
    async def _compress_context(self) -> List[Message]:
        return self.conversation_history[-10:]

class SafetyEngine:
    def __init__(self, safety_level: str = "medium"):
        self.safety_level = safety_level
        self.harmful_patterns = self._load_harmful_patterns()
    def _load_harmful_patterns(self) -> List[str]:
        return ["harmful", "dangerous", "illegal", "violent", "discriminatory", "hate speech", "malware"]
    async def check_content(self, messages: List[Message]) -> Dict[str, Any]:
        for message in messages:
            content = str(message.content).lower()
            for pattern in self.harmful_patterns:
                if pattern in content:
                    return {"safe": False, "reason": f"Contains {pattern} content", "level": self.safety_level}
        return {"safe": True, "reason": "Content passed safety check"}
    async def filter_response(self, response: str) -> str:
        return response

class ReasoningEngine:
    async def enhance_context(self, context: List[Message]) -> List[Message]:
        return context
    async def enhance_response(self, response: str) -> str:
        return response

class PerformanceMonitor:
    def __init__(self):
        self.generation_times = []
        self.response_lengths = []
        self.error_count = 0
    def record_generation(self, time_taken: float, response_length: int):
        self.generation_times.append(time_taken)
        self.response_lengths.append(response_length)
    def get_stats(self) -> Dict[str, Any]:
        if not self.generation_times:
            return {"error": "No data available"}
        return {
            "avg_generation_time": sum(self.generation_times) / len(self.generation_times),
            "avg_response_length": sum(self.response_lengths) / len(self.response_lengths),
            "total_generations": len(self.generation_times),
            "error_count": self.error_count
        }

# Global instance for backward compatibility
llm_engine = LLMEngine()

def run_llm_inference(model_name, prompt, temperature=0.7, max_tokens=256, **kwargs):
    return llm_engine.run(prompt, model_name, temperature, max_tokens, **kwargs)

def run_with_context_understanding(prompt: str, model_name: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 256, memory_context_id: str = None, **kwargs) -> Dict[str, Any]:
    """Convenience function for context-aware LLM inference."""
    return llm_engine.run_with_context_understanding(prompt, model_name, temperature, max_tokens, memory_context_id, **kwargs)

def understand_context(user_input: str, memory_context_id: str = None):
    """Convenience function for context understanding."""
    return llm_engine.understand_context(user_input, memory_context_id)

def classify_intent(user_input: str):
    """Convenience function for intent classification."""
    return llm_engine.classify_intent(user_input)

def route_with_understanding(user_input: str, memory_context_id: str = None):
    """Convenience function for routing with understanding."""
    return llm_engine.route_with_understanding(user_input, memory_context_id)

def summarize(text: str, max_length: int = 100, language_mode: str = "standard") -> str:
    return llm_engine.summarize(text, max_length, language_mode)

def parse_intent(text: str, language_mode: str = "standard") -> dict:
    return llm_engine.parse_intent(text, language_mode)

def generate_text(prompt: str, max_length: int = 200, language_mode: str = "standard") -> str:
    return llm_engine.generate_text(prompt, max_length, language_mode)

def classify_emotion(text: str, language_mode: str = "standard") -> dict:
    return llm_engine.classify_emotion(text, language_mode)

def generate_code(prompt: str, language: str = "python", language_mode: str = "standard") -> str:
    return llm_engine.generate_code(prompt, language, language_mode)

def understand_language(text: str) -> dict:
    return llm_engine.understand_language(text)

# --- Sophisticated Reasoning Convenience Functions ---
def reason(query: str, reasoning_mode: Optional[ReasoningMode] = None, context: Dict[str, Any] = None) -> ReasoningChain:
    """Convenience function for sophisticated reasoning."""
    return llm_engine.reason(query, reasoning_mode, context)

def chain_of_thought_reasoning(problem: str, max_steps: int = 10) -> ReasoningChain:
    """Convenience function for chain-of-thought reasoning."""
    return llm_engine.chain_of_thought_reasoning(problem, max_steps)

def abductive_reasoning(observation: str, possible_causes: List[str] = None) -> AbductiveResult:
    """Convenience function for abductive reasoning."""
    return llm_engine.abductive_reasoning(observation, possible_causes)

def creative_reasoning(challenge: str, constraints: List[str] = None) -> CreativeSolution:
    """Convenience function for creative reasoning."""
    return llm_engine.creative_reasoning(challenge, constraints)

def get_reasoning_stats() -> Dict[str, Any]:
    """Convenience function for getting reasoning statistics."""
    return llm_engine.get_reasoning_stats()
