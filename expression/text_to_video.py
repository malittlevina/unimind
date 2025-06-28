# text_to_video.py – AI-powered video generation from symbolic and narrative prompts

import os
from unimind.models.text_to_video import VideoGenerator
from unimind.memory.hippocampus import retrieve_context
from unimind.language.lam_engine import parse_symbolic_meaning

class TextToVideoEngine:
    def __init__(self):
        self.generator = VideoGenerator()

    def generate_video(self, prompt: str, style: str = "cinematic", duration: int = 30):
        # Parse symbolic meaning from the prompt
        symbolic_context = parse_symbolic_meaning(prompt)

        # Optionally retrieve related memory cues
        memory_context = retrieve_context(prompt)

        # Combine prompt, symbolic cues, and memory for rich generation
        enriched_prompt = f"{prompt}\nSymbolism: {symbolic_context}\nMemory: {memory_context}"

        # Generate video content
        video_path = self.generator.render_video(enriched_prompt, style=style, duration=duration)
        return video_path

# Optional CLI for testing
if __name__ == "__main__":
    engine = TextToVideoEngine()
    result_path = engine.generate_video("A dreamlike journey through the pineal temple", style="surreal", duration=45)
    print(f"Generated video saved at: {result_path}")