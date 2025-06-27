"""
Persona 6 – Creative Synthesist
Specialization: Broca's Area + Wernicke's Area + Occipital Lobe
Role: Linguistic-Visual Translator and Creative Generator
"""

from unimind.perception.brocas_area import analyze_syntax
from unimind.perception.wernickes_area import interpret_meaning
from unimind.perception.occipital_lobe import visualize_concepts
from unimind.visual.video_synthesis import generate_video
from unimind.cortex.occipital_lobe.text_to_3d import generate_3d_model
from unimind.cortex.brocas_area.text_to_text import rephrase_creatively

class CreativeSynthesist:
    def __init__(self):
        self.name = "Creative Synthesist"
        self.core_functions = [
            "transform_language_into_visuals",
            "synthesize media outputs",
            "rephrase and remix narratives"
        ]

    def process_input(self, input_text):
        syntax = analyze_syntax(input_text)
        meaning = interpret_meaning(input_text)
        visuals = visualize_concepts(meaning)

        return {
            "syntax": syntax,
            "meaning": meaning,
            "visuals": visuals
        }

    def generate_outputs(self, processed):
        video = generate_video(processed["meaning"])
        model = generate_3d_model(processed["visuals"])
        remix = rephrase_creatively(processed["meaning"])

        return {
            "video": video,
            "3d_model": model,
            "creative_text": remix
        }

    def synthesize(self, input_text):
        processed = self.process_input(input_text)
        return self.generate_outputs(processed)