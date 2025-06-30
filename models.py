from unimind.native_models import (
    lam_engine,
    llm_engine,
    text_to_text,
    text_to_sql,
    text_to_logic,
    text_to_shell,
    emotion_classifier,
    vision_model,
    voice_model,
    text_to_3d,
    text_to_video,
    text_to_code,
    context_model,
)

class DummyModelRegistry:
    def register_all_models(self):
        pass
    def summary(self):
        return "No models registered (dummy registry)."

model_registry = DummyModelRegistry() 