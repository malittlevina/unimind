# Amygdala.Py


# Amygdala.Py

from typing import Dict, Any

class Amygdala:
    def __init__(self):
        self.current_emotional_state = "neutral"
        self.triggers = {
            "fear": ["danger", "threat", "unknown"],
            "joy": ["success", "discovery", "praise"],
            "sadness": ["loss", "failure", "rejection"],
            "anger": ["injustice", "betrayal", "frustration"]
        }

    def assess_stimulus(self, stimulus: str) -> str:
        for emotion, keywords in self.triggers.items():
            if any(keyword in stimulus.lower() for keyword in keywords):
                self.current_emotional_state = emotion
                return emotion
        return "neutral"

    def get_current_state(self) -> str:
        return self.current_emotional_state

    def set_state(self, state: str):
        if state in self.triggers:
            self.current_emotional_state = state

    def add_trigger(self, emotion: str, keyword: str):
        if emotion in self.triggers:
            self.triggers[emotion].append(keyword)
        else:
            self.triggers[emotion] = [keyword]