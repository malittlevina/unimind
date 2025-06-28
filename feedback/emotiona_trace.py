

"""
emotiona_trace.py
Tracks emotional states and transitions across daemon activities, reflections, and user interactions.
"""

import time
from datetime import datetime
from typing import List, Dict

class EmotionTrace:
    def __init__(self):
        self.trace_log: List[Dict] = []

    def log_emotion(self, source: str, emotion: str, intensity: float, context: str = ""):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "source": source,
            "emotion": emotion,
            "intensity": intensity,
            "context": context
        }
        self.trace_log.append(entry)

    def get_recent_emotions(self, limit: int = 10) -> List[Dict]:
        return self.trace_log[-limit:]

    def summarize_emotions(self) -> Dict[str, float]:
        summary = {}
        for entry in self.trace_log:
            emotion = entry["emotion"]
            intensity = entry["intensity"]
            summary[emotion] = summary.get(emotion, 0) + intensity
        return summary

    def export_trace(self) -> List[Dict]:
        return self.trace_log