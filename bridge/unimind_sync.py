import json
import os
from datetime import datetime
from core.symbolic_map import get_symbolic_state
from memory.memory_graph import MemoryGraph
from cortex.prefrontal_cortex import analyze_decision
from emotion.amygdala import emotional_context

class UnimindSynchronizer:
    def __init__(self):
        self.last_sync_time = None
        self.memory_graph = MemoryGraph()

    def sync(self):
        print("Starting Unimind synchronization...")
        symbolic_state = get_symbolic_state()
        emotional_state = emotional_context()
        decisions = analyze_decision(symbolic_state, emotional_state)

        sync_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbolic_state": symbolic_state,
            "emotional_state": emotional_state,
            "decisions": decisions
        }

        self.memory_graph.log_sync(sync_report)
        self.last_sync_time = datetime.utcnow()
        print("Unimind synchronization complete.")

    def get_last_sync_time(self):
        return self.last_sync_time
