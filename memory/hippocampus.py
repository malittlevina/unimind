"""
Hippocampus - Long-term memory processing center for the Unimind architecture.
Responsible for encoding, storing, and retrieving symbolic and contextual memory traces.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

class Hippocampus:
    def __init__(self, memory_path: str = "unimind/memory/memory_graph.json"):
        self.memory_path = memory_path
        self.memory_graph = self.load_memory_graph()
        self.initialized_at = datetime.utcnow()

    def load_memory_graph(self) -> Dict[str, Any]:
        if os.path.exists(self.memory_path):
            with open(self.memory_path, 'r') as file:
                return json.load(file)
        return {}

    def save_memory_graph(self) -> None:
        with open(self.memory_path, 'w') as file:
            json.dump(self.memory_graph, file, indent=4)

    def encode_memory(self, context: str, data: Dict[str, Any]) -> str:
        timestamp = datetime.utcnow().isoformat()
        memory_id = f"memory_{len(self.memory_graph)}"
        self.memory_graph[memory_id] = {
            "timestamp": timestamp,
            "context": context,
            "data": data
        }
        self.save_memory_graph()
        return memory_id

    def get_memory_stats(self) -> Dict[str, Any]:
        return {
            "total_memories": len(self.memory_graph),
            "last_updated": max(
                (mem["timestamp"] for mem in self.memory_graph.values()),
                default="N/A"
            ),
            "initialized_at": self.initialized_at.isoformat()
        }

    def retrieve_memory(self, query: Optional[str] = None) -> Dict[str, Any]:
        if not query:
            return self.memory_graph

        results = {}
        for memory_id, memory_data in self.memory_graph.items():
            if query.lower() in memory_data.get("context", "").lower() or \
               query.lower() in json.dumps(memory_data.get("data", {})).lower():
                results[memory_id] = memory_data
        return results

    def forget_memory(self, memory_id: str) -> bool:
        if memory_id in self.memory_graph:
            del self.memory_graph[memory_id]
            self.save_memory_graph()
            return True
        return False

    def summarize_memory(self) -> Dict[str, str]:
        return {
            mem_id: mem_data.get("context", "")[:75] + "..."
            for mem_id, mem_data in self.memory_graph.items()
        }

# For integration with broader Unimind
if __name__ == "__main__":
    hippocampus = Hippocampus()
    print("Current memory summary:", hippocampus.summarize_memory())
    print("Memory stats:", hippocampus.get_memory_stats())
