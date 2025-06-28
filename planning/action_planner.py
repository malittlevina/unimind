import datetime
from typing import List, Dict, Any

class ActionPlanner:
    def __init__(self):
        self.plan_queue: List[Dict[str, Any]] = []
        self.completed_actions: List[Dict[str, Any]] = []

    def add_action(self, action_name: str, priority: int, metadata: Dict[str, Any]) -> None:
        action = {
            "name": action_name,
            "priority": priority,
            "metadata": metadata,
            "timestamp": datetime.datetime.now(),
        }
        self.plan_queue.append(action)
        self.plan_queue.sort(key=lambda x: x["priority"])

    def get_next_action(self) -> Dict[str, Any]:
        if self.plan_queue:
            return self.plan_queue[0]
        return {"message": "No actions in queue."}

    def mark_action_complete(self, action_name: str) -> bool:
        for i, action in enumerate(self.plan_queue):
            if action["name"] == action_name:
                self.completed_actions.append(self.plan_queue.pop(i))
                return True
        return False

    def get_plan_summary(self) -> List[str]:
        return [f"{action['name']} (Priority {action['priority']})" for action in self.plan_queue]

    def clear_all(self) -> None:
        self.plan_queue.clear()
        self.completed_actions.clear()