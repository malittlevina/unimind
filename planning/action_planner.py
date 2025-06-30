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

    def plan_evaluation(self) -> Dict[str, Any]:
        """
        Evaluates the current action plan and returns basic metrics:
        - Total pending actions
        - Total completed actions
        - Highest priority action
        """
        evaluation = {
            "total_pending": len(self.plan_queue),
            "total_completed": len(self.completed_actions),
            "highest_priority_action": None
        }
        if self.plan_queue:
            evaluation["highest_priority_action"] = self.plan_queue[0]["name"]
        return evaluation
    def plan_action(self, action_name: str, priority: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Quickly defines and queues a new action using standard or dynamic metadata.

        Args:
            action_name (str): The name of the action to plan.
            priority (int): The action's priority (lower is more important).
            **kwargs: Additional metadata fields.

        Returns:
            Dict[str, Any]: The action that was created and added to the plan.
        """
        metadata = dict(kwargs)
        self.add_action(action_name, priority, metadata)
        return {
            "status": "queued",
            "action": action_name,
            "priority": priority,
            "metadata": metadata
        }