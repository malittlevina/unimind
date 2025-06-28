
# motor_cortex.py

class MotorCortex:
    def __init__(self):
        self.intended_movements = []
        self.executed_movements = []
        self.gesture_plan = {}

    def plan_movement(self, movement_command):
        """Plan a movement based on symbolic or direct input."""
        self.intended_movements.append(movement_command)
        self.gesture_plan = {"command": movement_command, "status": "planned"}
        print(f"[MotorCortex] Planned movement: {movement_command}")
        return self.gesture_plan

    def execute_movement(self):
        """Execute the currently planned movement."""
        if not self.gesture_plan:
            print("[MotorCortex] No movement planned.")
            return None
        self.gesture_plan["status"] = "executed"
        self.executed_movements.append(self.gesture_plan["command"])
        print(f"[MotorCortex] Executed movement: {self.gesture_plan['command']}")
        return self.gesture_plan["command"]

    def reset_gesture_plan(self):
        """Clear current plan after execution or error."""
        print("[MotorCortex] Resetting gesture plan.")
        self.gesture_plan = {}

    def get_status(self):
        return {
            "planned": self.gesture_plan,
            "executed": self.executed_movements[-5:]  # last 5 movements
        }

