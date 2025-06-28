
"""
Cerebellum Module
Responsible for fine-tuning coordination of actions, motion smoothing, and error correction in motor outputs.
"""

class Cerebellum:
    def __init__(self):
        self.calibration_log = []
        self.error_history = []

    def log_movement(self, action_description: str, success: bool, error_margin: float = 0.0):
        """
        Logs the outcome of a movement-related action.
        """
        entry = {
            "action": action_description,
            "success": success,
            "error_margin": error_margin
        }
        self.calibration_log.append(entry)
        if not success:
            self.error_history.append(entry)

    def suggest_correction(self):
        """
        Provides suggestions for adjustments based on historical errors.
        """
        if not self.error_history:
            return "No corrections needed."
        # Simplified logic: average out the last few error margins
        avg_error = sum(e["error_margin"] for e in self.error_history[-5:]) / min(5, len(self.error_history))
        return f"Recommended motor correction factor: {round(avg_error * 0.8, 4)}"

    def reset_logs(self):
        self.calibration_log = []
        self.error_history = []

