

import os
import json
from threading import Lock

class StateServer:
    def __init__(self, state_file="runtime_state.json"):
        self.state_file = state_file
        self.lock = Lock()
        self.state = self.load_state()

    def load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {}
        return {}

    def save_state(self):
        with self.lock:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)

    def get(self, key, default=None):
        return self.state.get(key, default)

    def set(self, key, value):
        with self.lock:
            self.state[key] = value
            self.save_state()

    def delete(self, key):
        with self.lock:
            if key in self.state:
                del self.state[key]
                self.save_state()