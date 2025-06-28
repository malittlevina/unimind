


# codex_shard_bridge.py

import json
import os
from devices.device_registry import DeviceRegistry
from core.symbolic_map import translate_input

class CodexShardBridge:
    def __init__(self, shard_id):
        self.shard_id = shard_id
        self.device = DeviceRegistry().get_device(shard_id)
        self.linked = False if not self.device else True

    def receive_signal(self, input_data):
        """
        Simulates receiving input from the shard (e.g. RFID, vision, touch)
        """
        translated = translate_input(input_data)
        return {"raw": input_data, "translated": translated}

    def send_signal(self, output_data):
        """
        Sends data to the shard (e.g. haptic, LED, screen)
        """
        if not self.linked:
            raise Exception("Codex Shard not linked.")
        # Simulate transmission
        print(f"Sending to {self.shard_id}: {output_data}")

    def verify_connection(self):
        return self.linked

    def get_status(self):
        return {
            "shard_id": self.shard_id,
            "linked": self.linked,
            "device_meta": self.device if self.device else "Unregistered"
        }