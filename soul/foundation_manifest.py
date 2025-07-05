import json
import os

DAEMON_IDENTITY = {
    "name": "Unimind Daemon",
    "version": "0.1.0",
    "description": "Placeholder identity for Unimind system.",
    "founder_id": "malittlevina",
    "founder_ids": ["malittlevina"],
    "privileged_users": ["malittlevina"]
}

class Soul:
    """
    Represents the daemon's self-identity (soul), including core values and access control.
    Loads from JSON manifest if available, otherwise uses DAEMON_IDENTITY.
    """
    def __init__(self, manifest_path=None):
        self.identity = DAEMON_IDENTITY.copy()
        if manifest_path is None:
            manifest_path = os.path.join(os.path.dirname(__file__), "foundation_manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r") as f:
                    data = json.load(f)
                    if "daemon_identity" in data:
                        self.identity.update(data["daemon_identity"])
            except Exception as e:
                pass  # fallback to default

    def describe_self(self):
        """Return a human-readable description of the daemon's identity and values."""
        desc = f"I am {self.identity.get('name', 'an AI daemon')}, version {self.identity.get('version', '?')}."
        if self.identity.get("description"):
            desc += f" {self.identity['description']}"
        if self.identity.get("core_values"):
            desc += "\nCore values: " + ", ".join(self.identity["core_values"])
        if self.identity.get("ethical_tenets"):
            desc += "\nEthical tenets: " + ", ".join(self.identity["ethical_tenets"])
        return desc

    def get_founder_id(self):
        return self.identity.get("founder_id")

    def get_founder_ids(self):
        return set(self.identity.get("founder_ids", []))

    def get_privileged_users(self):
        return set(self.identity.get("privileged_users", []))

    def is_founder(self, user_id):
        return user_id in self.get_founder_ids()

    def is_privileged(self, user_id):
        return user_id in self.get_privileged_users() or self.is_founder(user_id)
