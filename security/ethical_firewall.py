

from security.encryption_service import encrypt_data, decrypt_data

class EthicalFirewall:
    def __init__(self):
        self.rules = []
        self.blocked_behaviors = set()
        self.audit_log = []

    def load_rules(self, rule_list):
        """Load ethical rules into the firewall."""
        self.rules.extend(rule_list)

    def evaluate_action(self, action_description: str) -> bool:
        """Evaluate whether the action is ethical."""
        for rule in self.rules:
            if not rule(action_description):
                self.blocked_behaviors.add(action_description)
                self.log_event(f"Blocked unethical action: {action_description}")
                return False
        self.log_event(f"Approved action: {action_description}")
        return True

    def log_event(self, message: str):
        """Encrypt and store an event in the audit log."""
        encrypted = encrypt_data(message)
        self.audit_log.append(encrypted)

    def get_audit_log(self) -> list:
        """Return decrypted audit log."""
        return [decrypt_data(entry) for entry in self.audit_log]