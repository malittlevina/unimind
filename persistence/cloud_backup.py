

import os
import datetime
import json

class CloudBackupManager:
    def __init__(self, backup_path="backups/cloud", file_prefix="unimind_backup"):
        self.backup_path = backup_path
        self.file_prefix = file_prefix
        os.makedirs(self.backup_path, exist_ok=True)

    def create_backup(self, data, tag=None):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.file_prefix}_{tag or timestamp}.json"
        full_path = os.path.join(self.backup_path, filename)
        with open(full_path, 'w') as f:
            json.dump(data, f, indent=2)
        return full_path

    def list_backups(self):
        return sorted([f for f in os.listdir(self.backup_path) if f.startswith(self.file_prefix)])

    def load_backup(self, filename):
        full_path = os.path.join(self.backup_path, filename)
        with open(full_path, 'r') as f:
            return json.load(f)

    def delete_backup(self, filename):
        full_path = os.path.join(self.backup_path, filename)
        if os.path.exists(full_path):
            os.remove(full_path)
            return True
        return False