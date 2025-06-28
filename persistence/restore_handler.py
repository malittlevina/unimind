

import os
import json
import logging
from datetime import datetime

BACKUP_DIR = "backups"
STATE_FILE = "system_state.json"

def load_backup(backup_name):
    """Load a backup file by name and return its content as a dictionary."""
    backup_path = os.path.join(BACKUP_DIR, backup_name)
    if not os.path.exists(backup_path):
        logging.error(f"Backup '{backup_name}' does not exist.")
        return None

    try:
        with open(backup_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load backup: {e}")
        return None

def restore_state(backup_name):
    """Restore system state from a given backup."""
    state_data = load_backup(backup_name)
    if not state_data:
        logging.error("No data to restore.")
        return False

    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state_data, f, indent=4)
        logging.info(f"System state restored from {backup_name}")
        return True
    except Exception as e:
        logging.error(f"Failed to restore system state: {e}")
        return False

def list_available_backups():
    """Return a list of available backup filenames."""
    if not os.path.exists(BACKUP_DIR):
        return []
    return [f for f in os.listdir(BACKUP_DIR) if f.endswith('.json')]