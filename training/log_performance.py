

import time
import json
import os

LOG_DIR = "logs/performance"
os.makedirs(LOG_DIR, exist_ok=True)

def log_performance(metric_name, value, tags=None):
    """
    Logs a performance metric to a JSON file.

    Args:
        metric_name (str): The name of the metric being logged.
        value (float): The value of the metric.
        tags (dict, optional): Additional metadata for the log.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_entry = {
        "timestamp": timestamp,
        "metric": metric_name,
        "value": value,
        "tags": tags or {}
    }

    filename = os.path.join(LOG_DIR, f"{metric_name}.json")
    with open(filename, "a") as log_file:
        log_file.write(json.dumps(log_entry) + "\n")