# /unimind/utils/logger.py

import logging

def log_event(event_type, message):
    logging.info(f"[{event_type.upper()}] {message}")