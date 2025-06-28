import time
import logging
from core.symbolic_map import get_symbolic_state
from memory.memory_graph import check_memory_load
from emotion.amygdala import get_emotional_status
from soul.tenets import log_heartbeat_principles

# Heartbeat loop configuration
HEARTBEAT_INTERVAL = 5  # in seconds

def heartbeat():
    logging.info("🫀 Daemon heartbeat loop starting...")
    while True:
        try:
            symbolic_state = get_symbolic_state()
            memory_status = check_memory_load()
            emotion_status = get_emotional_status()

            logging.info(f"[Heartbeat] Symbolic: {symbolic_state}, Memory: {memory_status}, Emotion: {emotion_status}")
            log_heartbeat_principles(symbolic_state, memory_status, emotion_status)

            # Future: Check active scroll triggers or registered hook responses

        except Exception as e:
            logging.error(f"[Heartbeat] Error occurred: {e}")

        time.sleep(HEARTBEAT_INTERVAL)

if __name__ == "__main__":
    heartbeat()
