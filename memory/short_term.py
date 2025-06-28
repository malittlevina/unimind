# short_term.py

import time
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ShortTermMemory:
    def __init__(self, max_items=50, decay_time=300):
        """
        Initializes the short-term memory.
        :param max_items: Maximum number of items to store.
        :param decay_time: Time in seconds before an item is considered decayed.
        """
        self.memory = deque()
        self.timestamps = deque()
        self.max_items = max_items
        self.decay_time = decay_time
        logger.info(f"ShortTermMemory initialized with max_items={max_items}, decay_time={decay_time}")

    def store(self, item):
        """
        Store a new item in short-term memory.
        """
        logger.info(f"Storing item: {item}")
        current_time = time.time()
        self.memory.append(item)
        self.timestamps.append(current_time)

        if len(self.memory) > self.max_items:
            logger.info("Memory limit reached. Removing oldest item.")
            self.memory.popleft()
            self.timestamps.popleft()

    def retrieve(self):
        """
        Retrieve non-expired items from short-term memory.
        """
        current_time = time.time()
        filtered_memory = []
        new_memory = deque()
        new_timestamps = deque()

        for i in range(len(self.memory)):
            age = current_time - self.timestamps[i]
            if age <= self.decay_time:
                filtered_memory.append(self.memory[i])
                new_memory.append(self.memory[i])
                new_timestamps.append(self.timestamps[i])

        self.memory = new_memory
        self.timestamps = new_timestamps
        logger.info(f"Retrieved {len(filtered_memory)} valid items from short-term memory.")
        return filtered_memory

    def clear(self):
        """
        Clear all items from short-term memory.
        """
        self.memory.clear()
        self.timestamps.clear()
        logger.info("Short-term memory cleared.")

    def __repr__(self):
        return f"<ShortTermMemory(size={len(self.memory)}, max_items={self.max_items})>"

    def summary(self):
        return {
            "current_size": len(self.memory),
            "max_items": self.max_items,
            "decay_time": self.decay_time,
            "oldest_item_age": time.time() - self.timestamps[0] if self.timestamps else None
        }
