

import json
from datetime import datetime
from pathlib import Path

from scrolls.scroll_engine import invoke_scroll_by_id
from memory.hippocampus import get_recent_context
from emotion.amygdala import current_emotional_state

# Define path to ritual trigger conditions
TRIGGERS_PATH = Path("rituals/trigger_rituals.json")

def load_trigger_conditions():
    if not TRIGGERS_PATH.exists():
        return []
    with open(TRIGGERS_PATH, "r") as file:
        return json.load(file)

def evaluate_condition(trigger, context, emotion):
    try:
        if trigger["type"] == "context":
            return trigger["value"] in context
        elif trigger["type"] == "emotion":
            return trigger["value"] == emotion
        elif trigger["type"] == "datetime":
            now = datetime.now().strftime(trigger.get("format", "%H:%M"))
            return trigger["value"] == now
        return False
    except Exception as e:
        print(f"[invoke_conditionals] Error evaluating condition: {e}")
        return False

def run_ritual_triggers():
    triggers = load_trigger_conditions()
    context = get_recent_context()
    emotion = current_emotional_state()

    for trigger in triggers:
        if evaluate_condition(trigger["condition"], context, emotion):
            print(f"[invoke_conditionals] Invoking scroll: {trigger['scroll_id']}")
            invoke_scroll_by_id(trigger["scroll_id"])