# core_router.py

from .persona_manifest import load_persona_manifest
from .personas import (
    persona_01, persona_02, persona_03, persona_04,
    persona_05, persona_06, persona_07, persona_08,
    persona_09
)

# Load manifest and route to correct persona handlers
PERSONA_HANDLERS = {
    "Navigator": persona_01.handle,
    "Engineer": persona_02.handle,
    "Mediator": persona_03.handle,
    "Dreamer": persona_04.handle,
    "Strategist": persona_05.handle,
    "Sentinel": persona_06.handle,
    "Storyweaver": persona_07.handle,
    "Mythic Mirror": persona_08.handle,
    "Meta-Ethical Philosopher": persona_09.handle,
}

def route_message(message, context):
    """
    Directs the message to the correct persona based on intent, emotional state, or explicit persona call.
    """
    manifest = load_persona_manifest()

    # Determine target persona (could be more complex with emotion/context routing)
    target = context.get("persona", manifest.get("default_persona", "Navigator"))

    handler = PERSONA_HANDLERS.get(target)

    if handler:
        return handler(message, context)
    else:
        return {"error": f"No handler for persona '{target}'"}