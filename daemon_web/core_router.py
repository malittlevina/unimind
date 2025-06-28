# core_router.py

from unimind.persona_manifest import load_persona_manifest
from unimind.personas import (
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

import logging

logging.basicConfig(level=logging.INFO)

def route_message(message, context):
    """
    Directs the message to the correct persona based on intent, emotional state, or explicit persona call.
    Includes fallback and logging.
    """
    manifest = load_persona_manifest()

    # Determine target persona
    target = context.get("persona", manifest.get("default_persona", "Navigator"))
    fallback = manifest.get("fallback_persona", "Sentinel")

    handler = PERSONA_HANDLERS.get(target)

    if handler:
        try:
            logging.info(f"Routing to persona: {target}")
            return handler(message, context)
        except Exception as e:
            logging.warning(f"Error in handler for {target}: {e}. Falling back to {fallback}.")
            fallback_handler = PERSONA_HANDLERS.get(fallback)
            if fallback_handler:
                return fallback_handler(message, context)
            else:
                return {"error": f"No fallback handler for persona '{fallback}'"}
    else:
        logging.error(f"No handler for persona '{target}'.")
        return {"error": f"No handler for persona '{target}'"}