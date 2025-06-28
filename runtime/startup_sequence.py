

import logging
from unimind.core.loader import initialize_system
from unimind.core.symbolic_map import map_symbolic_routes
from unimind.soul.foundation_manifest import load_foundation_manifest
from unimind.scrolls.scroll_engine import prepare_scroll_registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StartupSequence")

def run_startup_sequence():
    logger.info("Starting Unimind system initialization...")

    try:
        initialize_system()
        logger.info("System initialization successful.")
    except Exception as e:
        logger.error(f"System initialization failed: {e}")

    try:
        map_symbolic_routes()
        logger.info("Symbolic routing map initialized.")
    except Exception as e:
        logger.error(f"Symbolic routing failed: {e}")

    try:
        load_foundation_manifest()
        logger.info("Foundation manifest loaded.")
    except Exception as e:
        logger.warning(f"Foundation manifest load encountered an issue: {e}")

    try:
        prepare_scroll_registry()
        logger.info("Scroll registry prepared.")
    except Exception as e:
        logger.error(f"Scroll engine failed to load: {e}")

    logger.info("Startup sequence completed.")

if __name__ == "__main__":
    run_startup_sequence()