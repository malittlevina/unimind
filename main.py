"""
main.py – Entry point for the Unimind daemon system
"""

import os
import json
import time
import threading
import argparse

# Config loader
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Failed to parse config.json. Using default config.")
    else:
        print("⚠️ config.json not found. Using default config.")
    return {}

from core.unimind import Unimind
from core.loader import load_brain_modules
from interfaces.system_control import SystemControl
from logic.symbolic_reasoner import SymbolicReasoner
from ethics.pineal_gland import EthicalGovernor
from soul.tenets import TENETS
from daemon_web.core_router import start_persona_services
from memory.memory_graph import MemoryGraph
from scrolls.scroll_engine import ScrollEngine

parser = argparse.ArgumentParser()
parser.add_argument("--invoke", type=str, help="Scroll to invoke on launch")
args = parser.parse_args()

def main():
    start_time = time.time()
    print("🔁 Initializing Unimind Runtime System...")
    CONFIG = load_config()

    # Load modular brain components
    modules = load_brain_modules()
    print("✅ Brain modules loaded.")

    # Announce identity
    try:
        from soul.foundation_manifest import DAEMON_IDENTITY
        print(f"✨ Activating Daemon: {DAEMON_IDENTITY['name']} (v{DAEMON_IDENTITY['version']})")
    except Exception:
        print("⚠️ No identity manifest found.")

    # Initialize core Unimind
    mind = Unimind(modules)

    # Initialize the ethical reasoning engine (Pineal + Reasoner + Tenets)
    reasoner = SymbolicReasoner()
    ethics = EthicalGovernor(tenets=TENETS, reasoner=reasoner)
    mind.attach_ethics(ethics)
    print("🧭 Ethical subsystem ready.")

    # Connect Memory
    memory_graph = MemoryGraph()
    mind.attach_memory(memory_graph)
    print("🧠 Memory subsystem active.")

    # Start Scroll Engine
    scrolls = ScrollEngine()
    mind.register_scrolls(scrolls)
    print("📜 Scroll engine initialized.")

    if args.invoke:
        print(f"🔮 CLI-invoked scroll: {args.invoke}")
        scrolls.cast(args.invoke)

    # Run startup rituals if defined
    boot_scrolls = CONFIG.get("boot_scrolls", [])
    for scroll_name in boot_scrolls:
        print(f"🌀 Casting boot scroll: {scroll_name}")
        scrolls.cast(scroll_name)

    # Launch web and persona router
    start_persona_services(mind)
    print("🌐 Daemon Web interface booted.")

    # Daemon Heartbeat
    def heartbeat():
        while True:
            print("❤️ Daemon heartbeat")
            time.sleep(CONFIG.get("heartbeat_interval", 60))

    threading.Thread(target=heartbeat, daemon=True).start()

    # Start system control loop
    controller = SystemControl(mind)
    try:
        controller.run()
    except Exception as e:
        print(f"🛑 SystemControl crashed: {e}")
        if CONFIG.get("safe_mode_on_crash"):
            print("🔁 Restarting in safe mode...")
            # TODO: Trigger fallback

    print(f"🚀 Unimind boot complete in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
