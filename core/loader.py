"""
Loader.py – Centralized startup loader for the Unimind system
Responsible for initializing all Unimind subsystems and linking symbolic, emotional, logical, and memory modules.
"""

import importlib
import os
import sys

from unimind.core import unimind
from unimind.memory import hippocampus, short_term, memory_graph
from unimind.logic import symbolic_reasoner
from unimind.emotion import amygdala
from unimind.ethics import pineal_gland
from unimind.language import lam_engine, llm_engine, text_to_text
from unimind.models import (
    text_to_sql, text_to_logic, text_to_shell, emotion_classifier,
    vision_model, voice_model, text_to_3d, text_to_video,
    text_to_code, context_model
)
from unimind.perception import brocas_area, wernickes_area, occipital_lobe
from unimind.planning import action_planner
from unimind.interfaces import system_control
from unimind.scrolls import scroll_engine
from unimind.todo import tasks
from unimind.daemon_web import core_router
from unimind.soul import foundation_manifest, tenets

# Aliased for backward compatibility if needed
lam_model = lam_engine

def load_all_subsystems():
    print("🔁 [LOADER] Initializing Unimind subsystems...")

    # Core systems
    unimind.boot_sequence()

    # Memory
    hippocampus.initialize()
    short_term.warm_up()
    memory_graph.build_graph()

    # Logic
    symbolic_reasoner.activate()

    # Ethics / Tenets
    pineal_gland.load_tenets(tenets.TENETS)

    # Emotion
    amygdala.initialize_state()

    # Perception
    brocas_area.load_language_models()
    wernickes_area.load_comprehension_engines()
    occipital_lobe.load_vision_models()

    # Language and Models
    lam_engine.boot()
    llm_engine.load()
    lam_model.load()
    text_to_logic.prepare()
    text_to_text.prepare()
    text_to_3d.prepare()
    text_to_video.prepare()
    text_to_shell.prepare()
    text_to_sql.prepare()
    emotion_classifier.load_model()
    vision_model.load_model()
    voice_model.load_model()
    context_model.initialize_context()

    # Planning and Interface
    action_planner.load_goals()
    system_control.bind_to_system()

    # Scrolls and Daemon Web
    scroll_engine.index_scrolls()
    core_router.register_routes()

    # Soul and Tasks
    foundation_manifest.validate()
    tasks.load()

    print("✅ [LOADER] All subsystems initialized.")

if __name__ == "__main__":
    load_all_subsystems()
