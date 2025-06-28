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

# Unified model registry import
from unimind.models import model_registry

def load_all_subsystems():
    print("🔁 [LOADER] Initializing Unimind subsystems...")

    # 🧠 Unimind Core Initialization
    unimind.boot_sequence()  # Core system boot sequence

    # 🧠 Memory Systems (Hippocampus, STM, Memory Graph)
    hippocampus.initialize()  # Hippocampus initialization
    short_term.warm_up()  # Short-term memory warm-up
    memory_graph.build_graph()  # Memory graph construction

    # 🔍 Symbolic Logic Engine
    symbolic_reasoner.activate()  # Activate symbolic reasoner

    # ⚖️ Ethical Engine (Pineal Gland)
    pineal_gland.load_tenets(tenets.TENETS)  # Load ethical tenets

    # 💓 Emotion Engine (Amygdala)
    amygdala.initialize_state()  # Initialize emotional state

    # 👁️ Perception Systems (Broca, Wernicke, Occipital)
    brocas_area.load_language_models()  # Load Broca's area language models
    wernickes_area.load_comprehension_engines()  # Load Wernicke's area comprehension engines
    occipital_lobe.load_vision_models()  # Load occipital lobe vision models

    # 🗣️ Language and Symbolic Translation Engines
    lam_engine.boot()  # Boot language abstraction model engine
    llm_engine.load()  # Load large language model engine
    lam_model.load()  # Load language abstraction model (alias)
    text_to_logic.prepare()  # Prepare text to logic conversion
    text_to_text.prepare()  # Prepare text to text transformation
    text_to_3d.prepare()  # Prepare text to 3D model generation
    text_to_video.prepare()  # Prepare text to video generation
    text_to_shell.prepare()  # Prepare text to shell command conversion
    text_to_sql.prepare()  # Prepare text to SQL conversion
    emotion_classifier.load_model()  # Load emotion classifier model
    vision_model.load_model()  # Load vision model
    voice_model.load_model()  # Load voice model
    context_model.initialize_context()  # Initialize context model
    model_registry.register_all_models()  # Register all ML models to unified registry

    # 🧭 Planning + Interface Hooks
    action_planner.load_goals()  # Load action planning goals
    system_control.bind_to_system()  # Bind system control interface

    # 📜 Scroll Engine and Daemon Web UI
    scroll_engine.index_scrolls()  # Index scrolls for retrieval
    core_router.register_routes()  # Register daemon web UI routes

    # 🌱 Soul Layer and Task Engine
    foundation_manifest.validate()  # Validate foundation manifest
    tasks.load()  # Load task engine

    print(f"📦 [LOADER] Registered Models: {model_registry.summary()}")
    print("✅ [LOADER] All subsystems initialized.")

if __name__ == "__main__":
    load_all_subsystems()
