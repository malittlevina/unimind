"""
Loader.py – Centralized startup loader for the Unimind system
Responsible for initializing all Unimind subsystems and linking symbolic, emotional, logical, and memory modules.
"""

import importlib
import os
import sys
import time

from unimind.core import unimind
from unimind.memory import hippocampus, short_term, memory_graph
from unimind.logic import symbolic_reasoner
from unimind.emotion import amygdala
from unimind.ethics import pineal_gland
from unimind.native_models import (
    lam_engine, llm_engine, text_to_text, text_to_sql, text_to_logic, text_to_shell, emotion_classifier,
    vision_model, voice_model, text_to_3d, text_to_video,
    text_to_code, context_model
)
from unimind.perception import brocas_area, wernickes_area, occipital_lobe
from unimind.planning import action_planner
from unimind.interfaces import system_control
from unimind.scrolls import scroll_engine
from unimind.daemon_web import core_router
from unimind.soul import foundation_manifest, tenets


# Aliased for backward compatibility if needed
lam_model = lam_engine.LAMEngine()

# Unified model registry import
from unimind.models import model_registry

def load_all_subsystems():
    print("🔁 [LOADER] Initializing Unimind subsystems...")

    # 🧠 Unimind Core Initialization
    start = time.time()
    unimind.boot_sequence()  # Core system boot sequence
    print(f"⏱️ Unimind booted in {time.time() - start:.2f} seconds")

    # 🧠 Memory Systems (Hippocampus, STM, Memory Graph)
    start = time.time()
    hippocampus.initialize()  # Hippocampus initialization
    short_term.warm_up()  # Short-term memory warm-up
    memory_graph.build_graph()  # Memory graph construction
    print(f"⏱️ Memory systems initialized in {time.time() - start:.2f} seconds")

    # 🔍 Symbolic Logic Engine
    start = time.time()
    symbolic_reasoner.activate()  # Activate symbolic reasoner
    print(f"⏱️ Symbolic logic engine activated in {time.time() - start:.2f} seconds")

    # ⚖️ Ethical Engine (Pineal Gland)
    start = time.time()
    pineal_gland.load_tenets(tenets.TENETS)  # Load ethical tenets
    print(f"⏱️ Ethical engine loaded in {time.time() - start:.2f} seconds")

    # 💓 Emotion Engine (Amygdala)
    start = time.time()
    amygdala.initialize_state()  # Initialize emotional state
    print(f"⏱️ Emotion engine initialized in {time.time() - start:.2f} seconds")

    # 👁️ Perception Systems (Broca, Wernicke, Occipital)
    start = time.time()
    brocas_area.load_language_models()  # Load Broca's area language models
    wernickes_area.load_comprehension_engines()  # Load Wernicke's area comprehension engines
    occipital_lobe.load_vision_models()  # Load occipital lobe vision models
    print(f"⏱️ Perception systems loaded in {time.time() - start:.2f} seconds")

    # 🗣️ Language and Symbolic Translation Engines
    start = time.time()
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
    print(f"⏱️ Language and symbolic translation engines loaded in {time.time() - start:.2f} seconds")

    # 🧭 Planning + Interface Hooks
    start = time.time()
    action_planner.load_goals()  # Load action planning goals
    system_control.bind_to_system()  # Bind system control interface
    print(f"⏱️ Planning and interface hooks loaded in {time.time() - start:.2f} seconds")

    # 📜 Scroll Engine and Daemon Web UI
    start = time.time()
    scroll_engine.index_scrolls()  # Index scrolls for retrieval
    core_router.register_routes()  # Register daemon web UI routes
    print(f"⏱️ Scroll engine and daemon web UI loaded in {time.time() - start:.2f} seconds")

    # 🌱 Soul Layer and Task Engine
    start = time.time()
    foundation_manifest.validate()  # Validate foundation manifest
    tasks.load()  # Load task engine
    print(f"⏱️ Soul layer and task engine loaded in {time.time() - start:.2f} seconds")

    print(f"📦 [LOADER] Registered Models: {model_registry.summary()}")
    print("✅ [LOADER] All subsystems initialized.")
    print("🧠 [LOADER] Startup complete. Ready to engage.")

if __name__ == "__main__":
    load_all_subsystems()
