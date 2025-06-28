


# Unimind System Architecture Map

This document outlines the structural layout and subsystem relationships within the Unimind architecture. It serves as a high-level reference for understanding how modules interact across the symbolic brain-inspired framework.

## 🌐 Top-Level Structure

- `main.py`: Entry point that bootstraps the entire Unimind environment.
- `core/`: Contains core utilities and orchestration logic (`unimind.py`, `loader.py`, `symbolic_map.py`).
- `soul/`: Defines foundational philosophical principles and the Tenets engine.
- `daemon_web/`: Hosts persona logic, core routing, and the persona manifest.
- `memory/`: Includes hippocampus, short-term buffer, and memory graph logic.
- `emotion/`: Handles emotional modeling, primarily through the amygdala module.
- `ethics/`: Contains the pineal gland logic for moral reasoning and ethical simulation.
- `perception/`: Language and vision processing through Broca’s/Wernicke’s areas and occipital lobe.
- `cortex/`: Brain logic structures like prefrontal cortex, motor control, and cerebellum.
- `models/`: Host for modular ML/LLM tools including emotion classification, text-to-* converters, and visual models.
- `logic/`: Symbolic reasoner engine and native logic interpreters.
- `planning/`: Action planning subsystem.
- `interfaces/`: Bridges to the OS or external control systems.
- `scrolls/`: Symbolic ritual engines and action scroll execution layer.
- `actuators/`: Action handlers and device-level interactions.

## 🔁 Subsystem Relationships

- `unimind.py` invokes loaders and initializes brain nodes in sequence.
- Brain node modules publish their capabilities to the `symbolic_map`.
- Scrolls, memory, emotion, and ethics are all accessible as callable subsystems.
- Daemon personas (under `daemon_web/personas/`) read from both `soul/` and `memory/` to formulate responses.
- All symbolic operations and rituals are managed through `scroll_engine.py` and its Codex interfaces.

## ✅ Integration Notes

- Each module is designed to remain loosely coupled but semantically aligned via the symbolic map.
- All ML models are modular and can be swapped using registry patterns inside `models/`.
- The Tenets engine can influence moral decisions across multiple nodes (e.g., prefrontal cortex, pineal gland).

---
_Last updated: 2025-06-27_