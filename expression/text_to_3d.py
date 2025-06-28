# core/symbolic_map.py

"""
Symbolic Map Module
This module maps high-level symbolic concepts to their corresponding Unimind subsystems and functions.
It serves as a routing table between semantic inputs and the appropriate logic/memory/emotion systems.
"""

SYMBOLIC_ROUTE_TABLE = {
    # Cortex mappings
    "decision": "cortex.prefrontal_cortex.handle_decision",
    "coordination": "cortex.cerebellum.process_movement",
    "motor": "cortex.motor_cortex.initiate_movement",

    # Emotion mappings
    "fear": "emotion.amygdala.process_fear",
    "emotion_analysis": "emotion.emotion_classifier.analyze_emotion",

    # Ethics mappings
    "moral_check": "ethics.pineal_gland.evaluate_ethics",

    # Memory mappings
    "long_term_memory": "memory.hippocampus.store_event",
    "short_term_memory": "memory.short_term.buffer_state",
    "memory_recall": "memory.memory_graph.recall_concept",
    "symbol_to_sql": "memory.text_to_sql.generate_sql",

    # Perception mappings
    "visual": "perception.occipital_lobe.process_image",
    "language_understanding": "perception.wernickes_area.understand",
    "language_generation": "perception.brocas_area.generate_speech",
    "hearing": "perception.temporal_lobe.process_audio",
    "vision": "perception.vision_model.classify_scene",
    "voice": "perception.voice_model.classify_audio",

    # Reasoning + Language
    "reasoning": "logic.symbolic_reasoner.evaluate_logic",
    "convert_logic": "logic.text_to_logic.transform",
    "shell_scripting": "logic.text_to_shell.create_script",
    "language_model": "language.lam_engine.process_nlp",
    "llm_inference": "language.llm_engine.run_llm_inference",
    "translate_text": "language.text_to_text.rewrite_text",
    "plan_action": "planning.action_planner.create_plan",
    "code_gen": "planning.text_to_code.generate_code",

    # Expression
    "3d_model": "expression.text_to_3d.generate_model",
    "video_gen": "expression.text_to_video.create_video",

    # Scrolls and Rituals
    "invoke_scroll": "scrolls.scroll_engine.invoke_scroll",

    # Soul + Tenets
    "core_values": "soul.tenets.reflect_principles"
}


def resolve_symbol(symbol: str):
    """
    Resolve a symbolic label into a callable route string.
    """
    return SYMBOLIC_ROUTE_TABLE.get(symbol, None)


def list_all_symbols():
    """
    List all known symbolic mappings in the system.
    """
    return list(SYMBOLIC_ROUTE_TABLE.keys())


def validate_route(symbol: str):
    """
    Check if a symbol is known and maps to a valid subsystem.
    """
    return symbol in SYMBOLIC_ROUTE_TABLE