"""
models.py – Unified model interface for Unimind.
Provides access to the comprehensive model registry and unified input processor.
"""

from unimind.native_models.model_registry import (
    model_registry,
    get_model,
    list_models,
    execute_request,
    get_registry_stats,
    ModelCategory
)

from unimind.native_models.unified_input_processor import (
    unified_input_processor,
    process_input,
    get_processing_stats,
    ProcessingResult,
    ProcessingStage
)

# Re-export for backward compatibility
__all__ = [
    'model_registry',
    'get_model',
    'list_models', 
    'execute_request',
    'get_registry_stats',
    'ModelCategory',
    'unified_input_processor',
    'process_input',
    'get_processing_stats',
    'ProcessingResult',
    'ProcessingStage'
]

# Legacy compatibility - keep the old interface for existing code
class DummyModelRegistry:
    """Legacy model registry for backward compatibility."""
    def register_all_models(self):
        """Register all models (now handled by the new registry)."""
        pass
    
    def summary(self):
        """Get summary of registered models."""
        stats = get_registry_stats()
        return f"Model registry with {stats['total_models']} models, {stats['active_models']} active"

# Keep the old interface for backward compatibility
model_registry_legacy = DummyModelRegistry() 