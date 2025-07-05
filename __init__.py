"""
Unimind package initializer.

This file marks the 'unimind' folder as a Python package
and can be used to expose key modules or package-wide variables.
"""

# Package version
__version__ = "0.5.0"

# Core package information
__author__ = "Unimind Development Team"
__description__ = "Advanced AI system with unified input processing and native models"

# Import core components using relative imports
try:
    from .memory.memory_graph import MemoryGraph
except ImportError:
    MemoryGraph = None

try:
    from .emotion.amygdala import Amygdala
except ImportError:
    Amygdala = None

try:
    from .ethics.pineal_gland import PinealGland
except ImportError:
    PinealGland = None

# Export main components
__all__ = [
    'MemoryGraph',
    'Amygdala', 
    'PinealGland',
    '__version__',
    '__author__',
    '__description__'
]