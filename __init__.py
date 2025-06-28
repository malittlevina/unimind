

"""
Unimind package initializer.

This file marks the 'unimind' folder as a Python package
and can be used to expose key modules or package-wide variables.
"""

# Example: expose core modules for easy import
from unimind.core.unimind import Unimind
from unimind.logic.symbolic_reasoner import SymbolicReasoner
from unimind.memory.memory_graph import MemoryGraph
from unimind.emotion.amygdala import Amygdala
from unimind.ethics.pineal_gland import PinealGland

# Package version
__version__ = "0.5.0"