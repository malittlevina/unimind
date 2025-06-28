# symbolic_map.py

"""
The Symbolic Map module acts as a neural-symbolic interpreter that maintains
symbolic associations between brain regions, models, tasks, and daemon goals.
It enables structured reasoning and self-reflection based on symbolic cues
and abstract concept mappings.
"""

import logging
from typing import Dict, List, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SymbolicMap:
    def __init__(self):
        # Map of symbolic concepts to internal system modules or nodes
        self.symbol_to_node: Dict[str, str] = {}

        # Symbolic memory associations for goal chaining and reflection
        self.symbolic_memory_graph: Dict[str, List[str]] = {}

        # Semantic themes for meta-reasoning (e.g., ethics, emotion, logic)
        self.semantic_domains: Dict[str, List[str]] = {
            "ethics": ["justice", "compassion", "moral dilemma", "tenet"],
            "emotion": ["fear", "joy", "curiosity", "love", "awe"],
            "logic": ["deduction", "contradiction", "cause-effect"],
            "identity": ["persona", "tenet", "reflection", "soul"],
            "cognition": ["planning", "memory", "language", "reasoning"],
            "sensation": ["vision", "speech", "gesture", "touch"],
        }
        logger.info("SymbolicMap initialized with semantic domains: %s", list(self.semantic_domains.keys()))

    def register_symbol(self, symbol: str, node: str) -> None:
        """Link a symbolic concept to a functional brain node or model."""
        logger.debug("Registering symbol '%s' to node '%s'", symbol, node)
        self.symbol_to_node[symbol] = node

    def get_node_for_symbol(self, symbol: str) -> Union[str, None]:
        """Retrieve the node or model associated with a symbolic cue."""
        logger.debug("Retrieving node for symbol '%s': %s", symbol, self.symbol_to_node.get(symbol))
        return self.symbol_to_node.get(symbol)

    def link_symbols(self, source: str, target: str) -> None:
        """Create a symbolic relationship between two concepts."""
        logger.debug("Linking symbol '%s' to '%s'", source, target)
        if source not in self.symbolic_memory_graph:
            self.symbolic_memory_graph[source] = []
        self.symbolic_memory_graph[source].append(target)

    def get_related_symbols(self, symbol: str) -> List[str]:
        """Return a list of conceptually linked symbols."""
        logger.debug("Getting related symbols for '%s': %s", symbol, self.symbolic_memory_graph.get(symbol, []))
        return self.symbolic_memory_graph.get(symbol, [])

    def infer_domain(self, symbol: str) -> Union[str, None]:
        """Identify the semantic domain most relevant to a given symbol."""
        logger.debug("Inferring domain for symbol '%s'", symbol)
        for domain, keywords in self.semantic_domains.items():
            if symbol in keywords:
                return domain
        return None

    def debug_map(self) -> Dict[str, Union[Dict[str, str], Dict[str, List[str]]]]:
        """Return full debug information of the symbolic map system."""
        return {
            "symbol_to_node": self.symbol_to_node,
            "symbolic_memory_graph": self.symbolic_memory_graph,
            "semantic_domains": self.semantic_domains,
        }
