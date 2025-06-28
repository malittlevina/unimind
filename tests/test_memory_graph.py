


import unittest
from memory.memory_graph import MemoryGraph

class TestMemoryGraph(unittest.TestCase):
    def setUp(self):
        self.graph = MemoryGraph()

    def test_add_memory_node(self):
        self.graph.add_node("concept_1", data={"details": "test memory"})
        self.assertIn("concept_1", self.graph.nodes)

    def test_link_nodes(self):
        self.graph.add_node("concept_a")
        self.graph.add_node("concept_b")
        self.graph.link_nodes("concept_a", "concept_b", relation="supports")
        self.assertIn(("concept_a", "concept_b"), self.graph.edges)

    def test_retrieve_node_data(self):
        self.graph.add_node("concept_x", data={"mood": "reflective"})
        data = self.graph.get_node_data("concept_x")
        self.assertEqual(data.get("mood"), "reflective")

if __name__ == "__main__":
    unittest.main()