

import unittest
from scrolls.scroll_engine import ScrollEngine

class TestScrollEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ScrollEngine()

    def test_scroll_registration(self):
        self.engine.register_scroll("test_scroll", lambda: "invoked")
        self.assertIn("test_scroll", self.engine.scrolls)

    def test_scroll_invocation(self):
        self.engine.register_scroll("greet", lambda: "hello")
        result = self.engine.cast_scroll("greet")
        self.assertEqual(result, "hello")

    def test_unknown_scroll(self):
        result = self.engine.cast_scroll("nonexistent")
        self.assertIn("not found", result.lower())

if __name__ == "__main__":
    unittest.main()