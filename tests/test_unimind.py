

import unittest
from unimind.core.unimind import Unimind

class TestUnimind(unittest.TestCase):

    def setUp(self):
        self.unimind = Unimind()

    def test_initial_state(self):
        self.assertIsNotNone(self.unimind)
        self.assertTrue(hasattr(self.unimind, 'initialize_brain'))

    def test_initialize_brain_structure(self):
        result = self.unimind.initialize_brain()
        self.assertTrue(result)
        self.assertIn('prefrontal_cortex', self.unimind.brain_regions)
        self.assertIn('amygdala', self.unimind.brain_regions)

    def test_context_awareness(self):
        self.unimind.initialize_brain()
        context = self.unimind.get_context("Describe the environment.")
        self.assertIsInstance(context, dict)
        self.assertIn('perception', context)

if __name__ == '__main__':
    unittest.main()