import unittest
from predict import FormalityPredictor
import os

class TestFormalityPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize the predictor once for all tests"""
        cls.predictor = FormalityPredictor()
    
    def test_formal_text(self):
        """Test with formal text"""
        prediction = self.predictor.predict("Good afternoon. How may I assist you today?")
        self.assertEqual(prediction, "Formal")
    
    def test_informal_text(self):
        """Test with informal text"""
        prediction = self.predictor.predict("hey, what's up?")
        self.assertEqual(prediction, "Informal")
    
    def test_empty_text(self):
        """Test with empty string"""
        with self.assertRaises(ValueError):
            self.predictor.predict("")

if __name__ == "__main__":
    unittest.main()
