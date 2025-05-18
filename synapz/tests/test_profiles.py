"""Tests for the cognitive profiles functionality."""

import unittest
from synapz.core.profiles import CognitiveProfile, ProfileStrategy, ProfileManager

class TestCognitiveProfiles(unittest.TestCase):
    """Test suite for cognitive profiles."""
    
    def test_cognitive_profile_enum(self):
        """Test CognitiveProfile enum functionality."""
        # Test direct access
        self.assertEqual(CognitiveProfile.ADHD.value, "adhd")
        self.assertEqual(CognitiveProfile.DYSLEXIC.value, "dyslexic")
        self.assertEqual(CognitiveProfile.VISUAL.value, "visual")
        self.assertEqual(CognitiveProfile.CONTROL.value, "control")
        
        # Test from_string method
        self.assertEqual(CognitiveProfile.from_string("adhd"), CognitiveProfile.ADHD)
        self.assertEqual(CognitiveProfile.from_string("DYSLEXIC"), CognitiveProfile.DYSLEXIC)
        self.assertEqual(CognitiveProfile.from_string("Visual"), CognitiveProfile.VISUAL)
        self.assertEqual(CognitiveProfile.from_string("control"), CognitiveProfile.CONTROL)
        
        # Test invalid profile raises ValueError
        with self.assertRaises(ValueError):
            CognitiveProfile.from_string("invalid_profile")
            
    def test_profile_strategy(self):
        """Test ProfileStrategy dataclass."""
        # Create a strategy
        strategy = ProfileStrategy(
            profile=CognitiveProfile.ADHD,
            prompt_template_key="test_template",
            instruction_modifiers=["Test instruction 1", "Test instruction 2"],
            example_count=3,
            use_visuals=True,
            chunk_size=2
        )
        
        # Test properties
        self.assertEqual(strategy.profile, CognitiveProfile.ADHD)
        self.assertEqual(strategy.prompt_template_key, "test_template")
        self.assertEqual(len(strategy.instruction_modifiers), 2)
        self.assertEqual(strategy.example_count, 3)
        self.assertTrue(strategy.use_visuals)
        self.assertEqual(strategy.chunk_size, 2)
        self.assertFalse(strategy.is_control)
        
        # Test control profile
        control_strategy = ProfileStrategy(
            profile=CognitiveProfile.CONTROL,
            prompt_template_key="control_template",
            instruction_modifiers=[]
        )
        self.assertTrue(control_strategy.is_control)
        
    def test_profile_manager(self):
        """Test ProfileManager class."""
        manager = ProfileManager()
        
        # Test default strategies exist for all profiles
        for profile in CognitiveProfile:
            strategy = manager.get_strategy(profile)
            self.assertEqual(strategy.profile, profile)
            
        # Test update strategy
        adhd_strategy = manager.get_strategy(CognitiveProfile.ADHD)
        original_chunk_size = adhd_strategy.chunk_size
        original_example_count = adhd_strategy.example_count
        
        # Update a couple of parameters
        manager.update_strategy(
            CognitiveProfile.ADHD,
            {"chunk_size": 5, "example_count": 4}
        )
        
        # Check updated strategy
        updated_strategy = manager.get_strategy(CognitiveProfile.ADHD)
        self.assertEqual(updated_strategy.chunk_size, 5)
        self.assertEqual(updated_strategy.example_count, 4)
        
        # Other parameters should remain unchanged
        self.assertEqual(updated_strategy.prompt_template_key, adhd_strategy.prompt_template_key)
        self.assertEqual(updated_strategy.instruction_modifiers, adhd_strategy.instruction_modifiers)
        
if __name__ == "__main__":
    unittest.main() 