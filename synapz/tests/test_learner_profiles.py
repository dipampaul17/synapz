"""Tests for the learner profiles functionality."""

import unittest
import os
from pathlib import Path

from synapz.models.learner_profiles import (
    ADHD_PROFILE,
    DYSLEXIC_PROFILE,
    VISUAL_LEARNER_PROFILE,
    CONTROL_PROFILE,
    PROFILES_DIR,
    load_profile,
    get_all_profiles,
    get_profile_for_adaptation
)
from synapz.core.profiles import CognitiveProfile

class TestLearnerProfiles(unittest.TestCase):
    """Test suite for learner profiles."""
    
    def test_profile_structure(self):
        """Test that all profiles have the required structure."""
        profiles = [ADHD_PROFILE, DYSLEXIC_PROFILE, VISUAL_LEARNER_PROFILE, CONTROL_PROFILE]
        
        for profile in profiles:
            # Check required keys
            self.assertIn("id", profile)
            self.assertIn("name", profile)
            self.assertIn("cognitive_traits", profile)
            self.assertIn("modality_preferences", profile)
            self.assertIn("pedagogical_needs", profile)
            
            # Check cognitive traits structure
            cognitive = profile["cognitive_traits"]
            self.assertIn("attention_span", cognitive)
            self.assertIn("processing_style", cognitive)
            self.assertIn("strengths", cognitive)
            self.assertTrue(isinstance(cognitive["strengths"], list))
            
            # Check modality preferences structure
            modality = profile["modality_preferences"]
            self.assertIn("primary", modality)
            self.assertTrue(isinstance(modality["primary"], list))
            
            # Check pedagogical needs structure
            pedagogy = profile["pedagogical_needs"]
            self.assertIn("chunk_size", pedagogy)
            self.assertIn("organization", pedagogy)
            self.assertIn("example_types", pedagogy)
            self.assertTrue(isinstance(pedagogy["example_types"], list))
    
    def test_profile_files_created(self):
        """Test that profile JSON files are created on disk."""
        profiles = [
            CognitiveProfile.ADHD.value, 
            CognitiveProfile.DYSLEXIC.value,
            CognitiveProfile.VISUAL.value,
            CognitiveProfile.CONTROL.value
        ]
        
        for profile_id in profiles:
            file_path = PROFILES_DIR / f"{profile_id}.json"
            self.assertTrue(file_path.exists(), f"Profile file {file_path} does not exist")
    
    def test_load_profile(self):
        """Test loading profiles."""
        for profile_id in [p.value for p in CognitiveProfile]:
            profile = load_profile(profile_id)
            self.assertEqual(profile["id"], profile_id)
    
    def test_get_all_profiles(self):
        """Test getting all profiles."""
        profiles = get_all_profiles()
        self.assertEqual(len(profiles), 4)  # ADHD, Dyslexic, Visual, Control
        
        # Check that all profile types are included
        profile_ids = [p["id"] for p in profiles]
        self.assertIn(CognitiveProfile.ADHD.value, profile_ids)
        self.assertIn(CognitiveProfile.DYSLEXIC.value, profile_ids)
        self.assertIn(CognitiveProfile.VISUAL.value, profile_ids)
        self.assertIn(CognitiveProfile.CONTROL.value, profile_ids)
        
    def test_get_profile_for_adaptation(self):
        """Test getting profiles with adaptation parameters."""
        # Test ADHD profile
        adhd = get_profile_for_adaptation(CognitiveProfile.ADHD.value)
        self.assertIn("adaptation", adhd)
        self.assertEqual(adhd["adaptation"]["chunk_size"], 2)
        self.assertEqual(adhd["adaptation"]["example_count"], 2)
        
        # Test Dyslexic profile
        dyslexic = get_profile_for_adaptation(CognitiveProfile.DYSLEXIC.value)
        self.assertIn("adaptation", dyslexic)
        self.assertEqual(dyslexic["adaptation"]["example_count"], 3)
        self.assertIn("text_complexity", dyslexic["adaptation"])
        
        # Test Visual profile
        visual = get_profile_for_adaptation(CognitiveProfile.VISUAL.value)
        self.assertIn("adaptation", visual)
        self.assertIn("visual_elements", visual["adaptation"])
        self.assertEqual(visual["adaptation"]["spatial_organization"], "essential")
        
if __name__ == "__main__":
    unittest.main() 