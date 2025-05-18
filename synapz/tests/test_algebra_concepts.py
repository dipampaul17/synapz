"""Tests for the algebra concepts functionality."""

import unittest
import os
from pathlib import Path

from synapz.models.concepts.algebra_concepts import (
    ALGEBRA_CONCEPTS,
    CONCEPTS_DIR,
    load_concept,
    get_all_concepts,
    get_concepts_by_difficulty,
    get_concept_sequence
)

class TestAlgebraConcepts(unittest.TestCase):
    """Test suite for algebra concepts."""
    
    def test_concept_structure(self):
        """Test that all concepts have the required structure."""
        for concept in ALGEBRA_CONCEPTS:
            # Check required keys
            self.assertIn("id", concept)
            self.assertIn("title", concept)
            self.assertIn("difficulty", concept)
            self.assertIn("description", concept)
            self.assertIn("examples", concept)
            
            # Check types
            self.assertTrue(isinstance(concept["id"], str))
            self.assertTrue(isinstance(concept["title"], str))
            self.assertTrue(isinstance(concept["difficulty"], int))
            self.assertTrue(1 <= concept["difficulty"] <= 5)
            self.assertTrue(isinstance(concept["description"], str))
            self.assertTrue(isinstance(concept["examples"], list))
            self.assertTrue(len(concept["examples"]) >= 1)
            
            # Check description length
            self.assertTrue(len(concept["description"]) <= 150, 
                           f"Description for {concept['id']} exceeds 150 words")
    
    def test_concept_files_created(self):
        """Test that concept JSON files are created on disk."""
        concept_ids = [c["id"] for c in ALGEBRA_CONCEPTS]
        
        for concept_id in concept_ids:
            file_path = CONCEPTS_DIR / f"{concept_id}.json"
            self.assertTrue(file_path.exists(), f"Concept file {file_path} does not exist")
    
    def test_load_concept(self):
        """Test loading concepts."""
        for concept_id in [c["id"] for c in ALGEBRA_CONCEPTS]:
            concept = load_concept(concept_id)
            self.assertEqual(concept["id"], concept_id)
    
    def test_get_all_concepts(self):
        """Test getting all concepts."""
        concepts = get_all_concepts()
        self.assertEqual(len(concepts), 10)  # We should have 10 algebra concepts
        
        # Check that all concepts are included
        concept_ids = [c["id"] for c in concepts]
        expected_ids = ["variables", "expressions", "equations", "inequalities", 
                       "factoring", "quadratics", "systems", "functions", 
                       "exponents", "sequences"]
        for expected_id in expected_ids:
            self.assertIn(expected_id, concept_ids)
    
    def test_get_concepts_by_difficulty(self):
        """Test getting concepts by difficulty level."""
        # Test level 1
        level_1 = get_concepts_by_difficulty(1)
        self.assertEqual(len(level_1), 2)
        self.assertTrue(all(c["difficulty"] == 1 for c in level_1))
        
        # Test level 3
        level_3 = get_concepts_by_difficulty(3)
        self.assertEqual(len(level_3), 3)
        self.assertTrue(all(c["difficulty"] == 3 for c in level_3))
        
        # Test non-existent level
        level_6 = get_concepts_by_difficulty(6)
        self.assertEqual(len(level_6), 0)
    
    def test_get_concept_sequence(self):
        """Test getting a sequence of concepts by difficulty range."""
        # Test beginner sequence (levels 1-2)
        beginner = get_concept_sequence(1, 2)
        self.assertEqual(len(beginner), 4)
        self.assertTrue(all(1 <= c["difficulty"] <= 2 for c in beginner))
        
        # Test advanced sequence (levels 4-5)
        advanced = get_concept_sequence(4, 5)
        self.assertEqual(len(advanced), 3)
        self.assertTrue(all(4 <= c["difficulty"] <= 5 for c in advanced))
        
        # Test custom range
        custom = get_concept_sequence(2, 3)
        self.assertEqual(len(custom), 5)
        self.assertTrue(all(2 <= c["difficulty"] <= 3 for c in custom))
        
if __name__ == "__main__":
    unittest.main() 