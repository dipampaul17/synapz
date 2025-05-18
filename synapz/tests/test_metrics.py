#!/usr/bin/env python3
"""Unit tests for the metrics module."""

import unittest
import json
from typing import Dict, List, Any

from synapz.data.metrics import (
    calculate_readability_metrics,
    calculate_text_similarity,
    calculate_statistical_significance,
    extract_pedagogy_tags # Added for testing
)

class TestMetricsCalculations(unittest.TestCase):
    """Test suite for metrics calculation functions."""

    def test_calculate_readability_metrics_simple(self):
        """Test readability metrics with simple text."""
        text = """
        The mitochondria is the powerhouse of the cell. It is responsible for 
        generating ATP through cellular respiration. This process involves 
        the Krebs cycle and electron transport chain.
        """
        metrics = calculate_readability_metrics(text)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn("flesch_reading_ease", metrics)
        self.assertIn("flesch_kincaid_grade", metrics)
        self.assertGreater(metrics["lexicon_count"], 10) # Expect some words
        self.assertGreater(metrics["sentence_count"], 2) # Expect some sentences

        # Example: Check if Flesch Reading Ease is within a plausible range for this text
        # This text is fairly simple. Higher scores (60-100) are easier.
        self.assertGreaterEqual(metrics["flesch_reading_ease"], 50)
        self.assertLessEqual(metrics["flesch_reading_ease"], 100)


    def test_calculate_readability_metrics_complex(self):
        """Test readability metrics with more complex text."""
        complex_text = """
        The mitochondrion (plural mitochondria) is a double-membrane-bound organelle found 
        in most eukaryotic organisms. Mitochondria generate most of the cell's supply of 
        adenosine triphosphate (ATP), used as a source of chemical energy. The mitochondrion
        is popularly nicknamed the "powerhouse of the cell", a phrase coined by Philip Siekevitz in 1957.
        
        Mitochondria are commonly between 0.75 and 3 μm² in cross-section, but vary considerably in size
        and structure. Unless specifically stained, they are not visible under a light microscope.
        The number of mitochondria in a cell can vary widely by organism, tissue, and cell type.
        A mature red blood cell has no mitochondria, whereas a liver cell can have more than 2000.
        """
        metrics = calculate_readability_metrics(complex_text)
        self.assertIsInstance(metrics, dict)
        self.assertIn("flesch_reading_ease", metrics)
        # This text is more complex, so Flesch Reading Ease should be lower than the simple one.
        # Scores between 30-60 are typical for standard/difficult text.
        self.assertGreaterEqual(metrics["flesch_reading_ease"], 20)
        self.assertLessEqual(metrics["flesch_reading_ease"], 70)

    def test_calculate_readability_metrics_short_text(self):
        """Test readability with very short text (edge case)."""
        short_text = "Too short."
        metrics = calculate_readability_metrics(short_text)
        self.assertIsInstance(metrics, dict)
        # Check that default/fallback values are returned for too short content
        self.assertEqual(metrics["flesch_reading_ease"], 0.0)
        self.assertEqual(metrics["lexicon_count"], 0) # Based on current implementation for short text

    def test_calculate_text_similarity(self):
        """Test text similarity metrics."""
        text1 = """
        Python is a high-level, interpreted programming language known for its
        readability and simplicity. It was created by Guido van Rossum in 1991.
        """
        text2 = """
        Python is a popular programming language created by Guido van Rossum.
        It is known for being easy to read and simple to learn.
        """
        text3 = "This is completely different."

        similarity_text1_text2 = calculate_text_similarity(text1, text2)
        self.assertIsInstance(similarity_text1_text2, dict)
        self.assertIn("levenshtein_similarity", similarity_text1_text2)
        self.assertIn("jaccard_similarity", similarity_text1_text2)
        self.assertGreater(similarity_text1_text2["levenshtein_similarity"], 0.5) # Expect fairly high similarity
        self.assertGreater(similarity_text1_text2["jaccard_similarity"], 0.3)

        similarity_text1_text3 = calculate_text_similarity(text1, text3)
        self.assertLess(similarity_text1_text3["levenshtein_similarity"], 0.3) # Expect low similarity
        self.assertLess(similarity_text1_text3["jaccard_similarity"], 0.2)

        # Test empty strings
        similarity_empty = calculate_text_similarity("", "")
        self.assertEqual(similarity_empty["levenshtein_similarity"], 1.0)
        self.assertEqual(similarity_empty["jaccard_similarity"], 1.0)


    def test_calculate_statistical_significance(self):
        """Test statistical significance calculation."""
        adapted_metrics_data = [
            {"flesch_reading_ease": 65.2, "flesch_kincaid_grade": 7.8},
            {"flesch_reading_ease": 68.5, "flesch_kincaid_grade": 7.2},
            {"flesch_reading_ease": 70.1, "flesch_kincaid_grade": 6.9}
        ]
        control_metrics_data = [
            {"flesch_reading_ease": 55.3, "flesch_kincaid_grade": 9.1},
            {"flesch_reading_ease": 57.8, "flesch_kincaid_grade": 8.7},
            {"flesch_reading_ease": 59.2, "flesch_kincaid_grade": 8.5}
        ]

        significance_ease = calculate_statistical_significance(
            adapted_metrics_data, control_metrics_data, "flesch_reading_ease"
        )
        self.assertIsInstance(significance_ease, dict)
        self.assertIn("p_value", significance_ease)
        self.assertIn("effect_size", significance_ease)
        self.assertIn("is_significant", significance_ease)
        # For this data, adapted scores are higher, expect a p-value that might indicate significance
        # and a positive effect size. (Exact p-value depends on scipy's ttest_ind)
        self.assertLess(significance_ease["p_value"], 0.1) # Example threshold, might be significant
        self.assertGreater(significance_ease["effect_size"], 0)

        significance_grade = calculate_statistical_significance(
            adapted_metrics_data, control_metrics_data, "flesch_kincaid_grade"
        )
        self.assertIsInstance(significance_grade, dict)
        # For grade level, lower is often better, so adapted has lower (better) grades.
        # The mean_difference should be negative.
        self.assertLess(significance_grade["mean_difference"], 0)
        self.assertGreater(significance_grade["effect_size"], 0) # Effect size is absolute diff / std_dev

    def test_extract_pedagogy_tags(self):
        """Test extraction and counting of pedagogy tags."""
        interactions = [
            {"pedagogy_tags": ["visual", "example-driven", "chunked"]},
            {"pedagogy_tags": "visual"}, # Test string input
            {"pedagogy_tags": ["example-driven", "interactive"]},
            {"pedagogy_tags": []}, # Test empty list
            {"some_other_key": "value"} # Test missing key
        ]
        
        tag_counts = extract_pedagogy_tags(interactions)
        
        self.assertIsInstance(tag_counts, dict)
        self.assertEqual(tag_counts.get("visual"), 2)
        self.assertEqual(tag_counts.get("example-driven"), 2)
        self.assertEqual(tag_counts.get("chunked"), 1)
        self.assertEqual(tag_counts.get("interactive"), 1)
        self.assertNotIn("non_existent_tag", tag_counts)

if __name__ == "__main__":
    unittest.main() 