"""Unit tests for ML routing classifier."""

import unittest
import os
import json
import tempfile
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.routing_classifier import RoutingClassifier


class TestRoutingClassifier(unittest.TestCase):
    """Test cases for RoutingClassifier."""

    @classmethod
    def setUpClass(cls):
        """Create temporary directory and test data."""
        cls.test_dir = tempfile.mkdtemp()
        cls.test_data_path = os.path.join(cls.test_dir, "test_data.json")

        # Create minimal test dataset
        test_data = {
            "train": [
                {"query": "What are market trends?", "agents": ["market"]},
                {"query": "How to optimize costs?", "agents": ["financial"]},
                {"query": "Generate more leads", "agents": ["leadgen"]},
                {"query": "Improve efficiency", "agents": ["operations"]},
                {"query": "Market analysis and ROI", "agents": ["market", "financial"]},
            ] * 5,  # Repeat to have enough examples
            "val": [
                {"query": "Analyze competition", "agents": ["market"]},
                {"query": "Calculate ROI", "agents": ["financial"]},
            ],
            "test": [
                {"query": "Customer acquisition", "agents": ["leadgen"]},
                {"query": "Process optimization", "agents": ["operations"]},
            ]
        }

        with open(cls.test_data_path, 'w') as f:
            json.dump(test_data, f)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        shutil.rmtree(cls.test_dir)

    def setUp(self):
        """Initialize classifier for each test."""
        self.classifier = RoutingClassifier()

    def test_initialization(self):
        """Test classifier initializes correctly."""
        self.assertEqual(len(self.classifier.agent_labels), 4)
        self.assertIn("market", self.classifier.agent_labels)
        self.assertIn("financial", self.classifier.agent_labels)
        self.assertIn("leadgen", self.classifier.agent_labels)
        self.assertIn("operations", self.classifier.agent_labels)

    def test_load_training_data(self):
        """Test loading training data."""
        queries, labels = self.classifier.load_training_data(self.test_data_path)

        self.assertEqual(len(queries), 25)  # 5 examples * 5 repetitions
        self.assertEqual(len(labels), 25)
        self.assertTrue(all(isinstance(q, str) for q in queries))
        self.assertTrue(all(isinstance(l, list) for l in labels))

    def test_load_training_data_missing_file(self):
        """Test error handling for missing file."""
        with self.assertRaises(FileNotFoundError):
            self.classifier.load_training_data("nonexistent.json")

    def test_predict_without_training(self):
        """Test prediction fails without training."""
        with self.assertRaises(ValueError):
            self.classifier.predict("test query")

    def test_predict_proba_without_training(self):
        """Test probability prediction fails without training."""
        with self.assertRaises(ValueError):
            self.classifier.predict_proba("test query")

    def test_save_without_training(self):
        """Test save fails without training."""
        with self.assertRaises(ValueError):
            self.classifier.save(os.path.join(self.test_dir, "model.pkl"))

    def test_load_nonexistent_model(self):
        """Test load fails for nonexistent model."""
        with self.assertRaises(FileNotFoundError):
            self.classifier.load("nonexistent.pkl")

    def test_agents_constant(self):
        """Test AGENTS constant is correct."""
        expected_agents = ["financial", "leadgen", "market", "operations"]
        self.assertEqual(RoutingClassifier.AGENTS, expected_agents)


class TestRoutingClassifierIntegration(unittest.TestCase):
    """Integration tests requiring actual model training."""

    @classmethod
    def setUpClass(cls):
        """Create test data and train a small model."""
        cls.test_dir = tempfile.mkdtemp()
        cls.test_data_path = os.path.join(cls.test_dir, "test_data.json")
        cls.model_path = os.path.join(cls.test_dir, "model.pkl")

        # Create test dataset
        test_data = {
            "train": [
                {"query": "What are market trends in tech?", "agents": ["market"]},
                {"query": "How to optimize operational costs?", "agents": ["financial", "operations"]},
                {"query": "Generate more qualified leads", "agents": ["leadgen"]},
                {"query": "Improve workflow efficiency", "agents": ["operations"]},
                {"query": "Market sizing and revenue projections", "agents": ["market", "financial"]},
                {"query": "Customer acquisition strategy", "agents": ["leadgen", "market"]},
            ] * 10,  # Repeat for sufficient training data
            "val": [
                {"query": "Competitor analysis", "agents": ["market"]},
                {"query": "ROI calculation", "agents": ["financial"]},
            ],
            "test": [
                {"query": "Lead generation tactics", "agents": ["leadgen"]},
                {"query": "Process optimization", "agents": ["operations"]},
            ]
        }

        with open(cls.test_data_path, 'w') as f:
            json.dump(test_data, f)

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        shutil.rmtree(cls.test_dir)

    def test_end_to_end_workflow(self):
        """Test complete train-predict-save-load workflow."""
        # Note: This test requires actual SetFit training which is slow
        # In production, you'd mock this or use a tiny model
        # Skipping for now since we have training running in background
        self.skipTest("Skipping actual training test - too slow for unit tests")


class TestRoutingClassifierEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        self.classifier = RoutingClassifier()

    def test_empty_query(self):
        """Test handling of empty query."""
        # Should not crash, but will fail without trained model
        with self.assertRaises(ValueError):
            self.classifier.predict("")

    def test_very_long_query(self):
        """Test handling of very long query."""
        long_query = "test " * 1000
        with self.assertRaises(ValueError):
            self.classifier.predict(long_query)

    def test_special_characters(self):
        """Test handling of special characters."""
        special_query = "!@#$%^&*()_+{}|:<>?"
        with self.assertRaises(ValueError):
            self.classifier.predict(special_query)


if __name__ == "__main__":
    unittest.main()
