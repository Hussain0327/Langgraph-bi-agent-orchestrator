"""Unit tests for A/B testing framework."""

import unittest
import os
import json
import tempfile
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ab_testing import ABTestManager


class TestABTestManager(unittest.TestCase):
    """Test cases for ABTestManager."""

    def setUp(self):
        """Create temporary directory for each test."""
        self.test_dir = tempfile.mkdtemp()
        self.ab_test = ABTestManager(
            experiment_name="test_experiment",
            control="baseline",
            treatment="new_feature",
            split_ratio=0.5,
            results_dir=self.test_dir
        )

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test ABTestManager initializes correctly."""
        self.assertEqual(self.ab_test.experiment_name, "test_experiment")
        self.assertEqual(self.ab_test.control, "baseline")
        self.assertEqual(self.ab_test.treatment, "new_feature")
        self.assertEqual(self.ab_test.split_ratio, 0.5)

    def test_deterministic_assignment(self):
        """Test user assignment is deterministic."""
        user_id = "user_123"
        group1 = self.ab_test.assign_user(user_id)
        group2 = self.ab_test.assign_user(user_id)
        group3 = self.ab_test.assign_user(user_id)

        self.assertEqual(group1, group2)
        self.assertEqual(group2, group3)

    def test_split_ratio(self):
        """Test split ratio is approximately correct."""
        num_users = 1000
        assignments = [self.ab_test.assign_user(f"user_{i}") for i in range(num_users)]

        treatment_count = sum(1 for a in assignments if a == "treatment")
        treatment_ratio = treatment_count / num_users

        # Should be close to 50% with some tolerance
        self.assertGreater(treatment_ratio, 0.45)
        self.assertLess(treatment_ratio, 0.55)

    def test_log_result(self):
        """Test logging results."""
        self.ab_test.log_result(
            user_id="user_1",
            query="test query",
            response="test response",
            metrics={"latency": 1.5, "quality": 0.8}
        )

        # Check result was saved
        results_path = os.path.join(self.test_dir, "test_experiment.json")
        self.assertTrue(os.path.exists(results_path))

        with open(results_path, 'r') as f:
            data = json.load(f)

        # Check structure
        self.assertIn("control_results", data)
        self.assertIn("treatment_results", data)

        # One of them should have 1 result
        total_results = len(data["control_results"]) + len(data["treatment_results"])
        self.assertEqual(total_results, 1)

    def test_get_metric_values(self):
        """Test extracting metric values."""
        # Log some results
        for i in range(5):
            group = self.ab_test.assign_user(f"user_{i}")
            self.ab_test.log_result(
                user_id=f"user_{i}",
                query=f"query_{i}",
                response=f"response_{i}",
                metrics={"latency": i * 0.1, "quality": i * 0.2}
            )

        # Get latency values for each group
        control_latencies = self.ab_test.get_metric_values("control", "latency")
        treatment_latencies = self.ab_test.get_metric_values("treatment", "latency")

        # Should have some values in each
        self.assertTrue(len(control_latencies) + len(treatment_latencies) == 5)
        self.assertTrue(all(isinstance(v, float) for v in control_latencies))
        self.assertTrue(all(isinstance(v, float) for v in treatment_latencies))

    def test_calculate_statistics_empty(self):
        """Test statistics with empty data."""
        stats = self.ab_test.calculate_statistics([], [])

        self.assertEqual(stats["control_mean"], 0.0)
        self.assertEqual(stats["treatment_mean"], 0.0)
        self.assertEqual(stats["p_value"], 1.0)
        self.assertFalse(stats["significant"])

    def test_calculate_statistics_normal(self):
        """Test statistics with normal data."""
        control = [1.0, 1.1, 0.9, 1.2, 1.0]
        treatment = [1.5, 1.6, 1.4, 1.7, 1.5]

        stats = self.ab_test.calculate_statistics(control, treatment)

        self.assertGreater(stats["treatment_mean"], stats["control_mean"])
        self.assertGreater(stats["difference"], 0)
        self.assertIn("p_value", stats)
        self.assertIn("significant", stats)

    def test_save_and_load(self):
        """Test saving and loading results."""
        # Log a result
        self.ab_test.log_result(
            user_id="user_1",
            query="test",
            response="response",
            metrics={"latency": 1.0}
        )

        # Create new manager and load
        new_manager = ABTestManager(
            experiment_name="test_experiment",
            control="baseline",
            treatment="new_feature",
            results_dir=self.test_dir
        )

        # Should load existing results
        total_results = (
            len(new_manager.results["control_results"]) +
            len(new_manager.results["treatment_results"])
        )
        self.assertEqual(total_results, 1)

    def test_recommendation_logic(self):
        """Test recommendation generation."""
        # Mock analysis result
        analysis = {
            "summary": {
                "significant_improvements": 3,
                "total_metrics": 4
            }
        }

        recommendation = self.ab_test._get_recommendation(analysis)
        self.assertIn("DEPLOY", recommendation.upper())


class TestABTestManagerEdgeCases(unittest.TestCase):
    """Test edge cases."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_extreme_split_ratio(self):
        """Test extreme split ratios."""
        # 100% treatment
        ab_test = ABTestManager(
            experiment_name="test_100",
            control="a",
            treatment="b",
            split_ratio=1.0,
            results_dir=self.test_dir
        )

        assignments = [ab_test.assign_user(f"user_{i}") for i in range(100)]
        self.assertEqual(sum(1 for a in assignments if a == "treatment"), 100)

        # 0% treatment
        ab_test = ABTestManager(
            experiment_name="test_0",
            control="a",
            treatment="b",
            split_ratio=0.0,
            results_dir=self.test_dir
        )

        assignments = [ab_test.assign_user(f"user_{i}") for i in range(100)]
        self.assertEqual(sum(1 for a in assignments if a == "control"), 100)

    def test_missing_metrics(self):
        """Test handling of missing metrics."""
        ab_test = ABTestManager(
            experiment_name="test_missing",
            control="a",
            treatment="b",
            results_dir=self.test_dir
        )

        ab_test.log_result(
            user_id="user_1",
            query="test",
            response="response",
            metrics={"latency": 1.0}  # Only latency, no quality
        )

        # Should not crash when getting quality values
        quality_values = ab_test.get_metric_values("control", "quality")
        self.assertEqual(len(quality_values), 0)


if __name__ == "__main__":
    unittest.main()
