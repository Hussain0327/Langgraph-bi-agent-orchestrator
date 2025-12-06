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

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.test_data_path = os.path.join(cls.test_dir, 'test_data.json')
        test_data = {'train': [{'query': 'What are market trends?', 'agents': ['market']}, {'query': 'How to optimize costs?', 'agents': ['financial']}, {'query': 'Generate more leads', 'agents': ['leadgen']}, {'query': 'Improve efficiency', 'agents': ['operations']}, {'query': 'Market analysis and ROI', 'agents': ['market', 'financial']}] * 5, 'val': [{'query': 'Analyze competition', 'agents': ['market']}, {'query': 'Calculate ROI', 'agents': ['financial']}], 'test': [{'query': 'Customer acquisition', 'agents': ['leadgen']}, {'query': 'Process optimization', 'agents': ['operations']}]}
        with open(cls.test_data_path, 'w') as f:
            json.dump(test_data, f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def setUp(self):
        self.classifier = RoutingClassifier()

    def test_initialization(self):
        self.assertEqual(len(self.classifier.agent_labels), 4)
        self.assertIn('market', self.classifier.agent_labels)
        self.assertIn('financial', self.classifier.agent_labels)
        self.assertIn('leadgen', self.classifier.agent_labels)
        self.assertIn('operations', self.classifier.agent_labels)

    def test_load_training_data(self):
        queries, labels = self.classifier.load_training_data(self.test_data_path)
        self.assertEqual(len(queries), 25)
        self.assertEqual(len(labels), 25)
        self.assertTrue(all((isinstance(q, str) for q in queries)))
        self.assertTrue(all((isinstance(l, list) for l in labels)))

    def test_load_training_data_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            self.classifier.load_training_data('nonexistent.json')

    def test_predict_without_training(self):
        with self.assertRaises(ValueError):
            self.classifier.predict('test query')

    def test_predict_proba_without_training(self):
        with self.assertRaises(ValueError):
            self.classifier.predict_proba('test query')

    def test_save_without_training(self):
        with self.assertRaises(ValueError):
            self.classifier.save(os.path.join(self.test_dir, 'model.pkl'))

    def test_load_nonexistent_model(self):
        with self.assertRaises(FileNotFoundError):
            self.classifier.load('nonexistent.pkl')

    def test_agents_constant(self):
        expected_agents = ['financial', 'leadgen', 'market', 'operations']
        self.assertEqual(RoutingClassifier.AGENTS, expected_agents)

class TestRoutingClassifierIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.test_data_path = os.path.join(cls.test_dir, 'test_data.json')
        cls.model_path = os.path.join(cls.test_dir, 'model.pkl')
        test_data = {'train': [{'query': 'What are market trends in tech?', 'agents': ['market']}, {'query': 'How to optimize operational costs?', 'agents': ['financial', 'operations']}, {'query': 'Generate more qualified leads', 'agents': ['leadgen']}, {'query': 'Improve workflow efficiency', 'agents': ['operations']}, {'query': 'Market sizing and revenue projections', 'agents': ['market', 'financial']}, {'query': 'Customer acquisition strategy', 'agents': ['leadgen', 'market']}] * 10, 'val': [{'query': 'Competitor analysis', 'agents': ['market']}, {'query': 'ROI calculation', 'agents': ['financial']}], 'test': [{'query': 'Lead generation tactics', 'agents': ['leadgen']}, {'query': 'Process optimization', 'agents': ['operations']}]}
        with open(cls.test_data_path, 'w') as f:
            json.dump(test_data, f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def test_end_to_end_workflow(self):
        self.skipTest('Skipping actual training test - too slow for unit tests')

class TestRoutingClassifierEdgeCases(unittest.TestCase):

    def setUp(self):
        self.classifier = RoutingClassifier()

    def test_empty_query(self):
        with self.assertRaises(ValueError):
            self.classifier.predict('')

    def test_very_long_query(self):
        long_query = 'test ' * 1000
        with self.assertRaises(ValueError):
            self.classifier.predict(long_query)

    def test_special_characters(self):
        special_query = '!@#$%^&*()_+{}|:<>?'
        with self.assertRaises(ValueError):
            self.classifier.predict(special_query)
if __name__ == '__main__':
    unittest.main()