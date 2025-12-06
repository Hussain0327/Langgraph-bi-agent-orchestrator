import sys
sys.path.insert(0, '/workspaces/multi_agent_workflow')
from src.ml.routing_classifier import RoutingClassifier
print('Starting retrain with 5 epochs...')
classifier = RoutingClassifier()
classifier.train(data_path='models/training_data.json', num_epochs=5, save_path='models/routing_classifier.pkl')
print('\nRetraining complete!')
print('New model saved to: models/routing_classifier.pkl')