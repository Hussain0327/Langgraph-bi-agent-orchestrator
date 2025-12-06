import sys
sys.path.insert(0, '/workspaces/multi_agent_workflow')
from models.inspect_model import route_text
import json
with open('models/training_data.json', 'r') as f:
    data = json.load(f)
val_data = data['val']
correct = 0
total = len(val_data)
print(f'Testing on {total} validation examples...\n')
for i, example in enumerate(val_data, 1):
    query = example['query']
    expected = set(example['agents'])
    result = route_text(query)
    predicted = {result['label']}
    match = predicted.intersection(expected)
    if match:
        correct += 1
        status = '✓'
    else:
        status = '✗'
    if not match:
        print(f'{status} Query {i}: Expected {expected}, got {predicted}')
accuracy = correct / total
print(f'\nValidation Accuracy: {accuracy:.1%} ({correct}/{total})')
if accuracy >= 0.85:
    print('Target accuracy achieved!')
else:
    print(f'Need {int((0.85 - accuracy) * total)} more correct to reach 85%')