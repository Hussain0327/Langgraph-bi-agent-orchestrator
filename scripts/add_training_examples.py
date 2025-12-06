import json

def add_boundary_examples():
    with open('models/training_data.json', 'r') as f:
        data = json.load(f)
    new_examples = [{'query': 'Can you follow up with these 5 inbound leads?', 'agents': ['leadgen']}, {'query': 'I need help qualifying these prospects', 'agents': ['leadgen']}, {'query': 'Book sales calls with our top 10 leads', 'agents': ['leadgen']}, {'query': 'Send outreach emails to cold leads', 'agents': ['leadgen']}, {'query': 'Create a lead nurturing sequence', 'agents': ['leadgen']}, {'query': 'Optimize our manufacturing workflow', 'agents': ['operations']}, {'query': 'Improve team productivity and reduce bottlenecks', 'agents': ['operations']}, {'query': 'How do I automate our onboarding process?', 'agents': ['operations']}, {'query': 'Streamline our supply chain operations', 'agents': ['operations']}, {'query': 'Reduce operational costs in production', 'agents': ['operations']}, {'query': "What's the best pricing model for SaaS?", 'agents': ['financial', 'market']}, {'query': 'How can I reduce customer churn?', 'agents': ['market']}, {'query': 'Should I raise prices or increase volume?', 'agents': ['financial']}, {'query': "What's our target customer profile?", 'agents': ['market']}, {'query': 'How much revenue can we expect next quarter?', 'agents': ['financial']}, {'query': 'Set up automated email sequences', 'agents': ['operations']}, {'query': 'How much should I budget for paid ads?', 'agents': ['financial']}, {'query': 'Build a sales funnel automation', 'agents': ['operations', 'leadgen']}, {'query': 'Calculate our customer lifetime value', 'agents': ['financial']}, {'query': 'Design an onboarding workflow', 'agents': ['operations']}]
    data['train'].extend(new_examples)
    with open('models/training_data.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Added {len(new_examples)} boundary examples')
    print(f"Total training examples: {len(data['train'])}")
if __name__ == '__main__':
    add_boundary_examples()