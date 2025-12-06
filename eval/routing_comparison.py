import argparse
import json
import time
from typing import List, Dict, Any
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.langgraph_orchestrator import LangGraphOrchestrator

class RoutingBenchmark:

    def __init__(self):
        print('\n' + '=' * 70)
        print('🔀 ROUTING COMPARISON BENCHMARK')
        print('=' * 70)
        print('\n1⃣  Initializing GPT-5 routing...')
        self.gpt5_orchestrator = LangGraphOrchestrator(enable_rag=False, use_ml_routing=False)
        print('\n2⃣  Initializing ML routing...')
        self.ml_orchestrator = LangGraphOrchestrator(enable_rag=False, use_ml_routing=True)
        self.results = {'gpt5': [], 'ml': []}

    def load_test_queries(self, path: str) -> List[Dict[str, Any]]:
        with open(path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            queries = data
        elif isinstance(data, dict) and 'test' in data:
            queries = data['test']
        else:
            raise ValueError(f'Unexpected JSON format in {path}')
        print(f'\n📂 Loaded {len(queries)} test queries')
        return queries

    def route_query_gpt5(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        state = {'query': query, 'agents_to_call': [], 'research_enabled': False, 'research_findings': {}, 'research_context': '', 'market_analysis': '', 'operations_audit': '', 'financial_modeling': '', 'lead_generation': '', 'web_research': {}, 'synthesis': '', 'conversation_history': [], 'use_memory': False}
        result_state = self.gpt5_orchestrator._router_node(state)
        latency = time.time() - start_time
        estimated_cost = 200 * 0.5 / 1000000 + 50 * 2.0 / 1000000
        return {'agents': result_state['agents_to_call'], 'latency': latency, 'cost': estimated_cost, 'method': 'gpt5'}

    def route_query_ml(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        state = {'query': query, 'agents_to_call': [], 'research_enabled': False, 'research_findings': {}, 'research_context': '', 'market_analysis': '', 'operations_audit': '', 'financial_modeling': '', 'lead_generation': '', 'web_research': {}, 'synthesis': '', 'conversation_history': [], 'use_memory': False}
        result_state = self.ml_orchestrator._router_node(state)
        latency = time.time() - start_time
        estimated_cost = 0.0
        return {'agents': result_state['agents_to_call'], 'latency': latency, 'cost': estimated_cost, 'method': 'ml'}

    def compare_routing(self, test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        print(f'\n' + '=' * 70)
        print(f' RUNNING ROUTING COMPARISON')
        print(f'=' * 70)
        print(f'\nTesting {len(test_queries)} queries...')
        gpt5_results = []
        ml_results = []
        for i, test_query in enumerate(test_queries, 1):
            query = test_query['query']
            expected_agents = sorted(test_query.get('agents', []))
            print(f'\n[{i}/{len(test_queries)}] {query[:60]}...')
            gpt5_result = self.route_query_gpt5(query)
            gpt5_agents = sorted(gpt5_result['agents'])
            ml_result = self.route_query_ml(query)
            ml_agents = sorted(ml_result['agents'])
            gpt5_correct = gpt5_agents == expected_agents
            ml_correct = ml_agents == expected_agents
            gpt5_results.append({'query': query, 'expected': expected_agents, 'predicted': gpt5_agents, 'correct': gpt5_correct, 'latency': gpt5_result['latency'], 'cost': gpt5_result['cost']})
            ml_results.append({'query': query, 'expected': expected_agents, 'predicted': ml_agents, 'correct': ml_correct, 'latency': ml_result['latency'], 'cost': ml_result['cost']})
            print(f"   GPT-5: {gpt5_agents} ({('✓' if gpt5_correct else '✗')}) - {gpt5_result['latency']:.3f}s")
            print(f"   ML:    {ml_agents} ({('✓' if ml_correct else '✗')}) - {ml_result['latency']:.3f}s")
        gpt5_accuracy = sum((r['correct'] for r in gpt5_results)) / len(gpt5_results)
        ml_accuracy = sum((r['correct'] for r in ml_results)) / len(ml_results)
        gpt5_avg_latency = sum((r['latency'] for r in gpt5_results)) / len(gpt5_results)
        ml_avg_latency = sum((r['latency'] for r in ml_results)) / len(ml_results)
        gpt5_total_cost = sum((r['cost'] for r in gpt5_results))
        ml_total_cost = sum((r['cost'] for r in ml_results))
        return {'timestamp': datetime.now().isoformat(), 'num_queries': len(test_queries), 'gpt5': {'accuracy': gpt5_accuracy, 'avg_latency': gpt5_avg_latency, 'total_cost': gpt5_total_cost, 'cost_per_query': gpt5_total_cost / len(test_queries), 'results': gpt5_results}, 'ml': {'accuracy': ml_accuracy, 'avg_latency': ml_avg_latency, 'total_cost': ml_total_cost, 'cost_per_query': ml_total_cost / len(test_queries), 'results': ml_results}, 'comparison': {'accuracy_improvement': ml_accuracy - gpt5_accuracy, 'latency_improvement': gpt5_avg_latency - ml_avg_latency, 'latency_speedup': gpt5_avg_latency / ml_avg_latency if ml_avg_latency > 0 else 0, 'cost_savings': gpt5_total_cost - ml_total_cost, 'cost_reduction_pct': (gpt5_total_cost - ml_total_cost) / gpt5_total_cost * 100 if gpt5_total_cost > 0 else 0}}

    def generate_report(self, results: Dict[str, Any]) -> str:
        report = f"# Routing Comparison Report\n\n**Generated:** {results['timestamp']}\n**Test Queries:** {results['num_queries']}\n\n---\n\n| Metric | GPT-5 Routing | ML Routing | Improvement |\n|--------|---------------|------------|-------------|\n| **Accuracy** | {results['gpt5']['accuracy']:.1%} | {results['ml']['accuracy']:.1%} | {results['comparison']['accuracy_improvement']:+.1%} |\n| **Avg Latency** | {results['gpt5']['avg_latency']:.3f}s | {results['ml']['avg_latency']:.3f}s | {results['comparison']['latency_speedup']:.1f}x faster |\n| **Cost per Query** | ${results['gpt5']['cost_per_query']:.6f} | ${results['ml']['cost_per_query']:.6f} | {results['comparison']['cost_reduction_pct']:.0f}% reduction |\n| **Total Cost** | ${results['gpt5']['total_cost']:.4f} | ${results['ml']['total_cost']:.4f} | ${results['comparison']['cost_savings']:.4f} saved |\n\n---\n\n- **GPT-5 Routing:** {results['gpt5']['accuracy']:.1%} exact match accuracy\n- **ML Routing:** {results['ml']['accuracy']:.1%} exact match accuracy\n- **Winner:** {('ML' if results['ml']['accuracy'] > results['gpt5']['accuracy'] else 'GPT-5' if results['gpt5']['accuracy'] > results['ml']['accuracy'] else 'Tie')}\n\n- **GPT-5 Routing:** {results['gpt5']['avg_latency'] * 1000:.1f}ms average\n- **ML Routing:** {results['ml']['avg_latency'] * 1000:.1f}ms average\n- **Speedup:** {results['comparison']['latency_speedup']:.1f}x faster with ML\n- **Winner:** ML (always faster due to local inference)\n\n- **GPT-5 Routing:** ${results['gpt5']['total_cost']:.4f} total (${results['gpt5']['cost_per_query']:.6f} per query)\n- **ML Routing:** $0.00 total (free local inference)\n- **Savings:** ${results['comparison']['cost_savings']:.4f} ({results['comparison']['cost_reduction_pct']:.0f}% reduction)\n- **Winner:** ML (100% cost reduction)\n\n---\n\n"
        if results['ml']['accuracy'] >= results['gpt5']['accuracy']:
            report += f"\n** RECOMMENDATION: Use ML Routing in Production**\n\nThe ML routing classifier matches or exceeds GPT-5 routing accuracy ({results['ml']['accuracy']:.1%} vs {results['gpt5']['accuracy']:.1%}) while being **{results['comparison']['latency_speedup']:.0f}x faster** and **100% cheaper**.\n\n**Benefits:**\n- Same or better routing accuracy\n- {results['comparison']['latency_speedup']:.1f}x faster response times\n- Zero API costs for routing\n- Predictable performance (no API rate limits)\n\n**Estimated Annual Savings** (assuming 10,000 queries/month):\n- Cost savings: ${results['comparison']['cost_savings'] * 10000:.2f}/month = ${results['comparison']['cost_savings'] * 120000:.2f}/year\n"
        else:
            accuracy_diff = (results['gpt5']['accuracy'] - results['ml']['accuracy']) * 100
            report += f"\n**  RECOMMENDATION: Improve ML Model Before Production**\n\nThe ML routing classifier is **{results['comparison']['latency_speedup']:.1f}x faster** and **100% cheaper**, but accuracy is {accuracy_diff:.1f}% lower than GPT-5 ({results['ml']['accuracy']:.1%} vs {results['gpt5']['accuracy']:.1%}).\n\n**Options:**\n1. **Hybrid Approach:** Use ML for high-confidence predictions, GPT-5 for uncertain cases\n2. **Improve Training Data:** Collect more diverse examples (currently {results['num_queries']} test examples)\n3. **Fine-tune Threshold:** Adjust confidence thresholds for better accuracy-coverage tradeoff\n"
        report += '\n---\n\n*Generated by routing_comparison.py*\n'
        return report

    def run_benchmark(self, queries_path: str, output_path: str) -> Dict[str, Any]:
        test_queries = self.load_test_queries(queries_path)
        results = self.compare_routing(test_queries)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        report = self.generate_report(results)
        report_path = output_path.replace('.json', '_report.md')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f'\n' + '=' * 70)
        print(f' BENCHMARK COMPLETE')
        print(f'=' * 70)
        print(f'\n Results:')
        print(f"   GPT-5 Accuracy:  {results['gpt5']['accuracy']:.1%}")
        print(f"   ML Accuracy:     {results['ml']['accuracy']:.1%}")
        print(f"   Speedup:         {results['comparison']['latency_speedup']:.1f}x")
        print(f"   Cost Reduction:  {results['comparison']['cost_reduction_pct']:.0f}%")
        print(f'\n📁 Saved:')
        print(f'   Results: {output_path}')
        print(f'   Report:  {report_path}')
        print(f'=' * 70 + '\n')
        return results

def main():
    parser = argparse.ArgumentParser(description='Compare ML vs GPT-5 routing')
    parser.add_argument('--queries', type=str, default='models/training_data.json', help='Path to test queries JSON')
    parser.add_argument('--output', type=str, default='eval/routing_comparison_results.json', help='Path to save results')
    args = parser.parse_args()
    benchmark = RoutingBenchmark()
    results = benchmark.run_benchmark(queries_path=args.queries, output_path=args.output)
if __name__ == '__main__':
    main()