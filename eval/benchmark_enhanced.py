import json
import time
import re
import csv
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.langgraph_orchestrator import LangGraphOrchestrator
from src.gpt5_wrapper import GPT5Wrapper
from src.config import Config
from src.unified_llm import UnifiedLLM

class EnhancedBenchmarkRunner:

    def __init__(self, enable_rag: bool=True, use_ml_routing: bool=True, output_dir: str='./eval'):
        self.enable_rag = enable_rag
        self.use_ml_routing = use_ml_routing
        self.output_dir = Path(output_dir)
        self.model_strategy = Config.MODEL_STRATEGY
        print(f'Model Strategy: {self.model_strategy}')
        print(f"Initializing orchestrator (RAG={('ON' if enable_rag else 'OFF')}, ML Routing={('ON' if use_ml_routing else 'OFF')})...")
        self.orchestrator = LangGraphOrchestrator(enable_rag=enable_rag, use_ml_routing=use_ml_routing)
        self.judge = GPT5Wrapper()
        self.pricing = {'gpt5': {'input': 0.015 / 1000000, 'output': 0.06 / 1000000}, 'deepseek': {'input': 0.28 / 1000000, 'output': 0.42 / 1000000, 'input_cached': 0.028 / 1000000}}

    def load_test_queries(self, filepath: str='eval/test_queries.json') -> List[Dict]:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data['queries']

    def _track_agent_models(self, query: str) -> Dict[str, Dict[str, Any]]:
        agent_models = {}
        for agent_type in ['market', 'operations', 'financial', 'leadgen', 'research_synthesis']:
            llm = UnifiedLLM(agent_type=agent_type)
            provider, model_instance = llm._select_model()
            if provider == 'gpt5':
                model_name = 'GPT-5-nano'
            elif provider == 'deepseek':
                if hasattr(model_instance, 'model'):
                    model_name = model_instance.model
                else:
                    model_name = 'deepseek-unknown'
            else:
                model_name = 'unknown'
            agent_models[agent_type] = {'provider': provider, 'model_name': model_name, 'temperature': llm._get_optimal_temperature(), 'max_tokens': llm._get_optimal_max_tokens()}
        return agent_models

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def _calculate_cost(self, agent_type: str, input_tokens: int, output_tokens: int, provider: str) -> float:
        if provider == 'gpt5':
            input_cost = input_tokens * self.pricing['gpt5']['input']
            output_cost = output_tokens * self.pricing['gpt5']['output']
            return input_cost + output_cost
        elif provider == 'deepseek':
            input_cost = input_tokens * (0.5 * self.pricing['deepseek']['input_cached'] + 0.5 * self.pricing['deepseek']['input'])
            output_cost = output_tokens * self.pricing['deepseek']['output']
            return input_cost + output_cost
        return 0.0

    def run_single_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        query_id = query_data['id']
        query_text = query_data['query']
        expected_agents = query_data.get('expected_agents', [])
        print(f'\n[Query {query_id}] {query_text[:80]}...')
        agent_models = self._track_agent_models(query_text)
        start_time = time.time()
        try:
            result = self.orchestrator.orchestrate(query=query_text, use_memory=False)
            latency = time.time() - start_time
            success = True
            error = None
        except Exception as e:
            latency = time.time() - start_time
            success = False
            error = str(e)
            result = {'query': query_text, 'agents_consulted': [], 'recommendation': f'Error: {error}', 'detailed_findings': {}}
        agents_called = result.get('agents_consulted', [])
        agent_details = []
        total_cost = 0.0
        for agent in agents_called:
            model_info = agent_models.get(agent, {})
            provider = model_info.get('provider', 'unknown')
            model_name = model_info.get('model_name', 'unknown')
            agent_output = result.get('detailed_findings', {}).get(agent, '')
            input_tokens_estimate = 500 + self._estimate_tokens(query_text)
            if self.enable_rag and agent != 'research_synthesis':
                input_tokens_estimate += 2000
            output_tokens_estimate = self._estimate_tokens(agent_output)
            agent_cost = self._calculate_cost(agent, input_tokens_estimate, output_tokens_estimate, provider)
            total_cost += agent_cost
            agent_details.append({'agent': agent, 'provider': provider, 'model': model_name, 'temperature': model_info.get('temperature', 'unknown'), 'input_tokens_est': input_tokens_estimate, 'output_tokens_est': output_tokens_estimate, 'cost_est': round(agent_cost, 6), 'output_length': len(agent_output)})
        if self.enable_rag and 'research_synthesis' in agent_models:
            research_model = agent_models['research_synthesis']
            research_input = 5000
            research_output = 3000
            research_cost = self._calculate_cost('research_synthesis', research_input, research_output, research_model['provider'])
            total_cost += research_cost
            agent_details.append({'agent': 'research_synthesis', 'provider': research_model['provider'], 'model': research_model['model_name'], 'temperature': research_model['temperature'], 'input_tokens_est': research_input, 'output_tokens_est': research_output, 'cost_est': round(research_cost, 6), 'output_length': 0})
        routing_method = 'ML Classifier' if self.use_ml_routing else 'GPT-5 Semantic'
        routing_accuracy = self._calculate_routing_accuracy(agents_called, expected_agents)
        metrics = {'query_id': query_id, 'query': query_text, 'expected_agents': expected_agents, 'category': query_data.get('category', 'unknown'), 'complexity': query_data.get('complexity', 'unknown'), 'model_strategy': self.model_strategy, 'rag_enabled': self.enable_rag, 'ml_routing': self.use_ml_routing, 'routing_method': routing_method, 'success': success, 'error': error, 'latency_total': round(latency, 2), 'agents_called': agents_called, 'num_agents_called': len(agents_called), 'routing_accuracy': round(routing_accuracy, 3), 'routing_correct': routing_accuracy == 1.0, 'agent_details': agent_details, 'total_cost_est': round(total_cost, 6), 'response': result.get('recommendation', ''), 'response_length': len(result.get('recommendation', '')), 'citation_count': self._count_citations(result.get('recommendation', '')), 'has_references': 'References' in result.get('recommendation', '')}
        print(f"  ✓ Latency: {metrics['latency_total']}s")
        print(f'  ✓ Agents: {agents_called}')
        print(f'  ✓ Routing Accuracy: {routing_accuracy:.1%}')
        print(f'  ✓ Estimated Cost: ${total_cost:.6f}')
        for detail in agent_details:
            if detail['agent'] in agents_called or detail['agent'] == 'research_synthesis':
                print(f"    - {detail['agent']}: {detail['model']} (${detail['cost_est']:.6f})")
        return metrics

    def run_llm_judge_evaluation(self, query: str, response: str) -> Dict[str, float]:
        judge_prompt = f'Evaluate this business intelligence recommendation.\n\nQuery: {query}\n\nResponse: {response[:2000]}...  (truncated for evaluation)\n\nRate the response on these criteria (0.0 to 1.0 scale):\n\n1. **Factuality**: Are the claims accurate and well-supported?\n2. **Helpfulness**: Is the advice actionable and relevant?\n3. **Comprehensiveness**: Does it address all aspects of the query?\n\nReturn ONLY a JSON object (no markdown):\n{{"factuality": 0.8, "helpfulness": 0.9, "comprehensiveness": 0.85}}'
        try:
            judge_response = self.judge.generate(input_text=judge_prompt, reasoning_effort='low', max_output_tokens=100)
            judge_response_clean = judge_response.strip()
            judge_response_clean = re.sub('```json\\s*', '', judge_response_clean)
            judge_response_clean = re.sub('```\\s*', '', judge_response_clean)
            scores = json.loads(judge_response_clean)
            return {'factuality': scores.get('factuality', 0.5), 'helpfulness': scores.get('helpfulness', 0.5), 'comprehensiveness': scores.get('comprehensiveness', 0.5), 'quality_avg': (scores.get('factuality', 0.5) + scores.get('helpfulness', 0.5) + scores.get('comprehensiveness', 0.5)) / 3}
        except Exception as e:
            print(f'    LLM judge failed: {e}')
            return {'factuality': 0.5, 'helpfulness': 0.5, 'comprehensiveness': 0.5, 'quality_avg': 0.5}

    def run_benchmark(self, num_queries: Optional[int]=None, include_llm_judge: bool=True) -> List[Dict[str, Any]]:
        queries = self.load_test_queries()
        if num_queries:
            queries = queries[:num_queries]
        print('\n' + '=' * 80)
        print(f' ENHANCED BENCHMARK - Model Selection Tracking')
        print('=' * 80)
        print(f'Queries: {len(queries)}')
        print(f'Model Strategy: {self.model_strategy}')
        print(f"RAG: {('ON' if self.enable_rag else 'OFF')}")
        print(f"ML Routing: {('ON' if self.use_ml_routing else 'OFF')}")
        print(f"LLM Judge: {('ON' if include_llm_judge else 'OFF')}")
        print('=' * 80)
        results = []
        for i, query_data in enumerate(queries, 1):
            print(f"\n{'─' * 80}")
            print(f'Query {i}/{len(queries)}')
            print(f"{'─' * 80}")
            metrics = self.run_single_query(query_data)
            if include_llm_judge and metrics['success']:
                print(f'   Running LLM judge...')
                scores = self.run_llm_judge_evaluation(metrics['query'], metrics['response'])
                metrics.update(scores)
                print(f"  ✓ Quality: {scores['quality_avg']:.2f} (F={scores['factuality']:.2f}, H={scores['helpfulness']:.2f}, C={scores['comprehensiveness']:.2f})")
            results.append(metrics)
            time.sleep(2)
        return results

    def export_to_csv(self, results: List[Dict[str, Any]], filename: Optional[str]=None) -> Path:
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'benchmark_results_{timestamp}.csv'
        filepath = self.output_dir / filename
        csv_rows = []
        for result in results:
            base_row = {'query_id': result['query_id'], 'query': result['query'][:100] + '...' if len(result['query']) > 100 else result['query'], 'category': result['category'], 'complexity': result['complexity'], 'model_strategy': result['model_strategy'], 'rag_enabled': result['rag_enabled'], 'ml_routing': result['ml_routing'], 'routing_method': result['routing_method'], 'latency_total': result['latency_total'], 'agents_called': ', '.join(result['agents_called']), 'expected_agents': ', '.join(result['expected_agents']), 'routing_accuracy': result['routing_accuracy'], 'routing_correct': result['routing_correct'], 'total_cost': result['total_cost_est'], 'response_length': result['response_length'], 'citation_count': result['citation_count'], 'has_references': result['has_references'], 'factuality': result.get('factuality', 'N/A'), 'helpfulness': result.get('helpfulness', 'N/A'), 'comprehensiveness': result.get('comprehensiveness', 'N/A'), 'quality_avg': result.get('quality_avg', 'N/A')}
            for i, agent_detail in enumerate(result.get('agent_details', []), 1):
                base_row[f'agent{i}_name'] = agent_detail['agent']
                base_row[f'agent{i}_provider'] = agent_detail['provider']
                base_row[f'agent{i}_model'] = agent_detail['model']
                base_row[f'agent{i}_cost'] = agent_detail['cost_est']
                base_row[f'agent{i}_output_len'] = agent_detail['output_length']
            csv_rows.append(base_row)
        if csv_rows:
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(csv_rows)
        print(f'\n CSV exported to: {filepath}')
        return filepath

    def save_results(self, results: List[Dict[str, Any]], filename: Optional[str]=None) -> Path:
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'benchmark_detailed_{timestamp}.json'
        filepath = self.output_dir / filename
        output_data = {'metadata': {'timestamp': datetime.now().isoformat(), 'model_strategy': self.model_strategy, 'rag_enabled': self.enable_rag, 'ml_routing': self.use_ml_routing, 'num_queries': len(results)}, 'results': results, 'summary': self._generate_summary(results)}
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f' JSON results saved to: {filepath}')
        return filepath

    def _calculate_routing_accuracy(self, actual: List[str], expected: List[str]) -> float:
        if not expected:
            return 1.0
        actual_set = set(actual)
        expected_set = set(expected)
        intersection = len(actual_set & expected_set)
        union = len(actual_set | expected_set)
        return intersection / union if union > 0 else 0.0

    def _count_citations(self, text: str) -> int:
        pattern = '\\([A-Z][a-z]+\\s+et al\\.,\\s+\\d{4}\\)|\\[\\d+\\]'
        return len(re.findall(pattern, text))

    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        successful = [r for r in results if r['success']]
        if not successful:
            return {'error': 'No successful queries'}
        return {'total_queries': len(results), 'successful': len(successful), 'failed': len(results) - len(successful), 'avg_latency': round(sum((r['latency_total'] for r in successful)) / len(successful), 2), 'avg_cost': round(sum((r['total_cost_est'] for r in successful)) / len(successful), 6), 'total_cost': round(sum((r['total_cost_est'] for r in successful)), 4), 'avg_routing_accuracy': round(sum((r['routing_accuracy'] for r in successful)) / len(successful), 3), 'routing_perfect': sum((1 for r in successful if r['routing_correct'])), 'avg_response_length': round(sum((r['response_length'] for r in successful)) / len(successful)), 'avg_citations': round(sum((r['citation_count'] for r in successful)) / len(successful), 1), 'with_references': sum((1 for r in successful if r['has_references'])), 'avg_quality': round(sum((r.get('quality_avg', 0) for r in successful)) / len(successful), 3) if any(('quality_avg' in r for r in successful)) else None}

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run enhanced benchmark with model tracking')
    parser.add_argument('--num-queries', type=int, default=25, help='Number of queries to run')
    parser.add_argument('--no-rag', action='store_true', help='Disable RAG')
    parser.add_argument('--no-ml-routing', action='store_true', help='Disable ML routing')
    parser.add_argument('--no-judge', action='store_true', help='Disable LLM judge')
    args = parser.parse_args()
    runner = EnhancedBenchmarkRunner(enable_rag=not args.no_rag, use_ml_routing=not args.no_ml_routing)
    results = runner.run_benchmark(num_queries=args.num_queries, include_llm_judge=not args.no_judge)
    json_file = runner.save_results(results)
    csv_file = runner.export_to_csv(results)
    print('\n' + '=' * 80)
    print(' BENCHMARK SUMMARY')
    print('=' * 80)
    summary = runner._generate_summary(results)
    for key, value in summary.items():
        print(f'{key:25s}: {value}')
    print('=' * 80)
    print(f'\n Results saved:')
    print(f'   JSON: {json_file}')
    print(f'   CSV:  {csv_file}')
    print(f'\n Inspect CSV to identify weak routing decisions!')
if __name__ == '__main__':
    main()