import json
import time
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.langgraph_orchestrator import LangGraphOrchestrator
from src.gpt5_wrapper import GPT5Wrapper
from src.config import Config

class BenchmarkRunner:

    def __init__(self, enable_rag: bool=True, use_ml_routing: bool=False, output_dir: str='./eval'):
        self.enable_rag = enable_rag
        self.use_ml_routing = use_ml_routing
        self.output_dir = Path(output_dir)
        print(f"Initializing orchestrator (RAG={('ON' if enable_rag else 'OFF')})...")
        self.orchestrator = LangGraphOrchestrator(enable_rag=enable_rag)
        self.judge = GPT5Wrapper()
        self.input_token_cost = 0.05 / 1000000
        self.output_token_cost = 0.4 / 1000000

    def load_test_queries(self, filepath: str='eval/test_queries.json') -> List[Dict]:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data['queries']

    def run_single_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        query_id = query_data['id']
        query_text = query_data['query']
        expected_agents = query_data.get('expected_agents', [])
        print(f'\n[Query {query_id}] {query_text[:60]}...')
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
        metrics = {'query_id': query_id, 'query': query_text, 'mode': 'rag' if self.enable_rag else 'no_rag', 'success': success, 'error': error, 'latency': round(latency, 2), 'agents_called': result.get('agents_consulted', []), 'expected_agents': expected_agents, 'routing_accuracy': self._calculate_routing_accuracy(result.get('agents_consulted', []), expected_agents), 'response': result.get('recommendation', ''), 'response_length': len(result.get('recommendation', '')), 'citation_count': self._count_citations(result.get('recommendation', '')), 'has_references': 'References' in result.get('recommendation', ''), 'estimated_cost': self._estimate_cost(result), 'detailed_findings': result.get('detailed_findings', {})}
        print(f"  ✓ Latency: {metrics['latency']}s")
        print(f"  ✓ Agents: {metrics['agents_called']}")
        print(f"  ✓ Citations: {metrics['citation_count']}")
        return metrics

    def run_llm_judge_evaluation(self, query: str, response: str) -> Dict[str, float]:
        judge_prompt = f'Evaluate this business intelligence recommendation.\n\nQuery: {query}\n\nResponse: {response}\n\nRate the response on these criteria (0.0 to 1.0 scale):\n\n1. **Factuality**: Are the claims accurate and well-supported? Do citations (if present) add credibility?\n   - 0.0-0.3: Many factual errors or unsupported claims\n   - 0.4-0.6: Some accuracy but lacks support\n   - 0.7-0.8: Mostly accurate with good support\n   - 0.9-1.0: Highly accurate with strong evidence\n\n2. **Helpfulness**: Is the advice actionable and relevant to the query?\n   - 0.0-0.3: Generic or irrelevant advice\n   - 0.4-0.6: Somewhat helpful but lacks specifics\n   - 0.7-0.8: Actionable and relevant\n   - 0.9-1.0: Highly actionable with clear next steps\n\n3. **Comprehensiveness**: Does it address all aspects of the query?\n   - 0.0-0.3: Misses key aspects\n   - 0.4-0.6: Addresses some aspects\n   - 0.7-0.8: Covers most aspects well\n   - 0.9-1.0: Thoroughly addresses all aspects\n\nReturn ONLY a JSON object (no markdown, no explanation):\n{{"factuality": 0.8, "helpfulness": 0.9, "comprehensiveness": 0.85}}'
        try:
            judge_response = self.judge.generate(input_text=judge_prompt, reasoning_effort='high', text_verbosity='low', max_output_tokens=100)
            judge_response_clean = judge_response.strip()
            judge_response_clean = re.sub('```json\\s*', '', judge_response_clean)
            judge_response_clean = re.sub('```\\s*', '', judge_response_clean)
            scores = json.loads(judge_response_clean)
            return {'factuality': scores.get('factuality', 0.5), 'helpfulness': scores.get('helpfulness', 0.5), 'comprehensiveness': scores.get('comprehensiveness', 0.5)}
        except Exception as e:
            print(f'    LLM judge failed: {e}')
            return {'factuality': 0.5, 'helpfulness': 0.5, 'comprehensiveness': 0.5}

    def run_benchmark(self, num_queries: Optional[int]=None, include_llm_judge: bool=True) -> List[Dict[str, Any]]:
        queries = self.load_test_queries()
        if num_queries:
            queries = queries[:num_queries]
        print('\n' + '=' * 70)
        print(f'Running Benchmark: {len(queries)} queries')
        print(f"Mode: {('RAG' if self.enable_rag else 'No RAG')}")
        print(f"LLM Judge: {('ON' if include_llm_judge else 'OFF')}")
        print('=' * 70)
        results = []
        for i, query_data in enumerate(queries, 1):
            print(f'\n--- Query {i}/{len(queries)} ---')
            metrics = self.run_single_query(query_data)
            if include_llm_judge and metrics['success']:
                print(f'   Running LLM judge...')
                scores = self.run_llm_judge_evaluation(metrics['query'], metrics['response'])
                metrics.update(scores)
                print(f"  ✓ Scores: F={scores['factuality']:.2f}, H={scores['helpfulness']:.2f}, C={scores['comprehensiveness']:.2f}")
            results.append(metrics)
            time.sleep(1)
        return results

    def save_results(self, results: List[Dict[str, Any]], filename: Optional[str]=None):
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            mode = 'rag' if self.enable_rag else 'no_rag'
            filename = f'results_{mode}_{timestamp}.json'
        filepath = self.output_dir / filename
        output_data = {'metadata': {'timestamp': datetime.now().isoformat(), 'mode': 'rag' if self.enable_rag else 'no_rag', 'num_queries': len(results), 'model': Config.OPENAI_MODEL}, 'results': results, 'summary': self._generate_summary(results)}
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f'\n✓ Results saved to: {filepath}')
        return filepath

    def _calculate_routing_accuracy(self, actual: List[str], expected: List[str]) -> float:
        if not expected:
            return 1.0
        actual_set = set(actual)
        expected_set = set(expected)
        intersection = len(actual_set & expected_set)
        union = len(actual_set | expected_set)
        return round(intersection / union if union > 0 else 0.0, 2)

    def _count_citations(self, text: str) -> int:
        et_al_count = len(re.findall('et al\\.', text, re.IGNORECASE))
        paren_citations = len(re.findall('\\([A-Z][a-z]+.*?\\d{4}\\)', text))
        return max(et_al_count, paren_citations)

    def _estimate_cost(self, result: Dict[str, Any]) -> float:
        agents_count = len(result.get('agents_consulted', []))
        base_cost = 0.1
        agent_cost = agents_count * 0.05
        rag_cost = 0.1 if self.enable_rag else 0.0
        return round(base_cost + agent_cost + rag_cost, 3)

    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {}
        successful = [r for r in results if r.get('success', False)]

        def avg(values):
            return round(sum(values) / len(values), 3) if values else 0.0
        summary = {'total_queries': len(results), 'successful_queries': len(successful), 'failed_queries': len(results) - len(successful), 'avg_latency': avg([r['latency'] for r in successful]), 'avg_cost': avg([r['estimated_cost'] for r in successful]), 'avg_response_length': avg([r['response_length'] for r in successful]), 'avg_citations': avg([r.get('citation_count', 0) for r in successful]), 'citation_rate': round(sum((1 for r in successful if r.get('citation_count', 0) > 0)) / len(successful) * 100, 1) if successful else 0.0, 'has_references_rate': round(sum((1 for r in successful if r.get('has_references', False))) / len(successful) * 100, 1) if successful else 0.0, 'avg_routing_accuracy': avg([r.get('routing_accuracy', 0) for r in successful])}
        if successful and 'factuality' in successful[0]:
            summary.update({'avg_factuality': avg([r.get('factuality', 0) for r in successful]), 'avg_helpfulness': avg([r.get('helpfulness', 0) for r in successful]), 'avg_comprehensiveness': avg([r.get('comprehensiveness', 0) for r in successful]), 'avg_overall_quality': avg([(r.get('factuality', 0) + r.get('helpfulness', 0) + r.get('comprehensiveness', 0)) / 3 for r in successful])})
        return summary

def print_summary(summary: Dict[str, Any], mode: str):
    print('\n' + '=' * 70)
    print(f'BENCHMARK SUMMARY - {mode.upper()} MODE')
    print('=' * 70)
    print(f"\nQueries: {summary['successful_queries']}/{summary['total_queries']} successful")
    print('\n Performance Metrics:')
    print(f"  Average Latency:        {summary['avg_latency']}s")
    print(f"  Average Cost:           ${summary['avg_cost']}")
    print(f"  Average Response Length: {summary['avg_response_length']} chars")
    print('\n📚 Citation Metrics:')
    print(f"  Average Citations:      {summary['avg_citations']}")
    print(f"  Citation Rate:          {summary['citation_rate']}%")
    print(f"  Has References:         {summary['has_references_rate']}%")
    print('\n Routing Metrics:')
    print(f"  Routing Accuracy:       {summary['avg_routing_accuracy'] * 100:.1f}%")
    if 'avg_factuality' in summary:
        print('\n Quality Metrics (LLM Judge):')
        print(f"  Factuality:             {summary['avg_factuality']:.2f}/1.0")
        print(f"  Helpfulness:            {summary['avg_helpfulness']:.2f}/1.0")
        print(f"  Comprehensiveness:      {summary['avg_comprehensiveness']:.2f}/1.0")
        print(f"  Overall Quality:        {summary['avg_overall_quality']:.2f}/1.0")
    print('\n' + '=' * 70)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run benchmark evaluation')
    parser.add_argument('--mode', choices=['rag', 'no_rag', 'both'], default='both', help='Which mode to test')
    parser.add_argument('--num-queries', type=int, default=5, help='Number of queries to run (default: 5)')
    parser.add_argument('--no-judge', action='store_true', help='Skip LLM-as-judge evaluation')
    args = parser.parse_args()
    modes_to_test = []
    if args.mode in ['no_rag', 'both']:
        modes_to_test.append((False, 'no_rag'))
    if args.mode in ['rag', 'both']:
        modes_to_test.append((True, 'rag'))
    all_results = {}
    for enable_rag, mode_name in modes_to_test:
        runner = BenchmarkRunner(enable_rag=enable_rag)
        results = runner.run_benchmark(num_queries=args.num_queries, include_llm_judge=not args.no_judge)
        filepath = runner.save_results(results)
        with open(filepath, 'r') as f:
            data = json.load(f)
        print_summary(data['summary'], mode_name)
        all_results[mode_name] = data['summary']
    if len(all_results) == 2:
        print('\n' + '=' * 70)
        print('COMPARISON: RAG vs No RAG')
        print('=' * 70)
        baseline = all_results['no_rag']
        rag = all_results['rag']

        def compare(name, baseline_val, rag_val, unit='', invert=False):
            diff = rag_val - baseline_val
            pct_change = diff / baseline_val * 100 if baseline_val != 0 else 0
            if invert:
                pct_change = -pct_change
            symbol = '↑' if pct_change > 0 else '↓' if pct_change < 0 else '='
            color = '\x1b[92m' if pct_change > 0 else '\x1b[91m' if pct_change < 0 else '\x1b[93m'
            reset = '\x1b[0m'
            print(f'  {name:25} {baseline_val:8.2f}{unit} → {rag_val:8.2f}{unit}  {color}{symbol} {abs(pct_change):5.1f}%{reset}')
        print('\n Performance:')
        compare('Latency', baseline['avg_latency'], rag['avg_latency'], 's', invert=True)
        compare('Cost', baseline['avg_cost'], rag['avg_cost'], '$', invert=True)
        print('\n📚 Citations:')
        compare('Citation Count', baseline['avg_citations'], rag['avg_citations'], '')
        compare('Citation Rate', baseline['citation_rate'], rag['citation_rate'], '%')
        if 'avg_factuality' in baseline and 'avg_factuality' in rag:
            print('\n Quality:')
            compare('Factuality', baseline['avg_factuality'], rag['avg_factuality'], '')
            compare('Helpfulness', baseline['avg_helpfulness'], rag['avg_helpfulness'], '')
            compare('Overall Quality', baseline['avg_overall_quality'], rag['avg_overall_quality'], '')
        print('\n' + '=' * 70)
if __name__ == '__main__':
    main()