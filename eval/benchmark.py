

import json
import time
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.langgraph_orchestrator import LangGraphOrchestrator
from src.gpt5_wrapper import GPT5Wrapper
from src.config import Config


class BenchmarkRunner:

    def __init__(
        self,
        enable_rag: bool = True,
        use_ml_routing: bool = False,
        output_dir: str = "./eval"
    ):

        self.enable_rag = enable_rag
        self.use_ml_routing = use_ml_routing
        self.output_dir = Path(output_dir)

        # Initialize orchestrator
        print(f"Initializing orchestrator (RAG={'ON' if enable_rag else 'OFF'})...")
        self.orchestrator = LangGraphOrchestrator(
            enable_rag=enable_rag
            # use_ml_routing will be added in Task B3
        )

        # Initialize LLM-as-judge (using GPT-4 for quality assessment)
        self.judge = GPT5Wrapper()

        # Pricing (GPT-5-nano)
        self.input_token_cost = 0.05 / 1_000_000   # $0.05 per 1M tokens
        self.output_token_cost = 0.40 / 1_000_000  # $0.40 per 1M tokens

    def load_test_queries(self, filepath: str = "eval/test_queries.json") -> List[Dict]:
        """Load test queries from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data['queries']

    def run_single_query(
        self,
        query_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        
        query_id = query_data['id']
        query_text = query_data['query']
        expected_agents = query_data.get('expected_agents', [])

        print(f"\n[Query {query_id}] {query_text[:60]}...")

        # Run query and measure latency
        start_time = time.time()

        try:
            result = self.orchestrator.orchestrate(
                query=query_text,
                use_memory=False  # Disable memory for consistent testing
            )

            latency = time.time() - start_time
            success = True
            error = None

        except Exception as e:
            latency = time.time() - start_time
            success = False
            error = str(e)
            result = {
                "query": query_text,
                "agents_consulted": [],
                "recommendation": f"Error: {error}",
                "detailed_findings": {}
            }

        # Extract metrics
        metrics = {
            "query_id": query_id,
            "query": query_text,
            "mode": "rag" if self.enable_rag else "no_rag",
            "success": success,
            "error": error,

            # Performance metrics
            "latency": round(latency, 2),

            # Routing metrics
            "agents_called": result.get("agents_consulted", []),
            "expected_agents": expected_agents,
            "routing_accuracy": self._calculate_routing_accuracy(
                result.get("agents_consulted", []),
                expected_agents
            ),

            # Response metrics
            "response": result.get("recommendation", ""),
            "response_length": len(result.get("recommendation", "")),

            # Citation metrics
            "citation_count": self._count_citations(result.get("recommendation", "")),
            "has_references": "References" in result.get("recommendation", ""),

            # Cost metrics (estimated)
            "estimated_cost": self._estimate_cost(result),

            # Detailed findings
            "detailed_findings": result.get("detailed_findings", {}),
        }

        print(f"  ✓ Latency: {metrics['latency']}s")
        print(f"  ✓ Agents: {metrics['agents_called']}")
        print(f"  ✓ Citations: {metrics['citation_count']}")

        return metrics

    def run_llm_judge_evaluation(
        self,
        query: str,
        response: str
    ) -> Dict[str, float]:
        """
        Use LLM-as-judge to evaluate response quality.

        Args:
            query: User query
            response: System response

        Returns:
            Dict with factuality, helpfulness, comprehensiveness scores (0-1)
        """
        judge_prompt = f"""Evaluate this business intelligence recommendation.

Query: {query}

Response: {response}

Rate the response on these criteria (0.0 to 1.0 scale):

1. **Factuality**: Are the claims accurate and well-supported? Do citations (if present) add credibility?
   - 0.0-0.3: Many factual errors or unsupported claims
   - 0.4-0.6: Some accuracy but lacks support
   - 0.7-0.8: Mostly accurate with good support
   - 0.9-1.0: Highly accurate with strong evidence

2. **Helpfulness**: Is the advice actionable and relevant to the query?
   - 0.0-0.3: Generic or irrelevant advice
   - 0.4-0.6: Somewhat helpful but lacks specifics
   - 0.7-0.8: Actionable and relevant
   - 0.9-1.0: Highly actionable with clear next steps

3. **Comprehensiveness**: Does it address all aspects of the query?
   - 0.0-0.3: Misses key aspects
   - 0.4-0.6: Addresses some aspects
   - 0.7-0.8: Covers most aspects well
   - 0.9-1.0: Thoroughly addresses all aspects

Return ONLY a JSON object (no markdown, no explanation):
{{"factuality": 0.8, "helpfulness": 0.9, "comprehensiveness": 0.85}}"""

        try:
            judge_response = self.judge.generate(
                input_text=judge_prompt,
                reasoning_effort="high",  # Careful evaluation
                text_verbosity="low",     # Just JSON
                max_output_tokens=100
            )

            # Parse JSON from response
            # Handle potential markdown code blocks
            judge_response_clean = judge_response.strip()
            judge_response_clean = re.sub(r'```json\s*', '', judge_response_clean)
            judge_response_clean = re.sub(r'```\s*', '', judge_response_clean)

            scores = json.loads(judge_response_clean)

            return {
                "factuality": scores.get("factuality", 0.5),
                "helpfulness": scores.get("helpfulness", 0.5),
                "comprehensiveness": scores.get("comprehensiveness", 0.5),
            }

        except Exception as e:
            print(f"  ⚠️  LLM judge failed: {e}")
            return {
                "factuality": 0.5,
                "helpfulness": 0.5,
                "comprehensiveness": 0.5,
            }

    def run_benchmark(
        self,
        num_queries: Optional[int] = None,
        include_llm_judge: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run full benchmark suite.

        Args:
            num_queries: Number of queries to run (None = all)
            include_llm_judge: Whether to run LLM-as-judge evaluation

        Returns:
            List of result dicts
        """
        # Load queries
        queries = self.load_test_queries()

        if num_queries:
            queries = queries[:num_queries]

        print("\n" + "="*70)
        print(f"Running Benchmark: {len(queries)} queries")
        print(f"Mode: {'RAG' if self.enable_rag else 'No RAG'}")
        print(f"LLM Judge: {'ON' if include_llm_judge else 'OFF'}")
        print("="*70)

        results = []

        for i, query_data in enumerate(queries, 1):
            print(f"\n--- Query {i}/{len(queries)} ---")

            # Run query
            metrics = self.run_single_query(query_data)

            # Add LLM judge scores
            if include_llm_judge and metrics['success']:
                print(f"  🔍 Running LLM judge...")
                scores = self.run_llm_judge_evaluation(
                    metrics['query'],
                    metrics['response']
                )
                metrics.update(scores)
                print(f"  ✓ Scores: F={scores['factuality']:.2f}, "
                      f"H={scores['helpfulness']:.2f}, "
                      f"C={scores['comprehensiveness']:.2f}")

            results.append(metrics)

            # Brief pause to avoid rate limiting
            time.sleep(1)

        return results

    def save_results(
        self,
        results: List[Dict[str, Any]],
        filename: Optional[str] = None
    ):
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode = "rag" if self.enable_rag else "no_rag"
            filename = f"results_{mode}_{timestamp}.json"

        filepath = self.output_dir / filename

        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "mode": "rag" if self.enable_rag else "no_rag",
                "num_queries": len(results),
                "model": Config.OPENAI_MODEL,
            },
            "results": results,
            "summary": self._generate_summary(results)
        }

        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Results saved to: {filepath}")
        return filepath

    def _calculate_routing_accuracy(
        self,
        actual: List[str],
        expected: List[str]
    ) -> float:
        """Calculate routing accuracy (Jaccard similarity)."""
        if not expected:
            return 1.0  # No expectation = always correct

        actual_set = set(actual)
        expected_set = set(expected)

        intersection = len(actual_set & expected_set)
        union = len(actual_set | expected_set)

        return round(intersection / union if union > 0 else 0.0, 2)

    def _count_citations(self, text: str) -> int:
        """Count citations in text (looking for patterns like 'et al.' or parenthetical years)."""
        # Pattern 1: "et al." citations
        et_al_count = len(re.findall(r'et al\.', text, re.IGNORECASE))

        # Pattern 2: "(Author, Year)" or "(Author et al., Year)"
        paren_citations = len(re.findall(r'\([A-Z][a-z]+.*?\d{4}\)', text))

        return max(et_al_count, paren_citations)

    def _estimate_cost(self, result: Dict[str, Any]) -> float:
        """Estimate cost based on typical token usage."""
        # Rough estimates (will be refined with actual token tracking)
        agents_count = len(result.get("agents_consulted", []))

        # Baseline cost
        base_cost = 0.10  # Router + synthesis

        # Per-agent cost
        agent_cost = agents_count * 0.05

        # RAG overhead
        rag_cost = 0.10 if self.enable_rag else 0.0

        return round(base_cost + agent_cost + rag_cost, 3)

    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        if not results:
            return {}

        successful = [r for r in results if r.get('success', False)]

        def avg(values):
            return round(sum(values) / len(values), 3) if values else 0.0

        summary = {
            "total_queries": len(results),
            "successful_queries": len(successful),
            "failed_queries": len(results) - len(successful),

            "avg_latency": avg([r['latency'] for r in successful]),
            "avg_cost": avg([r['estimated_cost'] for r in successful]),
            "avg_response_length": avg([r['response_length'] for r in successful]),
            "avg_citations": avg([r.get('citation_count', 0) for r in successful]),

            "citation_rate": round(
                sum(1 for r in successful if r.get('citation_count', 0) > 0) / len(successful) * 100, 1
            ) if successful else 0.0,

            "has_references_rate": round(
                sum(1 for r in successful if r.get('has_references', False)) / len(successful) * 100, 1
            ) if successful else 0.0,

            "avg_routing_accuracy": avg([r.get('routing_accuracy', 0) for r in successful]),
        }

        # Add LLM judge scores if available
        if successful and 'factuality' in successful[0]:
            summary.update({
                "avg_factuality": avg([r.get('factuality', 0) for r in successful]),
                "avg_helpfulness": avg([r.get('helpfulness', 0) for r in successful]),
                "avg_comprehensiveness": avg([r.get('comprehensiveness', 0) for r in successful]),
                "avg_overall_quality": avg([
                    (r.get('factuality', 0) + r.get('helpfulness', 0) + r.get('comprehensiveness', 0)) / 3
                    for r in successful
                ]),
            })

        return summary


def print_summary(summary: Dict[str, Any], mode: str):
    """Print formatted summary."""
    print("\n" + "="*70)
    print(f"BENCHMARK SUMMARY - {mode.upper()} MODE")
    print("="*70)

    print(f"\nQueries: {summary['successful_queries']}/{summary['total_queries']} successful")

    print("\n📊 Performance Metrics:")
    print(f"  Average Latency:        {summary['avg_latency']}s")
    print(f"  Average Cost:           ${summary['avg_cost']}")
    print(f"  Average Response Length: {summary['avg_response_length']} chars")

    print("\n📚 Citation Metrics:")
    print(f"  Average Citations:      {summary['avg_citations']}")
    print(f"  Citation Rate:          {summary['citation_rate']}%")
    print(f"  Has References:         {summary['has_references_rate']}%")

    print("\n🎯 Routing Metrics:")
    print(f"  Routing Accuracy:       {summary['avg_routing_accuracy']*100:.1f}%")

    if 'avg_factuality' in summary:
        print("\n⭐ Quality Metrics (LLM Judge):")
        print(f"  Factuality:             {summary['avg_factuality']:.2f}/1.0")
        print(f"  Helpfulness:            {summary['avg_helpfulness']:.2f}/1.0")
        print(f"  Comprehensiveness:      {summary['avg_comprehensiveness']:.2f}/1.0")
        print(f"  Overall Quality:        {summary['avg_overall_quality']:.2f}/1.0")

    print("\n" + "="*70)


def main():
    """Run benchmark with command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description='Run benchmark evaluation')
    parser.add_argument('--mode', choices=['rag', 'no_rag', 'both'], default='both',
                       help='Which mode to test')
    parser.add_argument('--num-queries', type=int, default=5,
                       help='Number of queries to run (default: 5)')
    parser.add_argument('--no-judge', action='store_true',
                       help='Skip LLM-as-judge evaluation')

    args = parser.parse_args()

    modes_to_test = []
    if args.mode in ['no_rag', 'both']:
        modes_to_test.append((False, 'no_rag'))
    if args.mode in ['rag', 'both']:
        modes_to_test.append((True, 'rag'))

    all_results = {}

    for enable_rag, mode_name in modes_to_test:
        runner = BenchmarkRunner(enable_rag=enable_rag)

        results = runner.run_benchmark(
            num_queries=args.num_queries,
            include_llm_judge=not args.no_judge
        )

        filepath = runner.save_results(results)

        # Load and print summary
        with open(filepath, 'r') as f:
            data = json.load(f)

        print_summary(data['summary'], mode_name)
        all_results[mode_name] = data['summary']

    # Compare modes if both were run
    if len(all_results) == 2:
        print("\n" + "="*70)
        print("COMPARISON: RAG vs No RAG")
        print("="*70)

        baseline = all_results['no_rag']
        rag = all_results['rag']

        def compare(name, baseline_val, rag_val, unit="", invert=False):
            diff = rag_val - baseline_val
            pct_change = (diff / baseline_val * 100) if baseline_val != 0 else 0

            if invert:
                pct_change = -pct_change

            symbol = "↑" if pct_change > 0 else "↓" if pct_change < 0 else "="
            color = "\033[92m" if pct_change > 0 else "\033[91m" if pct_change < 0 else "\033[93m"
            reset = "\033[0m"

            print(f"  {name:25} {baseline_val:8.2f}{unit} → {rag_val:8.2f}{unit}  "
                  f"{color}{symbol} {abs(pct_change):5.1f}%{reset}")

        print("\n📊 Performance:")
        compare("Latency", baseline['avg_latency'], rag['avg_latency'], "s", invert=True)
        compare("Cost", baseline['avg_cost'], rag['avg_cost'], "$", invert=True)

        print("\n📚 Citations:")
        compare("Citation Count", baseline['avg_citations'], rag['avg_citations'], "")
        compare("Citation Rate", baseline['citation_rate'], rag['citation_rate'], "%")

        if 'avg_factuality' in baseline and 'avg_factuality' in rag:
            print("\n⭐ Quality:")
            compare("Factuality", baseline['avg_factuality'], rag['avg_factuality'], "")
            compare("Helpfulness", baseline['avg_helpfulness'], rag['avg_helpfulness'], "")
            compare("Overall Quality", baseline['avg_overall_quality'], rag['avg_overall_quality'], "")

        print("\n" + "="*70)


if __name__ == "__main__":
    main()
