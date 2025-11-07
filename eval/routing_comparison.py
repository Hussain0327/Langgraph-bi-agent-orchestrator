"""
Routing Comparison Benchmark: ML Classifier vs GPT-5

Compares ML routing classifier against GPT-5 semantic routing on:
- Accuracy (vs expected agents)
- Latency (routing time)
- Cost (GPT-5 API costs vs free ML)

Usage:
    python eval/routing_comparison.py --queries eval/test_queries.json --output eval/routing_comparison_results.json
"""

import argparse
import json
import time
from typing import List, Dict, Any
from datetime import datetime

# Add parent directory to path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.langgraph_orchestrator import LangGraphOrchestrator


class RoutingBenchmark:
    """Compare ML routing vs GPT-5 routing performance."""

    def __init__(self):
        """Initialize orchestrators for both routing methods."""
        print("\n" + "="*70)
        print("🔀 ROUTING COMPARISON BENCHMARK")
        print("="*70)

        # Initialize GPT-5 routing orchestrator
        print("\n1️⃣  Initializing GPT-5 routing...")
        self.gpt5_orchestrator = LangGraphOrchestrator(
            enable_rag=False,  # Disable RAG for faster routing tests
            use_ml_routing=False
        )

        # Initialize ML routing orchestrator
        print("\n2️⃣  Initializing ML routing...")
        self.ml_orchestrator = LangGraphOrchestrator(
            enable_rag=False,  # Disable RAG for faster routing tests
            use_ml_routing=True
        )

        self.results = {
            "gpt5": [],
            "ml": []
        }

    def load_test_queries(self, path: str) -> List[Dict[str, Any]]:
        """
        Load test queries from JSON file.

        Args:
            path: Path to test queries JSON

        Returns:
            List of test query dictionaries
        """
        with open(path, 'r') as f:
            data = json.load(f)

        # Handle different JSON formats
        if isinstance(data, list):
            queries = data
        elif isinstance(data, dict) and "test" in data:
            # Training data format
            queries = data["test"]
        else:
            raise ValueError(f"Unexpected JSON format in {path}")

        print(f"\n📂 Loaded {len(queries)} test queries")
        return queries

    def route_query_gpt5(self, query: str) -> Dict[str, Any]:
        """
        Route query using GPT-5 and measure performance.

        Args:
            query: Query string

        Returns:
            Routing result with metrics
        """
        start_time = time.time()

        # Initialize state
        state = {
            "query": query,
            "agents_to_call": [],
            "research_enabled": False,
            "research_findings": {},
            "research_context": "",
            "market_analysis": "",
            "operations_audit": "",
            "financial_modeling": "",
            "lead_generation": "",
            "web_research": {},
            "synthesis": "",
            "conversation_history": [],
            "use_memory": False
        }

        # Run router node only
        result_state = self.gpt5_orchestrator._router_node(state)

        latency = time.time() - start_time

        # Estimate cost (GPT-5-nano routing: ~200 input + ~50 output tokens)
        # GPT-5-nano pricing: ~$0.50 per 1M input tokens, ~$2.00 per 1M output tokens
        estimated_cost = (200 * 0.50 / 1_000_000) + (50 * 2.00 / 1_000_000)

        return {
            "agents": result_state["agents_to_call"],
            "latency": latency,
            "cost": estimated_cost,
            "method": "gpt5"
        }

    def route_query_ml(self, query: str) -> Dict[str, Any]:
        """
        Route query using ML classifier and measure performance.

        Args:
            query: Query string

        Returns:
            Routing result with metrics
        """
        start_time = time.time()

        # Initialize state
        state = {
            "query": query,
            "agents_to_call": [],
            "research_enabled": False,
            "research_findings": {},
            "research_context": "",
            "market_analysis": "",
            "operations_audit": "",
            "financial_modeling": "",
            "lead_generation": "",
            "web_research": {},
            "synthesis": "",
            "conversation_history": [],
            "use_memory": False
        }

        # Run router node only
        result_state = self.ml_orchestrator._router_node(state)

        latency = time.time() - start_time

        # ML routing is free (no API costs)
        estimated_cost = 0.0

        return {
            "agents": result_state["agents_to_call"],
            "latency": latency,
            "cost": estimated_cost,
            "method": "ml"
        }

    def compare_routing(self, test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare routing methods on test queries.

        Args:
            test_queries: List of test queries with expected agents

        Returns:
            Comparison results
        """
        print(f"\n" + "="*70)
        print(f"🚀 RUNNING ROUTING COMPARISON")
        print(f"="*70)
        print(f"\nTesting {len(test_queries)} queries...")

        gpt5_results = []
        ml_results = []

        for i, test_query in enumerate(test_queries, 1):
            query = test_query["query"]
            expected_agents = sorted(test_query.get("agents", []))

            print(f"\n[{i}/{len(test_queries)}] {query[:60]}...")

            # Route with GPT-5
            gpt5_result = self.route_query_gpt5(query)
            gpt5_agents = sorted(gpt5_result["agents"])

            # Route with ML
            ml_result = self.route_query_ml(query)
            ml_agents = sorted(ml_result["agents"])

            # Calculate accuracy (exact match)
            gpt5_correct = (gpt5_agents == expected_agents)
            ml_correct = (ml_agents == expected_agents)

            gpt5_results.append({
                "query": query,
                "expected": expected_agents,
                "predicted": gpt5_agents,
                "correct": gpt5_correct,
                "latency": gpt5_result["latency"],
                "cost": gpt5_result["cost"]
            })

            ml_results.append({
                "query": query,
                "expected": expected_agents,
                "predicted": ml_agents,
                "correct": ml_correct,
                "latency": ml_result["latency"],
                "cost": ml_result["cost"]
            })

            print(f"   GPT-5: {gpt5_agents} ({'✓' if gpt5_correct else '✗'}) - {gpt5_result['latency']:.3f}s")
            print(f"   ML:    {ml_agents} ({'✓' if ml_correct else '✗'}) - {ml_result['latency']:.3f}s")

        # Calculate aggregate metrics
        gpt5_accuracy = sum(r["correct"] for r in gpt5_results) / len(gpt5_results)
        ml_accuracy = sum(r["correct"] for r in ml_results) / len(ml_results)

        gpt5_avg_latency = sum(r["latency"] for r in gpt5_results) / len(gpt5_results)
        ml_avg_latency = sum(r["latency"] for r in ml_results) / len(ml_results)

        gpt5_total_cost = sum(r["cost"] for r in gpt5_results)
        ml_total_cost = sum(r["cost"] for r in ml_results)

        return {
            "timestamp": datetime.now().isoformat(),
            "num_queries": len(test_queries),
            "gpt5": {
                "accuracy": gpt5_accuracy,
                "avg_latency": gpt5_avg_latency,
                "total_cost": gpt5_total_cost,
                "cost_per_query": gpt5_total_cost / len(test_queries),
                "results": gpt5_results
            },
            "ml": {
                "accuracy": ml_accuracy,
                "avg_latency": ml_avg_latency,
                "total_cost": ml_total_cost,
                "cost_per_query": ml_total_cost / len(test_queries),
                "results": ml_results
            },
            "comparison": {
                "accuracy_improvement": ml_accuracy - gpt5_accuracy,
                "latency_improvement": gpt5_avg_latency - ml_avg_latency,
                "latency_speedup": gpt5_avg_latency / ml_avg_latency if ml_avg_latency > 0 else 0,
                "cost_savings": gpt5_total_cost - ml_total_cost,
                "cost_reduction_pct": ((gpt5_total_cost - ml_total_cost) / gpt5_total_cost * 100) if gpt5_total_cost > 0 else 0
            }
        }

    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate markdown report from results.

        Args:
            results: Comparison results

        Returns:
            Markdown report string
        """
        report = f"""# Routing Comparison Report

**Generated:** {results['timestamp']}
**Test Queries:** {results['num_queries']}

---

## Results Summary

| Metric | GPT-5 Routing | ML Routing | Improvement |
|--------|---------------|------------|-------------|
| **Accuracy** | {results['gpt5']['accuracy']:.1%} | {results['ml']['accuracy']:.1%} | {results['comparison']['accuracy_improvement']:+.1%} |
| **Avg Latency** | {results['gpt5']['avg_latency']:.3f}s | {results['ml']['avg_latency']:.3f}s | {results['comparison']['latency_speedup']:.1f}x faster |
| **Cost per Query** | ${results['gpt5']['cost_per_query']:.6f} | ${results['ml']['cost_per_query']:.6f} | {results['comparison']['cost_reduction_pct']:.0f}% reduction |
| **Total Cost** | ${results['gpt5']['total_cost']:.4f} | ${results['ml']['total_cost']:.4f} | ${results['comparison']['cost_savings']:.4f} saved |

---

## Detailed Analysis

### Accuracy
- **GPT-5 Routing:** {results['gpt5']['accuracy']:.1%} exact match accuracy
- **ML Routing:** {results['ml']['accuracy']:.1%} exact match accuracy
- **Winner:** {"ML" if results['ml']['accuracy'] > results['gpt5']['accuracy'] else "GPT-5" if results['gpt5']['accuracy'] > results['ml']['accuracy'] else "Tie"}

### Latency
- **GPT-5 Routing:** {results['gpt5']['avg_latency']*1000:.1f}ms average
- **ML Routing:** {results['ml']['avg_latency']*1000:.1f}ms average
- **Speedup:** {results['comparison']['latency_speedup']:.1f}x faster with ML
- **Winner:** ML (always faster due to local inference)

### Cost
- **GPT-5 Routing:** ${results['gpt5']['total_cost']:.4f} total (${results['gpt5']['cost_per_query']:.6f} per query)
- **ML Routing:** $0.00 total (free local inference)
- **Savings:** ${results['comparison']['cost_savings']:.4f} ({results['comparison']['cost_reduction_pct']:.0f}% reduction)
- **Winner:** ML (100% cost reduction)

---

## Recommendations

"""

        # Add recommendations based on results
        if results['ml']['accuracy'] >= results['gpt5']['accuracy']:
            report += f"""
**✅ RECOMMENDATION: Use ML Routing in Production**

The ML routing classifier matches or exceeds GPT-5 routing accuracy ({results['ml']['accuracy']:.1%} vs {results['gpt5']['accuracy']:.1%}) while being **{results['comparison']['latency_speedup']:.0f}x faster** and **100% cheaper**.

**Benefits:**
- Same or better routing accuracy
- {results['comparison']['latency_speedup']:.1f}x faster response times
- Zero API costs for routing
- Predictable performance (no API rate limits)

**Estimated Annual Savings** (assuming 10,000 queries/month):
- Cost savings: ${results['comparison']['cost_savings'] * 10000:.2f}/month = ${results['comparison']['cost_savings'] * 120000:.2f}/year
"""
        else:
            accuracy_diff = (results['gpt5']['accuracy'] - results['ml']['accuracy']) * 100
            report += f"""
**⚠️  RECOMMENDATION: Improve ML Model Before Production**

The ML routing classifier is **{results['comparison']['latency_speedup']:.1f}x faster** and **100% cheaper**, but accuracy is {accuracy_diff:.1f}% lower than GPT-5 ({results['ml']['accuracy']:.1%} vs {results['gpt5']['accuracy']:.1%}).

**Options:**
1. **Hybrid Approach:** Use ML for high-confidence predictions, GPT-5 for uncertain cases
2. **Improve Training Data:** Collect more diverse examples (currently {results['num_queries']} test examples)
3. **Fine-tune Threshold:** Adjust confidence thresholds for better accuracy-coverage tradeoff
"""

        report += "\n---\n\n*Generated by routing_comparison.py*\n"

        return report

    def run_benchmark(
        self,
        queries_path: str,
        output_path: str
    ) -> Dict[str, Any]:
        """
        Run complete routing benchmark.

        Args:
            queries_path: Path to test queries
            output_path: Path to save results

        Returns:
            Comparison results
        """
        # Load test queries
        test_queries = self.load_test_queries(queries_path)

        # Run comparison
        results = self.compare_routing(test_queries)

        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Generate and save report
        report = self.generate_report(results)
        report_path = output_path.replace(".json", "_report.md")
        with open(report_path, 'w') as f:
            f.write(report)

        # Print summary
        print(f"\n" + "="*70)
        print(f"✅ BENCHMARK COMPLETE")
        print(f"="*70)
        print(f"\n📊 Results:")
        print(f"   GPT-5 Accuracy:  {results['gpt5']['accuracy']:.1%}")
        print(f"   ML Accuracy:     {results['ml']['accuracy']:.1%}")
        print(f"   Speedup:         {results['comparison']['latency_speedup']:.1f}x")
        print(f"   Cost Reduction:  {results['comparison']['cost_reduction_pct']:.0f}%")
        print(f"\n📁 Saved:")
        print(f"   Results: {output_path}")
        print(f"   Report:  {report_path}")
        print(f"="*70 + "\n")

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare ML vs GPT-5 routing")
    parser.add_argument(
        "--queries",
        type=str,
        default="models/training_data.json",
        help="Path to test queries JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/routing_comparison_results.json",
        help="Path to save results"
    )

    args = parser.parse_args()

    # Run benchmark
    benchmark = RoutingBenchmark()
    results = benchmark.run_benchmark(
        queries_path=args.queries,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
