import argparse
import json
import os
from typing import List, Dict, Any, Tuple
from datetime import datetime
import numpy as np
from scipy import stats


class StatisticalAnalyzer:

    def __init__(self, baseline_results: Dict[str, Any], treatment_results: Dict[str, Any]):

        self.baseline = baseline_results
        self.treatment = treatment_results

        print("\n" + "="*70)
        print("📊 STATISTICAL ANALYSIS MODULE")
        print("="*70)

        print(f"\nBaseline: {len(self.baseline.get('results', []))} queries")
        print(f"Treatment: {len(self.treatment.get('results', []))} queries")

    def extract_metrics(self, results: Dict[str, Any]) -> Dict[str, List[float]]:
        
        metrics = {
            "factuality": [],
            "helpfulness": [],
            "comprehensiveness": [],
            "latency": [],
            "cost": [],
            "citation_count": [],
            "routing_accuracy": []
        }

        for result in results.get("results", []):
            # Quality metrics
            quality = result.get("quality_scores", {})
            metrics["factuality"].append(quality.get("factuality", 0))
            metrics["helpfulness"].append(quality.get("helpfulness", 0))
            metrics["comprehensiveness"].append(quality.get("comprehensiveness", 0))

            # Performance metrics
            metrics["latency"].append(result.get("latency", 0))
            metrics["cost"].append(result.get("cost", 0))

            # Citation metrics
            metrics["citation_count"].append(result.get("citation_count", 0))

            # Routing accuracy (1 if match, 0 if not)
            expected = sorted(result.get("expected_agents", []))
            actual = sorted(result.get("agents_to_call", []))
            metrics["routing_accuracy"].append(1.0 if expected == actual else 0.0)

        return metrics

    def calculate_ttest(self, baseline_values: List[float], treatment_values: List[float]) -> Tuple[float, float]:

        if not baseline_values or not treatment_values:
            return 0.0, 1.0

        t_stat, p_value = stats.ttest_ind(treatment_values, baseline_values)
        return float(t_stat), float(p_value)

    def calculate_effect_size(self, baseline_values: List[float], treatment_values: List[float]) -> float:

        if not baseline_values or not treatment_values:
            return 0.0

        baseline_mean = np.mean(baseline_values)
        treatment_mean = np.mean(treatment_values)

        # Pooled standard deviation
        baseline_std = np.std(baseline_values, ddof=1)
        treatment_std = np.std(treatment_values, ddof=1)
        n1, n2 = len(baseline_values), len(treatment_values)

        pooled_std = np.sqrt(((n1-1)*baseline_std**2 + (n2-1)*treatment_std**2) / (n1+n2-2))

        if pooled_std == 0:
            return 0.0

        cohens_d = (treatment_mean - baseline_mean) / pooled_std
        return float(cohens_d)

    def interpret_effect_size(self, d: float) -> str:

        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def cost_benefit_analysis(
        self,
        baseline_metrics: Dict[str, List[float]],
        treatment_metrics: Dict[str, List[float]]
    ) -> Dict[str, Any]:

        # Calculate average quality (mean of factuality, helpfulness, comprehensiveness)
        baseline_quality = np.mean([
            np.mean(baseline_metrics["factuality"]),
            np.mean(baseline_metrics["helpfulness"]),
            np.mean(baseline_metrics["comprehensiveness"])
        ])

        treatment_quality = np.mean([
            np.mean(treatment_metrics["factuality"]),
            np.mean(treatment_metrics["helpfulness"]),
            np.mean(treatment_metrics["comprehensiveness"])
        ])

        # Calculate costs
        baseline_cost = np.mean(baseline_metrics["cost"])
        treatment_cost = np.mean(treatment_metrics["cost"])

        # Quality improvement
        quality_improvement = treatment_quality - baseline_quality
        quality_improvement_pct = (quality_improvement / baseline_quality * 100) if baseline_quality > 0 else 0

        # Cost increase
        cost_increase = treatment_cost - baseline_cost
        cost_increase_pct = (cost_increase / baseline_cost * 100) if baseline_cost > 0 else 0

        # Quality per dollar
        baseline_qpd = baseline_quality / baseline_cost if baseline_cost > 0 else 0
        treatment_qpd = treatment_quality / treatment_cost if treatment_cost > 0 else 0

        return {
            "baseline_quality": float(baseline_quality),
            "treatment_quality": float(treatment_quality),
            "quality_improvement": float(quality_improvement),
            "quality_improvement_pct": float(quality_improvement_pct),
            "baseline_cost": float(baseline_cost),
            "treatment_cost": float(treatment_cost),
            "cost_increase": float(cost_increase),
            "cost_increase_pct": float(cost_increase_pct),
            "baseline_quality_per_dollar": float(baseline_qpd),
            "treatment_quality_per_dollar": float(treatment_qpd),
            "quality_per_dollar_improvement": float(treatment_qpd - baseline_qpd),
            "roi": float((quality_improvement / cost_increase) if cost_increase > 0 else 0)
        }

    def citation_correlation(self, treatment_metrics: Dict[str, List[float]]) -> Dict[str, Any]:

        citations = treatment_metrics["citation_count"]

        if not citations or all(c == 0 for c in citations):
            return {
                "correlation_with_factuality": 0.0,
                "correlation_with_helpfulness": 0.0,
                "correlation_with_comprehensiveness": 0.0,
                "avg_citations": 0.0,
                "citation_rate": 0.0
            }

        # Pearson correlations
        corr_factuality = np.corrcoef(citations, treatment_metrics["factuality"])[0, 1] if len(citations) > 1 else 0.0
        corr_helpfulness = np.corrcoef(citations, treatment_metrics["helpfulness"])[0, 1] if len(citations) > 1 else 0.0
        corr_comprehensiveness = np.corrcoef(citations, treatment_metrics["comprehensiveness"])[0, 1] if len(citations) > 1 else 0.0

        # Citation statistics
        avg_citations = np.mean(citations)
        citation_rate = sum(1 for c in citations if c > 0) / len(citations)

        return {
            "correlation_with_factuality": float(corr_factuality) if not np.isnan(corr_factuality) else 0.0,
            "correlation_with_helpfulness": float(corr_helpfulness) if not np.isnan(corr_helpfulness) else 0.0,
            "correlation_with_comprehensiveness": float(corr_comprehensiveness) if not np.isnan(corr_comprehensiveness) else 0.0,
            "avg_citations": float(avg_citations),
            "citation_rate": float(citation_rate)
        }

    def analyze(self) -> Dict[str, Any]:

        print(f"\nRunning statistical analysis...")

        # Extract metrics
        baseline_metrics = self.extract_metrics(self.baseline)
        treatment_metrics = self.extract_metrics(self.treatment)

        # Run tests for each quality metric
        results = {
            "timestamp": datetime.now().isoformat(),
            "num_queries": len(self.baseline.get("results", [])),
            "metrics": {}
        }

        quality_metrics = ["factuality", "helpfulness", "comprehensiveness"]

        for metric in quality_metrics:
            baseline_vals = baseline_metrics[metric]
            treatment_vals = treatment_metrics[metric]

            t_stat, p_value = self.calculate_ttest(baseline_vals, treatment_vals)
            effect_size = self.calculate_effect_size(baseline_vals, treatment_vals)

            results["metrics"][metric] = {
                "baseline_mean": float(np.mean(baseline_vals)),
                "baseline_std": float(np.std(baseline_vals)),
                "treatment_mean": float(np.mean(treatment_vals)),
                "treatment_std": float(np.std(treatment_vals)),
                "improvement": float(np.mean(treatment_vals) - np.mean(baseline_vals)),
                "improvement_pct": float((np.mean(treatment_vals) - np.mean(baseline_vals)) / np.mean(baseline_vals) * 100) if np.mean(baseline_vals) > 0 else 0.0,
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
                "cohens_d": float(effect_size),
                "effect_size_interpretation": self.interpret_effect_size(effect_size)
            }

        # Performance metrics
        results["performance"] = {
            "latency": {
                "baseline_mean": float(np.mean(baseline_metrics["latency"])),
                "treatment_mean": float(np.mean(treatment_metrics["latency"])),
                "increase_pct": float((np.mean(treatment_metrics["latency"]) - np.mean(baseline_metrics["latency"])) / np.mean(baseline_metrics["latency"]) * 100) if np.mean(baseline_metrics["latency"]) > 0 else 0.0
            },
            "cost": {
                "baseline_mean": float(np.mean(baseline_metrics["cost"])),
                "treatment_mean": float(np.mean(treatment_metrics["cost"])),
                "increase_pct": float((np.mean(treatment_metrics["cost"]) - np.mean(baseline_metrics["cost"])) / np.mean(baseline_metrics["cost"]) * 100) if np.mean(baseline_metrics["cost"]) > 0 else 0.0
            }
        }

        # Cost-benefit analysis
        results["cost_benefit"] = self.cost_benefit_analysis(baseline_metrics, treatment_metrics)

        # Citation analysis (treatment only)
        results["citations"] = self.citation_correlation(treatment_metrics)

        # Routing accuracy
        results["routing"] = {
            "baseline_accuracy": float(np.mean(baseline_metrics["routing_accuracy"])),
            "treatment_accuracy": float(np.mean(treatment_metrics["routing_accuracy"]))
        }

        print(f"✓ Analysis complete")

        return results

    def generate_report(self, results: Dict[str, Any]) -> str:

        report = f"""# Statistical Analysis Report

**Generated:** {results['timestamp']}
**Sample Size:** {results['num_queries']} queries per condition

---

## Executive Summary

"""

        # Determine overall conclusion
        significant_improvements = sum(
            1 for metric in ["factuality", "helpfulness", "comprehensiveness"]
            if results["metrics"][metric]["significant"] and results["metrics"][metric]["improvement"] > 0
        )

        if significant_improvements >= 2:
            report += """**✅ RAG SYSTEM SIGNIFICANTLY IMPROVES QUALITY**

The research-augmented generation system shows statistically significant improvements in quality metrics while maintaining acceptable cost and latency trade-offs.

"""
        elif significant_improvements == 1:
            report += """**⚠️  RAG SYSTEM SHOWS MIXED RESULTS**

The research-augmented generation system shows some quality improvements, but statistical significance is limited. Further optimization may be needed.

"""
        else:
            report += """**❌ RAG SYSTEM DOES NOT SHOW SIGNIFICANT IMPROVEMENTS**

The research-augmented generation system does not demonstrate statistically significant quality improvements over the baseline. The additional cost and latency may not be justified.

"""

        report += """---

## Quality Metrics

| Metric | Baseline | Treatment | Improvement | p-value | Significant? | Effect Size |
|--------|----------|-----------|-------------|---------|--------------|-------------|
"""

        for metric in ["factuality", "helpfulness", "comprehensiveness"]:
            m = results["metrics"][metric]
            sig_marker = "✓" if m["significant"] else "✗"
            report += f"| **{metric.capitalize()}** | {m['baseline_mean']:.3f} ± {m['baseline_std']:.3f} | {m['treatment_mean']:.3f} ± {m['treatment_std']:.3f} | {m['improvement']:+.3f} ({m['improvement_pct']:+.1f}%) | {m['p_value']:.4f} | {sig_marker} | {m['cohens_d']:.3f} ({m['effect_size_interpretation']}) |\n"

        report += f"""

### Interpretation

- **Statistical Significance:** p < 0.05 indicates the improvement is unlikely due to chance
- **Effect Size:** Cohen's d measures practical significance (small: 0.2-0.5, medium: 0.5-0.8, large: >0.8)

"""

        for metric in ["factuality", "helpfulness", "comprehensiveness"]:
            m = results["metrics"][metric]
            if m["significant"]:
                report += f"- **{metric.capitalize()}:** {m['improvement_pct']:+.1f}% improvement (p={m['p_value']:.4f}, d={m['cohens_d']:.2f}) - {m['effect_size_interpretation']} effect\n"

        report += f"""

---

## Performance Metrics

### Latency
- **Baseline:** {results['performance']['latency']['baseline_mean']:.2f}s
- **Treatment:** {results['performance']['latency']['treatment_mean']:.2f}s
- **Change:** {results['performance']['latency']['increase_pct']:+.1f}%

### Cost
- **Baseline:** ${results['performance']['cost']['baseline_mean']:.4f} per query
- **Treatment:** ${results['performance']['cost']['treatment_mean']:.4f} per query
- **Change:** {results['performance']['cost']['increase_pct']:+.1f}%

---

## Cost-Benefit Analysis

| Metric | Value |
|--------|-------|
| Quality Improvement | {results['cost_benefit']['quality_improvement']:+.3f} ({results['cost_benefit']['quality_improvement_pct']:+.1f}%) |
| Cost Increase | ${results['cost_benefit']['cost_increase']:+.4f} ({results['cost_benefit']['cost_increase_pct']:+.1f}%) |
| Quality per Dollar (Baseline) | {results['cost_benefit']['baseline_quality_per_dollar']:.2f} |
| Quality per Dollar (Treatment) | {results['cost_benefit']['treatment_quality_per_dollar']:.2f} |
| ROI | {results['cost_benefit']['roi']:.2f} quality points per dollar |

**Interpretation:**
- For every dollar spent on RAG features, you get **{results['cost_benefit']['roi']:.2f} quality points** of improvement
- Treatment system delivers **{results['cost_benefit']['treatment_quality_per_dollar'] / results['cost_benefit']['baseline_quality_per_dollar']:.2f}x** more quality per dollar

---

## Citation Analysis

| Metric | Value |
|--------|-------|
| Average Citations | {results['citations']['avg_citations']:.1f} |
| Citation Rate | {results['citations']['citation_rate']:.1%} |
| Correlation with Factuality | {results['citations']['correlation_with_factuality']:.3f} |
| Correlation with Helpfulness | {results['citations']['correlation_with_helpfulness']:.3f} |
| Correlation with Comprehensiveness | {results['citations']['correlation_with_comprehensiveness']:.3f} |

**Interpretation:**
- {results['citations']['citation_rate']*100:.0f}% of responses include citations
- Citations show {"positive" if results['citations']['correlation_with_factuality'] > 0 else "negative" if results['citations']['correlation_with_factuality'] < 0 else "no"} correlation with quality metrics

---

## Recommendations

"""

        cb = results["cost_benefit"]
        if cb["roi"] > 1.0 and significant_improvements >= 2:
            report += f"""
**✅ DEPLOY RAG SYSTEM TO PRODUCTION**

The data supports deploying the RAG system:
- Statistically significant quality improvements ({significant_improvements}/3 metrics)
- Positive ROI ({cb["roi"]:.2f} quality points per dollar)
- Quality improvement ({cb["quality_improvement_pct"]:.1f}%) justifies cost increase ({cb["cost_increase_pct"]:.1f}%)

**Expected Value:**
- For 1,000 queries/month: ${cb["cost_increase"]*1000:.2f}/month additional cost
- Quality improvement: {cb["quality_improvement_pct"]:.1f}% better recommendations
- Can justify premium pricing or improved customer satisfaction
"""
        elif significant_improvements >= 1:
            report += f"""
**⚠️  CONDITIONAL DEPLOYMENT RECOMMENDED**

Consider deploying RAG for specific use cases:
- Use RAG for high-value queries where quality matters most
- Monitor citation usage and quality correlation
- Optimize to reduce latency/cost ({results['performance']['latency']['increase_pct']:.0f}% increase, {results['performance']['cost']['increase_pct']:.0f}% increase)

**Next Steps:**
- A/B test with subset of users
- Optimize research retrieval (reduce latency)
- Consider hybrid approach (RAG for complex queries only)
"""
        else:
            report += f"""
**❌ DO NOT DEPLOY - NEEDS IMPROVEMENT**

The RAG system does not show sufficient benefit:
- Limited statistical significance ({significant_improvements}/3 metrics improved)
- ROI too low ({cb["roi"]:.2f} quality points per dollar)
- Cost/latency increases not justified

**Recommendations:**
1. Improve research retrieval relevance
2. Better prompt engineering for synthesis
3. Increase paper quality/recency filters
4. Consider alternative RAG architectures
"""

        report += "\n---\n\n*Generated by analysis.py*\n"

        return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Statistical analysis of evaluation results")
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline results JSON"
    )
    parser.add_argument(
        "--treatment",
        type=str,
        required=True,
        help="Path to treatment results JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/statistical_analysis.json",
        help="Path to save analysis results"
    )

    args = parser.parse_args()

    # Load results
    with open(args.baseline, 'r') as f:
        baseline = json.load(f)

    with open(args.treatment, 'r') as f:
        treatment = json.load(f)

    # Run analysis
    analyzer = StatisticalAnalyzer(baseline, treatment)
    results = analyzer.analyze()

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate report
    report = analyzer.generate_report(results)
    report_path = args.output.replace(".json", "_report.md")
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\n✅ Analysis complete!")
    print(f"   Results: {args.output}")
    print(f"   Report: {report_path}\n")


if __name__ == "__main__":
    main()
