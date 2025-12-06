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
        print('\n' + '=' * 70)
        print(' STATISTICAL ANALYSIS MODULE')
        print('=' * 70)
        print(f"\nBaseline: {len(self.baseline.get('results', []))} queries")
        print(f"Treatment: {len(self.treatment.get('results', []))} queries")

    def extract_metrics(self, results: Dict[str, Any]) -> Dict[str, List[float]]:
        metrics = {'factuality': [], 'helpfulness': [], 'comprehensiveness': [], 'latency': [], 'cost': [], 'citation_count': [], 'routing_accuracy': []}
        for result in results.get('results', []):
            quality = result.get('quality_scores', {})
            metrics['factuality'].append(quality.get('factuality', 0))
            metrics['helpfulness'].append(quality.get('helpfulness', 0))
            metrics['comprehensiveness'].append(quality.get('comprehensiveness', 0))
            metrics['latency'].append(result.get('latency', 0))
            metrics['cost'].append(result.get('cost', 0))
            metrics['citation_count'].append(result.get('citation_count', 0))
            expected = sorted(result.get('expected_agents', []))
            actual = sorted(result.get('agents_to_call', []))
            metrics['routing_accuracy'].append(1.0 if expected == actual else 0.0)
        return metrics

    def calculate_ttest(self, baseline_values: List[float], treatment_values: List[float]) -> Tuple[float, float]:
        if not baseline_values or not treatment_values:
            return (0.0, 1.0)
        t_stat, p_value = stats.ttest_ind(treatment_values, baseline_values)
        return (float(t_stat), float(p_value))

    def calculate_effect_size(self, baseline_values: List[float], treatment_values: List[float]) -> float:
        if not baseline_values or not treatment_values:
            return 0.0
        baseline_mean = np.mean(baseline_values)
        treatment_mean = np.mean(treatment_values)
        baseline_std = np.std(baseline_values, ddof=1)
        treatment_std = np.std(treatment_values, ddof=1)
        n1, n2 = (len(baseline_values), len(treatment_values))
        pooled_std = np.sqrt(((n1 - 1) * baseline_std ** 2 + (n2 - 1) * treatment_std ** 2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        cohens_d = (treatment_mean - baseline_mean) / pooled_std
        return float(cohens_d)

    def interpret_effect_size(self, d: float) -> str:
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'

    def cost_benefit_analysis(self, baseline_metrics: Dict[str, List[float]], treatment_metrics: Dict[str, List[float]]) -> Dict[str, Any]:
        baseline_quality = np.mean([np.mean(baseline_metrics['factuality']), np.mean(baseline_metrics['helpfulness']), np.mean(baseline_metrics['comprehensiveness'])])
        treatment_quality = np.mean([np.mean(treatment_metrics['factuality']), np.mean(treatment_metrics['helpfulness']), np.mean(treatment_metrics['comprehensiveness'])])
        baseline_cost = np.mean(baseline_metrics['cost'])
        treatment_cost = np.mean(treatment_metrics['cost'])
        quality_improvement = treatment_quality - baseline_quality
        quality_improvement_pct = quality_improvement / baseline_quality * 100 if baseline_quality > 0 else 0
        cost_increase = treatment_cost - baseline_cost
        cost_increase_pct = cost_increase / baseline_cost * 100 if baseline_cost > 0 else 0
        baseline_qpd = baseline_quality / baseline_cost if baseline_cost > 0 else 0
        treatment_qpd = treatment_quality / treatment_cost if treatment_cost > 0 else 0
        return {'baseline_quality': float(baseline_quality), 'treatment_quality': float(treatment_quality), 'quality_improvement': float(quality_improvement), 'quality_improvement_pct': float(quality_improvement_pct), 'baseline_cost': float(baseline_cost), 'treatment_cost': float(treatment_cost), 'cost_increase': float(cost_increase), 'cost_increase_pct': float(cost_increase_pct), 'baseline_quality_per_dollar': float(baseline_qpd), 'treatment_quality_per_dollar': float(treatment_qpd), 'quality_per_dollar_improvement': float(treatment_qpd - baseline_qpd), 'roi': float(quality_improvement / cost_increase if cost_increase > 0 else 0)}

    def citation_correlation(self, treatment_metrics: Dict[str, List[float]]) -> Dict[str, Any]:
        citations = treatment_metrics['citation_count']
        if not citations or all((c == 0 for c in citations)):
            return {'correlation_with_factuality': 0.0, 'correlation_with_helpfulness': 0.0, 'correlation_with_comprehensiveness': 0.0, 'avg_citations': 0.0, 'citation_rate': 0.0}
        corr_factuality = np.corrcoef(citations, treatment_metrics['factuality'])[0, 1] if len(citations) > 1 else 0.0
        corr_helpfulness = np.corrcoef(citations, treatment_metrics['helpfulness'])[0, 1] if len(citations) > 1 else 0.0
        corr_comprehensiveness = np.corrcoef(citations, treatment_metrics['comprehensiveness'])[0, 1] if len(citations) > 1 else 0.0
        avg_citations = np.mean(citations)
        citation_rate = sum((1 for c in citations if c > 0)) / len(citations)
        return {'correlation_with_factuality': float(corr_factuality) if not np.isnan(corr_factuality) else 0.0, 'correlation_with_helpfulness': float(corr_helpfulness) if not np.isnan(corr_helpfulness) else 0.0, 'correlation_with_comprehensiveness': float(corr_comprehensiveness) if not np.isnan(corr_comprehensiveness) else 0.0, 'avg_citations': float(avg_citations), 'citation_rate': float(citation_rate)}

    def analyze(self) -> Dict[str, Any]:
        print(f'\nRunning statistical analysis...')
        baseline_metrics = self.extract_metrics(self.baseline)
        treatment_metrics = self.extract_metrics(self.treatment)
        results = {'timestamp': datetime.now().isoformat(), 'num_queries': len(self.baseline.get('results', [])), 'metrics': {}}
        quality_metrics = ['factuality', 'helpfulness', 'comprehensiveness']
        for metric in quality_metrics:
            baseline_vals = baseline_metrics[metric]
            treatment_vals = treatment_metrics[metric]
            t_stat, p_value = self.calculate_ttest(baseline_vals, treatment_vals)
            effect_size = self.calculate_effect_size(baseline_vals, treatment_vals)
            results['metrics'][metric] = {'baseline_mean': float(np.mean(baseline_vals)), 'baseline_std': float(np.std(baseline_vals)), 'treatment_mean': float(np.mean(treatment_vals)), 'treatment_std': float(np.std(treatment_vals)), 'improvement': float(np.mean(treatment_vals) - np.mean(baseline_vals)), 'improvement_pct': float((np.mean(treatment_vals) - np.mean(baseline_vals)) / np.mean(baseline_vals) * 100) if np.mean(baseline_vals) > 0 else 0.0, 't_statistic': float(t_stat), 'p_value': float(p_value), 'significant': bool(p_value < 0.05), 'cohens_d': float(effect_size), 'effect_size_interpretation': self.interpret_effect_size(effect_size)}
        results['performance'] = {'latency': {'baseline_mean': float(np.mean(baseline_metrics['latency'])), 'treatment_mean': float(np.mean(treatment_metrics['latency'])), 'increase_pct': float((np.mean(treatment_metrics['latency']) - np.mean(baseline_metrics['latency'])) / np.mean(baseline_metrics['latency']) * 100) if np.mean(baseline_metrics['latency']) > 0 else 0.0}, 'cost': {'baseline_mean': float(np.mean(baseline_metrics['cost'])), 'treatment_mean': float(np.mean(treatment_metrics['cost'])), 'increase_pct': float((np.mean(treatment_metrics['cost']) - np.mean(baseline_metrics['cost'])) / np.mean(baseline_metrics['cost']) * 100) if np.mean(baseline_metrics['cost']) > 0 else 0.0}}
        results['cost_benefit'] = self.cost_benefit_analysis(baseline_metrics, treatment_metrics)
        results['citations'] = self.citation_correlation(treatment_metrics)
        results['routing'] = {'baseline_accuracy': float(np.mean(baseline_metrics['routing_accuracy'])), 'treatment_accuracy': float(np.mean(treatment_metrics['routing_accuracy']))}
        print(f'✓ Analysis complete')
        return results

    def generate_report(self, results: Dict[str, Any]) -> str:
        report = f"# Statistical Analysis Report\n\n**Generated:** {results['timestamp']}\n**Sample Size:** {results['num_queries']} queries per condition\n\n---\n\n"
        significant_improvements = sum((1 for metric in ['factuality', 'helpfulness', 'comprehensiveness'] if results['metrics'][metric]['significant'] and results['metrics'][metric]['improvement'] > 0))
        if significant_improvements >= 2:
            report += '** RAG SYSTEM SIGNIFICANTLY IMPROVES QUALITY**\n\nThe research-augmented generation system shows statistically significant improvements in quality metrics while maintaining acceptable cost and latency trade-offs.\n\n'
        elif significant_improvements == 1:
            report += '**  RAG SYSTEM SHOWS MIXED RESULTS**\n\nThe research-augmented generation system shows some quality improvements, but statistical significance is limited. Further optimization may be needed.\n\n'
        else:
            report += '** RAG SYSTEM DOES NOT SHOW SIGNIFICANT IMPROVEMENTS**\n\nThe research-augmented generation system does not demonstrate statistically significant quality improvements over the baseline. The additional cost and latency may not be justified.\n\n'
        report += '---\n\n| Metric | Baseline | Treatment | Improvement | p-value | Significant? | Effect Size |\n|--------|----------|-----------|-------------|---------|--------------|-------------|\n'
        for metric in ['factuality', 'helpfulness', 'comprehensiveness']:
            m = results['metrics'][metric]
            sig_marker = '✓' if m['significant'] else '✗'
            report += f"| **{metric.capitalize()}** | {m['baseline_mean']:.3f} ± {m['baseline_std']:.3f} | {m['treatment_mean']:.3f} ± {m['treatment_std']:.3f} | {m['improvement']:+.3f} ({m['improvement_pct']:+.1f}%) | {m['p_value']:.4f} | {sig_marker} | {m['cohens_d']:.3f} ({m['effect_size_interpretation']}) |\n"
        report += f"\n\n- **Statistical Significance:** p < 0.05 indicates the improvement is unlikely due to chance\n- **Effect Size:** Cohen's d measures practical significance (small: 0.2-0.5, medium: 0.5-0.8, large: >0.8)\n\n"
        for metric in ['factuality', 'helpfulness', 'comprehensiveness']:
            m = results['metrics'][metric]
            if m['significant']:
                report += f"- **{metric.capitalize()}:** {m['improvement_pct']:+.1f}% improvement (p={m['p_value']:.4f}, d={m['cohens_d']:.2f}) - {m['effect_size_interpretation']} effect\n"
        report += f"\n\n---\n\n- **Baseline:** {results['performance']['latency']['baseline_mean']:.2f}s\n- **Treatment:** {results['performance']['latency']['treatment_mean']:.2f}s\n- **Change:** {results['performance']['latency']['increase_pct']:+.1f}%\n\n- **Baseline:** ${results['performance']['cost']['baseline_mean']:.4f} per query\n- **Treatment:** ${results['performance']['cost']['treatment_mean']:.4f} per query\n- **Change:** {results['performance']['cost']['increase_pct']:+.1f}%\n\n---\n\n| Metric | Value |\n|--------|-------|\n| Quality Improvement | {results['cost_benefit']['quality_improvement']:+.3f} ({results['cost_benefit']['quality_improvement_pct']:+.1f}%) |\n| Cost Increase | ${results['cost_benefit']['cost_increase']:+.4f} ({results['cost_benefit']['cost_increase_pct']:+.1f}%) |\n| Quality per Dollar (Baseline) | {results['cost_benefit']['baseline_quality_per_dollar']:.2f} |\n| Quality per Dollar (Treatment) | {results['cost_benefit']['treatment_quality_per_dollar']:.2f} |\n| ROI | {results['cost_benefit']['roi']:.2f} quality points per dollar |\n\n**Interpretation:**\n- For every dollar spent on RAG features, you get **{results['cost_benefit']['roi']:.2f} quality points** of improvement\n- Treatment system delivers **{results['cost_benefit']['treatment_quality_per_dollar'] / results['cost_benefit']['baseline_quality_per_dollar']:.2f}x** more quality per dollar\n\n---\n\n| Metric | Value |\n|--------|-------|\n| Average Citations | {results['citations']['avg_citations']:.1f} |\n| Citation Rate | {results['citations']['citation_rate']:.1%} |\n| Correlation with Factuality | {results['citations']['correlation_with_factuality']:.3f} |\n| Correlation with Helpfulness | {results['citations']['correlation_with_helpfulness']:.3f} |\n| Correlation with Comprehensiveness | {results['citations']['correlation_with_comprehensiveness']:.3f} |\n\n**Interpretation:**\n- {results['citations']['citation_rate'] * 100:.0f}% of responses include citations\n- Citations show {('positive' if results['citations']['correlation_with_factuality'] > 0 else 'negative' if results['citations']['correlation_with_factuality'] < 0 else 'no')} correlation with quality metrics\n\n---\n\n"
        cb = results['cost_benefit']
        if cb['roi'] > 1.0 and significant_improvements >= 2:
            report += f"\n** DEPLOY RAG SYSTEM TO PRODUCTION**\n\nThe data supports deploying the RAG system:\n- Statistically significant quality improvements ({significant_improvements}/3 metrics)\n- Positive ROI ({cb['roi']:.2f} quality points per dollar)\n- Quality improvement ({cb['quality_improvement_pct']:.1f}%) justifies cost increase ({cb['cost_increase_pct']:.1f}%)\n\n**Expected Value:**\n- For 1,000 queries/month: ${cb['cost_increase'] * 1000:.2f}/month additional cost\n- Quality improvement: {cb['quality_improvement_pct']:.1f}% better recommendations\n- Can justify premium pricing or improved customer satisfaction\n"
        elif significant_improvements >= 1:
            report += f"\n**  CONDITIONAL DEPLOYMENT RECOMMENDED**\n\nConsider deploying RAG for specific use cases:\n- Use RAG for high-value queries where quality matters most\n- Monitor citation usage and quality correlation\n- Optimize to reduce latency/cost ({results['performance']['latency']['increase_pct']:.0f}% increase, {results['performance']['cost']['increase_pct']:.0f}% increase)\n\n**Next Steps:**\n- A/B test with subset of users\n- Optimize research retrieval (reduce latency)\n- Consider hybrid approach (RAG for complex queries only)\n"
        else:
            report += f"\n** DO NOT DEPLOY - NEEDS IMPROVEMENT**\n\nThe RAG system does not show sufficient benefit:\n- Limited statistical significance ({significant_improvements}/3 metrics improved)\n- ROI too low ({cb['roi']:.2f} quality points per dollar)\n- Cost/latency increases not justified\n\n**Recommendations:**\n1. Improve research retrieval relevance\n2. Better prompt engineering for synthesis\n3. Increase paper quality/recency filters\n4. Consider alternative RAG architectures\n"
        report += '\n---\n\n*Generated by analysis.py*\n'
        return report

def main():
    parser = argparse.ArgumentParser(description='Statistical analysis of evaluation results')
    parser.add_argument('--baseline', type=str, required=True, help='Path to baseline results JSON')
    parser.add_argument('--treatment', type=str, required=True, help='Path to treatment results JSON')
    parser.add_argument('--output', type=str, default='eval/statistical_analysis.json', help='Path to save analysis results')
    args = parser.parse_args()
    with open(args.baseline, 'r') as f:
        baseline = json.load(f)
    with open(args.treatment, 'r') as f:
        treatment = json.load(f)
    analyzer = StatisticalAnalyzer(baseline, treatment)
    results = analyzer.analyze()
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    report = analyzer.generate_report(results)
    report_path = args.output.replace('.json', '_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f'\n Analysis complete!')
    print(f'   Results: {args.output}')
    print(f'   Report: {report_path}\n')
if __name__ == '__main__':
    main()