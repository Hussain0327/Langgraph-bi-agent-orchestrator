import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from scipy import stats
import numpy as np
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ABTestManager:

    def __init__(self, experiment_name: str, control: str, treatment: str, split_ratio: float=0.5, results_dir: str='ab_tests'):
        self.experiment_name = experiment_name
        self.control = control
        self.treatment = treatment
        self.split_ratio = split_ratio
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.results = {'experiment_name': experiment_name, 'control': control, 'treatment': treatment, 'split_ratio': split_ratio, 'created_at': datetime.now().isoformat(), 'control_results': [], 'treatment_results': []}
        self.results_path = os.path.join(results_dir, f'{experiment_name}.json')
        if os.path.exists(self.results_path):
            self.load_results(self.results_path)
            print(f'✓ Loaded existing A/B test: {experiment_name}')
            print(f"   Control: {len(self.results['control_results'])} samples")
            print(f"   Treatment: {len(self.results['treatment_results'])} samples")
        else:
            print(f'✓ Created new A/B test: {experiment_name}')

    def assign_user(self, user_id: str) -> str:
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        hash_pct = hash_val % 100 / 100.0
        if hash_pct < self.split_ratio:
            return 'treatment'
        else:
            return 'control'

    def log_result(self, user_id: str, query: str, response: str, metrics: Dict[str, Any], metadata: Optional[Dict[str, Any]]=None):
        group = self.assign_user(user_id)
        result = {'user_id': user_id, 'query': query, 'response': response, 'metrics': metrics, 'metadata': metadata or {}, 'timestamp': datetime.now().isoformat()}
        if group == 'treatment':
            self.results['treatment_results'].append(result)
        else:
            self.results['control_results'].append(result)
        self.save_results()

    def get_metric_values(self, group: str, metric_name: str) -> List[float]:
        results_key = f'{group}_results'
        results = self.results.get(results_key, [])
        values = []
        for result in results:
            metric_value = result.get('metrics', {}).get(metric_name)
            if metric_value is not None:
                values.append(float(metric_value))
        return values

    def calculate_statistics(self, control_values: List[float], treatment_values: List[float]) -> Dict[str, Any]:
        if not control_values or not treatment_values:
            return {'control_mean': 0.0, 'treatment_mean': 0.0, 'difference': 0.0, 'difference_pct': 0.0, 't_statistic': 0.0, 'p_value': 1.0, 'significant': False, 'confidence_interval_95': (0.0, 0.0)}
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)
        difference = treatment_mean - control_mean
        difference_pct = difference / control_mean * 100 if control_mean != 0 else 0.0
        t_stat, p_value = stats.ttest_ind(treatment_values, control_values)
        control_se = stats.sem(control_values)
        treatment_se = stats.sem(treatment_values)
        se_diff = np.sqrt(control_se ** 2 + treatment_se ** 2)
        ci_95 = (difference - 1.96 * se_diff, difference + 1.96 * se_diff)
        return {'control_mean': float(control_mean), 'control_std': float(np.std(control_values)), 'treatment_mean': float(treatment_mean), 'treatment_std': float(np.std(treatment_values)), 'difference': float(difference), 'difference_pct': float(difference_pct), 't_statistic': float(t_stat), 'p_value': float(p_value), 'significant': bool(p_value < 0.05), 'confidence_interval_95': (float(ci_95[0]), float(ci_95[1])), 'sample_size_control': len(control_values), 'sample_size_treatment': len(treatment_values)}

    def analyze(self, metrics: Optional[List[str]]=None) -> Dict[str, Any]:
        print(f'\n' + '=' * 70)
        print(f' A/B TEST ANALYSIS: {self.experiment_name}')
        print(f'=' * 70)
        control_count = len(self.results['control_results'])
        treatment_count = len(self.results['treatment_results'])
        print(f'\nSample Sizes:')
        print(f'   Control ({self.control}): {control_count}')
        print(f'   Treatment ({self.treatment}): {treatment_count}')
        if metrics is None:
            all_metrics = set()
            for result in self.results['control_results'] + self.results['treatment_results']:
                all_metrics.update(result.get('metrics', {}).keys())
            metrics = sorted(list(all_metrics))
        analysis = {'experiment_name': self.experiment_name, 'timestamp': datetime.now().isoformat(), 'control': self.control, 'treatment': self.treatment, 'sample_size_control': control_count, 'sample_size_treatment': treatment_count, 'metrics': {}}
        print(f'\nAnalyzing {len(metrics)} metrics...')
        for metric in metrics:
            control_vals = self.get_metric_values('control', metric)
            treatment_vals = self.get_metric_values('treatment', metric)
            stats_result = self.calculate_statistics(control_vals, treatment_vals)
            analysis['metrics'][metric] = stats_result
            sig_marker = '✓' if stats_result['significant'] else '✗'
            print(f'\n   {metric}:')
            print(f"      Control:   {stats_result['control_mean']:.3f} ± {stats_result['control_std']:.3f}")
            print(f"      Treatment: {stats_result['treatment_mean']:.3f} ± {stats_result['treatment_std']:.3f}")
            print(f"      Difference: {stats_result['difference']:+.3f} ({stats_result['difference_pct']:+.1f}%)")
            print(f"      p-value: {stats_result['p_value']:.4f} {sig_marker}")
        significant_improvements = sum((1 for m in analysis['metrics'].values() if m['significant'] and m['difference'] > 0))
        analysis['summary'] = {'significant_improvements': significant_improvements, 'total_metrics': len(metrics), 'recommendation': self._get_recommendation(analysis)}
        print(f'\n' + '=' * 70)
        print(f"RECOMMENDATION: {analysis['summary']['recommendation']}")
        print(f'=' * 70 + '\n')
        return analysis

    def _get_recommendation(self, analysis: Dict[str, Any]) -> str:
        sig_improvements = analysis['summary']['significant_improvements']
        total_metrics = analysis['summary']['total_metrics']
        if sig_improvements >= total_metrics * 0.6:
            return 'DEPLOY TREATMENT - Significant improvements across majority of metrics'
        elif sig_improvements >= total_metrics * 0.3:
            return 'CONDITIONAL DEPLOYMENT - Some improvements, monitor closely'
        elif sig_improvements > 0:
            return 'CONTINUE TESTING - Mixed results, need more data'
        else:
            return 'DO NOT DEPLOY - No significant improvements detected'

    def generate_report(self, analysis: Dict[str, Any]) -> str:
        report = f"# A/B Test Report: {analysis['experiment_name']}\n\n**Generated:** {analysis['timestamp']}\n**Control:** {analysis['control']}\n**Treatment:** {analysis['treatment']}\n\n---\n\n- **Control:** {analysis['sample_size_control']} samples\n- **Treatment:** {analysis['sample_size_treatment']} samples\n\n---\n\n| Metric | Control | Treatment | Difference | p-value | Significant? |\n|--------|---------|-----------|------------|---------|--------------|\n"
        for metric, stats in analysis['metrics'].items():
            sig_marker = '✓' if stats['significant'] else '✗'
            report += f"| {metric} | {stats['control_mean']:.3f} ± {stats['control_std']:.3f} | {stats['treatment_mean']:.3f} ± {stats['treatment_std']:.3f} | {stats['difference']:+.3f} ({stats['difference_pct']:+.1f}%) | {stats['p_value']:.4f} | {sig_marker} |\n"
        report += f"\n\n---\n\n- **Significant Improvements:** {analysis['summary']['significant_improvements']}/{analysis['summary']['total_metrics']} metrics\n- **Recommendation:** {analysis['summary']['recommendation']}\n\n---\n\n*Generated by ABTestManager*\n"
        return report

    def save_results(self, filepath: Optional[str]=None):
        filepath = filepath or self.results_path
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)

    def load_results(self, filepath: str):
        with open(filepath, 'r') as f:
            self.results = json.load(f)

    def export_analysis(self, output_dir: str='ab_tests/reports'):
        os.makedirs(output_dir, exist_ok=True)
        analysis = self.analyze()
        analysis_path = os.path.join(output_dir, f'{self.experiment_name}_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        report = self.generate_report(analysis)
        report_path = os.path.join(output_dir, f'{self.experiment_name}_report.md')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f'\n Exported analysis:')
        print(f'   JSON: {analysis_path}')
        print(f'   Report: {report_path}\n')
        return analysis
if __name__ == '__main__':
    ab_test = ABTestManager(experiment_name='example_test', control='baseline', treatment='new_feature', split_ratio=0.5)
    import random
    for i in range(50):
        user_id = f'user_{i}'
        group = ab_test.assign_user(user_id)
        if group == 'treatment':
            latency = random.gauss(2.5, 0.5)
            quality = random.gauss(0.85, 0.1)
        else:
            latency = random.gauss(3.0, 0.5)
            quality = random.gauss(0.75, 0.1)
        ab_test.log_result(user_id=user_id, query=f'Test query {i}', response=f'Test response {i}', metrics={'latency': latency, 'quality': quality, 'cost': random.gauss(0.05, 0.01)})
    ab_test.export_analysis()