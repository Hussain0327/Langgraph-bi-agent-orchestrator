import sys
sys.path.insert(0, '/workspaces/multi_agent_workflow')
from eval.analysis import EvaluationAnalyzer
import glob
import json
no_rag_files = sorted(glob.glob('eval/results_no_rag_*.json'))
rag_files = sorted(glob.glob('eval/results_rag_*.json'))
if not no_rag_files or not rag_files:
    print('Error: Missing evaluation results')
    print(f'No RAG files: {len(no_rag_files)}')
    print(f'RAG files: {len(rag_files)}')
    sys.exit(1)
no_rag = no_rag_files[-1]
rag = rag_files[-1]
print(f'Analyzing:')
print(f'  Baseline: {no_rag}')
print(f'  RAG:      {rag}')
print()
analyzer = EvaluationAnalyzer(no_rag, rag)
report = analyzer.generate_full_report()
output_file = 'eval/ANALYSIS_REPORT.md'
with open(output_file, 'w') as f:
    f.write(report)
print(f'\nAnalysis complete: {output_file}')
with open(no_rag, 'r') as f:
    baseline = json.load(f)
with open(rag, 'r') as f:
    rag_data = json.load(f)
print('\nKey Metrics:')
print(f"  Baseline quality: {baseline.get('avg_quality', 'N/A')}")
print(f"  RAG quality:      {rag_data.get('avg_quality', 'N/A')}")
print(f"  Citation rate:    {rag_data.get('citation_rate', 'N/A')}")