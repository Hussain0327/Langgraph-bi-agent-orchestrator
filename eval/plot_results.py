import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from pathlib import Path
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
csv_file = Path('eval/benchmark_results_10queries.csv')
df = pd.read_csv(csv_file)
print(f' Loaded {len(df)} queries from {csv_file}')
df = df.sort_values('query_id')
pdf_file = Path('eval/benchmark_analysis.pdf')
with PdfPages(pdf_file) as pdf:
    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(df['query_id'], df['routing_accuracy'], color=['#2ecc71' if acc == 100 else '#e74c3c' for acc in df['routing_accuracy']], edgecolor='black', linewidth=1.2, alpha=0.8)
    for i, (qid, acc) in enumerate(zip(df['query_id'], df['routing_accuracy'])):
        ax.text(qid, acc + 2, f'{acc:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.axhline(y=100, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect (100%)')
    avg_accuracy = df['routing_accuracy'].mean()
    ax.axhline(y=avg_accuracy, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Average ({avg_accuracy:.1f}%)')
    ax.set_xlabel('Query ID', fontsize=14, fontweight='bold')
    ax.set_ylabel('Routing Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Routing Accuracy by Query\nGreen = Perfect (100%), Red = Missed Agents', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 110)
    ax.set_xticks(df['query_id'])
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    textstr = f'Total Queries: {len(df)}\n'
    textstr += f"Perfect Routing: {sum(df['routing_accuracy'] == 100)}/{len(df)}\n"
    textstr += f'Avg Accuracy: {avg_accuracy:.1f}%\n'
    textstr += f"Min Accuracy: {df['routing_accuracy'].min():.0f}%"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=props)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print(' Plot 1: Routing Accuracy created')
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.YlOrRd(df['latency_sec'] / df['latency_sec'].max())
    bars = ax.bar(df['query_id'], df['latency_sec'], color=colors, edgecolor='black', linewidth=1.2, alpha=0.8)
    for i, (qid, lat) in enumerate(zip(df['query_id'], df['latency_sec'])):
        ax.text(qid, lat + 3, f'{lat:.0f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    avg_latency = df['latency_sec'].mean()
    ax.axhline(y=avg_latency, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Average ({avg_latency:.1f}s)')
    target_latency = 120
    ax.axhline(y=target_latency, color='green', linestyle='--', linewidth=2, alpha=0.5, label=f'Target ({target_latency}s)')
    ax.set_xlabel('Query ID', fontsize=14, fontweight='bold')
    ax.set_ylabel('Latency (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('Query Latency by Query\nYellow = Fast, Red = Slow', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(df['query_id'])
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    textstr = f'Avg Latency: {avg_latency:.1f}s\n'
    textstr += f"Min: {df['latency_sec'].min():.0f}s\n"
    textstr += f"Max: {df['latency_sec'].max():.0f}s\n"
    textstr += f"Std Dev: {df['latency_sec'].std():.1f}s"
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=props)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print(' Plot 2: Latency created')
print(f'\n🎉 PDF created: {pdf_file}')
print(f' Contains 2 plots:')
print(f'   1. Routing Accuracy by Query')
print(f'   2. Latency by Query')
print(f'\n Open with: open {pdf_file}')