import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_system_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    input_color = '#E8F4F8'
    classifier_color = '#D4E6F1'
    agent_color = '#A9DFBF'
    synthesis_color = '#F9E79F'
    output_color = '#FAD7A0'
    ax.text(5, 11.5, 'Business Intelligence Orchestrator Architecture', ha='center', va='top', fontsize=18, fontweight='bold')
    user_box = FancyBboxPatch((4, 10.2), 2, 0.6, boxstyle='round,pad=0.1', edgecolor='black', facecolor=input_color, linewidth=2)
    ax.add_patch(user_box)
    ax.text(5, 10.5, 'User Query', ha='center', va='center', fontsize=11, fontweight='bold')
    arrow1 = FancyArrowPatch((5, 10.2), (5, 9.4), arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)
    classifier_box = FancyBboxPatch((3.5, 8.6), 3, 0.8, boxstyle='round,pad=0.1', edgecolor='black', facecolor=classifier_color, linewidth=2)
    ax.add_patch(classifier_box)
    ax.text(5, 9.1, 'Query Complexity Classifier', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5, 8.8, '(simple / business / complex)', ha='center', va='center', fontsize=8)
    arrow_simple = FancyArrowPatch((3.8, 8.6), (1.5, 7.5), arrowstyle='->', mutation_scale=15, linewidth=1.5, color='#2874A6')
    ax.add_patch(arrow_simple)
    simple_box = FancyBboxPatch((0.5, 7), 2, 0.6, boxstyle='round,pad=0.1', edgecolor='#2874A6', facecolor=input_color, linewidth=1.5)
    ax.add_patch(simple_box)
    ax.text(1.5, 7.3, 'Fast Answer', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(1.5, 7.05, '(5 seconds)', ha='center', va='center', fontsize=7)
    arrow_business = FancyArrowPatch((5, 8.6), (5, 7.8), arrowstyle='->', mutation_scale=15, linewidth=1.5, color='#239B56')
    ax.add_patch(arrow_business)
    arrow_complex = FancyArrowPatch((6.2, 8.6), (8.5, 7.5), arrowstyle='->', mutation_scale=15, linewidth=1.5, color='#B03A2E')
    ax.add_patch(arrow_complex)
    research_box = FancyBboxPatch((7.5, 6.8), 2, 0.8, boxstyle='round,pad=0.1', edgecolor='#B03A2E', facecolor='#FADBD8', linewidth=1.5)
    ax.add_patch(research_box)
    ax.text(8.5, 7.3, 'Research Retrieval', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(8.5, 7, 'Semantic Scholar', ha='center', va='center', fontsize=7)
    ax.text(8.5, 6.85, '+ arXiv', ha='center', va='center', fontsize=7)
    router_box = FancyBboxPatch((3.5, 7), 3, 0.7, boxstyle='round,pad=0.1', edgecolor='black', facecolor=classifier_color, linewidth=2)
    ax.add_patch(router_box)
    ax.text(5, 7.35, 'Agent Router', ha='center', va='center', fontsize=10, fontweight='bold')
    arrow_research_router = FancyArrowPatch((7.5, 7.2), (6.5, 7.2), arrowstyle='->', mutation_scale=15, linewidth=1.5, color='#B03A2E')
    ax.add_patch(arrow_research_router)
    agents = [('Market\nAnalysis', 1.5, 5), ('Operations\nAudit', 3.5, 5), ('Financial\nModeling', 5.5, 5), ('Lead\nGeneration', 7.5, 5)]
    for name, x, y in agents:
        agent_box = FancyBboxPatch((x - 0.6, y), 1.2, 0.8, boxstyle='round,pad=0.05', edgecolor='#239B56', facecolor=agent_color, linewidth=1.5)
        ax.add_patch(agent_box)
        ax.text(x, y + 0.4, name, ha='center', va='center', fontsize=8, fontweight='bold')
        arrow_to_agent = FancyArrowPatch((5, 7), (x, 5.8), arrowstyle='->', mutation_scale=12, linewidth=1, color='#239B56')
        ax.add_patch(arrow_to_agent)
    ax.text(4.5, 5.9, 'Parallel Execution (asyncio)', ha='center', va='center', fontsize=7, style='italic', color='#239B56')
    synthesis_box = FancyBboxPatch((3.5, 3.5), 3, 0.8, boxstyle='round,pad=0.1', edgecolor='black', facecolor=synthesis_color, linewidth=2)
    ax.add_patch(synthesis_box)
    ax.text(5, 3.9, 'Synthesis Agent', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5, 3.65, 'Combines all agent outputs', ha='center', va='center', fontsize=8)
    for name, x, y in agents:
        arrow_to_synthesis = FancyArrowPatch((x, 5), (5, 4.3), arrowstyle='->', mutation_scale=12, linewidth=1, color='#F39C12')
        ax.add_patch(arrow_to_synthesis)
    outputs = [('JSON\nAnalysis', 2, 2), ('PowerPoint\nDeck', 5, 2), ('Excel\nWorkbook', 8, 2)]
    for name, x, y in outputs:
        output_box = FancyBboxPatch((x - 0.7, y), 1.4, 0.7, boxstyle='round,pad=0.05', edgecolor='#D68910', facecolor=output_color, linewidth=1.5)
        ax.add_patch(output_box)
        ax.text(x, y + 0.35, name, ha='center', va='center', fontsize=9, fontweight='bold')
        arrow_to_output = FancyArrowPatch((5, 3.5), (x, 2.7), arrowstyle='->', mutation_scale=15, linewidth=1.5, color='#D68910')
        ax.add_patch(arrow_to_output)
    ax.text(0.5, 1, 'Paths:', fontsize=9, fontweight='bold')
    ax.text(0.5, 0.7, '— Simple queries (fast answer)', fontsize=7, color='#2874A6')
    ax.text(0.5, 0.4, '— Business queries (agents only)', fontsize=7, color='#239B56')
    ax.text(0.5, 0.1, '— Complex queries (research + agents)', fontsize=7, color='#B03A2E')
    plt.tight_layout()
    plt.savefig('docs/screenshots/system_architecture.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print('Created: docs/screenshots/system_architecture.png')

def create_performance_comparison_chart():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    configurations = ['GPT-5 Only', 'Hybrid\n(Current)', 'DeepSeek Only']
    monthly_costs = [900, 129, 9]
    colors = ['#E74C3C', '#27AE60', '#3498DB']
    bars1 = ax1.bar(configurations, monthly_costs, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Monthly Cost ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Monthly Cost Comparison\n(100 queries/day)', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1000)
    for bar, cost in zip(bars1, monthly_costs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f'${cost}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.annotate('86% savings', xy=(0.5, 500), xytext=(1.5, 700), arrowprops=dict(arrowstyle='->', lw=2, color='#27AE60'), fontsize=11, color='#27AE60', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    query_types = ['Simple', 'Business', 'Complex']
    sequential = [5, 145, 235]
    parallel = [5, 69, 153]
    cached = [0.1, 0.5, 1]
    x = np.arange(len(query_types))
    width = 0.25
    bars2 = ax2.bar(x - width, sequential, width, label='Sequential', color='#E74C3C', edgecolor='black', linewidth=1)
    bars3 = ax2.bar(x, parallel, width, label='Parallel (Current)', color='#27AE60', edgecolor='black', linewidth=1)
    bars4 = ax2.bar(x + width, cached, width, label='Cached', color='#3498DB', edgecolor='black', linewidth=1)
    ax2.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Query Speed Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(query_types)
    ax2.legend(fontsize=10, loc='upper left')
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.annotate('2.1x faster', xy=(1, 69), xytext=(1.3, 100), arrowprops=dict(arrowstyle='->', lw=1.5, color='#27AE60'), fontsize=9, color='#27AE60', fontweight='bold')
    ax2.annotate('138x faster', xy=(1 + width, 0.5), xytext=(1.5, 10), arrowprops=dict(arrowstyle='->', lw=1.5, color='#3498DB'), fontsize=9, color='#3498DB', fontweight='bold')
    plt.tight_layout()
    plt.savefig('docs/screenshots/performance_comparison.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print('Created: docs/screenshots/performance_comparison.png')

def create_deliverables_overview():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    deliverables = [{'title': 'JSON Analysis', 'color': '#E8F4F8', 'items': ['Machine-readable output', 'Structured data format', 'Agent consultation logs', 'Key metrics and findings', 'Research citations', 'Recommendation summary']}, {'title': 'PowerPoint Deck', 'color': '#FAD7A0', 'items': ['10-12 professional slides', 'Executive summary', 'Context and methodology', 'Key findings with metrics', 'Risk analysis', 'Detailed recommendations', 'Appendix']}, {'title': 'Excel Workbook', 'color': '#A9DFBF', 'items': ['5 comprehensive sheets', 'Executive KPI dashboard', 'Complete raw data', 'Financial calculations', 'Charts and visualizations', 'Methodology and sources']}]
    for ax, deliverable in zip(axes, deliverables):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        title_box = FancyBboxPatch((1, 8.5), 8, 1.2, boxstyle='round,pad=0.1', edgecolor='black', facecolor=deliverable['color'], linewidth=2)
        ax.add_patch(title_box)
        ax.text(5, 9.1, deliverable['title'], ha='center', va='center', fontsize=14, fontweight='bold')
        y_pos = 7.5
        for item in deliverable['items']:
            ax.text(1.5, y_pos, f'• {item}', ha='left', va='center', fontsize=9)
            y_pos -= 0.9
    plt.suptitle('Auto-Generated Deliverables from Single Query', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('docs/screenshots/deliverables_overview.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print('Created: docs/screenshots/deliverables_overview.png')
if __name__ == '__main__':
    print('Generating documentation diagrams...')
    create_system_architecture_diagram()
    create_performance_comparison_chart()
    create_deliverables_overview()
    print('\nAll diagrams created successfully!')
    print('Files saved in: docs/screenshots/')