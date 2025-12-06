import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Optional
import io
from PIL import Image
matplotlib.use('Agg')
from src.schemas import ChartSpec

class ChartGenerator:

    def __init__(self, dpi: int=150, figsize: tuple=(10, 6)):
        self.dpi = dpi
        self.figsize = figsize
        self.default_colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12', '#9B59B6', '#1ABC9C']

    def generate(self, chart_spec: ChartSpec, output_path: Optional[str]=None, return_bytes: bool=False):
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        colors = chart_spec.colors if chart_spec.colors else self.default_colors
        if chart_spec.type == 'bar':
            self._generate_bar(ax, chart_spec, colors)
        elif chart_spec.type == 'line':
            self._generate_line(ax, chart_spec, colors)
        elif chart_spec.type == 'scatter':
            self._generate_scatter(ax, chart_spec, colors)
        elif chart_spec.type == 'pie':
            self._generate_pie(ax, chart_spec, colors)
        elif chart_spec.type == 'area':
            self._generate_area(ax, chart_spec, colors)
        else:
            raise ValueError(f'Unsupported chart type: {chart_spec.type}')
        ax.set_title(chart_spec.title, fontsize=16, fontweight='bold', pad=20)
        if chart_spec.x_label:
            ax.set_xlabel(chart_spec.x_label, fontsize=12)
        if chart_spec.y_label:
            ax.set_ylabel(chart_spec.y_label, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        if return_bytes:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            return buf.getvalue()
        else:
            if not output_path:
                output_path = f'chart_{chart_spec.type}_{id(chart_spec)}.png'
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            return output_path

    def _generate_bar(self, ax, spec: ChartSpec, colors):
        bars = ax.bar(spec.x_data, spec.y_data, color=colors[:len(spec.x_data)])
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:,.0f}', ha='center', va='bottom', fontsize=10)

    def _generate_line(self, ax, spec: ChartSpec, colors):
        ax.plot(spec.x_data, spec.y_data, color=colors[0], linewidth=2.5, marker='o', markersize=6)
        for x, y in zip(spec.x_data, spec.y_data):
            ax.text(x, y, f'{y:,.0f}', ha='center', va='bottom', fontsize=9)

    def _generate_scatter(self, ax, spec: ChartSpec, colors):
        ax.scatter(spec.x_data, spec.y_data, color=colors[0], s=100, alpha=0.6, edgecolors='black', linewidth=1)

    def _generate_pie(self, ax, spec: ChartSpec, colors):
        wedges, texts, autotexts = ax.pie(spec.y_data, labels=spec.x_data, colors=colors[:len(spec.x_data)], autopct='%1.1f%%', startangle=90)
        for text in texts:
            text.set_fontsize(11)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)

    def _generate_area(self, ax, spec: ChartSpec, colors):
        ax.fill_between(spec.x_data, spec.y_data, color=colors[0], alpha=0.3)
        ax.plot(spec.x_data, spec.y_data, color=colors[0], linewidth=2)

    def generate_metric_comparison(self, metrics: dict, title: str='Key Metrics', output_path: Optional[str]=None):
        from src.schemas import ChartSpec
        spec = ChartSpec(type='bar', title=title, x_label='Metric', y_label='Value', x_data=list(metrics.keys()), y_data=list(metrics.values()))
        return self.generate(spec, output_path)