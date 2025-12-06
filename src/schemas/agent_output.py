from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

class Metric(BaseModel):
    value: float | int | str = Field(..., description='The metric value')
    unit: str = Field(..., description='Unit of measurement (USD, percent, months, etc.)')
    confidence: Literal['high', 'medium', 'low'] = Field('medium', description='Confidence level in this metric')
    source: Literal['calculation', 'research', 'assumption', 'user_provided', 'industry_benchmark'] = Field('calculation', description='Where this metric came from')
    formula: Optional[str] = Field(None, description="Excel formula if applicable (e.g., '=B2/B3')")

    class Config:
        json_schema_extra = {'example': {'value': 5000, 'unit': 'USD', 'confidence': 'high', 'source': 'calculation', 'formula': '=SUM(B2:B10)/COUNT(B2:B10)'}}

class DataTable(BaseModel):
    name: str = Field(..., description='Table identifier (snake_case)')
    description: str = Field(..., description='What this table shows')
    columns: List[str] = Field(..., description='Column headers')
    rows: List[List[Any]] = Field(..., description='Data rows')
    formatting: Optional[Dict[str, Any]] = Field(None, description='Optional formatting rules (colors, number formats, etc.)')

    @field_validator('rows')
    @classmethod
    def validate_row_length(cls, v, info):
        columns = info.data.get('columns', [])
        for row in v:
            if len(row) != len(columns):
                raise ValueError(f"Row length {len(row)} doesn't match columns {len(columns)}")
        return v

    class Config:
        json_schema_extra = {'example': {'name': 'monthly_cohorts', 'description': 'Monthly cohort retention analysis', 'columns': ['month', 'customers', 'revenue', 'churn'], 'rows': [[1, 100, 10000, 5], [2, 95, 9500, 4], [3, 91, 9100, 3]]}}

class ChartSpec(BaseModel):
    type: Literal['bar', 'line', 'scatter', 'pie', 'area', 'heatmap'] = Field(..., description='Chart type')
    title: str = Field(..., description='Chart title')
    x_label: Optional[str] = Field(None, description='X-axis label')
    y_label: Optional[str] = Field(None, description='Y-axis label')
    x_data: List[Any] = Field(..., description='X-axis data points')
    y_data: List[Any] = Field(..., description='Y-axis data points')
    colors: Optional[List[str]] = Field(None, description='Color palette (hex codes)')
    style: Optional[Dict[str, Any]] = Field(None, description='Additional matplotlib styling options')

    @field_validator('y_data')
    @classmethod
    def validate_data_length(cls, v, info):
        x_data = info.data.get('x_data', [])
        if len(v) != len(x_data):
            raise ValueError(f"y_data length {len(v)} doesn't match x_data length {len(x_data)}")
        return v

    class Config:
        json_schema_extra = {'example': {'type': 'bar', 'title': 'CAC vs LTV Comparison', 'x_label': 'Metric', 'y_label': 'Value (USD)', 'x_data': ['CAC', 'LTV'], 'y_data': [5000, 15000], 'colors': ['#E74C3C', '#2ECC71']}}

class Recommendation(BaseModel):
    title: str = Field(..., description='Short recommendation title')
    priority: Literal['high', 'medium', 'low'] = Field(..., description='Priority level')
    impact: str = Field(..., description='Expected impact (quantified if possible)')
    rationale: str = Field(..., description='Why this recommendation makes sense')
    action_items: List[str] = Field(..., description='Specific next steps')
    estimated_effort: Optional[str] = Field(None, description='Implementation effort estimate')
    timeline: Optional[str] = Field(None, description='Suggested timeline')

    class Config:
        json_schema_extra = {'example': {'title': 'Reduce churn rate to 3%', 'priority': 'high', 'impact': 'Would increase LTV by 40% ($15k → $21k)', 'rationale': 'Current 5% churn is eroding long-term value', 'action_items': ['Implement customer success program', 'Add quarterly business reviews', 'Build churn prediction model'], 'estimated_effort': '2-3 months', 'timeline': 'Q1 2026'}}

class Citation(BaseModel):
    title: str = Field(..., description='Paper/article title')
    authors: List[str] = Field(..., description='Author names')
    year: int = Field(..., description='Publication year')
    url: Optional[str] = Field(None, description='URL to source')
    relevance: str = Field(..., description='Why this source is relevant')
    citation_format: Optional[str] = Field(None, description='Pre-formatted APA citation')

    class Config:
        json_schema_extra = {'example': {'title': 'SaaS Metrics 2.0', 'authors': ['David Skok'], 'year': 2015, 'url': 'https://www.forentrepreneurs.com/saas-metrics-2/', 'relevance': 'Industry benchmarks for LTV:CAC ratio', 'citation_format': 'Skok, D. (2015). SaaS Metrics 2.0. For Entrepreneurs.'}}

class Findings(BaseModel):
    executive_summary: str = Field(..., description='1-2 paragraph high-level summary')
    metrics: Dict[str, Metric] = Field(default_factory=dict, description='Key metrics with metadata')
    data_tables: List[DataTable] = Field(default_factory=list, description='Data tables for Excel or PowerPoint')
    chart_specs: List[ChartSpec] = Field(default_factory=list, description='Chart specifications for visualization')
    narrative: str = Field(..., description='Detailed narrative explanation (2-3 paragraphs)')
    key_findings: List[str] = Field(..., description='Bullet-point key findings (3-5 items)')
    risks: List[str] = Field(default_factory=list, description='Potential risks or caveats')
    recommendations: List[Recommendation] = Field(default_factory=list, description='Actionable recommendations')

class AgentMetadata(BaseModel):
    confidence: Literal['high', 'medium', 'low'] = Field('medium', description='Overall confidence in the analysis')
    model: str = Field(..., description="LLM model used (e.g., 'DeepSeek-V3.2-Exp')")
    tokens_used: Optional[int] = Field(None, description='Total tokens consumed')
    cost_usd: Optional[float] = Field(None, description='Cost in USD')
    processing_time_seconds: Optional[float] = Field(None, description='Processing time')
    cache_hit: bool = Field(False, description='Whether response was cached')

class AgentOutput(BaseModel):
    query: str = Field(..., description='Original user query')
    agent: Literal['financial', 'market', 'operations', 'leadgen', 'research_synthesis'] = Field(..., description='Which agent produced this output')
    timestamp: datetime = Field(default_factory=datetime.now, description='When this analysis was generated')
    findings: Findings = Field(..., description='Core analysis findings')
    research_citations: List[Citation] = Field(default_factory=list, description='Research papers cited (if RAG enabled)')
    metadata: AgentMetadata = Field(..., description='Execution metadata')

    class Config:
        json_schema_extra = {'example': {'query': 'What are my SaaS unit economics?', 'agent': 'financial', 'timestamp': '2025-11-14T21:30:00Z', 'findings': {'executive_summary': 'Your unit economics are healthy with a 3:1 LTV:CAC ratio...', 'metrics': {'CAC': {'value': 5000, 'unit': 'USD', 'confidence': 'high', 'source': 'calculation'}, 'LTV': {'value': 15000, 'unit': 'USD', 'confidence': 'high', 'source': 'calculation'}}, 'data_tables': [], 'chart_specs': [], 'narrative': 'Detailed analysis...', 'key_findings': ['LTV:CAC ratio is 3:1', '12-month payback period'], 'risks': ['Churn rate trending upward'], 'recommendations': []}, 'research_citations': [], 'metadata': {'confidence': 'high', 'model': 'DeepSeek-V3.2-Exp (Chat)', 'tokens_used': 2500, 'cost_usd': 0.002, 'processing_time_seconds': 3.5, 'cache_hit': False}}}

class SynthesisOutput(BaseModel):
    query: str = Field(..., description='Original user query')
    agents_consulted: List[str] = Field(..., description='Which agents were consulted')
    timestamp: datetime = Field(default_factory=datetime.now)
    agent_outputs: Dict[str, AgentOutput] = Field(..., description='Outputs from each agent, keyed by agent name')
    synthesis: Findings = Field(..., description='Combined/synthesized findings across all agents')
    total_cost_usd: float = Field(0.0, description='Total cost across all agents')
    total_processing_time: float = Field(0.0, description='Total processing time')
    overall_confidence: Literal['high', 'medium', 'low'] = Field('medium', description='Overall confidence in the synthesis')

    class Config:
        json_schema_extra = {'example': {'query': 'How can I improve my SaaS business?', 'agents_consulted': ['financial', 'market', 'operations'], 'timestamp': '2025-11-14T21:30:00Z', 'agent_outputs': {'financial': {'query': '...', 'agent': 'financial'}, 'market': {'query': '...', 'agent': 'market'}}, 'synthesis': {'executive_summary': 'Combined analysis shows...', 'metrics': {}, 'narrative': '...', 'key_findings': [], 'recommendations': []}, 'total_cost_usd': 0.05, 'total_processing_time': 12.5, 'overall_confidence': 'high'}}