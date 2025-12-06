from typing import Dict, Any
import json
import re
from datetime import datetime
from src.unified_llm import UnifiedLLM
from src.schemas import AgentOutput, Findings, Metric, AgentMetadata

class FinancialModelingAgent:

    def __init__(self):
        self.llm = UnifiedLLM(agent_type='financial')
        self.name = 'financial_modeling_agent'
        self.description = 'Specialized agent for financial modeling, ROI calculations, revenue projections, cost analysis, and financial planning.'
        self.system_prompt = 'You are a Financial Modeling Agent specializing in financial analysis and projections.\n\nYour expertise includes:\n- Financial modeling and forecasting\n- ROI and NPV calculations\n- Revenue and cost projections\n- Profitability analysis\n- Budget planning and optimization\n- Financial risk assessment\n- Investment evaluation and decision support\n\nWhen creating financial models, provide:\n1. Clear financial assumptions and methodology\n2. Revenue projections with growth scenarios\n3. Cost structure analysis and optimization opportunities\n4. ROI calculations and payback periods\n5. Profitability metrics and financial KPIs\n6. Risk factors and sensitivity analysis\n7. Financial recommendations with supporting data\n\n**Citation Requirements**:\n- When academic research is provided, reference it to support your financial models\n- Format citations as: [Your insight] (Source: Author et al., Year)\n- Include a "References" section at the end with full citations\n\nUse the calculator tool for precise financial calculations. Present findings with clear metrics and actionable financial guidance.'

    def model_financials(self, query: str, calculator_results: Dict[str, Any]=None, research_context: str=None) -> str:
        user_prompt = f'Create detailed financial models and analysis for the following business query:\n\n{query}'
        if research_context:
            user_prompt += f'\n\n{research_context}'
        if calculator_results:
            user_prompt += f'\n\nCalculation Results:\n{calculator_results}'
        user_prompt += '\n\nProvide comprehensive financial analysis including:\n- Revenue and cost projections\n- ROI calculations and metrics\n- Profitability assessment\n- Budget recommendations\n- Financial risks and opportunities\n- Actionable financial guidance\n\nUse specific numbers and financial metrics where possible.'
        if research_context:
            user_prompt += '\n\nCRITICAL CITATION REQUIREMENTS:'
            user_prompt += '\n- Use the EXACT citation format: (Source: Author et al., Year)'
            user_prompt += '\n- Cite sources for EVERY major claim or recommendation'
            user_prompt += "\n- Include a 'References' section at the end with full citations"
        try:
            return self.llm.generate(input_text=user_prompt, instructions=self.system_prompt, reasoning_effort='low', text_verbosity='high', max_tokens=1500)
        except Exception as e:
            return f'Error in financial modeling: {str(e)}'

    def model_financials_structured(self, query: str, calculator_results: Dict[str, Any]=None, research_context: str=None) -> AgentOutput:
        text_analysis = self.model_financials(query, calculator_results, research_context)
        extraction_prompt = f'Given this financial analysis, extract structured data in JSON format:\n\nANALYSIS:\n{text_analysis}\n\nExtract the following in valid JSON format:\n\n{{\n  "executive_summary": "1-2 paragraph summary",\n  "metrics": {{\n    "metric_name": {{"value": number, "unit": "string", "confidence": "high|medium|low", "source": "calculation|assumption"}},\n    ...\n  }},\n  "key_findings": ["finding 1", "finding 2", "finding 3"],\n  "risks": ["risk 1", "risk 2"],\n  "recommendations": [\n    {{\n      "title": "recommendation title",\n      "priority": "high|medium|low",\n      "impact": "expected impact description",\n      "rationale": "why this recommendation",\n      "action_items": ["action 1", "action 2"]\n    }}\n  ]\n}}\n\nExtract ALL metrics mentioned (CAC, LTV, ROI, revenue, costs, etc.) with their actual values.\nReturn ONLY valid JSON, no additional text.'
        try:
            json_response = self.llm.generate(input_text=extraction_prompt, instructions='You are a data extraction assistant. Extract structured financial data from analysis text. Return ONLY valid JSON.', reasoning_effort='low', max_tokens=2000)
            json_str = json_response.strip()
            json_str = re.sub('```json\\s*', '', json_str)
            json_str = re.sub('```\\s*$', '', json_str)
            extracted_data = json.loads(json_str)
            metrics = {}
            for key, val in extracted_data.get('metrics', {}).items():
                metrics[key] = Metric(**val)
            findings = Findings(executive_summary=extracted_data.get('executive_summary', ''), metrics=metrics, narrative=text_analysis, key_findings=extracted_data.get('key_findings', []), risks=extracted_data.get('risks', []), recommendations=extracted_data.get('recommendations', []))
            metadata = AgentMetadata(confidence='high', model=self.llm.get_current_provider(), tokens_used=None, cost_usd=None, processing_time_seconds=None)
            return AgentOutput(query=query, agent='financial', timestamp=datetime.now(), findings=findings, research_citations=[], metadata=metadata)
        except json.JSONDecodeError as e:
            print(f'  JSON extraction failed: {e}')
            print(f'Response was: {json_response[:200]}...')
            fallback_findings = Findings(executive_summary='See narrative for full analysis', narrative=text_analysis, key_findings=['Analysis generated successfully - see narrative'], recommendations=[])
            return AgentOutput(query=query, agent='financial', timestamp=datetime.now(), findings=fallback_findings, metadata=AgentMetadata(confidence='medium', model=self.llm.get_current_provider()))
        except Exception as e:
            error_findings = Findings(executive_summary=f'Error during financial analysis: {str(e)}', narrative=str(e), key_findings=[], recommendations=[])
            return AgentOutput(query=query, agent='financial', timestamp=datetime.now(), findings=error_findings, metadata=AgentMetadata(confidence='low', model='error'))