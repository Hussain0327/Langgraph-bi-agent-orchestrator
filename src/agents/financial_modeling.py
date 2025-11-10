from typing import Dict, Any, Optional
import asyncio
from src.unified_llm import UnifiedLLM


class FinancialModelingAgent:

    def __init__(self):
        self.llm = UnifiedLLM(agent_type="financial")
        self.name = "financial_modeling_agent"
        self.description = "Specialized agent for financial modeling, ROI calculations, revenue projections, cost analysis, and financial planning."

        self.system_prompt = """You are a Financial Modeling Agent specializing in financial analysis and projections.

Your expertise includes:
- Financial modeling and forecasting
- ROI and NPV calculations
- Revenue and cost projections
- Profitability analysis
- Budget planning and optimization
- Financial risk assessment
- Investment evaluation and decision support

When creating financial models, provide:
1. Clear financial assumptions and methodology
2. Revenue projections with growth scenarios
3. Cost structure analysis and optimization opportunities
4. ROI calculations and payback periods
5. Profitability metrics and financial KPIs
6. Risk factors and sensitivity analysis
7. Financial recommendations with supporting data

**Citation Requirements**:
- When academic research is provided, reference it to support your financial models
- Format citations as: [Your insight] (Source: Author et al., Year)
- Include a "References" section at the end with full citations

Use the calculator tool for precise financial calculations. Present findings with clear metrics and actionable financial guidance."""

    async def model_financials(
        self,
        query: str,
        calculator_results: Optional[Dict[str, Any]] = None,
        research_context: Optional[str] = None
    ) -> str:
        """Create detailed financial models and analysis.

        Args:
            query: Business query requiring financial analysis
            calculator_results: Optional calculation results to incorporate
            research_context: Optional academic research context with citations

        Returns:
            Financial analysis and recommendations (with citations if research provided)
        """
        user_prompt = f"""Create detailed financial models and analysis for the following business query:

{query}"""

        if research_context:
            user_prompt += f"\n\n{research_context}"

        if calculator_results:
            user_prompt += f"\n\nCalculation Results:\n{calculator_results}"

        user_prompt += """

Provide comprehensive financial analysis including:
- Revenue and cost projections
- ROI calculations and metrics
- Profitability assessment
- Budget recommendations
- Financial risks and opportunities
- Actionable financial guidance

Use specific numbers and financial metrics where possible."""

        if research_context:
            user_prompt += "\n\nCRITICAL CITATION REQUIREMENTS:"
            user_prompt += "\n- Use the EXACT citation format: (Source: Author et al., Year)"
            user_prompt += "\n- Cite sources for EVERY major claim or recommendation"
            user_prompt += "\n- Include a 'References' section at the end with full citations"

        try:
            return await asyncio.to_thread(
                self.llm.generate,
                input_text=user_prompt,
                instructions=self.system_prompt,
                reasoning_effort="low",
                text_verbosity="high",
                max_tokens=1500
            )

        except Exception as e:
            return f"Error in financial modeling: {str(e)}"
