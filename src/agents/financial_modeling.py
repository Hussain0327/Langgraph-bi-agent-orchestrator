"""Financial Modeling Agent - specializes in financial analysis and projections."""
import os
from openai import OpenAI
from typing import Dict, Any


class FinancialModelingAgent:
    """Specialized agent for financial modeling and analysis."""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
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

Use the calculator tool for precise financial calculations. Present findings with clear metrics and actionable financial guidance."""

    def model_financials(self, query: str, calculator_results: Dict[str, Any] = None) -> str:
        """Create detailed financial models and analysis.

        Args:
            query: Business query requiring financial analysis
            calculator_results: Optional calculation results to incorporate

        Returns:
            Financial analysis and recommendations
        """
        user_prompt = f"""Create detailed financial models and analysis for the following business query:

{query}"""

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

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error in financial modeling: {str(e)}"
