"""Market Analysis Agent - specializes in market research and competitive analysis."""
import os
from openai import OpenAI
from typing import Dict, Any


class MarketAnalysisAgent:
    """Specialized agent for market analysis, competitive research, industry trends."""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.name = "market_analysis_agent"
        self.description = "Specialized agent for market analysis, competitive research, industry trends, market sizing, and customer segmentation."

        self.system_prompt = """You are a Market Analysis Agent specializing in market research and competitive intelligence.

Your expertise includes:
- Market trends and industry dynamics
- Competitive landscape analysis
- Market sizing and opportunity assessment
- Customer segmentation and targeting
- Industry benchmarking and best practices

When analyzing markets, provide:
1. Current market trends and growth drivers
2. Competitive positioning and key players
3. Market opportunities and threats
4. Customer segments and target personas
5. Strategic recommendations based on market insights

Always base your analysis on data and provide actionable insights."""

    def analyze(self, query: str, web_research_results: Dict[str, Any] = None) -> str:
        """Conduct comprehensive market analysis.

        Args:
            query: Business query to analyze
            web_research_results: Optional web research data to inform analysis

        Returns:
            Market analysis findings and recommendations
        """
        # Build the prompt with context
        user_prompt = f"""Conduct comprehensive market analysis for the following business query:

{query}"""

        if web_research_results:
            user_prompt += f"\n\nWeb Research Data:\n{web_research_results.get('insights', '')}"

        user_prompt += "\n\nProvide actionable market insights and strategic recommendations."

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
            return f"Error in market analysis: {str(e)}"
