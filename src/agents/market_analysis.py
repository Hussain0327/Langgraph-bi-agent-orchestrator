from typing import Dict, Any, Optional
import asyncio
from src.unified_llm import UnifiedLLM


class MarketAnalysisAgent:

    def __init__(self):
        self.llm = UnifiedLLM(agent_type="market")
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

**Citation Requirements**:
- When academic research is provided, reference it to support your analysis
- Format citations as: [Your insight] (Source: Author et al., Year)
- Include a "References" section at the end with full citations
- Prioritize evidence-based insights over speculation

Always base your analysis on data and provide actionable insights."""

    async def analyze(
        self,
        query: str,
        web_research_results: Optional[Dict[str, Any]] = None,
        research_context: Optional[str] = None
    ) -> str:
        """Conduct comprehensive market analysis.

        Args:
            query: Business query to analyze
            web_research_results: Optional web research data to inform analysis
            research_context: Optional academic research context with citations

        Returns:
            Market analysis findings and recommendations (with citations if research provided)
        """
        # Build the prompt with context
        user_prompt = f"""Conduct comprehensive market analysis for the following business query:

{query}"""

        # Add research context if available (highest priority)
        if research_context:
            user_prompt += f"\n\n{research_context}"

        # Add web research as supplementary context
        if web_research_results:
            user_prompt += f"\n\nWeb Research Data:\n{web_research_results.get('insights', '')}"

        user_prompt += "\n\nProvide actionable market insights and strategic recommendations."

        if research_context:
            user_prompt += "\n\nCRITICAL CITATION REQUIREMENTS:"
            user_prompt += "\n- Use the EXACT citation format: (Source: Author et al., Year)"
            user_prompt += "\n- Cite sources for EVERY major claim or recommendation"
            user_prompt += "\n- Include a 'References' section at the end with full citations"
            user_prompt += "\n- Example: 'SaaS churn averages 5-7% monthly (Source: Smith et al., 2024).'"

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
            return f"Error in market analysis: {str(e)}"
