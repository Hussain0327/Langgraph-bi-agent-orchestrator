"""Market Analysis Agent - specializes in market research and competitive analysis."""
from typing import Dict, Any
from src.gpt5_wrapper import GPT5Wrapper


class MarketAnalysisAgent:
    """Specialized agent for market analysis, competitive research, industry trends."""

    def __init__(self):
        self.gpt5 = GPT5Wrapper()
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

    def analyze(
        self,
        query: str,
        web_research_results: Dict[str, Any] = None,
        research_context: str = None
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
            user_prompt += "\n\nIMPORTANT: Reference the academic sources above in your analysis using proper citations."

        try:
            return self.gpt5.generate(
                input_text=user_prompt,
                instructions=self.system_prompt,
                reasoning_effort="medium",
                text_verbosity="high",
                max_output_tokens=1500
            )

        except Exception as e:
            return f"Error in market analysis: {str(e)}"
