from typing import Dict, Any
from src.unified_llm import UnifiedLLM

class MarketAnalysisAgent:

    def __init__(self):
        self.llm = UnifiedLLM(agent_type='market')
        self.name = 'market_analysis_agent'
        self.description = 'Specialized agent for market analysis, competitive research, industry trends, market sizing, and customer segmentation.'
        self.system_prompt = 'You are a Market Analysis Agent specializing in market research and competitive intelligence.\n\nYour expertise includes:\n- Market trends and industry dynamics\n- Competitive landscape analysis\n- Market sizing and opportunity assessment\n- Customer segmentation and targeting\n- Industry benchmarking and best practices\n\nWhen analyzing markets, provide:\n1. Current market trends and growth drivers\n2. Competitive positioning and key players\n3. Market opportunities and threats\n4. Customer segments and target personas\n5. Strategic recommendations based on market insights\n\n**Citation Requirements**:\n- When academic research is provided, reference it to support your analysis\n- Format citations as: [Your insight] (Source: Author et al., Year)\n- Include a "References" section at the end with full citations\n- Prioritize evidence-based insights over speculation\n\nAlways base your analysis on data and provide actionable insights.'

    def analyze(self, query: str, web_research_results: Dict[str, Any]=None, research_context: str=None) -> str:
        user_prompt = f'Conduct comprehensive market analysis for the following business query:\n\n{query}'
        if research_context:
            user_prompt += f'\n\n{research_context}'
        if web_research_results:
            user_prompt += f"\n\nWeb Research Data:\n{web_research_results.get('insights', '')}"
        user_prompt += '\n\nProvide actionable market insights and strategic recommendations.'
        if research_context:
            user_prompt += '\n\nCRITICAL CITATION REQUIREMENTS:'
            user_prompt += '\n- Use the EXACT citation format: (Source: Author et al., Year)'
            user_prompt += '\n- Cite sources for EVERY major claim or recommendation'
            user_prompt += "\n- Include a 'References' section at the end with full citations"
            user_prompt += "\n- Example: 'SaaS churn averages 5-7% monthly (Source: Smith et al., 2024).'"
        try:
            return self.llm.generate(input_text=user_prompt, instructions=self.system_prompt, reasoning_effort='low', text_verbosity='high', max_tokens=1500)
        except Exception as e:
            return f'Error in market analysis: {str(e)}'