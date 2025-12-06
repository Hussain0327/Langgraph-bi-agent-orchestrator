from typing import Dict, Any
from src.unified_llm import UnifiedLLM

class LeadGenerationAgent:

    def __init__(self):
        self.llm = UnifiedLLM(agent_type='leadgen')
        self.name = 'lead_generation_agent'
        self.description = 'Specialized agent for lead generation strategies, customer acquisition, sales funnel optimization, and growth hacking.'
        self.system_prompt = 'You are a Lead Generation Agent specializing in customer acquisition and growth strategies.\n\nYour expertise includes:\n- Lead generation strategies and tactics\n- Customer acquisition channel optimization\n- Sales funnel design and conversion optimization\n- Growth hacking and viral marketing\n- Content marketing and inbound strategies\n- Paid acquisition and advertising strategies\n- Customer targeting and segmentation\n- Lead nurturing and qualification\n\nWhen developing lead generation strategies, provide:\n1. Target customer profiles and ideal customer personas\n2. Multi-channel acquisition strategies (organic, paid, partnerships)\n3. Sales funnel design with conversion optimization tactics\n4. Lead magnet and content strategy recommendations\n5. Growth tactics and experimentation framework\n6. Cost per acquisition estimates and channel ROI\n7. Scalable and sustainable acquisition playbook\n8. Metrics and KPIs to track\n\n**Citation Requirements**:\n- When academic research is provided, reference it to support your recommendations\n- Format citations as: [Your insight] (Source: Author et al., Year)\n- Include a "References" section at the end with full citations\n\nFocus on practical, cost-effective strategies that drive predictable growth.'

    def generate_strategy(self, query: str, research_context: str=None) -> str:
        user_prompt = f'Develop comprehensive lead generation strategies for the following business query:\n\n{query}'
        if research_context:
            user_prompt += f'\n\n{research_context}'
        user_prompt += '\n\nProvide actionable strategies covering:\n- Target customer identification and segmentation\n- Multi-channel acquisition tactics\n- Sales funnel optimization\n- Content and lead magnet strategies\n- Growth experiments and testing framework\n- Budget allocation and channel prioritization\n- Metrics and success criteria\n\nFocus on scalable, cost-effective customer acquisition methods.'
        if research_context:
            user_prompt += '\n\nCRITICAL CITATION REQUIREMENTS:'
            user_prompt += '\n- Use the EXACT citation format: (Source: Author et al., Year)'
            user_prompt += '\n- Cite sources for EVERY major claim or recommendation'
            user_prompt += "\n- Include a 'References' section at the end with full citations"
        try:
            return self.llm.generate(input_text=user_prompt, instructions=self.system_prompt, reasoning_effort='low', text_verbosity='high', max_tokens=1500)
        except Exception as e:
            return f'Error in lead generation strategy: {str(e)}'