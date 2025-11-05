"""Lead Generation Agent - specializes in customer acquisition and growth strategies."""
from typing import Dict, Any
from src.gpt5_wrapper import GPT5Wrapper


class LeadGenerationAgent:
    """Specialized agent for lead generation and customer acquisition strategies."""

    def __init__(self):
        self.gpt5 = GPT5Wrapper()
        self.name = "lead_generation_agent"
        self.description = "Specialized agent for lead generation strategies, customer acquisition, sales funnel optimization, and growth hacking."

        self.system_prompt = """You are a Lead Generation Agent specializing in customer acquisition and growth strategies.

Your expertise includes:
- Lead generation strategies and tactics
- Customer acquisition channel optimization
- Sales funnel design and conversion optimization
- Growth hacking and viral marketing
- Content marketing and inbound strategies
- Paid acquisition and advertising strategies
- Customer targeting and segmentation
- Lead nurturing and qualification

When developing lead generation strategies, provide:
1. Target customer profiles and ideal customer personas
2. Multi-channel acquisition strategies (organic, paid, partnerships)
3. Sales funnel design with conversion optimization tactics
4. Lead magnet and content strategy recommendations
5. Growth tactics and experimentation framework
6. Cost per acquisition estimates and channel ROI
7. Scalable and sustainable acquisition playbook
8. Metrics and KPIs to track

**Citation Requirements**:
- When academic research is provided, reference it to support your recommendations
- Format citations as: [Your insight] (Source: Author et al., Year)
- Include a "References" section at the end with full citations

Focus on practical, cost-effective strategies that drive predictable growth."""

    def generate_strategy(self, query: str, research_context: str = None) -> str:
        """Develop comprehensive lead generation strategies.

        Args:
            query: Business query focused on customer acquisition
            research_context: Optional academic research context with citations

        Returns:
            Lead generation strategies and growth recommendations (with citations if research provided)
        """
        user_prompt = f"""Develop comprehensive lead generation strategies for the following business query:

{query}"""

        if research_context:
            user_prompt += f"\n\n{research_context}"

        user_prompt += """

Provide actionable strategies covering:
- Target customer identification and segmentation
- Multi-channel acquisition tactics
- Sales funnel optimization
- Content and lead magnet strategies
- Growth experiments and testing framework
- Budget allocation and channel prioritization
- Metrics and success criteria

Focus on scalable, cost-effective customer acquisition methods."""

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
            return f"Error in lead generation strategy: {str(e)}"
