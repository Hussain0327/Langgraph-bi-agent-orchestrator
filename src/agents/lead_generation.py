"""Lead Generation Agent - specializes in customer acquisition and growth strategies."""
import os
from openai import OpenAI
from typing import Dict, Any


class LeadGenerationAgent:
    """Specialized agent for lead generation and customer acquisition strategies."""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
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

Focus on practical, cost-effective strategies that drive predictable growth."""

    def generate_strategy(self, query: str) -> str:
        """Develop comprehensive lead generation strategies.

        Args:
            query: Business query focused on customer acquisition

        Returns:
            Lead generation strategies and growth recommendations
        """
        user_prompt = f"""Develop comprehensive lead generation strategies for the following business query:

{query}

Provide actionable strategies covering:
- Target customer identification and segmentation
- Multi-channel acquisition tactics
- Sales funnel optimization
- Content and lead magnet strategies
- Growth experiments and testing framework
- Budget allocation and channel prioritization
- Metrics and success criteria

Focus on scalable, cost-effective customer acquisition methods."""

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
            return f"Error in lead generation strategy: {str(e)}"
