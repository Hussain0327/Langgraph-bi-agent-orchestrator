from typing import Dict, Any
from src.unified_llm import UnifiedLLM

class OperationsAuditAgent:

    def __init__(self):
        self.llm = UnifiedLLM(agent_type='operations')
        self.name = 'operations_audit_agent'
        self.description = 'Specialized agent for operations audit, process optimization, efficiency analysis, workflow improvement, and operational excellence.'
        self.system_prompt = 'You are an Operations Audit Agent specializing in process optimization and operational efficiency.\n\nYour expertise includes:\n- Process analysis and workflow optimization\n- Efficiency assessment and bottleneck identification\n- Operational best practices and frameworks\n- Scalability and capacity planning\n- Automation opportunities and digital transformation\n- Quality management and continuous improvement\n\nWhen auditing operations, provide:\n1. Current state assessment of processes and workflows\n2. Identification of inefficiencies, bottlenecks, and pain points\n3. Process optimization recommendations\n4. Automation and technology opportunities\n5. Scalability considerations and growth planning\n6. Implementation roadmap and priorities\n\n**Citation Requirements**:\n- When academic research is provided, reference it to support your recommendations\n- Format citations as: [Your insight] (Source: Author et al., Year)\n- Include a "References" section at the end with full citations\n\nFocus on practical, actionable improvements that drive efficiency and scalability.'

    def audit(self, query: str, research_context: str=None) -> str:
        user_prompt = f'Perform a thorough operations audit for the following business query:\n\n{query}'
        if research_context:
            user_prompt += f'\n\n{research_context}'
        user_prompt += '\n\nAnalyze current processes, identify inefficiencies, and recommend optimizations focusing on:\n- Efficiency improvements\n- Bottleneck elimination\n- Automation opportunities\n- Scalability enhancements\n- Best practices implementation\n\nProvide specific, actionable recommendations with implementation priorities.'
        if research_context:
            user_prompt += '\n\nCRITICAL CITATION REQUIREMENTS:'
            user_prompt += '\n- Use the EXACT citation format: (Source: Author et al., Year)'
            user_prompt += '\n- Cite sources for EVERY major claim or recommendation'
            user_prompt += "\n- Include a 'References' section at the end with full citations"
        try:
            return self.llm.generate(input_text=user_prompt, instructions=self.system_prompt, reasoning_effort='low', text_verbosity='high', max_tokens=1500)
        except Exception as e:
            return f'Error in operations audit: {str(e)}'