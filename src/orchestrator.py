import os
from openai import OpenAI
from typing import Dict, Any, List
from src.agents.market_analysis import MarketAnalysisAgent
from src.agents.operations_audit import OperationsAuditAgent
from src.agents.financial_modeling import FinancialModelingAgent
from src.agents.lead_generation import LeadGenerationAgent
from src.tools.calculator import CalculatorTool
from src.tools.web_research import WebResearchTool
from src.memory import ConversationMemory

class PrimaryOrchestrator:

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        self.market_agent = MarketAnalysisAgent()
        self.operations_agent = OperationsAuditAgent()
        self.financial_agent = FinancialModelingAgent()
        self.lead_gen_agent = LeadGenerationAgent()
        self.calculator = CalculatorTool()
        self.web_research = WebResearchTool()
        self.memory = ConversationMemory(max_messages=10)
        self.system_prompt = 'You are a Business Intelligence Orchestrator that coordinates specialized agents to provide comprehensive business recommendations.\n\nYour role is to:\n1. Analyze incoming business queries\n2. Determine which specialized agents should be consulted\n3. Synthesize findings from multiple agents into cohesive recommendations\n\nAvailable specialized agents:\n- market_analysis_agent: Market research, trends, competition, market sizing, customer segmentation\n- operations_audit_agent: Process optimization, efficiency analysis, workflow improvement, operational excellence\n- financial_modeling_agent: Financial projections, ROI calculations, revenue/cost analysis, financial planning\n- lead_generation_agent: Customer acquisition strategies, sales funnel optimization, growth tactics\n\nFor complex business decisions, consult multiple relevant agents and coordinate their findings into actionable recommendations.'

    def determine_agents_needed(self, query: str) -> Dict[str, bool]:
        query_lower = query.lower()
        agents_needed = {'market': False, 'operations': False, 'financial': False, 'leadgen': False}
        market_keywords = ['market', 'competition', 'competitor', 'industry', 'trend', 'customer segment', 'target audience']
        if any((keyword in query_lower for keyword in market_keywords)):
            agents_needed['market'] = True
        ops_keywords = ['process', 'efficiency', 'workflow', 'operation', 'optimize', 'automate', 'scale', 'bottleneck']
        if any((keyword in query_lower for keyword in ops_keywords)):
            agents_needed['operations'] = True
        financial_keywords = ['financial', 'revenue', 'cost', 'profit', 'roi', 'budget', 'pricing', 'investment', 'money']
        if any((keyword in query_lower for keyword in financial_keywords)):
            agents_needed['financial'] = True
        leadgen_keywords = ['lead', 'customer acquisition', 'growth', 'sales', 'marketing', 'funnel', 'conversion', 'acquire']
        if any((keyword in query_lower for keyword in leadgen_keywords)):
            agents_needed['leadgen'] = True
        if not any(agents_needed.values()):
            agents_needed = {k: True for k in agents_needed}
        return agents_needed

    def orchestrate(self, query: str, use_memory: bool=True) -> Dict[str, Any]:
        if use_memory:
            self.memory.add_message('user', query)
        agents_needed = self.determine_agents_needed(query)
        results = {}
        agent_outputs = []
        if agents_needed['market']:
            web_results = self.web_research.execute(query)
            market_analysis = self.market_agent.analyze(query, web_results)
            results['market_analysis'] = market_analysis
            agent_outputs.append(f'MARKET ANALYSIS:\n{market_analysis}')
        if agents_needed['operations']:
            ops_audit = self.operations_agent.audit(query)
            results['operations_audit'] = ops_audit
            agent_outputs.append(f'OPERATIONS AUDIT:\n{ops_audit}')
        if agents_needed['financial']:
            financial_model = self.financial_agent.model_financials(query)
            results['financial_modeling'] = financial_model
            agent_outputs.append(f'FINANCIAL ANALYSIS:\n{financial_model}')
        if agents_needed['leadgen']:
            leadgen_strategy = self.lead_gen_agent.generate_strategy(query)
            results['lead_generation'] = leadgen_strategy
            agent_outputs.append(f'LEAD GENERATION STRATEGY:\n{leadgen_strategy}')
        synthesis = self.synthesize_findings(query, agent_outputs, use_memory)
        results['synthesis'] = synthesis
        if use_memory:
            self.memory.add_message('assistant', synthesis)
        return {'query': query, 'agents_consulted': [k for k, v in agents_needed.items() if v], 'detailed_findings': results, 'recommendation': synthesis}

    def synthesize_findings(self, query: str, agent_outputs: List[str], use_memory: bool=True) -> str:
        context = ''
        if use_memory:
            context = f'\n\nConversation History:\n{self.memory.get_context_string()}\n\n'
        synthesis_prompt = f'As the Business Intelligence Orchestrator, synthesize the following findings from specialized agents into a comprehensive, actionable recommendation.\n\nOriginal Query: {query}\n{context}\nAgent Findings:\n\n{chr(10).join(agent_outputs)}\n\nYour task:\n1. Identify key themes and insights across all agent analyses\n2. Highlight any conflicts or trade-offs between recommendations\n3. Provide a clear, prioritized action plan\n4. Offer a holistic strategic recommendation\n\nProvide an executive summary followed by detailed recommendations.'
        try:
            response = self.client.chat.completions.create(model=self.model, messages=[{'role': 'system', 'content': self.system_prompt}, {'role': 'user', 'content': synthesis_prompt}], temperature=0.7, max_tokens=2000)
            return response.choices[0].message.content
        except Exception as e:
            return f'Error synthesizing findings: {str(e)}'

    def get_conversation_history(self) -> List[Dict[str, str]]:
        return self.memory.get_messages()

    def clear_memory(self):
        self.memory.clear()