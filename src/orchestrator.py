"""Primary Orchestrator Agent - coordinates specialized agents and synthesizes findings."""
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
    """Primary orchestrator that delegates to specialized agents and synthesizes results."""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        # Initialize specialized agents
        self.market_agent = MarketAnalysisAgent()
        self.operations_agent = OperationsAuditAgent()
        self.financial_agent = FinancialModelingAgent()
        self.lead_gen_agent = LeadGenerationAgent()

        # Initialize tools
        self.calculator = CalculatorTool()
        self.web_research = WebResearchTool()

        # Initialize memory
        self.memory = ConversationMemory(max_messages=10)

        self.system_prompt = """You are a Business Intelligence Orchestrator that coordinates specialized agents to provide comprehensive business recommendations.

Your role is to:
1. Analyze incoming business queries
2. Determine which specialized agents should be consulted
3. Synthesize findings from multiple agents into cohesive recommendations

Available specialized agents:
- market_analysis_agent: Market research, trends, competition, market sizing, customer segmentation
- operations_audit_agent: Process optimization, efficiency analysis, workflow improvement, operational excellence
- financial_modeling_agent: Financial projections, ROI calculations, revenue/cost analysis, financial planning
- lead_generation_agent: Customer acquisition strategies, sales funnel optimization, growth tactics

For complex business decisions, consult multiple relevant agents and coordinate their findings into actionable recommendations."""

    def determine_agents_needed(self, query: str) -> Dict[str, bool]:
        """Determine which agents should be called based on the query.

        Args:
            query: User's business query

        Returns:
            Dictionary indicating which agents to call
        """
        query_lower = query.lower()

        agents_needed = {
            "market": False,
            "operations": False,
            "financial": False,
            "leadgen": False
        }

        # Market analysis keywords
        market_keywords = ["market", "competition", "competitor", "industry", "trend", "customer segment", "target audience"]
        if any(keyword in query_lower for keyword in market_keywords):
            agents_needed["market"] = True

        # Operations keywords
        ops_keywords = ["process", "efficiency", "workflow", "operation", "optimize", "automate", "scale", "bottleneck"]
        if any(keyword in query_lower for keyword in ops_keywords):
            agents_needed["operations"] = True

        # Financial keywords
        financial_keywords = ["financial", "revenue", "cost", "profit", "roi", "budget", "pricing", "investment", "money"]
        if any(keyword in query_lower for keyword in financial_keywords):
            agents_needed["financial"] = True

        # Lead generation keywords
        leadgen_keywords = ["lead", "customer acquisition", "growth", "sales", "marketing", "funnel", "conversion", "acquire"]
        if any(keyword in query_lower for keyword in leadgen_keywords):
            agents_needed["leadgen"] = True

        # If no specific keywords detected, use all agents for comprehensive analysis
        if not any(agents_needed.values()):
            agents_needed = {k: True for k in agents_needed}

        return agents_needed

    def orchestrate(self, query: str, use_memory: bool = True) -> Dict[str, Any]:
        """Orchestrate the multi-agent analysis of a business query.

        Args:
            query: User's business query
            use_memory: Whether to use conversation memory

        Returns:
            Dictionary containing agent findings and synthesized recommendation
        """
        # Add to memory
        if use_memory:
            self.memory.add_message("user", query)

        # Determine which agents to call
        agents_needed = self.determine_agents_needed(query)

        results = {}
        agent_outputs = []

        # Call Market Analysis Agent if needed
        if agents_needed["market"]:
            # First do web research
            web_results = self.web_research.execute(query)
            market_analysis = self.market_agent.analyze(query, web_results)
            results["market_analysis"] = market_analysis
            agent_outputs.append(f"MARKET ANALYSIS:\n{market_analysis}")

        # Call Operations Audit Agent if needed
        if agents_needed["operations"]:
            ops_audit = self.operations_agent.audit(query)
            results["operations_audit"] = ops_audit
            agent_outputs.append(f"OPERATIONS AUDIT:\n{ops_audit}")

        # Call Financial Modeling Agent if needed
        if agents_needed["financial"]:
            financial_model = self.financial_agent.model_financials(query)
            results["financial_modeling"] = financial_model
            agent_outputs.append(f"FINANCIAL ANALYSIS:\n{financial_model}")

        # Call Lead Generation Agent if needed
        if agents_needed["leadgen"]:
            leadgen_strategy = self.lead_gen_agent.generate_strategy(query)
            results["lead_generation"] = leadgen_strategy
            agent_outputs.append(f"LEAD GENERATION STRATEGY:\n{leadgen_strategy}")

        # Synthesize findings from all agents
        synthesis = self.synthesize_findings(query, agent_outputs, use_memory)
        results["synthesis"] = synthesis

        # Add synthesis to memory
        if use_memory:
            self.memory.add_message("assistant", synthesis)

        return {
            "query": query,
            "agents_consulted": [k for k, v in agents_needed.items() if v],
            "detailed_findings": results,
            "recommendation": synthesis
        }

    def synthesize_findings(self, query: str, agent_outputs: List[str], use_memory: bool = True) -> str:
        """Synthesize findings from multiple agents into a cohesive recommendation.

        Args:
            query: Original user query
            agent_outputs: List of outputs from specialized agents
            use_memory: Whether to include conversation context

        Returns:
            Synthesized recommendation
        """
        context = ""
        if use_memory:
            context = f"\n\nConversation History:\n{self.memory.get_context_string()}\n\n"

        synthesis_prompt = f"""As the Business Intelligence Orchestrator, synthesize the following findings from specialized agents into a comprehensive, actionable recommendation.

Original Query: {query}
{context}
Agent Findings:

{chr(10).join(agent_outputs)}

Your task:
1. Identify key themes and insights across all agent analyses
2. Highlight any conflicts or trade-offs between recommendations
3. Provide a clear, prioritized action plan
4. Offer a holistic strategic recommendation

Provide an executive summary followed by detailed recommendations."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": synthesis_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error synthesizing findings: {str(e)}"

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history.

        Returns:
            List of conversation messages
        """
        return self.memory.get_messages()

    def clear_memory(self):
        """Clear the conversation memory."""
        self.memory.clear()
