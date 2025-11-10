import asyncio
from typing import Dict, Any, List, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langsmith import traceable

from src.gpt5_wrapper import GPT5Wrapper
from src.agents.market_analysis import MarketAnalysisAgent
from src.agents.operations_audit import OperationsAuditAgent
from src.agents.financial_modeling import FinancialModelingAgent
from src.agents.lead_generation import LeadGenerationAgent
from src.agents.research_synthesis import ResearchSynthesisAgent
from src.tools.web_research import WebResearchTool
from src.memory import ConversationMemory
from src.config import Config


class AgentState(TypedDict):
    query: str
    agents_to_call: List[str]
    research_enabled: bool
    research_findings: Dict[str, Any]
    research_context: str
    market_analysis: str
    operations_audit: str
    financial_modeling: str
    lead_generation: str
    web_research: Dict[str, Any]
    synthesis: str
    conversation_history: List[Dict[str, str]]
    use_memory: bool


class LangGraphOrchestrator:

    def __init__(self, enable_rag: bool = True, use_ml_routing: bool = False):
        self.gpt5 = GPT5Wrapper()
        self.enable_rag = enable_rag
        self.use_ml_routing = use_ml_routing

        # Initialize agents
        self.market_agent = MarketAnalysisAgent()
        self.operations_agent = OperationsAuditAgent()
        self.financial_agent = FinancialModelingAgent()
        self.lead_gen_agent = LeadGenerationAgent()

        # Initialize research agent (RAG)
        if self.enable_rag:
            self.research_agent = ResearchSynthesisAgent()
            print("[+] RAG enabled - Research Synthesis Agent initialized")
        else:
            self.research_agent = None
            print("[!] RAG disabled - Running without research augmentation")

        # Initialize ML routing classifier (if enabled)
        self.ml_router = None
        if self.use_ml_routing:
            try:
                import os
                if os.path.exists("models/routing_classifier.pkl"):
                    from src.ml.routing_classifier import RoutingClassifier
                    self.ml_router = RoutingClassifier()
                    self.ml_router.load("models/routing_classifier.pkl")
                    print("[+] ML routing enabled - Classifier loaded")
                else:
                    print("[!] ML routing requested but model not found. Using GPT-5 routing.")
                    self.use_ml_routing = False
            except Exception as e:
                print(f"[!] ML routing failed to load: {e}. Using GPT-5 routing.")
                self.use_ml_routing = False

        if not self.use_ml_routing:
            print("[+] Using GPT-5 semantic routing")

        # Initialize tools
        self.web_research = WebResearchTool()

        # Initialize memory
        self.memory = ConversationMemory(max_messages=Config.MAX_MEMORY_MESSAGES)

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("router", self._router_node)

        if self.enable_rag:
            workflow.add_node("research_synthesis", self._research_synthesis_node)

        workflow.add_node("agents_parallel", self._agents_parallel_node)
        workflow.add_node("synthesis", self._synthesis_node)

        workflow.set_entry_point("router")

        if self.enable_rag:
            workflow.add_edge("router", "research_synthesis")
            workflow.add_edge("research_synthesis", "agents_parallel")
        else:
            workflow.add_edge("router", "agents_parallel")

        workflow.add_edge("agents_parallel", "synthesis")
        workflow.add_edge("synthesis", END)

        return workflow.compile()

    @traceable(name="router_node")
    def _router_node(self, state: AgentState) -> AgentState:
        query = state["query"]

        # Use ML routing if enabled
        if self.use_ml_routing and self.ml_router:
            try:
                agents_to_call = self.ml_router.predict(query)
                probas = self.ml_router.predict_proba(query)

                print(f"[ML Router] Selected: {agents_to_call}")
                print(f"[ML Router] Confidence: {probas}")

                state["agents_to_call"] = agents_to_call
                return state

            except Exception as e:
                print(f"[!] ML routing failed: {e}, falling back to GPT-5")
                # Fall through to GPT-5 routing

        # Use GPT-5 to analyze which agents are needed
        routing_prompt = f"""Analyze the following business query and determine which specialized agents should be consulted.

Available agents:
- market: Market research, trends, competition, market sizing, customer segmentation
- operations: Process optimization, efficiency analysis, workflow improvement
- financial: Financial projections, ROI calculations, revenue/cost analysis, pricing
- leadgen: Customer acquisition, sales funnel, growth strategies, marketing

Query: {query}

Respond with a JSON array of agent names that should be consulted. For comprehensive business decisions, include multiple relevant agents.
Example: ["market", "financial", "leadgen"]

Only output the JSON array, nothing else."""

        try:
            response = self.gpt5.generate(
                input_text=routing_prompt,
                reasoning_effort="low",  # Low reasoning for fast routing
                text_verbosity="low",
            )

            # Parse agent list from response
            import json
            # Extract JSON array from response (handle potential markdown formatting)
            response_clean = response.strip().replace("```json", "").replace("```", "").strip()
            agents_to_call = json.loads(response_clean)

            # If no agents selected, use all for comprehensive analysis
            if not agents_to_call:
                agents_to_call = ["market", "operations", "financial", "leadgen"]

            print(f"[GPT-5 Router] Selected: {agents_to_call}")

        except Exception as e:
            print(f"Routing error: {e}, using all agents")
            # Fallback to all agents on error
            agents_to_call = ["market", "operations", "financial", "leadgen"]

        state["agents_to_call"] = agents_to_call
        return state

    @traceable(name="research_synthesis")
    def _research_synthesis_node(self, state: AgentState) -> AgentState:
        if not self.enable_rag or not self.research_agent:
            state["research_findings"] = {}
            state["research_context"] = ""
            return state

        query = state["query"]

        print("\n[Research] Retrieving academic research...")

        try:
            # Retrieve and synthesize research
            research_result = self.research_agent.synthesize(
                query=query,
                retrieve_papers=True,
                top_k_papers=3
            )

            state["research_findings"] = research_result
            state["research_context"] = research_result.get("research_context", "")

            paper_count = research_result.get("paper_count", 0)
            if paper_count > 0:
                print(f"[Research] Retrieved {paper_count} relevant papers")
                print(f"[Research] Synthesis complete")
            else:
                print("[Research] No relevant research found - continuing without RAG")

        except Exception as e:
            print(f"[Research] Synthesis failed: {e}")
            print("[Research] Continuing without research augmentation")
            state["research_findings"] = {}
            state["research_context"] = ""

        return state

    @traceable(name="agents_parallel")
    def _agents_parallel_node(self, state: AgentState) -> AgentState:
        agents_to_call = state.get("agents_to_call", [])
        if not agents_to_call:
            return state

        print(f"\nExecuting {len(agents_to_call)} agents in parallel...")

        results = asyncio.run(self._execute_agents_parallel(state))

        state["market_analysis"] = results.get("market_analysis", "")
        state["operations_audit"] = results.get("operations_audit", "")
        state["financial_modeling"] = results.get("financial_modeling", "")
        state["lead_generation"] = results.get("lead_generation", "")

        return state

    @traceable(name="synthesis_node")
    def _synthesis_node(self, state: AgentState) -> AgentState:
        query = state["query"]

        # Collect agent outputs
        agent_outputs = []
        if state.get("market_analysis"):
            agent_outputs.append(f"MARKET ANALYSIS:\n{state['market_analysis']}")
        if state.get("operations_audit"):
            agent_outputs.append(f"OPERATIONS AUDIT:\n{state['operations_audit']}")
        if state.get("financial_modeling"):
            agent_outputs.append(f"FINANCIAL ANALYSIS:\n{state['financial_modeling']}")
        if state.get("lead_generation"):
            agent_outputs.append(f"LEAD GENERATION STRATEGY:\n{state['lead_generation']}")

        # Build synthesis prompt with conversation context
        context = ""
        if state.get("use_memory", True):
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

        synthesis = self.gpt5.generate(
            input_text=synthesis_prompt,
            reasoning_effort="low",  # Fixed: "high" uses all tokens for reasoning, no output
            text_verbosity="high",
        )

        state["synthesis"] = synthesis
        return state

    async def _execute_agents_parallel(self, state: AgentState) -> Dict[str, str]:
        agents_to_call = state.get("agents_to_call", [])
        query = state["query"]
        research_context = state.get("research_context", "")

        tasks = {}

        if "market" in agents_to_call:
            web_results = state.get("web_research")
            if not web_results:
                web_results = await asyncio.to_thread(self.web_research.execute, query)
            tasks["market_analysis"] = self.market_agent.analyze(
                query, web_results, research_context
            )

        if "operations" in agents_to_call:
            tasks["operations_audit"] = self.operations_agent.audit(
                query, research_context
            )

        if "financial" in agents_to_call:
            tasks["financial_modeling"] = self.financial_agent.model_financials(
                query, None, research_context
            )

        if "leadgen" in agents_to_call:
            tasks["lead_generation"] = self.lead_gen_agent.generate_strategy(
                query, research_context
            )

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        output = {}
        for key, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                output[key] = f"Error: {str(result)}"
            else:
                output[key] = result

        return output

    @traceable(name="orchestrate_query")
    def orchestrate(self, query: str, use_memory: bool = True) -> Dict[str, Any]:
        # Add to memory
        if use_memory:
            self.memory.add_message("user", query)

        # Initialize state
        initial_state: AgentState = {
            "query": query,
            "agents_to_call": [],
            "research_enabled": self.enable_rag,
            "research_findings": {},
            "research_context": "",
            "market_analysis": "",
            "operations_audit": "",
            "financial_modeling": "",
            "lead_generation": "",
            "web_research": {},
            "synthesis": "",
            "conversation_history": self.memory.get_messages(),
            "use_memory": use_memory,
        }

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        # Add synthesis to memory
        if use_memory:
            self.memory.add_message("assistant", final_state["synthesis"])

        # Return formatted results
        return {
            "query": query,
            "agents_consulted": final_state.get("agents_to_call", []),
            "detailed_findings": {
                "market_analysis": final_state.get("market_analysis", ""),
                "operations_audit": final_state.get("operations_audit", ""),
                "financial_modeling": final_state.get("financial_modeling", ""),
                "lead_generation": final_state.get("lead_generation", ""),
            },
            "recommendation": final_state["synthesis"],
        }

    def get_conversation_history(self) -> List[Dict[str, str]]:
        return self.memory.get_messages()

    def clear_memory(self) -> None:
        self.memory.clear()
