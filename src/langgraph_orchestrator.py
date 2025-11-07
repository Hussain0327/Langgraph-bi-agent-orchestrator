"""LangGraph-based orchestrator with state machine and parallel execution."""
import asyncio
from typing import Dict, Any, List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
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
    """State object passed between nodes in the graph."""
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
    """
    LangGraph-based orchestrator with state machine routing.

    Architecture:
        User Query → Router Node → Parallel Agent Execution → Synthesis Node → Result
    """

    def __init__(self, enable_rag: bool = True, use_ml_routing: bool = False):
        """
        Initialize the LangGraph Orchestrator.

        Args:
            enable_rag: Enable research-augmented generation (default: True)
            use_ml_routing: Use ML classifier for routing instead of GPT-5 (default: False)
        """
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
            print("✓ RAG enabled - Research Synthesis Agent initialized")
        else:
            self.research_agent = None
            print("⚠️  RAG disabled - Running without research augmentation")

        # Initialize ML routing classifier (if enabled)
        self.ml_router = None
        if self.use_ml_routing:
            try:
                import os
                if os.path.exists("models/routing_classifier.pkl"):
                    from src.ml.routing_classifier import RoutingClassifier
                    self.ml_router = RoutingClassifier()
                    self.ml_router.load("models/routing_classifier.pkl")
                    print("✓ ML routing enabled - Classifier loaded")
                else:
                    print("⚠️  ML routing requested but model not found. Using GPT-5 routing.")
                    self.use_ml_routing = False
            except Exception as e:
                print(f"⚠️  ML routing failed to load: {e}. Using GPT-5 routing.")
                self.use_ml_routing = False

        if not self.use_ml_routing:
            print("✓ Using GPT-5 semantic routing")

        # Initialize tools
        self.web_research = WebResearchTool()

        # Initialize memory
        self.memory = ConversationMemory(max_messages=Config.MAX_MEMORY_MESSAGES)

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("router", self._router_node)

        # Add research synthesis node (RAG) if enabled
        if self.enable_rag:
            workflow.add_node("research_synthesis", self._research_synthesis_node)

        workflow.add_node("market_agent", self._market_agent_node)
        workflow.add_node("operations_agent", self._operations_agent_node)
        workflow.add_node("financial_agent", self._financial_agent_node)
        workflow.add_node("leadgen_agent", self._leadgen_agent_node)
        workflow.add_node("synthesis", self._synthesis_node)

        # Set entry point
        workflow.set_entry_point("router")

        # Router → Research Synthesis (if RAG enabled) or directly to agents
        if self.enable_rag:
            workflow.add_edge("router", "research_synthesis")
            # Research synthesis → market agent (start of sequential chain)
            workflow.add_edge("research_synthesis", "market_agent")
        else:
            # Direct routing without research synthesis → market agent
            workflow.add_edge("router", "market_agent")

        # Sequential agent execution (all agents run, each checks if needed)
        # This fixes Bug #4: Now ALL agents in agents_to_call will execute
        workflow.add_edge("market_agent", "operations_agent")
        workflow.add_edge("operations_agent", "financial_agent")
        workflow.add_edge("financial_agent", "leadgen_agent")
        workflow.add_edge("leadgen_agent", "synthesis")

        # Synthesis is the end
        workflow.add_edge("synthesis", END)

        return workflow.compile()

    @traceable(name="router_node")
    def _router_node(self, state: AgentState) -> AgentState:
        """
        Router node: Determines which agents to call based on query analysis.

        Uses ML classifier (if enabled) or GPT-5 for semantic routing.
        """
        query = state["query"]

        # Use ML routing if enabled
        if self.use_ml_routing and self.ml_router:
            try:
                agents_to_call = self.ml_router.predict(query)
                probas = self.ml_router.predict_proba(query)

                print(f"🤖 ML Router: {agents_to_call}")
                print(f"   Confidence: {probas}")

                state["agents_to_call"] = agents_to_call
                return state

            except Exception as e:
                print(f"⚠️  ML routing failed: {e}, falling back to GPT-5")
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

            print(f"🧠 GPT-5 Router: {agents_to_call}")

        except Exception as e:
            print(f"Routing error: {e}, using all agents")
            # Fallback to all agents on error
            agents_to_call = ["market", "operations", "financial", "leadgen"]

        state["agents_to_call"] = agents_to_call
        return state

    @traceable(name="research_synthesis")
    def _research_synthesis_node(self, state: AgentState) -> AgentState:
        """
        Research synthesis node: Retrieves and synthesizes academic research.

        Only runs if RAG is enabled.
        """
        if not self.enable_rag or not self.research_agent:
            state["research_findings"] = {}
            state["research_context"] = ""
            return state

        query = state["query"]

        print("\n📚 Retrieving academic research...")

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
                print(f"✓ Retrieved {paper_count} relevant papers")
                print(f"✓ Research synthesis complete")
            else:
                print("⚠️  No relevant research found - continuing without RAG")

        except Exception as e:
            print(f"⚠️  Research synthesis failed: {e}")
            print("   Continuing without research augmentation...")
            state["research_findings"] = {}
            state["research_context"] = ""

        return state

    def _route_to_agents(self, state: AgentState) -> str:
        """Conditional edge function: Routes to appropriate agents based on router decision."""
        agents_to_call = state.get("agents_to_call", [])

        if not agents_to_call:
            return "synthesis"

        # Return first agent to call (others will be called in parallel)
        # Note: This is a limitation of the current routing approach
        # For true parallelization, we'll execute all agents asynchronously in one node
        return agents_to_call[0] if len(agents_to_call) == 1 else "market"

    @traceable(name="market_agent")
    def _market_agent_node(self, state: AgentState) -> AgentState:
        """Market analysis agent node."""
        if "market" in state.get("agents_to_call", []):
            # Do web research first
            web_results = state.get("web_research")
            if not web_results:
                web_results = self.web_research.execute(state["query"])
                state["web_research"] = web_results

            # Get research context if available
            research_context = state.get("research_context", "")

            state["market_analysis"] = self.market_agent.analyze(
                query=state["query"],
                web_research_results=web_results,
                research_context=research_context
            )
        return state

    @traceable(name="operations_agent")
    def _operations_agent_node(self, state: AgentState) -> AgentState:
        """Operations audit agent node."""
        if "operations" in state.get("agents_to_call", []):
            research_context = state.get("research_context", "")

            state["operations_audit"] = self.operations_agent.audit(
                query=state["query"],
                research_context=research_context
            )
        return state

    @traceable(name="financial_agent")
    def _financial_agent_node(self, state: AgentState) -> AgentState:
        """Financial modeling agent node."""
        if "financial" in state.get("agents_to_call", []):
            research_context = state.get("research_context", "")

            state["financial_modeling"] = self.financial_agent.model_financials(
                query=state["query"],
                research_context=research_context
            )
        return state

    @traceable(name="leadgen_agent")
    def _leadgen_agent_node(self, state: AgentState) -> AgentState:
        """Lead generation agent node."""
        if "leadgen" in state.get("agents_to_call", []):
            research_context = state.get("research_context", "")

            state["lead_generation"] = self.lead_gen_agent.generate_strategy(
                query=state["query"],
                research_context=research_context
            )
        return state

    @traceable(name="synthesis_node")
    def _synthesis_node(self, state: AgentState) -> AgentState:
        """Synthesis node: Combines all agent outputs into final recommendation."""
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

    async def _execute_agents_parallel(
        self, state: AgentState
    ) -> AgentState:
        """Execute all required agents in parallel using asyncio."""
        agents_to_call = state.get("agents_to_call", [])

        # Create tasks for each agent
        tasks = []
        if "market" in agents_to_call:
            tasks.append(self._run_market_agent_async(state))
        if "operations" in agents_to_call:
            tasks.append(self._run_operations_agent_async(state))
        if "financial" in agents_to_call:
            tasks.append(self._run_financial_agent_async(state))
        if "leadgen" in agents_to_call:
            tasks.append(self._run_leadgen_agent_async(state))

        # Execute all agents in parallel
        results = await asyncio.gather(*tasks)

        # Update state with results
        for result in results:
            state.update(result)

        return state

    async def _run_market_agent_async(self, state: AgentState) -> Dict[str, str]:
        """Run market agent asynchronously."""
        web_results = state.get("web_research") or self.web_research.execute(state["query"])
        analysis = self.market_agent.analyze(state["query"], web_results)
        return {"market_analysis": analysis, "web_research": web_results}

    async def _run_operations_agent_async(self, state: AgentState) -> Dict[str, str]:
        """Run operations agent asynchronously."""
        audit = self.operations_agent.audit(state["query"])
        return {"operations_audit": audit}

    async def _run_financial_agent_async(self, state: AgentState) -> Dict[str, str]:
        """Run financial agent asynchronously."""
        modeling = self.financial_agent.model_financials(state["query"])
        return {"financial_modeling": modeling}

    async def _run_leadgen_agent_async(self, state: AgentState) -> Dict[str, str]:
        """Run leadgen agent asynchronously."""
        strategy = self.lead_gen_agent.generate_strategy(state["query"])
        return {"lead_generation": strategy}

    @traceable(name="orchestrate_query")
    def orchestrate(self, query: str, use_memory: bool = True) -> Dict[str, Any]:
        """
        Orchestrate a business intelligence query using LangGraph.

        Args:
            query: User's business query
            use_memory: Whether to use conversation memory

        Returns:
            Dictionary containing detailed findings and synthesis
        """
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
        """Get the conversation history."""
        return self.memory.get_messages()

    def clear_memory(self):
        """Clear the conversation memory."""
        self.memory.clear()
