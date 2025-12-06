import asyncio
from typing import Dict, Any, List, TypedDict, Annotated, Literal
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
from src.cache import QueryCache

class AgentState(TypedDict):
    query: str
    query_complexity: Literal['simple', 'business', 'complex']
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

    def __init__(self, enable_rag: bool=True, use_ml_routing: bool=False):
        self.gpt5 = GPT5Wrapper()
        self.enable_rag = enable_rag
        self.use_ml_routing = use_ml_routing
        self.market_agent = MarketAnalysisAgent()
        self.operations_agent = OperationsAuditAgent()
        self.financial_agent = FinancialModelingAgent()
        self.lead_gen_agent = LeadGenerationAgent()
        if self.enable_rag:
            self.research_agent = ResearchSynthesisAgent()
            print('✓ RAG enabled - Research Synthesis Agent initialized')
        else:
            self.research_agent = None
            print('  RAG disabled - Running without research augmentation')
        self.ml_router = None
        if self.use_ml_routing:
            try:
                import os
                if os.path.exists('models/routing_classifier.pkl'):
                    from src.ml.routing_classifier import RoutingClassifier
                    self.ml_router = RoutingClassifier()
                    self.ml_router.load('models/routing_classifier.pkl')
                    print('✓ ML routing enabled - Classifier loaded')
                else:
                    print('  ML routing requested but model not found. Using GPT-5 routing.')
                    self.use_ml_routing = False
            except Exception as e:
                print(f'  ML routing failed to load: {e}. Using GPT-5 routing.')
                self.use_ml_routing = False
        if not self.use_ml_routing:
            print('✓ Using GPT-5 semantic routing')
        self.web_research = WebResearchTool()
        self.memory = ConversationMemory(max_messages=Config.MAX_MEMORY_MESSAGES)
        self.cache = QueryCache()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node('complexity_classifier', self._complexity_classifier_node)
        workflow.add_node('fast_answer', self._fast_answer_node)
        workflow.add_node('router', self._router_node)
        workflow.add_node('research_synthesis', self._research_synthesis_node)
        workflow.add_node('parallel_agents', self._parallel_agents_node)
        workflow.add_node('synthesis', self._synthesis_node)
        workflow.set_entry_point('complexity_classifier')
        workflow.add_conditional_edges('complexity_classifier', self._route_by_complexity, {'simple': 'fast_answer', 'business': 'router', 'complex': 'router'})
        workflow.add_edge('fast_answer', END)
        workflow.add_conditional_edges('router', self._route_after_router, {'research': 'research_synthesis', 'agents': 'parallel_agents'})
        workflow.add_edge('research_synthesis', 'parallel_agents')
        workflow.add_edge('parallel_agents', 'synthesis')
        workflow.add_edge('synthesis', END)
        return workflow.compile()

    @traceable(name='complexity_classifier')
    def _complexity_classifier_node(self, state: AgentState) -> AgentState:
        query = state['query']
        classification_prompt = f"""Classify this query's complexity for a business intelligence system:\n\nQuery: {query}\n\nCategories:\n- simple: Non-business questions, general knowledge, casual conversation (answer in <3 seconds)\n- business: Business questions that can be answered by our agents without research papers\n- complex: Deep business questions requiring academic research backing\n\nRespond with ONLY ONE WORD: simple, business, or complex\n\nExamples:\n"What's the color of the sky?" → simple\n"How do I improve customer retention?" → business\n"What's the optimal pricing strategy for B2B SaaS with 500+ customers based on latest research?" → complex\n\nClassification:"""
        try:
            response = self.gpt5.generate(input_text=classification_prompt, reasoning_effort='low', text_verbosity='low').strip().lower()
            if 'simple' in response:
                complexity = 'simple'
            elif 'complex' in response:
                complexity = 'complex'
            elif 'business' in response:
                complexity = 'business'
            else:
                complexity = 'business'
            print(f' Query Complexity: {complexity.upper()}')
            state['query_complexity'] = complexity
        except Exception as e:
            print(f"  Complexity classification failed: {e}, defaulting to 'business'")
            state['query_complexity'] = 'business'
        return state

    def _route_by_complexity(self, state: AgentState) -> str:
        complexity = state.get('query_complexity', 'business')
        return complexity

    @traceable(name='fast_answer')
    def _fast_answer_node(self, state: AgentState) -> AgentState:
        query = state['query']
        print('⚡ Fast Answer Mode - Direct response')
        cached_answer = self.cache.get_simple_answer(query)
        if cached_answer:
            print('   ⚡ Using cached answer (instant!)')
            state['synthesis'] = cached_answer
            state['agents_to_call'] = []
            return state
        answer = self.gpt5.generate(input_text=f'Answer this question concisely:\n\n{query}', reasoning_effort='low', text_verbosity='medium')
        self.cache.set_simple_answer(query, answer)
        state['synthesis'] = answer
        state['agents_to_call'] = []
        return state

    def _route_after_router(self, state: AgentState) -> str:
        complexity = state.get('query_complexity', 'business')
        if complexity == 'complex' and self.enable_rag:
            return 'research'
        return 'agents'

    @traceable(name='router_node')
    def _router_node(self, state: AgentState) -> AgentState:
        query = state['query']
        if self.use_ml_routing and self.ml_router:
            try:
                ADAPTIVE_THRESHOLDS = {'market': 0.55, 'financial': 0.45, 'operations': 0.45, 'leadgen': 0.35}
                agents_to_call = self.ml_router.predict(query, adaptive_thresholds=ADAPTIVE_THRESHOLDS)
                probas = self.ml_router.predict_proba(query)
                print(f'🤖 ML Router: {agents_to_call}')
                print(f'   Confidence: {probas}')
                confidence_scores = list(probas.values())
                max_confidence = max(confidence_scores)
                min_confidence = min(confidence_scores)
                all_in_middle = all((0.3 <= score <= 0.7 for score in confidence_scores))
                max_spread = max_confidence - min_confidence
                indecisive = max_spread < 0.3
                if all_in_middle or (indecisive and max_confidence < 0.7):
                    print(f'  ML router uncertain (max={max_confidence:.2f}, spread={max_spread:.2f})')
                    print(f'   Falling back to GPT-5 for verification...')
                else:
                    print(f'   ✓ High confidence (max={max_confidence:.2f})')
                    state['agents_to_call'] = agents_to_call
                    return state
            except Exception as e:
                print(f'  ML routing failed: {e}, falling back to GPT-5')
        routing_prompt = f'Analyze the following business query and determine which specialized agents should be consulted.\n\nAvailable agents:\n- market: Market research, trends, competition, market sizing, customer segmentation\n- operations: Process optimization, efficiency analysis, workflow improvement\n- financial: Financial projections, ROI calculations, revenue/cost analysis, pricing\n- leadgen: Customer acquisition, sales funnel, growth strategies, marketing\n\nQuery: {query}\n\nRespond with a JSON array of agent names that should be consulted. For comprehensive business decisions, include multiple relevant agents.\nExample: ["market", "financial", "leadgen"]\n\nOnly output the JSON array, nothing else.'
        try:
            response = self.gpt5.generate(input_text=routing_prompt, reasoning_effort='low', text_verbosity='low')
            import json
            response_clean = response.strip().replace('```json', '').replace('```', '').strip()
            agents_to_call = json.loads(response_clean)
            if not agents_to_call:
                agents_to_call = ['market', 'operations', 'financial', 'leadgen']
            print(f'🧠 GPT-5 Router: {agents_to_call}')
        except Exception as e:
            print(f'Routing error: {e}, using all agents')
            agents_to_call = ['market', 'operations', 'financial', 'leadgen']
        state['agents_to_call'] = agents_to_call
        return state

    @traceable(name='research_synthesis')
    def _research_synthesis_node(self, state: AgentState) -> AgentState:
        if not self.enable_rag or not self.research_agent:
            state['research_findings'] = {}
            state['research_context'] = ''
            return state
        query = state['query']
        print('\n📚 Retrieving academic research...')
        cached_research = self.cache.get_research(query)
        if cached_research:
            print('   ⚡ Using cached research papers')
            state['research_findings'] = cached_research
            state['research_context'] = cached_research.get('research_context', '')
            paper_count = cached_research.get('paper_count', 0)
            if paper_count > 0:
                print(f'✓ {paper_count} papers loaded from cache')
            return state
        try:
            research_result = self.research_agent.synthesize(query=query, retrieve_papers=True, top_k_papers=3)
            state['research_findings'] = research_result
            state['research_context'] = research_result.get('research_context', '')
            paper_count = research_result.get('paper_count', 0)
            if paper_count > 0:
                print(f'✓ Retrieved {paper_count} relevant papers')
                print(f'✓ Research synthesis complete')
                self.cache.set_research(query, research_result)
            else:
                print('  No relevant research found - continuing without RAG')
        except Exception as e:
            print(f'  Research synthesis failed: {e}')
            print('   Continuing without research augmentation...')
            state['research_findings'] = {}
            state['research_context'] = ''
        return state

    @traceable(name='parallel_agents')
    def _parallel_agents_node(self, state: AgentState) -> AgentState:
        agents_to_call = state.get('agents_to_call', [])
        if not agents_to_call:
            print('  No agents selected - skipping agent execution')
            return state
        print(f' Parallel Execution: {len(agents_to_call)} agents running concurrently')
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            state = loop.run_until_complete(self._execute_agents_parallel(state))
        finally:
            loop.close()
        print('✓ All agents completed')
        return state

    @traceable(name='synthesis_node')
    def _synthesis_node(self, state: AgentState) -> AgentState:
        query = state['query']
        agent_names = []
        agent_outputs = []
        if state.get('market_analysis'):
            agent_names.append('market')
            agent_outputs.append(f"MARKET ANALYSIS:\n{state['market_analysis']}")
        if state.get('operations_audit'):
            agent_names.append('operations')
            agent_outputs.append(f"OPERATIONS AUDIT:\n{state['operations_audit']}")
        if state.get('financial_modeling'):
            agent_names.append('financial')
            agent_outputs.append(f"FINANCIAL ANALYSIS:\n{state['financial_modeling']}")
        if state.get('lead_generation'):
            agent_names.append('leadgen')
            agent_outputs.append(f"LEAD GENERATION STRATEGY:\n{state['lead_generation']}")
        if not agent_outputs:
            print('  No agent outputs to synthesize')
            state['synthesis'] = 'No analysis available. Please try again.'
            return state
        if len(agent_outputs) == 1:
            print(' Single agent output - skipping synthesis overhead')
            state['synthesis'] = agent_outputs[0]
            return state
        cached_synthesis = self.cache.get_synthesis(query, agent_names)
        if cached_synthesis:
            print('   ⚡ Using cached synthesis')
            state['synthesis'] = cached_synthesis
            return state
        context = ''
        if state.get('use_memory', True):
            context = f'\n\nConversation History:\n{self.memory.get_context_string()}\n\n'
        synthesis_prompt = f'As the Business Intelligence Orchestrator, synthesize the following findings from specialized agents into a comprehensive, actionable recommendation.\n\nOriginal Query: {query}\n{context}\nAgent Findings:\n\n{chr(10).join(agent_outputs)}\n\nYour task:\n1. Identify key themes and insights across all agent analyses\n2. Highlight any conflicts or trade-offs between recommendations\n3. Provide a clear, prioritized action plan\n4. Offer a holistic strategic recommendation\n\nProvide an executive summary followed by detailed recommendations.'
        print(f'🔄 Synthesizing {len(agent_outputs)} agent outputs')
        synthesis = self.gpt5.generate(input_text=synthesis_prompt, reasoning_effort='low', text_verbosity='high')
        self.cache.set_synthesis(query, agent_names, synthesis)
        state['synthesis'] = synthesis
        return state

    async def _execute_agents_parallel(self, state: AgentState) -> AgentState:
        agents_to_call = state.get('agents_to_call', [])
        tasks = []
        if 'market' in agents_to_call:
            tasks.append(self._run_market_agent_async(state))
        if 'operations' in agents_to_call:
            tasks.append(self._run_operations_agent_async(state))
        if 'financial' in agents_to_call:
            tasks.append(self._run_financial_agent_async(state))
        if 'leadgen' in agents_to_call:
            tasks.append(self._run_leadgen_agent_async(state))
        results = await asyncio.gather(*tasks)
        for result in results:
            state.update(result)
        return state

    async def _run_market_agent_async(self, state: AgentState) -> Dict[str, str]:
        loop = asyncio.get_event_loop()
        web_results = state.get('web_research')
        if not web_results:
            web_results = await loop.run_in_executor(None, self.web_research.execute, state['query'])
        research_context = state.get('research_context', '')
        analysis = await loop.run_in_executor(None, lambda: self.market_agent.analyze(query=state['query'], web_research_results=web_results, research_context=research_context))
        return {'market_analysis': analysis, 'web_research': web_results}

    async def _run_operations_agent_async(self, state: AgentState) -> Dict[str, str]:
        loop = asyncio.get_event_loop()
        research_context = state.get('research_context', '')
        audit = await loop.run_in_executor(None, lambda: self.operations_agent.audit(query=state['query'], research_context=research_context))
        return {'operations_audit': audit}

    async def _run_financial_agent_async(self, state: AgentState) -> Dict[str, str]:
        loop = asyncio.get_event_loop()
        research_context = state.get('research_context', '')
        modeling = await loop.run_in_executor(None, lambda: self.financial_agent.model_financials(query=state['query'], research_context=research_context))
        return {'financial_modeling': modeling}

    async def _run_leadgen_agent_async(self, state: AgentState) -> Dict[str, str]:
        loop = asyncio.get_event_loop()
        research_context = state.get('research_context', '')
        strategy = await loop.run_in_executor(None, lambda: self.lead_gen_agent.generate_strategy(query=state['query'], research_context=research_context))
        return {'lead_generation': strategy}

    @traceable(name='orchestrate_query')
    def orchestrate(self, query: str, use_memory: bool=True) -> Dict[str, Any]:
        if use_memory:
            self.memory.add_message('user', query)
        initial_state: AgentState = {'query': query, 'query_complexity': 'business', 'agents_to_call': [], 'research_enabled': self.enable_rag, 'research_findings': {}, 'research_context': '', 'market_analysis': '', 'operations_audit': '', 'financial_modeling': '', 'lead_generation': '', 'web_research': {}, 'synthesis': '', 'conversation_history': self.memory.get_messages(), 'use_memory': use_memory}
        final_state = self.graph.invoke(initial_state)
        if use_memory:
            self.memory.add_message('assistant', final_state['synthesis'])
        return {'query': query, 'agents_consulted': final_state.get('agents_to_call', []), 'detailed_findings': {'market_analysis': final_state.get('market_analysis', ''), 'operations_audit': final_state.get('operations_audit', ''), 'financial_modeling': final_state.get('financial_modeling', ''), 'lead_generation': final_state.get('lead_generation', '')}, 'recommendation': final_state['synthesis']}

    def get_conversation_history(self) -> List[Dict[str, str]]:
        return self.memory.get_messages()

    def clear_memory(self):
        self.memory.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        return self.cache.get_stats()

    def clear_cache(self):
        self.cache.clear()