import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from src.langgraph_orchestrator import LangGraphOrchestrator, AgentState


class TestLangGraphOrchestrator:
    
    @pytest.fixture
    def orchestrator(self):
        with patch('src.langgraph_orchestrator.GPT5Wrapper'), \
             patch('src.langgraph_orchestrator.MarketAnalysisAgent'), \
             patch('src.langgraph_orchestrator.OperationsAuditAgent'), \
             patch('src.langgraph_orchestrator.FinancialModelingAgent'), \
             patch('src.langgraph_orchestrator.LeadGenerationAgent'), \
             patch('src.langgraph_orchestrator.ResearchSynthesisAgent'), \
             patch('src.langgraph_orchestrator.WebResearchTool'), \
             patch('src.langgraph_orchestrator.ConversationMemory'):
            
            return LangGraphOrchestrator(enable_rag=True, use_ml_routing=False)
    
    def test_initialization(self, orchestrator):
        assert orchestrator.enable_rag is True
        assert orchestrator.use_ml_routing is False
        assert orchestrator.graph is not None
    
    def test_router_node_gpt5_routing(self, orchestrator, sample_agent_state):
        orchestrator.gpt5.generate = Mock(return_value='["market", "financial"]')
        
        result = orchestrator._router_node(sample_agent_state)
        
        assert "agents_to_call" in result
        assert isinstance(result["agents_to_call"], list)
    
    def test_router_node_fallback_on_error(self, orchestrator, sample_agent_state):
        orchestrator.gpt5.generate = Mock(side_effect=Exception("API error"))
        
        result = orchestrator._router_node(sample_agent_state)
        
        assert result["agents_to_call"] == ["market", "operations", "financial", "leadgen"]
    
    def test_research_synthesis_node_rag_disabled(self, sample_agent_state):
        with patch('src.langgraph_orchestrator.GPT5Wrapper'), \
             patch('src.langgraph_orchestrator.MarketAnalysisAgent'), \
             patch('src.langgraph_orchestrator.OperationsAuditAgent'), \
             patch('src.langgraph_orchestrator.FinancialModelingAgent'), \
             patch('src.langgraph_orchestrator.LeadGenerationAgent'), \
             patch('src.langgraph_orchestrator.WebResearchTool'), \
             patch('src.langgraph_orchestrator.ConversationMemory'):
            
            orchestrator = LangGraphOrchestrator(enable_rag=False)
            
            result = orchestrator._research_synthesis_node(sample_agent_state)
            
            assert result["research_findings"] == {}
            assert result["research_context"] == ""
    
    def test_research_synthesis_node_with_rag(self, orchestrator, sample_agent_state):
        mock_research_result = {
            "research_context": "Test research context",
            "paper_count": 3
        }
        orchestrator.research_agent.synthesize = Mock(return_value=mock_research_result)
        
        result = orchestrator._research_synthesis_node(sample_agent_state)
        
        assert result["research_context"] == "Test research context"
        assert result["research_findings"] == mock_research_result
    
    @pytest.mark.asyncio
    async def test_execute_agents_parallel(self, orchestrator, sample_agent_state):
        sample_agent_state["agents_to_call"] = ["market", "financial"]
        
        orchestrator.market_agent.analyze = Mock(return_value=asyncio.coroutine(lambda: "Market analysis")())
        orchestrator.financial_agent.model_financials = Mock(return_value=asyncio.coroutine(lambda: "Financial model")())
        orchestrator.web_research.execute = Mock(return_value={"insights": "test"})
        
        results = await orchestrator._execute_agents_parallel(sample_agent_state)
        
        assert "market_analysis" in results
        assert "financial_modeling" in results
    
    def test_get_conversation_history(self, orchestrator):
        orchestrator.memory.get_messages = Mock(return_value=[
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Response"}
        ])
        
        history = orchestrator.get_conversation_history()
        
        assert len(history) == 2
        assert history[0]["role"] == "user"
    
    def test_clear_memory(self, orchestrator):
        orchestrator.memory.clear = Mock()
        
        orchestrator.clear_memory()
        
        orchestrator.memory.clear.assert_called_once()


class TestAgentState:
    
    def test_agent_state_structure(self, sample_agent_state):
        assert "query" in sample_agent_state
        assert "agents_to_call" in sample_agent_state
        assert "research_enabled" in sample_agent_state
        assert "market_analysis" in sample_agent_state
        assert "synthesis" in sample_agent_state
    
    def test_agent_state_types(self, sample_agent_state):
        assert isinstance(sample_agent_state["query"], str)
        assert isinstance(sample_agent_state["agents_to_call"], list)
        assert isinstance(sample_agent_state["research_enabled"], bool)
        assert isinstance(sample_agent_state["conversation_history"], list)
