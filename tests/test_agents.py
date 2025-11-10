import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from src.agents.market_analysis import MarketAnalysisAgent
from src.agents.operations_audit import OperationsAuditAgent
from src.agents.financial_modeling import FinancialModelingAgent
from src.agents.lead_generation import LeadGenerationAgent


class TestMarketAnalysisAgent:
    
    @pytest.fixture
    def agent(self, mock_unified_llm):
        with patch('src.agents.market_analysis.UnifiedLLM', return_value=mock_unified_llm):
            return MarketAnalysisAgent()
    
    @pytest.mark.asyncio
    async def test_analyze_with_research_context(self, agent, sample_query, sample_research_context):
        agent.llm.generate.return_value = "Market analysis with citations (Source: Smith et al., 2024)"
        
        result = await agent.analyze(
            query=sample_query,
            research_context=sample_research_context
        )
        
        assert result is not None
        assert isinstance(result, str)
        agent.llm.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_without_research_context(self, agent, sample_query):
        agent.llm.generate.return_value = "Market analysis without research"
        
        result = await agent.analyze(query=sample_query)
        
        assert result is not None
        assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_analyze_with_web_research(self, agent, sample_query, sample_web_research_results):
        agent.llm.generate.return_value = "Market analysis with web research"
        
        result = await agent.analyze(
            query=sample_query,
            web_research_results=sample_web_research_results
        )
        
        assert result is not None
        assert isinstance(result, str)


class TestOperationsAuditAgent:
    
    @pytest.fixture
    def agent(self, mock_unified_llm):
        with patch('src.agents.operations_audit.UnifiedLLM', return_value=mock_unified_llm):
            return OperationsAuditAgent()
    
    @pytest.mark.asyncio
    async def test_audit_with_research_context(self, agent, sample_query, sample_research_context):
        agent.llm.generate.return_value = "Operations audit with research"
        
        result = await agent.audit(
            query=sample_query,
            research_context=sample_research_context
        )
        
        assert result is not None
        assert isinstance(result, str)
        agent.llm.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_audit_handles_llm_error(self, agent, sample_query):
        agent.llm.generate.side_effect = Exception("LLM API error")
        
        result = await agent.audit(query=sample_query)
        
        assert "Error in operations audit" in result


class TestFinancialModelingAgent:
    
    @pytest.fixture
    def agent(self, mock_unified_llm):
        with patch('src.agents.financial_modeling.UnifiedLLM', return_value=mock_unified_llm):
            return FinancialModelingAgent()
    
    @pytest.mark.asyncio
    async def test_model_financials_with_research(self, agent, sample_query, sample_research_context):
        agent.llm.generate.return_value = "Financial model with projections"
        
        result = await agent.model_financials(
            query=sample_query,
            research_context=sample_research_context
        )
        
        assert result is not None
        assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_model_financials_with_calculator_results(self, agent, sample_query):
        agent.llm.generate.return_value = "Financial model with calculations"
        calculator_results = {"roi": 25.5, "payback_period": 18}
        
        result = await agent.model_financials(
            query=sample_query,
            calculator_results=calculator_results
        )
        
        assert result is not None


class TestLeadGenerationAgent:
    
    @pytest.fixture
    def agent(self, mock_unified_llm):
        with patch('src.agents.lead_generation.UnifiedLLM', return_value=mock_unified_llm):
            return LeadGenerationAgent()
    
    @pytest.mark.asyncio
    async def test_generate_strategy(self, agent, sample_query):
        agent.llm.generate.return_value = "Lead generation strategy"
        
        result = await agent.generate_strategy(query=sample_query)
        
        assert result is not None
        assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_generate_strategy_handles_error(self, agent, sample_query):
        agent.llm.generate.side_effect = Exception("API timeout")
        
        result = await agent.generate_strategy(query=sample_query)
        
        assert "Error in lead generation strategy" in result


class TestAgentParallelExecution:
    
    @pytest.mark.asyncio
    async def test_all_agents_can_run_in_parallel(self, mock_unified_llm):
        with patch('src.agents.market_analysis.UnifiedLLM', return_value=mock_unified_llm), \
             patch('src.agents.operations_audit.UnifiedLLM', return_value=mock_unified_llm), \
             patch('src.agents.financial_modeling.UnifiedLLM', return_value=mock_unified_llm), \
             patch('src.agents.lead_generation.UnifiedLLM', return_value=mock_unified_llm):
            
            market_agent = MarketAnalysisAgent()
            ops_agent = OperationsAuditAgent()
            financial_agent = FinancialModelingAgent()
            leadgen_agent = LeadGenerationAgent()
            
            mock_unified_llm.generate.return_value = "Agent response"
            
            query = "Test parallel execution"
            
            results = await asyncio.gather(
                market_agent.analyze(query),
                ops_agent.audit(query),
                financial_agent.model_financials(query),
                leadgen_agent.generate_strategy(query)
            )
            
            assert len(results) == 4
            assert all(isinstance(r, str) for r in results)
