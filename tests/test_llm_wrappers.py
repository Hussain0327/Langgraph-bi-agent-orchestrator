import pytest
from unittest.mock import Mock, patch
from src.unified_llm import UnifiedLLM


class TestUnifiedLLM:
    
    def test_initialization_gpt5_strategy(self):
        with patch('src.unified_llm.GPT5Wrapper') as mock_gpt5:
            llm = UnifiedLLM(agent_type="market")
            assert llm.agent_type == "market"
    
    def test_hybrid_routing_research_synthesis(self):
        with patch('src.unified_llm.Config') as mock_config, \
             patch('src.unified_llm.GPT5Wrapper') as mock_gpt5, \
             patch('src.unified_llm.DeepSeekWrapper') as mock_deepseek:
            
            mock_config.MODEL_STRATEGY = "hybrid"
            llm = UnifiedLLM(agent_type="research_synthesis")
            
            provider, _ = llm._select_model()
            assert provider == "deepseek"
    
    def test_hybrid_routing_financial(self):
        with patch('src.unified_llm.Config') as mock_config, \
             patch('src.unified_llm.GPT5Wrapper') as mock_gpt5, \
             patch('src.unified_llm.DeepSeekWrapper') as mock_deepseek:
            
            mock_config.MODEL_STRATEGY = "hybrid"
            llm = UnifiedLLM(agent_type="financial")
            
            provider, _ = llm._select_model()
            assert provider == "deepseek"
    
    def test_optimal_temperature_financial(self):
        with patch('src.unified_llm.Config') as mock_config, \
             patch('src.unified_llm.GPT5Wrapper'), \
             patch('src.unified_llm.DeepSeekWrapper'):
            
            mock_config.MODEL_STRATEGY = "hybrid"
            mock_config.TEMPERATURE_CODING = 0.0
            mock_config.TEMPERATURE_CONVERSATION = 1.3
            mock_config.TEMPERATURE_ANALYSIS = 1.0
            
            llm = UnifiedLLM(agent_type="financial")
            temp = llm._get_optimal_temperature()
            
            assert temp == 0.0
    
    def test_optimal_temperature_market(self):
        with patch('src.unified_llm.Config') as mock_config, \
             patch('src.unified_llm.GPT5Wrapper'), \
             patch('src.unified_llm.DeepSeekWrapper'):
            
            mock_config.MODEL_STRATEGY = "hybrid"
            mock_config.TEMPERATURE_CODING = 0.0
            mock_config.TEMPERATURE_CONVERSATION = 1.3
            mock_config.TEMPERATURE_ANALYSIS = 1.0
            
            llm = UnifiedLLM(agent_type="market")
            temp = llm._get_optimal_temperature()
            
            assert temp == 1.3
    
    def test_optimal_max_tokens_research_synthesis(self):
        with patch('src.unified_llm.Config') as mock_config, \
             patch('src.unified_llm.GPT5Wrapper'), \
             patch('src.unified_llm.DeepSeekWrapper'):
            
            mock_config.MODEL_STRATEGY = "hybrid"
            llm = UnifiedLLM(agent_type="research_synthesis")
            max_tokens = llm._get_optimal_max_tokens()
            
            assert max_tokens == 32000
    
    def test_fallback_to_gpt5_on_deepseek_error(self):
        with patch('src.unified_llm.Config') as mock_config, \
             patch('src.unified_llm.GPT5Wrapper') as mock_gpt5_class, \
             patch('src.unified_llm.DeepSeekWrapper') as mock_deepseek_class:
            
            mock_config.MODEL_STRATEGY = "hybrid"
            
            mock_gpt5 = Mock()
            mock_gpt5.generate.return_value = "GPT-5 fallback response"
            mock_gpt5_class.return_value = mock_gpt5
            
            mock_deepseek_chat = Mock()
            mock_deepseek_chat.generate.side_effect = Exception("API error")
            mock_deepseek_reasoner = Mock()
            mock_deepseek_class.return_value = mock_deepseek_chat
            
            llm = UnifiedLLM(agent_type="market")
            llm.gpt5 = mock_gpt5
            llm.deepseek_chat = mock_deepseek_chat
            
            result = llm.generate(input_text="Test query")
            
            assert result == "GPT-5 fallback response"
            mock_gpt5.generate.assert_called_once()
