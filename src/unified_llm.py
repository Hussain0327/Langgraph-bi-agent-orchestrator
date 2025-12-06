from typing import List, Dict, Any, Optional
from src.config import Config
from src.gpt5_wrapper import GPT5Wrapper
from src.deepseek_wrapper import DeepSeekWrapper

class UnifiedLLM:

    def __init__(self, agent_type: Optional[str]=None):
        self.agent_type = agent_type
        self.strategy = Config.MODEL_STRATEGY
        self.gpt5 = None
        self.deepseek_chat = None
        self.deepseek_reasoner = None
        if self.strategy in ['gpt5', 'hybrid']:
            self.gpt5 = GPT5Wrapper()
        if self.strategy in ['deepseek', 'hybrid']:
            self.deepseek_chat = DeepSeekWrapper(model=Config.DEEPSEEK_CHAT_MODEL)
            self.deepseek_reasoner = DeepSeekWrapper(model=Config.DEEPSEEK_REASONER_MODEL)

    def generate(self, messages: List[Dict[str, str]]=None, input_text: str=None, instructions: str=None, temperature: float=None, max_tokens: int=None, tools: List[Dict[str, Any]]=None, **kwargs) -> str:
        provider, model_name = self._select_model()
        if temperature is None:
            temperature = self._get_optimal_temperature()
        if max_tokens is None:
            max_tokens = self._get_optimal_max_tokens()
        try:
            if provider == 'gpt5':
                return self.gpt5.generate(messages=messages, input_text=input_text, instructions=instructions, max_output_tokens=max_tokens, tools=tools, **kwargs)
            elif provider == 'deepseek':
                return model_name.generate(messages=messages, input_text=input_text, instructions=instructions, temperature=temperature, max_tokens=max_tokens, tools=tools, **kwargs)
        except Exception as e:
            if self.strategy == 'hybrid' and provider == 'deepseek' and self.gpt5:
                print(f' DeepSeek failed ({str(e)}), falling back to GPT-5...')
                return self.gpt5.generate(messages=messages, input_text=input_text, instructions=instructions, max_output_tokens=max_tokens, tools=tools, **kwargs)
            else:
                raise

    def _select_model(self) -> tuple:
        if self.strategy == 'gpt5':
            return ('gpt5', self.gpt5)
        if self.strategy == 'deepseek':
            if self.agent_type == 'research_synthesis':
                return ('deepseek', self.deepseek_reasoner)
            else:
                return ('deepseek', self.deepseek_chat)
        if self.strategy == 'hybrid':
            return self._hybrid_routing()
        return ('gpt5', self.gpt5)

    def _hybrid_routing(self) -> tuple:
        routing_map = {'research_synthesis': ('deepseek', self.deepseek_reasoner), 'financial': ('deepseek', self.deepseek_chat), 'market': ('deepseek', self.deepseek_chat), 'operations': ('deepseek', self.deepseek_chat), 'leadgen': ('deepseek', self.deepseek_chat), 'router': ('deepseek', self.deepseek_chat), 'synthesis': ('deepseek', self.deepseek_chat)}
        return routing_map.get(self.agent_type, ('gpt5', self.gpt5))

    def _get_optimal_temperature(self) -> float:
        temperature_map = {'financial': Config.TEMPERATURE_CODING, 'market': Config.TEMPERATURE_CONVERSATION, 'operations': Config.TEMPERATURE_ANALYSIS, 'leadgen': Config.TEMPERATURE_CONVERSATION, 'research_synthesis': Config.TEMPERATURE_ANALYSIS, 'router': Config.TEMPERATURE_CODING, 'synthesis': Config.TEMPERATURE_ANALYSIS}
        return temperature_map.get(self.agent_type, Config.TEMPERATURE_ANALYSIS)

    def _get_optimal_max_tokens(self) -> int:
        if self.agent_type == 'research_synthesis':
            return 32000 if self.strategy == 'hybrid' else 16000
        if self.agent_type == 'financial':
            return 8000
        return 4000

    def get_current_provider(self) -> str:
        provider, _ = self._select_model()
        if provider == 'gpt5':
            return 'GPT-5-nano'
        elif provider == 'deepseek':
            model = _
            if 'reasoner' in model.model:
                return 'DeepSeek-V3.2-Exp (Reasoner)'
            else:
                return 'DeepSeek-V3.2-Exp (Chat)'
        return 'Unknown'

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        provider, model = self._select_model()
        if provider == 'gpt5':
            return (input_tokens * 0.015 + output_tokens * 0.06) / 1000000
        elif provider == 'deepseek':
            input_cost = input_tokens * 0.28 / 1000000
            output_cost = output_tokens * 0.42 / 1000000
            return input_cost + output_cost
        return 0.0