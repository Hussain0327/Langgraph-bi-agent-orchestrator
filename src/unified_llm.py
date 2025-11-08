"""Unified LLM wrapper supporting GPT-5, DeepSeek, and Hybrid strategies."""
from typing import List, Dict, Any, Optional
from src.config import Config
from src.gpt5_wrapper import GPT5Wrapper
from src.deepseek_wrapper import DeepSeekWrapper


class UnifiedLLM:

    def __init__(self, agent_type: Optional[str] = None):

        self.agent_type = agent_type
        self.strategy = Config.MODEL_STRATEGY

        # Initialize providers based on strategy
        self.gpt5 = None
        self.deepseek_chat = None
        self.deepseek_reasoner = None

        if self.strategy in ["gpt5", "hybrid"]:
            self.gpt5 = GPT5Wrapper()

        if self.strategy in ["deepseek", "hybrid"]:
            self.deepseek_chat = DeepSeekWrapper(model=Config.DEEPSEEK_CHAT_MODEL)
            self.deepseek_reasoner = DeepSeekWrapper(model=Config.DEEPSEEK_REASONER_MODEL)

    def generate(
        self,
        messages: List[Dict[str, str]] = None,
        input_text: str = None,
        instructions: str = None,
        temperature: float = None,
        max_tokens: int = None,
        tools: List[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Generate response using optimal model based on strategy.

        Args:
            messages: List of message dicts
            input_text: Direct input string
            instructions: System instructions
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            tools: Function calling tools

        Returns:
            Generated text response
        """
        # Select the best model for this task
        provider, model_name = self._select_model()

        # Set optimal temperature if not specified
        if temperature is None:
            temperature = self._get_optimal_temperature()

        # Set optimal max_tokens if not specified
        if max_tokens is None:
            max_tokens = self._get_optimal_max_tokens()

        try:
            # Generate with selected provider
            if provider == "gpt5":
                return self.gpt5.generate(
                    messages=messages,
                    input_text=input_text,
                    instructions=instructions,
                    max_output_tokens=max_tokens,
                    tools=tools,
                    **kwargs
                )

            elif provider == "deepseek":
                return model_name.generate(
                    messages=messages,
                    input_text=input_text,
                    instructions=instructions,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    **kwargs
                )

        except Exception as e:
            # Fallback to GPT-5 if hybrid mode and DeepSeek fails
            if self.strategy == "hybrid" and provider == "deepseek" and self.gpt5:
                print(f"⚠️ DeepSeek failed ({str(e)}), falling back to GPT-5...")
                return self.gpt5.generate(
                    messages=messages,
                    input_text=input_text,
                    instructions=instructions,
                    max_output_tokens=max_tokens,
                    tools=tools,
                    **kwargs
                )
            else:
                raise

    def _select_model(self) -> tuple:
        """
        Select the optimal model based on strategy and agent type.

        Returns:
            Tuple of (provider_name, provider_instance)
        """
        if self.strategy == "gpt5":
            return ("gpt5", self.gpt5)

        if self.strategy == "deepseek":
            # Use reasoner for research synthesis, chat for everything else
            if self.agent_type == "research_synthesis":
                return ("deepseek", self.deepseek_reasoner)
            else:
                return ("deepseek", self.deepseek_chat)

        # Hybrid strategy: intelligent routing
        if self.strategy == "hybrid":
            return self._hybrid_routing()

        # Default to GPT-5
        return ("gpt5", self.gpt5)

    def _hybrid_routing(self) -> tuple:
        """
        Hybrid routing logic: choose best model for each agent type.

        Strategy:
        - Research Synthesis: DeepSeek-reasoner (needs deep thinking, long output)
        - Financial Modeling: DeepSeek-chat (good at math)
        - Market/Operations/LeadGen: DeepSeek-chat (fast and cheap)
        - Router: DeepSeek-chat (simple classification)
        - Synthesis: DeepSeek-chat (aggregation task)
        - Unknown: GPT-5 (safe default)
        """
        routing_map = {
            "research_synthesis": ("deepseek", self.deepseek_reasoner),  # Deep thinking
            "financial": ("deepseek", self.deepseek_chat),               # Math
            "market": ("deepseek", self.deepseek_chat),                  # Analysis
            "operations": ("deepseek", self.deepseek_chat),              # Analysis
            "leadgen": ("deepseek", self.deepseek_chat),                 # Analysis
            "router": ("deepseek", self.deepseek_chat),                  # Classification
            "synthesis": ("deepseek", self.deepseek_chat),               # Aggregation
        }

        # Use mapping if agent type is known, otherwise default to GPT-5
        return routing_map.get(self.agent_type, ("gpt5", self.gpt5))

    def _get_optimal_temperature(self) -> float:
        """Get optimal temperature for this agent type."""
        temperature_map = {
            "financial": Config.TEMPERATURE_CODING,         # 0.0 for math
            "market": Config.TEMPERATURE_CONVERSATION,      # 1.3 for analysis
            "operations": Config.TEMPERATURE_ANALYSIS,      # 1.0 for analysis
            "leadgen": Config.TEMPERATURE_CONVERSATION,     # 1.3 for suggestions
            "research_synthesis": Config.TEMPERATURE_ANALYSIS,  # 1.0 for synthesis
            "router": Config.TEMPERATURE_CODING,            # 0.0 for classification
            "synthesis": Config.TEMPERATURE_ANALYSIS,       # 1.0 for aggregation
        }

        return temperature_map.get(self.agent_type, Config.TEMPERATURE_ANALYSIS)

    def _get_optimal_max_tokens(self) -> int:
        """Get optimal max tokens for this agent type."""
        # Research synthesis needs more tokens for comprehensive output
        if self.agent_type == "research_synthesis":
            return 32000 if self.strategy == "hybrid" else 16000

        # Financial modeling may need detailed calculations
        if self.agent_type == "financial":
            return 8000

        # Default for other agents
        return 4000

    def get_current_provider(self) -> str:
        """Get the name of the currently selected provider."""
        provider, _ = self._select_model()

        if provider == "gpt5":
            return "GPT-5-nano"
        elif provider == "deepseek":
            model = _
            if "reasoner" in model.model:
                return "DeepSeek-V3.2-Exp (Reasoner)"
            else:
                return "DeepSeek-V3.2-Exp (Chat)"

        return "Unknown"

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost for a query.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        provider, model = self._select_model()

        if provider == "gpt5":
            # GPT-5-nano pricing (approximate)
            return (input_tokens * 0.015 + output_tokens * 0.060) / 1_000_000

        elif provider == "deepseek":
            # DeepSeek pricing (assuming cache miss for conservative estimate)
            input_cost = input_tokens * 0.28 / 1_000_000
            output_cost = output_tokens * 0.42 / 1_000_000
            return input_cost + output_cost

        return 0.0
