"""DeepSeek V3.2-Exp API wrapper for unified interface."""
import os
from openai import OpenAI
from typing import List, Dict, Any, Optional
from src.config import Config


class DeepSeekWrapper:
    """Wrapper for DeepSeek API using OpenAI-compatible format."""

    def __init__(self, model: str = "deepseek-chat"):
        """
        Initialize DeepSeek client.

        Args:
            model: Either 'deepseek-chat' (non-thinking) or 'deepseek-reasoner' (thinking)
        """
        self.client = OpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )
        self.model = model
        self.is_reasoner = "reasoner" in model.lower()

    def generate(
        self,
        messages: List[Dict[str, str]] = None,
        input_text: str = None,
        instructions: str = None,
        temperature: float = None,
        max_tokens: int = None,
        stream: bool = False,
        tools: List[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Generate a response using DeepSeek API.

        Args:
            messages: List of message dicts [{"role": "user", "content": "..."}]
            input_text: Direct input string (converted to messages)
            instructions: System instructions (converted to system message)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            stream: Enable streaming responses
            tools: List of tools for function calling (chat mode only)

        Returns:
            Generated text response
        """
        # Convert to messages format if needed
        if messages is None:
            messages = []
            if instructions:
                messages.append({"role": "system", "content": instructions})
            if input_text:
                messages.append({"role": "user", "content": input_text})

        # Set default temperature based on DeepSeek docs
        if temperature is None:
            temperature = Config.TEMPERATURE_ANALYSIS  # 1.0 for data analysis

        # Set default max_tokens based on model
        if max_tokens is None:
            max_tokens = 32000 if self.is_reasoner else 4000

        # Build request params
        request_params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        # Add tools if provided (only for chat mode)
        if tools and not self.is_reasoner:
            request_params["tools"] = tools

        try:
            response = self.client.chat.completions.create(**request_params)

            if stream:
                return response  # Return generator for streaming
            else:
                content = response.choices[0].message.content

                # Store usage info for cost tracking
                if hasattr(response, 'usage'):
                    self._log_usage(response.usage)

                return content

        except Exception as e:
            return f"DeepSeek API Error: {str(e)}"

    def _log_usage(self, usage):
        """Log token usage for cost tracking."""
        # Calculate cost based on DeepSeek pricing
        input_tokens = getattr(usage, 'prompt_tokens', 0)
        output_tokens = getattr(usage, 'completion_tokens', 0)

        # Assume cache miss for now (conservative estimate)
        input_cost = input_tokens * 0.28 / 1_000_000
        output_cost = output_tokens * 0.42 / 1_000_000
        total_cost = input_cost + output_cost

        # Log to console (could be sent to monitoring system)
        print(f"[DeepSeek] Tokens: {input_tokens} in + {output_tokens} out = ${total_cost:.4f}")

    def generate_streaming(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        max_tokens: int = 4000,
        **kwargs
    ):
        """
        Generate streaming response.

        Yields chunks of text as they arrive.
        """
        response = self.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
