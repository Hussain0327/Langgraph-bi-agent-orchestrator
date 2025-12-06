import os
from openai import OpenAI
from typing import List, Dict, Any, Optional
from src.config import Config

class DeepSeekWrapper:

    def __init__(self, model: str='deepseek-chat'):
        self.client = OpenAI(api_key=Config.DEEPSEEK_API_KEY, base_url='https://api.deepseek.com')
        self.model = model
        self.is_reasoner = 'reasoner' in model.lower()

    def generate(self, messages: List[Dict[str, str]]=None, input_text: str=None, instructions: str=None, temperature: float=None, max_tokens: int=None, stream: bool=False, tools: List[Dict[str, Any]]=None, **kwargs) -> str:
        if messages is None:
            messages = []
            if instructions:
                messages.append({'role': 'system', 'content': instructions})
            if input_text:
                messages.append({'role': 'user', 'content': input_text})
        if temperature is None:
            temperature = Config.TEMPERATURE_ANALYSIS
        if max_tokens is None:
            max_tokens = 32000 if self.is_reasoner else 4000
        request_params = {'model': self.model, 'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens, 'stream': stream}
        if tools and (not self.is_reasoner):
            request_params['tools'] = tools
        try:
            response = self.client.chat.completions.create(**request_params)
            if stream:
                return response
            else:
                content = response.choices[0].message.content
                if hasattr(response, 'usage'):
                    self._log_usage(response.usage)
                return content
        except Exception as e:
            return f'DeepSeek API Error: {str(e)}'

    def _log_usage(self, usage):
        input_tokens = getattr(usage, 'prompt_tokens', 0)
        output_tokens = getattr(usage, 'completion_tokens', 0)
        input_cost = input_tokens * 0.28 / 1000000
        output_cost = output_tokens * 0.42 / 1000000
        total_cost = input_cost + output_cost
        print(f'[DeepSeek] Tokens: {input_tokens} in + {output_tokens} out = ${total_cost:.4f}')

    def generate_streaming(self, messages: List[Dict[str, str]], temperature: float=1.0, max_tokens: int=4000, **kwargs):
        response = self.generate(messages=messages, temperature=temperature, max_tokens=max_tokens, stream=True, **kwargs)
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content