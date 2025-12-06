import os
from openai import OpenAI
from typing import List, Dict, Any, Optional
from src.config import Config

class GPT5Wrapper:

    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = Config.OPENAI_MODEL
        self.is_gpt5 = Config.is_gpt5()

    def generate(self, messages: List[Dict[str, str]]=None, input_text: str=None, instructions: str=None, reasoning_effort: str=None, text_verbosity: str=None, max_output_tokens: int=None, tools: List[Dict[str, Any]]=None) -> str:
        try:
            if self.is_gpt5:
                return self._generate_gpt5(messages=messages, input_text=input_text, instructions=instructions, reasoning_effort=reasoning_effort or Config.REASONING_EFFORT, text_verbosity=text_verbosity or Config.TEXT_VERBOSITY, max_output_tokens=max_output_tokens or Config.MAX_OUTPUT_TOKENS, tools=tools)
            else:
                return self._generate_chat_completions(messages=messages, max_tokens=max_output_tokens or Config.MAX_OUTPUT_TOKENS, tools=tools)
        except Exception as e:
            return f'Error generating response: {str(e)}'

    def _generate_gpt5(self, messages: List[Dict[str, str]]=None, input_text: str=None, instructions: str=None, reasoning_effort: str='medium', text_verbosity: str='medium', max_output_tokens: int=2000, tools: List[Dict[str, Any]]=None) -> str:
        if input_text is None and messages:
            system_msgs = [m for m in messages if m.get('role') == 'system']
            if system_msgs and (not instructions):
                instructions = system_msgs[0]['content']
            user_msgs = [m for m in messages if m.get('role') != 'system']
            if user_msgs:
                input_text = user_msgs if len(user_msgs) > 1 else user_msgs[0]['content']
            else:
                input_text = ''
        request_params = {'model': self.model, 'input': input_text, 'reasoning': {'effort': reasoning_effort}, 'text': {'verbosity': text_verbosity}, 'max_output_tokens': max_output_tokens}
        if instructions:
            request_params['instructions'] = instructions
        if tools:
            request_params['tools'] = self._convert_tools_to_gpt5(tools)
        response = self.client.responses.create(**request_params)
        return self._extract_text_from_response(response)

    def _generate_chat_completions(self, messages: List[Dict[str, str]], max_tokens: int=2000, tools: List[Dict[str, Any]]=None) -> str:
        request_params = {'model': self.model, 'messages': messages, 'max_tokens': max_tokens}
        if tools:
            request_params['tools'] = tools
        response = self.client.chat.completions.create(**request_params)
        return response.choices[0].message.content

    def _convert_tools_to_gpt5(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        converted_tools = []
        for tool in tools:
            if tool.get('type') == 'function' and 'function' in tool:
                func = tool['function']
                converted_tools.append({'type': 'function', 'name': func.get('name'), 'description': func.get('description'), 'parameters': func.get('parameters', {})})
            else:
                converted_tools.append(tool)
        return converted_tools

    def _extract_text_from_response(self, response: Any) -> str:
        if hasattr(response, 'output_text'):
            return response.output_text
        if hasattr(response, 'output'):
            for item in response.output:
                if item.get('type') == 'message':
                    content = item.get('content', [])
                    for content_item in content:
                        if content_item.get('type') == 'output_text':
                            return content_item.get('text', '')
        return 'No text output found in response'