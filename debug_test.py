from src.config import Config
from openai import OpenAI
print(f'API Key present: {Config.OPENAI_API_KEY is not None}')
print(f'API Key length: {(len(Config.OPENAI_API_KEY) if Config.OPENAI_API_KEY else 0)}')
print(f'Model: {Config.OPENAI_MODEL}')
client = OpenAI(api_key=Config.OPENAI_API_KEY)
response = client.responses.create(model=Config.OPENAI_MODEL, input='What is 2+2?\nWhat is Chainsaw Man?\nWhat is the current temp in Japan?', max_output_tokens=20)
print(f'\nAPI Response: {response.output_text}')
print(f"  output_text: '{response.output_text}'")
print(f'  length: {len(response.output_text)}')
print(f'  type: {type(response.output_text)}')