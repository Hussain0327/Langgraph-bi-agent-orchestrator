import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-5-nano')
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
    DEEPSEEK_CHAT_MODEL = os.getenv('DEEPSEEK_CHAT_MODEL', 'deepseek-chat')
    DEEPSEEK_REASONER_MODEL = os.getenv('DEEPSEEK_REASONER_MODEL', 'deepseek-reasoner')
    MODEL_STRATEGY = os.getenv('MODEL_STRATEGY', 'hybrid')
    LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2', 'true').lower() == 'true'
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
    LANGCHAIN_PROJECT = os.getenv('LANGCHAIN_PROJECT', 'business-intelligence-orchestrator')
    LANGCHAIN_ENDPOINT = os.getenv('LANGCHAIN_ENDPOINT', 'https://api.smith.langchain.com')
    MAX_MEMORY_MESSAGES = 10
    REASONING_EFFORT = 'low'
    TEXT_VERBOSITY = 'medium'
    MAX_OUTPUT_TOKENS = 2000
    TEMPERATURE_CODING = 0.0
    TEMPERATURE_ANALYSIS = 1.0
    TEMPERATURE_CONVERSATION = 1.3
    TEMPERATURE_CREATIVE = 1.5

    @classmethod
    def validate(cls):
        if cls.MODEL_STRATEGY in ['gpt5', 'hybrid']:
            if not cls.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY required when MODEL_STRATEGY is 'gpt5' or 'hybrid'")
        if cls.MODEL_STRATEGY in ['deepseek', 'hybrid']:
            if not cls.DEEPSEEK_API_KEY:
                raise ValueError("DEEPSEEK_API_KEY required when MODEL_STRATEGY is 'deepseek' or 'hybrid'")
        if not cls.LANGCHAIN_API_KEY and cls.LANGCHAIN_TRACING_V2:
            print('Warning: LANGCHAIN_TRACING_V2 is enabled but LANGCHAIN_API_KEY is not set')
            print('Get your API key from: https://smith.langchain.com/settings')
            cls.LANGCHAIN_TRACING_V2 = False

    @classmethod
    def is_gpt5(cls) -> bool:
        return 'gpt-5' in cls.OPENAI_MODEL.lower()

    @classmethod
    def is_deepseek(cls) -> bool:
        return cls.MODEL_STRATEGY in ['deepseek', 'hybrid']

    @classmethod
    def is_hybrid(cls) -> bool:
        return cls.MODEL_STRATEGY == 'hybrid'
Config.validate()