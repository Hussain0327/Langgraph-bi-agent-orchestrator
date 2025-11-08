"""Configuration management for the Business Intelligence Orchestrator."""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Central configuration for the application."""

    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")

    # DeepSeek Configuration
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_CHAT_MODEL = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
    DEEPSEEK_REASONER_MODEL = os.getenv("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner")

    # Model Selection Strategy
    # Options: "gpt5", "deepseek", "hybrid"
    MODEL_STRATEGY = os.getenv("MODEL_STRATEGY", "hybrid")

    # LangSmith Configuration
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true").lower() == "true"
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "business-intelligence-orchestrator")
    LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    # Application Configuration
    MAX_MEMORY_MESSAGES = 10

    # GPT-5 Specific Configuration
    REASONING_EFFORT = "low"    # minimal, low, medium, high (using 'low' to avoid token issues)
    TEXT_VERBOSITY = "medium"   # low, medium, high
    MAX_OUTPUT_TOKENS = 2000

    # DeepSeek Temperature Settings (from docs)
    TEMPERATURE_CODING = 0.0         # For coding/math tasks
    TEMPERATURE_ANALYSIS = 1.0       # For data analysis
    TEMPERATURE_CONVERSATION = 1.3   # For general conversation
    TEMPERATURE_CREATIVE = 1.5       # For creative writing

    @classmethod
    def validate(cls):
        """Validate that required configuration is present."""
        # Check model strategy and validate appropriate API keys
        if cls.MODEL_STRATEGY in ["gpt5", "hybrid"]:
            if not cls.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY required when MODEL_STRATEGY is 'gpt5' or 'hybrid'")

        if cls.MODEL_STRATEGY in ["deepseek", "hybrid"]:
            if not cls.DEEPSEEK_API_KEY:
                raise ValueError("DEEPSEEK_API_KEY required when MODEL_STRATEGY is 'deepseek' or 'hybrid'")

        # LangSmith is optional, just warn if not configured
        if not cls.LANGCHAIN_API_KEY and cls.LANGCHAIN_TRACING_V2:
            print("Warning: LANGCHAIN_TRACING_V2 is enabled but LANGCHAIN_API_KEY is not set")
            print("Get your API key from: https://smith.langchain.com/settings")
            cls.LANGCHAIN_TRACING_V2 = False

    @classmethod
    def is_gpt5(cls) -> bool:
        """Check if using GPT-5 model."""
        return "gpt-5" in cls.OPENAI_MODEL.lower()

    @classmethod
    def is_deepseek(cls) -> bool:
        """Check if using DeepSeek model."""
        return cls.MODEL_STRATEGY in ["deepseek", "hybrid"]

    @classmethod
    def is_hybrid(cls) -> bool:
        """Check if using hybrid model strategy."""
        return cls.MODEL_STRATEGY == "hybrid"


# Validate configuration on import
Config.validate()
