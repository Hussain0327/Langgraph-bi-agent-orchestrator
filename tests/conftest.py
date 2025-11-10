import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, Any


@pytest.fixture
def mock_gpt5_wrapper():
    mock = Mock()
    mock.generate.return_value = "Test GPT-5 response"
    return mock


@pytest.fixture
def mock_deepseek_wrapper():
    mock = Mock()
    mock.generate.return_value = "Test DeepSeek response"
    return mock


@pytest.fixture
def mock_unified_llm(monkeypatch):
    mock_llm = Mock()
    mock_llm.generate.return_value = "Test LLM response"
    return mock_llm


@pytest.fixture
def sample_query():
    return "How can I improve my SaaS product's customer retention?"


@pytest.fixture
def sample_research_context():
    return """Research Context:
Key Research Themes:
- Customer retention strategies improve LTV by 25-40% (Source: Smith et al., 2024)
- Onboarding quality correlates with 60% retention improvement (Source: Jones et al., 2023)

References:
1. Smith, J. et al. (2024). "Customer Retention in SaaS"
2. Jones, A. et al. (2023). "Onboarding Best Practices"
"""


@pytest.fixture
def sample_agent_state():
    return {
        "query": "Test business query",
        "agents_to_call": ["market", "financial"],
        "research_enabled": True,
        "research_findings": {},
        "research_context": "",
        "market_analysis": "",
        "operations_audit": "",
        "financial_modeling": "",
        "lead_generation": "",
        "web_research": {},
        "synthesis": "",
        "conversation_history": [],
        "use_memory": True,
    }


@pytest.fixture
def sample_web_research_results():
    return {
        "insights": "Market growing at 15% CAGR. Key competitors: CompanyA, CompanyB.",
        "sources": ["https://example.com/market-report"]
    }
