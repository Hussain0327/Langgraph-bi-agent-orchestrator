"""Test script for DeepSeek integration."""
import sys
sys.path.insert(0, '/workspaces/multi_agent_workflow')

from src.unified_llm import UnifiedLLM
from src.config import Config

def test_deepseek_chat():
    """Test DeepSeek chat model."""
    print("=" * 60)
    print("TEST 1: DeepSeek Chat Model")
    print("=" * 60)

    llm = UnifiedLLM(agent_type="market")

    print(f"Strategy: {Config.MODEL_STRATEGY}")
    print(f"Selected Provider: {llm.get_current_provider()}")
    print()

    messages = [
        {"role": "system", "content": "You are a market analyst."},
        {"role": "user", "content": "Briefly explain SaaS pricing models in 2 sentences."}
    ]

    print("Generating response...")
    response = llm.generate(messages=messages)

    print("\n✅ Response:")
    print(response)
    print()

    return response

def test_deepseek_reasoner():
    """Test DeepSeek reasoner model for research synthesis."""
    print("=" * 60)
    print("TEST 2: DeepSeek Reasoner Model (Research Synthesis)")
    print("=" * 60)

    llm = UnifiedLLM(agent_type="research_synthesis")

    print(f"Strategy: {Config.MODEL_STRATEGY}")
    print(f"Selected Provider: {llm.get_current_provider()}")
    print()

    messages = [
        {"role": "system", "content": "You are a research analyst."},
        {"role": "user", "content": "What are 3 key factors in SaaS customer retention? Be concise."}
    ]

    print("Generating response...")
    response = llm.generate(messages=messages)

    print("\n✅ Response:")
    print(response)
    print()

    return response

def test_cost_comparison():
    """Test cost estimation."""
    print("=" * 60)
    print("TEST 3: Cost Comparison")
    print("=" * 60)

    input_tokens = 10000
    output_tokens = 10000

    # GPT-5 cost
    llm_gpt5 = UnifiedLLM(agent_type="market")
    llm_gpt5.strategy = "gpt5"
    gpt5_cost = llm_gpt5.estimate_cost(input_tokens, output_tokens)

    # DeepSeek cost
    llm_deepseek = UnifiedLLM(agent_type="market")
    llm_deepseek.strategy = "deepseek"
    deepseek_cost = llm_deepseek.estimate_cost(input_tokens, output_tokens)

    print(f"Query: {input_tokens:,} input tokens + {output_tokens:,} output tokens")
    print()
    print(f"GPT-5 Cost:      ${gpt5_cost:.4f}")
    print(f"DeepSeek Cost:   ${deepseek_cost:.4f}")
    print(f"Savings:         ${gpt5_cost - deepseek_cost:.4f} ({((gpt5_cost - deepseek_cost) / gpt5_cost * 100):.1f}%)")
    print()

    # Monthly projection
    queries_per_day = 100
    days_per_month = 30

    monthly_gpt5 = gpt5_cost * queries_per_day * days_per_month
    monthly_deepseek = deepseek_cost * queries_per_day * days_per_month

    print(f"Monthly Cost (100 queries/day):")
    print(f"GPT-5:           ${monthly_gpt5:,.2f}/month")
    print(f"DeepSeek:        ${monthly_deepseek:,.2f}/month")
    print(f"Monthly Savings: ${monthly_gpt5 - monthly_deepseek:,.2f}/month")
    print()

def test_hybrid_routing():
    """Test hybrid routing logic."""
    print("=" * 60)
    print("TEST 4: Hybrid Routing Strategy")
    print("=" * 60)

    agent_types = [
        "research_synthesis",
        "financial",
        "market",
        "operations",
        "leadgen",
        "router",
        "synthesis"
    ]

    print("Agent Type           → Selected Provider")
    print("-" * 60)

    for agent_type in agent_types:
        llm = UnifiedLLM(agent_type=agent_type)
        provider = llm.get_current_provider()
        temp = llm._get_optimal_temperature()
        max_tokens = llm._get_optimal_max_tokens()

        print(f"{agent_type:20} → {provider:30} (temp={temp}, max={max_tokens:,})")

    print()

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🚀 DEEPSEEK INTEGRATION TEST SUITE")
    print("=" * 60)
    print()

    try:
        # Test 1: DeepSeek Chat
        chat_response = test_deepseek_chat()

        if not chat_response or "Error" in chat_response:
            print("❌ DeepSeek Chat test failed!")
            return False

        # Test 2: DeepSeek Reasoner
        reasoner_response = test_deepseek_reasoner()

        if not reasoner_response or "Error" in reasoner_response:
            print("❌ DeepSeek Reasoner test failed!")
            return False

        # Test 3: Cost Comparison
        test_cost_comparison()

        # Test 4: Hybrid Routing
        test_hybrid_routing()

        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Update agents to use UnifiedLLM wrapper")
        print("2. Run full system test with CLI")
        print("3. Run evaluation comparison (GPT-5 vs DeepSeek)")
        print()

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
