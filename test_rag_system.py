"""
Test script for RAG-enabled Business Intelligence Orchestrator

Tests the full Phase 2 integration:
- Vector store
- Research retrieval
- Research synthesis agent
- Updated agents with citations
- LangGraph orchestrator with RAG
"""

import sys
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


def print_header(text: str):
    """Print a colored header."""
    print("\n" + "=" * 70)
    print(Fore.CYAN + Style.BRIGHT + text)
    print("=" * 70)


def print_section(text: str):
    """Print a colored section header."""
    print("\n" + Fore.YELLOW + "─" * 70)
    print(Fore.YELLOW + Style.BRIGHT + text)
    print(Fore.YELLOW + "─" * 70)


def print_success(text: str):
    """Print a success message."""
    print(Fore.GREEN + f"✓ {text}")


def print_error(text: str):
    """Print an error message."""
    print(Fore.RED + f"✗ {text}")


def print_info(text: str):
    """Print an info message."""
    print(Fore.CYAN + f"ℹ {text}")


def test_imports():
    """Test that all modules import correctly."""
    print_section("Test 1: Module Imports")

    try:
        from src.vector_store import VectorStore
        print_success("Vector Store imported")
    except Exception as e:
        print_error(f"Vector Store import failed: {e}")
        return False

    try:
        from src.tools.research_retrieval import ResearchRetriever
        print_success("Research Retriever imported")
    except Exception as e:
        print_error(f"Research Retriever import failed: {e}")
        return False

    try:
        from src.agents.research_synthesis import ResearchSynthesisAgent
        print_success("Research Synthesis Agent imported")
    except Exception as e:
        print_error(f"Research Synthesis Agent import failed: {e}")
        return False

    try:
        from src.langgraph_orchestrator import LangGraphOrchestrator
        print_success("LangGraph Orchestrator imported (with RAG)")
    except Exception as e:
        print_error(f"LangGraph Orchestrator import failed: {e}")
        return False

    print_success("All imports successful!")
    return True


def test_research_retrieval():
    """Test research retrieval functionality."""
    print_section("Test 2: Research Retrieval")

    try:
        from src.tools.research_retrieval import ResearchRetriever

        retriever = ResearchRetriever()
        print_success("Research Retriever initialized")

        # Test query
        query = "SaaS pricing strategies"
        print_info(f"Test query: '{query}'")

        papers = retriever.retrieve_papers(query, top_k=2)

        if len(papers) > 0:
            print_success(f"Retrieved {len(papers)} papers")

            for i, paper in enumerate(papers, 1):
                print(f"\n  Paper {i}:")
                print(f"    Title: {paper['title'][:80]}...")
                print(f"    Authors: {', '.join(paper['authors'][:2])}")
                print(f"    Year: {paper['year']}")
                print(f"    Source: {paper['source']}")
        else:
            print_info("No papers retrieved (may be due to rate limiting or API issues)")

        return True

    except Exception as e:
        print_error(f"Research retrieval test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_research_synthesis_agent():
    """Test research synthesis agent."""
    print_section("Test 3: Research Synthesis Agent")

    try:
        from src.agents.research_synthesis import ResearchSynthesisAgent

        agent = ResearchSynthesisAgent()
        print_success("Research Synthesis Agent initialized")

        # Test synthesis
        query = "What are best practices for reducing customer churn in SaaS?"
        print_info(f"Test query: '{query}'")

        print_info("Running research synthesis (this may take 20-30 seconds)...")

        result = agent.synthesize(query, top_k_papers=2)

        if result['paper_count'] > 0:
            print_success(f"Retrieved and synthesized {result['paper_count']} papers")

            print("\n  Research Synthesis Preview:")
            synthesis_preview = result['synthesis'][:300]
            print(f"    {synthesis_preview}...")

            print("\n  Research Context Preview:")
            context_preview = result['research_context'][:200]
            print(f"    {context_preview}...")
        else:
            print_info("No papers found for synthesis")

        return True

    except Exception as e:
        print_error(f"Research synthesis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_orchestrator_with_rag():
    """Test the full orchestrator with RAG enabled."""
    print_section("Test 4: LangGraph Orchestrator with RAG")

    try:
        from src.langgraph_orchestrator import LangGraphOrchestrator

        # Initialize with RAG enabled
        orchestrator = LangGraphOrchestrator(enable_rag=True)
        print_success("LangGraph Orchestrator initialized with RAG enabled")

        # Test query
        query = "How can I improve customer retention for my B2B SaaS product?"
        print_info(f"Test query: '{query}'")

        print_info("Running full orchestration (this may take 30-60 seconds)...")
        print_info("This will:")
        print_info("  1. Route to appropriate agents")
        print_info("  2. Retrieve academic research")
        print_info("  3. Synthesize research findings")
        print_info("  4. Run agents with research context")
        print_info("  5. Create final synthesis with citations")

        result = orchestrator.orchestrate(query, use_memory=False)

        print_success("Orchestration complete!")

        print("\n  Agents Consulted:")
        for agent in result['agents_consulted']:
            print(f"    • {agent}")

        print("\n  Final Recommendation Preview:")
        recommendation_preview = result['recommendation'][:400]
        print(f"    {recommendation_preview}...")

        # Check for citations
        if "et al." in result['recommendation'] or "(" in result['recommendation'] and ")" in result['recommendation']:
            print_success("Citations detected in output!")
        else:
            print_info("No obvious citations detected (may vary based on query)")

        return True

    except Exception as e:
        print_error(f"Orchestrator RAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison():
    """Test comparison between RAG and non-RAG modes."""
    print_section("Test 5: RAG vs Non-RAG Comparison")

    try:
        from src.langgraph_orchestrator import LangGraphOrchestrator

        query = "What pricing model should I use for a new SaaS product?"
        print_info(f"Test query: '{query}'")

        # Test without RAG
        print_info("\n  Running without RAG...")
        orchestrator_no_rag = LangGraphOrchestrator(enable_rag=False)
        result_no_rag = orchestrator_no_rag.orchestrate(query, use_memory=False)

        # Test with RAG
        print_info("\n  Running with RAG...")
        orchestrator_rag = LangGraphOrchestrator(enable_rag=True)
        result_rag = orchestrator_rag.orchestrate(query, use_memory=False)

        print_success("Both modes completed successfully!")

        print("\n  Output Length Comparison:")
        print(f"    Without RAG: {len(result_no_rag['recommendation'])} characters")
        print(f"    With RAG:    {len(result_rag['recommendation'])} characters")

        # Check for citations
        has_citations = ("et al." in result_rag['recommendation'] or
                        "References" in result_rag['recommendation'])

        if has_citations:
            print_success("RAG output includes citations!")
        else:
            print_info("RAG output may not include visible citations")

        return True

    except Exception as e:
        print_error(f"Comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print_header("Business Intelligence Orchestrator - Phase 2 RAG Testing")

    print_info("This test suite validates the complete RAG integration:")
    print_info("  • Vector store infrastructure")
    print_info("  • Research retrieval (Semantic Scholar + arXiv)")
    print_info("  • Research synthesis agent")
    print_info("  • RAG-enabled LangGraph orchestrator")
    print_info("  • Citation-aware agents")

    input("\nPress Enter to start tests (will make API calls to OpenAI and research APIs)...")

    tests = [
        ("Module Imports", test_imports),
        ("Research Retrieval", test_research_retrieval),
        ("Research Synthesis Agent", test_research_synthesis_agent),
        ("Orchestrator with RAG", test_orchestrator_with_rag),
        ("RAG vs Non-RAG Comparison", test_comparison),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except KeyboardInterrupt:
            print("\n\nTests interrupted by user.")
            sys.exit(1)
        except Exception as e:
            print_error(f"Unexpected error in {test_name}: {e}")
            results.append((test_name, False))

    # Summary
    print_header("Test Summary")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        if success:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")

    print(f"\n{Fore.CYAN}Results: {passed}/{total} tests passed")

    if passed == total:
        print(Fore.GREEN + Style.BRIGHT + "\n🎉 All tests passed! Phase 2 RAG integration is working!")
    else:
        print(Fore.YELLOW + f"\n⚠️  {total - passed} test(s) failed. Review errors above.")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
