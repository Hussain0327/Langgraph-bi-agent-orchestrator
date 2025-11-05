"""
Research Synthesis Agent

Retrieves and synthesizes academic research relevant to business queries.
Acts as a preprocessing step to provide research-backed context to specialist agents.
"""

from src.gpt5_wrapper import GPT5Wrapper
from src.tools.research_retrieval import ResearchRetriever
from typing import Dict, Any, List


class ResearchSynthesisAgent:
    """
    Research Synthesis Agent for retrieving and summarizing academic research.

    This agent:
    1. Receives a business query
    2. Retrieves relevant academic papers
    3. Synthesizes key findings into actionable insights
    4. Provides research context to downstream specialist agents
    """

    def __init__(self):
        """Initialize the Research Synthesis Agent."""
        self.gpt5 = GPT5Wrapper()
        self.retriever = ResearchRetriever()
        self.name = "research_synthesis"
        self.description = "Academic research retrieval and synthesis for evidence-backed recommendations"

        self.system_prompt = """You are an expert research analyst specializing in business intelligence.

Your role is to:
1. Analyze business queries to identify relevant research topics
2. Review academic papers and extract key insights
3. Synthesize findings into actionable business recommendations
4. Identify evidence-based best practices and frameworks

When analyzing research papers:
- Focus on practical applications and real-world implications
- Highlight validated frameworks and methodologies
- Note empirical findings and statistical evidence
- Connect academic insights to business contexts
- Identify knowledge gaps or conflicting findings

Your output should be:
- Concise and business-focused (not overly academic)
- Organized by key themes or topics
- Supported by specific paper citations
- Actionable for business decision-making"""

    def synthesize(
        self,
        query: str,
        retrieve_papers: bool = True,
        top_k_papers: int = 3
    ) -> Dict[str, Any]:
        """
        Retrieve and synthesize research relevant to the query.

        Args:
            query: Business intelligence query
            retrieve_papers: Whether to retrieve new papers (vs using cached)
            top_k_papers: Number of papers to retrieve

        Returns:
            Dict with:
                - papers: List of retrieved papers
                - synthesis: Synthesized research insights
                - research_context: Formatted context for downstream agents
        """
        # Step 1: Retrieve relevant papers
        papers = []
        if retrieve_papers:
            print(f"\n🔍 Retrieving research for: {query[:60]}...")
            papers = self.retriever.retrieve_papers(
                query=query,
                top_k=top_k_papers
            )

            if not papers:
                print("⚠️  No research papers found. Continuing without RAG.")
                return {
                    "papers": [],
                    "synthesis": "No relevant academic research was found for this query.",
                    "research_context": ""
                }

            print(f"✓ Retrieved {len(papers)} relevant papers")

        # Step 2: Format research context
        research_context = self._format_papers_for_llm(papers)

        # Step 3: Synthesize insights using GPT-5
        print("📝 Synthesizing research insights...")

        synthesis_prompt = f"""Business Query: {query}

You have access to the following academic research papers:

{research_context}

Your task:
1. Identify the key findings most relevant to the business query
2. Synthesize insights across papers (note where findings align or conflict)
3. Extract evidence-based recommendations and frameworks
4. Highlight empirical findings with statistical support
5. Note any limitations or gaps in the current research

Provide a concise synthesis (300-500 words) organized by key themes.
Use this format:

**Key Research Themes:**

1. [Theme 1 - e.g., "Pricing Model Selection"]
   - Finding: [Insight from research]
   - Evidence: [Author et al., Year]
   - Implication: [Business application]

2. [Theme 2]
   ...

**Evidence-Based Recommendations:**
- [Specific recommendation based on research]
- [Another recommendation]

**Knowledge Gaps:**
- [Areas where research is limited or conflicting]
"""

        synthesis = self.gpt5.generate(
            input_text=synthesis_prompt,
            instructions=self.system_prompt,
            reasoning_effort="high",      # Deep analysis of research
            text_verbosity="high",        # Comprehensive synthesis
            max_output_tokens=1500
        )

        # Step 4: Create lightweight context for downstream agents
        agent_context = self._create_agent_context(papers, synthesis)

        return {
            "papers": papers,
            "synthesis": synthesis,
            "research_context": agent_context,
            "paper_count": len(papers)
        }

    def _format_papers_for_llm(self, papers: List[Dict[str, Any]]) -> str:
        """
        Format retrieved papers for LLM consumption.

        Args:
            papers: List of paper metadata dicts

        Returns:
            Formatted string with paper details
        """
        if not papers:
            return "No papers retrieved."

        formatted = ""

        for i, paper in enumerate(papers, 1):
            formatted += f"--- Paper {i} ---\n"
            formatted += f"Title: {paper['title']}\n"
            formatted += f"Authors: {', '.join(paper['authors'][:3])}"
            if len(paper['authors']) > 3:
                formatted += " et al."
            formatted += f"\nYear: {paper['year']}\n"
            formatted += f"Source: {paper['source']}\n"

            if paper.get('citation_count', 0) > 0:
                formatted += f"Citations: {paper['citation_count']}\n"

            # Include full abstract for synthesis
            formatted += f"\nAbstract:\n{paper['abstract']}\n"
            formatted += f"\nCitation: {paper['citation']}\n"
            formatted += f"\n{'='*70}\n\n"

        return formatted

    def _create_agent_context(
        self,
        papers: List[Dict[str, Any]],
        synthesis: str
    ) -> str:
        """
        Create concise research context for downstream agents.

        Args:
            papers: Retrieved papers
            synthesis: Synthesized insights

        Returns:
            Formatted context string for agent prompts
        """
        if not papers:
            return ""

        context = "\n## Research-Backed Insights\n\n"
        context += synthesis + "\n\n"

        context += "## Academic Sources\n"
        for i, paper in enumerate(papers, 1):
            context += f"{i}. {paper['citation']}\n"
            context += f"   URL: {paper['url']}\n\n"

        return context

    def quick_research_summary(self, query: str, top_k: int = 2) -> str:
        """
        Get a quick research summary without full synthesis (faster).

        Args:
            query: Search query
            top_k: Number of papers to retrieve

        Returns:
            Brief summary with paper titles and citations
        """
        papers = self.retriever.retrieve_papers(query=query, top_k=top_k)

        if not papers:
            return "No relevant research found."

        summary = f"Found {len(papers)} relevant papers:\n\n"

        for i, paper in enumerate(papers, 1):
            summary += f"{i}. {paper['title']}\n"
            summary += f"   {paper['citation']}\n\n"

        return summary


# Testing function
def test_research_synthesis_agent():
    """
    Test the Research Synthesis Agent.
    """
    print("\n" + "="*70)
    print("Testing Research Synthesis Agent")
    print("="*70)

    agent = ResearchSynthesisAgent()

    # Test query
    query = "What are best practices for SaaS pricing strategies?"

    print(f"\n Query: {query}")
    print("-" * 70)

    # Run synthesis
    result = agent.synthesize(query=query, top_k_papers=2)

    print(f"\n✓ Retrieved {result['paper_count']} papers")
    print("\n📄 Papers:")
    for i, paper in enumerate(result['papers'], 1):
        print(f"   {i}. {paper['title']} ({paper['year']})")

    print("\n📝 Research Synthesis:")
    print("-" * 70)
    print(result['synthesis'][:500] + "...\n")

    print("\n📋 Agent Context (preview):")
    print("-" * 70)
    print(result['research_context'][:400] + "...\n")

    print("="*70)
    print("✓ Research Synthesis Agent test complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_research_synthesis_agent()
