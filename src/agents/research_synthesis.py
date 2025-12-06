from src.unified_llm import UnifiedLLM
from src.tools.research_retrieval import ResearchRetriever
from typing import Dict, Any, List

class ResearchSynthesisAgent:

    def __init__(self):
        self.llm = UnifiedLLM(agent_type='research_synthesis')
        self.retriever = ResearchRetriever()
        self.name = 'research_synthesis'
        self.description = 'Academic research retrieval and synthesis for evidence-backed recommendations'
        self.system_prompt = 'You are an expert research analyst specializing in business intelligence.\n\nYour role is to:\n1. Analyze business queries to identify relevant research topics\n2. Review academic papers and extract key insights\n3. Synthesize findings into actionable business recommendations\n4. Identify evidence-based best practices and frameworks\n\nWhen analyzing research papers:\n- Focus on practical applications and real-world implications\n- Highlight validated frameworks and methodologies\n- Note empirical findings and statistical evidence\n- Connect academic insights to business contexts\n- Identify knowledge gaps or conflicting findings\n\nYour output should be:\n- Concise and business-focused (not overly academic)\n- Organized by key themes or topics\n- Supported by specific paper citations\n- Actionable for business decision-making'

    def synthesize(self, query: str, retrieve_papers: bool=True, top_k_papers: int=3) -> Dict[str, Any]:
        papers = []
        if retrieve_papers:
            print(f'\n Retrieving research for: {query[:60]}...')
            papers = self.retriever.retrieve_papers(query=query, top_k=top_k_papers)
            if not papers:
                print('  No research papers found. Continuing without RAG.')
                return {'papers': [], 'synthesis': 'No relevant academic research was found for this query.', 'research_context': ''}
            print(f'✓ Retrieved {len(papers)} relevant papers')
        research_context = self._format_papers_for_llm(papers)
        print(' Synthesizing research insights...')
        synthesis_prompt = f'Business Query: {query}\n\nYou have access to the following academic research papers:\n\n{research_context}\n\nYour task:\n1. Identify the key findings most relevant to the business query\n2. Synthesize insights across papers (note where findings align or conflict)\n3. Extract evidence-based recommendations and frameworks\n4. Highlight empirical findings with statistical support\n5. Note any limitations or gaps in the current research\n\nProvide a concise synthesis (300-500 words) organized by key themes.\nUse this EXACT citation format: (Source: Author et al., Year)\n\nExample: "Customer churn is driven primarily by poor onboarding (Source: Smith et al., 2024)."\n\n**Key Research Themes:**\n\n1. [Theme 1]\n   - Finding with citation (Source: Author et al., Year)\n   - Implication for business\n\n2. [Theme 2]\n   - Finding with citation (Source: Author et al., Year)\n\n**Evidence-Based Recommendations:**\n- [Recommendation] (Source: Author et al., Year)\n\n**Knowledge Gaps:**\n- [Gaps in research]\n'
        synthesis = self.llm.generate(input_text=synthesis_prompt, instructions=self.system_prompt, reasoning_effort='low', text_verbosity='high', max_tokens=1500)
        agent_context = self._create_agent_context(papers, synthesis)
        return {'papers': papers, 'synthesis': synthesis, 'research_context': agent_context, 'paper_count': len(papers)}

    def _format_papers_for_llm(self, papers: List[Dict[str, Any]]) -> str:
        if not papers:
            return 'No papers retrieved.'
        formatted = ''
        for i, paper in enumerate(papers, 1):
            formatted += f'--- Paper {i} ---\n'
            formatted += f"Title: {paper['title']}\n"
            formatted += f"Authors: {', '.join(paper['authors'][:3])}"
            if len(paper['authors']) > 3:
                formatted += ' et al.'
            formatted += f"\nYear: {paper['year']}\n"
            formatted += f"Source: {paper['source']}\n"
            if paper.get('citation_count', 0) > 0:
                formatted += f"Citations: {paper['citation_count']}\n"
            formatted += f"\nAbstract:\n{paper['abstract']}\n"
            formatted += f"\nCitation: {paper['citation']}\n"
            formatted += f"\n{'=' * 70}\n\n"
        return formatted

    def _create_agent_context(self, papers: List[Dict[str, Any]], synthesis: str) -> str:
        if not papers:
            return ''
        context = '\n## Research-Backed Insights\n\n'
        context += synthesis + '\n\n'
        context += '## Academic Sources\n'
        for i, paper in enumerate(papers, 1):
            context += f"{i}. {paper['citation']}\n"
            context += f"   URL: {paper['url']}\n\n"
        return context

    def quick_research_summary(self, query: str, top_k: int=2) -> str:
        papers = self.retriever.retrieve_papers(query=query, top_k=top_k)
        if not papers:
            return 'No relevant research found.'
        summary = f'Found {len(papers)} relevant papers:\n\n'
        for i, paper in enumerate(papers, 1):
            summary += f"{i}. {paper['title']}\n"
            summary += f"   {paper['citation']}\n\n"
        return summary

def test_research_synthesis_agent():
    print('\n' + '=' * 70)
    print('Testing Research Synthesis Agent')
    print('=' * 70)
    agent = ResearchSynthesisAgent()
    query = 'What are best practices for SaaS pricing strategies?'
    print(f'\n Query: {query}')
    print('-' * 70)
    result = agent.synthesize(query=query, top_k_papers=2)
    print(f"\n✓ Retrieved {result['paper_count']} papers")
    print('\n📄 Papers:')
    for i, paper in enumerate(result['papers'], 1):
        print(f"   {i}. {paper['title']} ({paper['year']})")
    print('\n Research Synthesis:')
    print('-' * 70)
    print(result['synthesis'][:500] + '...\n')
    print('\n📋 Agent Context (preview):')
    print('-' * 70)
    print(result['research_context'][:400] + '...\n')
    print('=' * 70)
    print('✓ Research Synthesis Agent test complete!')
    print('=' * 70 + '\n')
if __name__ == '__main__':
    test_research_synthesis_agent()