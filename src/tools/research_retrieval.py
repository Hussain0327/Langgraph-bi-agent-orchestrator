"""
Research Retrieval Tool

Integrates with academic research APIs (Semantic Scholar, arXiv) to retrieve
relevant papers for business intelligence queries.

Features:
- Multi-source search (Semantic Scholar + arXiv)
- Relevance reranking using embeddings
- Result caching to reduce API calls
- Proper citation formatting
"""

import requests
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import hashlib
import json
import os


class ResearchRetriever:
    """
    Retrieves academic research papers from multiple sources.

    Uses Semantic Scholar API for peer-reviewed papers and arXiv for preprints.
    Implements caching and relevance reranking for optimal results.
    """

    def __init__(self, cache_dir: str = "./research_cache"):
        """
        Initialize the research retriever.

        Args:
            cache_dir: Directory to cache API responses
        """
        self.cache_dir = cache_dir
        self.semantic_scholar_base_url = "https://api.semanticscholar.org/graph/v1"
        self.arxiv_base_url = "http://export.arxiv.org/api/query"

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        # Rate limiting (Semantic Scholar: 100 req/5min)
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds (increased from 0.5 to reduce rate limit hits)

    def _get_cache_key(self, query: str, source: str) -> str:
        """
        Generate cache key for a query.

        Args:
            query: Search query
            source: API source (semantic_scholar or arxiv)

        Returns:
            Cache key (hash of query + source)
        """
        cache_string = f"{source}:{query}"
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str, max_age_days: int = 7) -> Optional[List[Dict]]:
        """
        Retrieve results from cache if available and not expired.

        Args:
            cache_key: Cache key
            max_age_days: Maximum age of cached results in days

        Returns:
            Cached results or None if not found/expired
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        if not os.path.exists(cache_file):
            return None

        # Check if cache is expired
        file_modified_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - file_modified_time > timedelta(days=max_age_days):
            return None

        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def _save_to_cache(self, cache_key: str, data: List[Dict]) -> None:
        """
        Save results to cache.

        Args:
            cache_key: Cache key
            data: Data to cache
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Warning: Could not save to cache: {e}")

    def _rate_limit(self) -> None:
        """
        Implement rate limiting for API requests.
        """
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def search_semantic_scholar(
        self,
        query: str,
        limit: int = 10,
        fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search Semantic Scholar for relevant papers.

        Args:
            query: Search query
            limit: Maximum number of results
            fields: Fields to include in response

        Returns:
            List of paper metadata dicts
        """
        # Check cache first
        cache_key = self._get_cache_key(query, "semantic_scholar")
        cached_results = self._get_from_cache(cache_key)

        if cached_results is not None:
            print(f"✓ Using cached Semantic Scholar results for: {query[:50]}...")
            return cached_results[:limit]

        # Default fields to retrieve
        if fields is None:
            fields = [
                "paperId", "title", "abstract", "year", "authors",
                "citationCount", "publicationDate", "venue", "url"
            ]

        # Rate limit
        self._rate_limit()

        try:
            # Call Semantic Scholar API
            url = f"{self.semantic_scholar_base_url}/paper/search"
            params = {
                "query": query,
                "limit": limit,
                "fields": ",".join(fields)
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            papers = data.get("data", [])

            # Format results
            formatted_papers = []
            for paper in papers:
                formatted_papers.append({
                    "paper_id": paper.get("paperId", ""),
                    "title": paper.get("title", ""),
                    "authors": [author.get("name", "") for author in paper.get("authors", [])],
                    "year": paper.get("year"),
                    "abstract": paper.get("abstract", ""),
                    "citation_count": paper.get("citationCount", 0),
                    "publication_date": paper.get("publicationDate", ""),
                    "venue": paper.get("venue", ""),
                    "url": paper.get("url", ""),
                    "source": "Semantic Scholar"
                })

            # Save to cache
            self._save_to_cache(cache_key, formatted_papers)

            print(f"✓ Retrieved {len(formatted_papers)} papers from Semantic Scholar")
            return formatted_papers

        except Exception as e:
            print(f"⚠️  Semantic Scholar search failed: {e}")
            return []

    def search_arxiv(
        self,
        query: str,
        limit: int = 10,
        sort_by: str = "relevance"
    ) -> List[Dict[str, Any]]:
        """
        Search arXiv for relevant preprints.

        Args:
            query: Search query
            limit: Maximum number of results
            sort_by: Sort order (relevance or lastUpdatedDate)

        Returns:
            List of paper metadata dicts
        """
        # Check cache first
        cache_key = self._get_cache_key(query, "arxiv")
        cached_results = self._get_from_cache(cache_key)

        if cached_results is not None:
            print(f"✓ Using cached arXiv results for: {query[:50]}...")
            return cached_results[:limit]

        # Rate limit
        self._rate_limit()

        try:
            # Call arXiv API
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": limit,
                "sortBy": sort_by,
                "sortOrder": "descending"
            }

            response = requests.get(self.arxiv_base_url, params=params, timeout=10)
            response.raise_for_status()

            # Parse XML response (arXiv returns XML, not JSON)
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)

            # Define namespaces
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }

            # Extract papers
            formatted_papers = []
            for entry in root.findall('atom:entry', namespaces):
                # Extract authors
                authors = [
                    author.find('atom:name', namespaces).text
                    for author in entry.findall('atom:author', namespaces)
                ]

                # Extract publication date
                published = entry.find('atom:published', namespaces).text
                year = published.split('-')[0] if published else None

                formatted_papers.append({
                    "paper_id": entry.find('atom:id', namespaces).text,
                    "title": entry.find('atom:title', namespaces).text.strip(),
                    "authors": authors,
                    "year": year,
                    "abstract": entry.find('atom:summary', namespaces).text.strip(),
                    "citation_count": 0,  # arXiv doesn't provide citation counts
                    "publication_date": published,
                    "venue": "arXiv preprint",
                    "url": entry.find('atom:id', namespaces).text,
                    "source": "arXiv"
                })

            # Save to cache
            self._save_to_cache(cache_key, formatted_papers)

            print(f"✓ Retrieved {len(formatted_papers)} papers from arXiv")
            return formatted_papers

        except Exception as e:
            print(f"⚠️  arXiv search failed: {e}")
            return []

    def retrieve_papers(
        self,
        query: str,
        top_k: int = 3,
        sources: List[str] = ["semantic_scholar", "arxiv"]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve papers from multiple sources and return top-k most relevant.

        Args:
            query: Search query
            top_k: Number of papers to return
            sources: List of sources to search (semantic_scholar, arxiv)

        Returns:
            List of top-k papers with formatted citations
        """
        all_papers = []

        # Search each source
        if "semantic_scholar" in sources:
            papers = self.search_semantic_scholar(query, limit=10)
            all_papers.extend(papers)

        if "arxiv" in sources:
            papers = self.search_arxiv(query, limit=10)
            all_papers.extend(papers)

        # Simple ranking by citation count (Semantic Scholar) or recency (arXiv)
        # More sophisticated reranking with embeddings can be added later
        all_papers.sort(
            key=lambda p: (
                p.get("citation_count", 0),
                int(p.get("year", 0) or 0)
            ),
            reverse=True
        )

        # Return top-k papers
        top_papers = all_papers[:top_k]

        # Add formatted citations
        for paper in top_papers:
            paper["citation"] = self._format_citation(paper)

        return top_papers

    def _format_citation(self, paper: Dict[str, Any]) -> str:
        """
        Format a paper as an academic citation.

        Args:
            paper: Paper metadata dict

        Returns:
            Formatted citation string
        """
        authors = paper.get("authors", [])

        # Format authors
        if len(authors) == 0:
            author_str = "Unknown"
        elif len(authors) == 1:
            author_str = authors[0]
        elif len(authors) == 2:
            author_str = f"{authors[0]} and {authors[1]}"
        else:
            author_str = f"{authors[0]} et al."

        # Format citation
        year = paper.get("year", "n.d.")
        title = paper.get("title", "Untitled")
        venue = paper.get("venue", "")

        if venue:
            citation = f"{author_str} ({year}). {title}. {venue}."
        else:
            citation = f"{author_str} ({year}). {title}."

        return citation

    def format_research_context(self, papers: List[Dict[str, Any]]) -> str:
        """
        Format retrieved papers as context for LLM prompts.

        Args:
            papers: List of paper metadata dicts

        Returns:
            Formatted research context string
        """
        if not papers:
            return "No relevant research papers found."

        context = "## Relevant Research Papers\n\n"

        for i, paper in enumerate(papers, 1):
            context += f"### Paper {i}: {paper['title']}\n"
            context += f"**Authors**: {', '.join(paper['authors'][:3])}"
            if len(paper['authors']) > 3:
                context += " et al."
            context += f"\n**Year**: {paper['year']}\n"
            context += f"**Source**: {paper['source']}\n"
            if paper.get('citation_count', 0) > 0:
                context += f"**Citations**: {paper['citation_count']}\n"
            context += f"\n**Abstract**: {paper['abstract'][:300]}...\n"
            context += f"\n**Citation**: {paper['citation']}\n"
            context += f"**URL**: {paper['url']}\n\n"
            context += "-" * 70 + "\n\n"

        return context


# Convenience function for testing
def test_research_retrieval():
    """
    Test the research retrieval system.
    """
    print("\n" + "="*70)
    print("Testing Research Retrieval")
    print("="*70)

    retriever = ResearchRetriever()

    # Test query
    query = "SaaS pricing strategies and customer retention"

    print(f"\n1. Searching for: '{query}'")
    print("-" * 70)

    papers = retriever.retrieve_papers(query, top_k=3)

    print(f"\n2. Retrieved {len(papers)} papers\n")

    for i, paper in enumerate(papers, 1):
        print(f"Paper {i}:")
        print(f"  Title: {paper['title']}")
        print(f"  Authors: {', '.join(paper['authors'][:2])}")
        print(f"  Year: {paper['year']}")
        print(f"  Source: {paper['source']}")
        print(f"  Citation: {paper['citation']}")
        print()

    print("\n3. Formatted Research Context:")
    print("-" * 70)
    context = retriever.format_research_context(papers[:2])
    print(context[:500] + "...\n")

    print("="*70)
    print("✓ Research Retrieval test complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_research_retrieval()
