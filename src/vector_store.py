"""
Vector Store Wrapper for ChromaDB

Provides a clean interface for storing and retrieving business research documents
using ChromaDB with OpenAI embeddings.
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import os
from src.config import Config


class VectorStore:
    """
    Wrapper around ChromaDB for document storage and semantic search.

    Uses OpenAI's text-embedding-3-small for embeddings.
    Persists data locally for development, can be configured for production.
    """

    def __init__(
        self,
        collection_name: str = "business-research",
        persist_directory: str = "./chroma_db"
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Use OpenAI embeddings (same as GPT-5 ecosystem)
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=Config.OPENAI_API_KEY,
            model_name="text-embedding-3-small"
        )

        # Get or create collection
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """
        Get existing collection or create a new one.

        Returns:
            ChromaDB collection instance
        """
        try:
            # Try to get existing collection
            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            print(f"✓ Loaded existing collection: {self.collection_name}")
        except Exception:
            # Create new collection if it doesn't exist
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Business and academic research papers"}
            )
            print(f"✓ Created new collection: {self.collection_name}")

        return collection

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of document texts to add
            metadatas: Optional list of metadata dicts for each document
            ids: Optional list of unique IDs for each document
        """
        if ids is None:
            # Generate IDs based on collection count
            start_id = self.collection.count()
            ids = [f"doc_{start_id + i}" for i in range(len(documents))]

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        print(f"✓ Added {len(documents)} documents to {self.collection_name}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.

        Args:
            query: Search query text
            top_k: Number of results to return
            filter_metadata: Optional metadata filter (e.g., {"year": {"$gte": 2020}})

        Returns:
            List of dicts with keys: id, document, metadata, distance
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_metadata
        )

        # Format results
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None
                })

        return formatted_results

    def get_by_id(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve documents by their IDs.

        Args:
            ids: List of document IDs

        Returns:
            List of documents with metadata
        """
        results = self.collection.get(ids=ids)

        formatted_results = []
        for i in range(len(results['ids'])):
            formatted_results.append({
                'id': results['ids'][i],
                'document': results['documents'][i],
                'metadata': results['metadatas'][i] if results['metadatas'] else {}
            })

        return formatted_results

    def count(self) -> int:
        """
        Get the number of documents in the collection.

        Returns:
            Document count
        """
        return self.collection.count()

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents by their IDs.

        Args:
            ids: List of document IDs to delete
        """
        self.collection.delete(ids=ids)
        print(f"✓ Deleted {len(ids)} documents from {self.collection_name}")

    def reset(self) -> None:
        """
        Delete all documents in the collection.

        WARNING: This will permanently delete all data!
        """
        self.client.delete_collection(name=self.collection_name)
        self.collection = self._get_or_create_collection()
        print(f"⚠️  Reset collection: {self.collection_name}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dict with collection stats
        """
        return {
            "collection_name": self.collection_name,
            "document_count": self.count(),
            "persist_directory": self.persist_directory,
            "embedding_model": "text-embedding-3-small"
        }


# Convenience function for quick testing
def test_vector_store():
    """
    Test the vector store with sample documents.
    """
    print("\n" + "="*70)
    print("Testing Vector Store")
    print("="*70)

    # Initialize
    vs = VectorStore(collection_name="test-collection")

    # Add sample documents
    sample_docs = [
        "SaaS pricing strategies include value-based, tiered, and usage-based models.",
        "Customer retention improves with regular engagement and personalized onboarding.",
        "Market segmentation helps target the right customers with tailored messaging.",
        "Financial modeling for startups should include burn rate and runway calculations."
    ]

    sample_metadata = [
        {"topic": "pricing", "source": "test"},
        {"topic": "retention", "source": "test"},
        {"topic": "marketing", "source": "test"},
        {"topic": "finance", "source": "test"}
    ]

    print("\n1. Adding sample documents...")
    vs.add_documents(
        documents=sample_docs,
        metadatas=sample_metadata
    )

    # Test search
    print("\n2. Searching for 'pricing strategies'...")
    results = vs.search("pricing strategies", top_k=2)

    for i, result in enumerate(results, 1):
        print(f"\n   Result {i}:")
        print(f"   Document: {result['document'][:80]}...")
        print(f"   Metadata: {result['metadata']}")
        print(f"   Distance: {result['distance']:.4f}")

    # Get stats
    print("\n3. Collection stats:")
    stats = vs.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Clean up
    print("\n4. Cleaning up test collection...")
    vs.reset()

    print("\n" + "="*70)
    print("✓ Vector Store test complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Run tests if executed directly
    test_vector_store()
