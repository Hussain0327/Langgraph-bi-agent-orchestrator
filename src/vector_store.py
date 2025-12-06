import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import os
from src.config import Config

class VectorStore:

    def __init__(self, collection_name: str='business-research', persist_directory: str='./chroma_db'):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False, allow_reset=True))
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(api_key=Config.OPENAI_API_KEY, model_name='text-embedding-3-small')
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        try:
            collection = self.client.get_collection(name=self.collection_name, embedding_function=self.embedding_function)
            print(f'✓ Loaded existing collection: {self.collection_name}')
        except Exception:
            collection = self.client.create_collection(name=self.collection_name, embedding_function=self.embedding_function, metadata={'description': 'Business and academic research papers'})
            print(f'✓ Created new collection: {self.collection_name}')
        return collection

    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]]=None, ids: Optional[List[str]]=None) -> None:
        if ids is None:
            start_id = self.collection.count()
            ids = [f'doc_{start_id + i}' for i in range(len(documents))]
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        print(f'✓ Added {len(documents)} documents to {self.collection_name}')

    def search(self, query: str, top_k: int=5, filter_metadata: Optional[Dict[str, Any]]=None) -> List[Dict[str, Any]]:
        results = self.collection.query(query_texts=[query], n_results=top_k, where=filter_metadata)
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                formatted_results.append({'id': results['ids'][0][i], 'document': results['documents'][0][i], 'metadata': results['metadatas'][0][i] if results['metadatas'] else {}, 'distance': results['distances'][0][i] if results['distances'] else None})
        return formatted_results

    def get_by_id(self, ids: List[str]) -> List[Dict[str, Any]]:
        results = self.collection.get(ids=ids)
        formatted_results = []
        for i in range(len(results['ids'])):
            formatted_results.append({'id': results['ids'][i], 'document': results['documents'][i], 'metadata': results['metadatas'][i] if results['metadatas'] else {}})
        return formatted_results

    def count(self) -> int:
        return self.collection.count()

    def delete(self, ids: List[str]) -> None:
        self.collection.delete(ids=ids)
        print(f'✓ Deleted {len(ids)} documents from {self.collection_name}')

    def reset(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        self.collection = self._get_or_create_collection()
        print(f'  Reset collection: {self.collection_name}')

    def get_stats(self) -> Dict[str, Any]:
        return {'collection_name': self.collection_name, 'document_count': self.count(), 'persist_directory': self.persist_directory, 'embedding_model': 'text-embedding-3-small'}

def test_vector_store():
    print('\n' + '=' * 70)
    print('Testing Vector Store')
    print('=' * 70)
    vs = VectorStore(collection_name='test-collection')
    sample_docs = ['SaaS pricing strategies include value-based, tiered, and usage-based models.', 'Customer retention improves with regular engagement and personalized onboarding.', 'Market segmentation helps target the right customers with tailored messaging.', 'Financial modeling for startups should include burn rate and runway calculations.']
    sample_metadata = [{'topic': 'pricing', 'source': 'test'}, {'topic': 'retention', 'source': 'test'}, {'topic': 'marketing', 'source': 'test'}, {'topic': 'finance', 'source': 'test'}]
    print('\n1. Adding sample documents...')
    vs.add_documents(documents=sample_docs, metadatas=sample_metadata)
    print("\n2. Searching for 'pricing strategies'...")
    results = vs.search('pricing strategies', top_k=2)
    for i, result in enumerate(results, 1):
        print(f'\n   Result {i}:')
        print(f"   Document: {result['document'][:80]}...")
        print(f"   Metadata: {result['metadata']}")
        print(f"   Distance: {result['distance']:.4f}")
    print('\n3. Collection stats:')
    stats = vs.get_stats()
    for key, value in stats.items():
        print(f'   {key}: {value}')
    print('\n4. Cleaning up test collection...')
    vs.reset()
    print('\n' + '=' * 70)
    print('✓ Vector Store test complete!')
    print('=' * 70 + '\n')
if __name__ == '__main__':
    test_vector_store()