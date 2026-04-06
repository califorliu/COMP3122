"""
BM25 Keyword Search
Implements BM25Okapi algorithm for keyword-based document retrieval.
"""
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
import re


class KeywordSearch:
    """BM25-based keyword search engine."""
    
    def __init__(self):
        self.bm25 = None
        self.chunk_ids = []
        self.documents = []
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization: lowercase, remove punctuation, split on whitespace.
        
        Args:
            text: Input text
        
        Returns:
            List of tokens
        """
        # Lowercase and extract words (alphanumeric + underscores)
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def build_index(self, chunks: List[Dict]):
        """
        Build BM25 index from chunks.
        
        Args:
            chunks: List of chunk dicts with 'chunk_id' and 'content' keys
        """
        if not chunks:
            print("Warning: No chunks provided to build_index()")
            self.chunk_ids = []
            self.documents = []
            self.bm25 = None
            return
        
        self.chunk_ids = [chunk['chunk_id'] for chunk in chunks]
        self.documents = [chunk['content'] for chunk in chunks]
        
        # Tokenize all documents
        tokenized_docs = [self._tokenize(doc) for doc in self.documents]
        
        # Check if all documents are empty (no tokens)
        total_tokens = sum(len(doc) for doc in tokenized_docs)
        if total_tokens == 0:
            print("Warning: All documents are empty (no tokens found). Cannot build BM25 index.")
            self.bm25 = None
            return
        
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for relevant chunks using BM25.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
        
        Returns:
            List of (chunk_id, bm25_score) tuples, sorted by score descending
        """
        if not self.bm25:
            print("Warning: BM25 index not built. Call build_index() first.")
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)
        
        # Create (chunk_id, score) pairs
        results = list(zip(self.chunk_ids, scores))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return results[:top_k]
    
    def update_index(self, new_chunks: List[Dict]):
        """
        Add new chunks to existing index.
        Note: This rebuilds the entire index (BM25Okapi doesn't support incremental updates).
        
        Args:
            new_chunks: List of new chunk dicts to add
        """
        # Combine existing and new chunks
        all_chunks = [
            {'chunk_id': cid, 'content': doc}
            for cid, doc in zip(self.chunk_ids, self.documents)
        ]
        all_chunks.extend(new_chunks)
        
        # Rebuild index
        self.build_index(all_chunks)
    
    def get_stats(self) -> Dict:
        """
        Get index statistics.
        
        Returns:
            Dict with index stats
        """
        return {
            'total_documents': len(self.chunk_ids),
            'avg_doc_length': sum(len(self._tokenize(doc)) for doc in self.documents) / len(self.documents) if self.documents else 0,
            'index_built': self.bm25 is not None
        }
