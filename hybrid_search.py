"""
Hybrid Search with RRF (Reciprocal Rank Fusion)
Combines vector similarity and BM25 keyword search results.
"""
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from config import SearchConfig


class HybridSearch:
    """Combines vector and BM25 search using Reciprocal Rank Fusion."""
    
    def __init__(
        self,
        vector_weight: float = None,
        bm25_weight: float = None,
        rrf_k: int = None
    ):
        """
        Initialize hybrid search.
        
        Args:
            vector_weight: Weight for vector search results (default from config)
            bm25_weight: Weight for BM25 results (default from config)
            rrf_k: RRF constant (default: 60)
        """
        self.vector_weight = vector_weight or SearchConfig.VECTOR_WEIGHT
        self.bm25_weight = bm25_weight or SearchConfig.BM25_WEIGHT
        self.rrf_k = rrf_k or SearchConfig.RRF_K
    
    def rrf_fusion(
        self,
        vector_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF formula: score(d) = Σ 1/(k + rank_i(d))
        
        Args:
            vector_results: List of (chunk_id, score) from vector search
            bm25_results: List of (chunk_id, score) from BM25 search
        
        Returns:
            Merged and ranked list of (chunk_id, rrf_score) tuples
        """
        rrf_scores = defaultdict(float)
        
        # Add vector search scores
        for rank, (chunk_id, _) in enumerate(vector_results, start=1):
            rrf_scores[chunk_id] += self.vector_weight * (1.0 / (self.rrf_k + rank))
        
        # Add BM25 scores
        for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
            rrf_scores[chunk_id] += self.bm25_weight * (1.0 / (self.rrf_k + rank))
        
        # Sort by RRF score descending
        merged_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return merged_results
    
    def hybrid_search(
        self,
        semantic_search,
        keyword_search,
        query: str,
        top_k: int = 10,
        course_id: Optional[str] = None,
        chunk_level: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Perform hybrid search combining semantic and keyword methods.
        
        Args:
            semantic_search: SemanticSearch instance
            keyword_search: KeywordSearch instance
            query: Search query
            top_k: Number of final results
            course_id: Optional course filter for semantic search
            chunk_level: Optional level filter for semantic search
        
        Returns:
            List of (chunk_id, rrf_score) tuples
        """
        # Get vector search results
        vector_results = semantic_search.search(
            query=query,
            top_k=top_k * 2,  # Get more candidates for fusion
            course_id=course_id,
            chunk_level=chunk_level
        )
        
        # Get BM25 keyword search results
        bm25_results = keyword_search.search(
            query=query,
            top_k=top_k * 2
        )
        
        # Fuse results using RRF
        merged_results = self.rrf_fusion(vector_results, bm25_results)
        
        # Return top-k
        return merged_results[:top_k]
    
    def hybrid_search_with_chunks(
        self,
        semantic_search,
        keyword_search,
        query: str,
        top_k: int = 10,
        course_id: Optional[str] = None,
        chunk_level: Optional[str] = None
    ) -> List[Dict]:
        """
        Perform hybrid search and return full chunk data.
        
        Args:
            semantic_search: SemanticSearch instance
            keyword_search: KeywordSearch instance
            query: Search query
            top_k: Number of results
            course_id: Optional course filter
            chunk_level: Optional level filter
        
        Returns:
            List of dicts with chunk_id, rrf_score, metadata, content
        """
        # Get merged chunk IDs
        merged_results = self.hybrid_search(
            semantic_search,
            keyword_search,
            query,
            top_k,
            course_id,
            chunk_level
        )
        
        # Retrieve full chunk data from vector store
        chunk_ids = [chunk_id for chunk_id, _ in merged_results]
        chunks_data = semantic_search.vector_store.get_by_ids(chunk_ids)
        
        # Create score lookup
        score_map = dict(merged_results)
        
        # Combine data
        result_chunks = []
        for i, chunk_id in enumerate(chunks_data['ids']):
            result_chunks.append({
                'chunk_id': chunk_id,
                'rrf_score': score_map.get(chunk_id, 0.0),
                'metadata': chunks_data['metadatas'][i],
                'content': chunks_data['documents'][i]
            })
        
        # Sort by RRF score to maintain order
        result_chunks.sort(key=lambda x: x['rrf_score'], reverse=True)
        
        return result_chunks
