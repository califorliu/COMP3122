"""
Progressive Retrieval Engine
Multi-stage search with progressive disclosure (L1→L2→L3).
"""
from typing import List, Dict, Optional
from config import SearchConfig


class ProgressiveRetrieval:
    """Multi-stage retrieval engine with progressive expansion."""
    
    def __init__(self, hybrid_search, reranker):
        """
        Initialize progressive retrieval.
        
        Args:
            hybrid_search: HybridSearch instance
            reranker: RerankerClient instance
        """
        self.hybrid_search = hybrid_search
        self.reranker = reranker
        self.retrieval_trace = []
    
    def retrieve_progressive(
        self,
        semantic_search,
        keyword_search,
        query: str,
        course_id: str,
        search_config: Dict,
        is_followup: bool = False
    ) -> Dict:
        """
        Perform progressive retrieval with automatic expansion.
        
        Args:
            semantic_search: SemanticSearch instance
            keyword_search: KeywordSearch instance
            query: User question
            course_id: Course identifier
            search_config: Intent analysis from SearchRouter
            is_followup: Whether this is a follow-up question
        
        Returns:
            Dict with keys:
                - chunks: List of retrieved chunks (deduplicated)
                - retrieval_path: List of retrieval stages
                - max_relevance: Highest relevance score
                - total_chunks_retrieved: Total count
        """
        self.retrieval_trace = []
        all_chunks = []
        chunk_ids_seen = set()
        
        # Extract config
        top_k_config = search_config['top_k_per_level']
        max_depth = search_config['search_depth_limit']
        
        # Stage 1: Description-level search
        desc_chunks = self._retrieve_level(
            semantic_search,
            keyword_search,
            query,
            course_id,
            level='description',
            top_k=top_k_config.get('description', SearchConfig.LEVEL1_TOP_K)
        )
        
        # Rerank description chunks
        desc_chunks_reranked = self._rerank_chunks(query, desc_chunks)
        
        # Track stage 1
        self.retrieval_trace.append({
            'stage': 1,
            'level': 'description',
            'query': query,
            'chunks_retrieved': len(desc_chunks_reranked),
            'max_relevance': desc_chunks_reranked[0]['relevance_score'] if desc_chunks_reranked else 0.0,
            'expansion_trigger': None
        })
        
        # Add to results
        for chunk in desc_chunks_reranked:
            if chunk['chunk_id'] not in chunk_ids_seen:
                all_chunks.append(chunk)
                chunk_ids_seen.add(chunk['chunk_id'])
        
        # Check if should expand to Stage 2
        max_relevance_l1 = desc_chunks_reranked[0]['relevance_score'] if desc_chunks_reranked else 0.0
        should_expand = self._should_expand(1, max_depth, max_relevance_l1, is_followup)
        
        if should_expand:
            # Stage 2: Header-level search (expand from descriptions or direct search)
            header_chunks = self._retrieve_level(
                semantic_search,
                keyword_search,
                query,
                course_id,
                level='header',
                top_k=top_k_config.get('header', SearchConfig.LEVEL2_TOP_K)
            )
            
            # Rerank header chunks
            header_chunks_reranked = self._rerank_chunks(query, header_chunks)
            
            # Track stage 2
            self.retrieval_trace.append({
                'stage': 2,
                'level': 'header',
                'query': query,
                'chunks_retrieved': len(header_chunks_reranked),
                'max_relevance': header_chunks_reranked[0]['relevance_score'] if header_chunks_reranked else 0.0,
                'expansion_trigger': 'low_relevance' if max_relevance_l1 < SearchConfig.RELEVANCE_THRESHOLD else 'followup'
            })
            
            # Add header chunks
            for chunk in header_chunks_reranked:
                if chunk['chunk_id'] not in chunk_ids_seen:
                    all_chunks.append(chunk)
                    chunk_ids_seen.add(chunk['chunk_id'])
            
            # Check if should expand to Stage 3
            max_relevance_l2 = header_chunks_reranked[0]['relevance_score'] if header_chunks_reranked else 0.0
            should_expand_l3 = self._should_expand(2, max_depth, max_relevance_l2, is_followup)
            
            if should_expand_l3:
                # Stage 3: Detail-level search
                detail_chunks = self._retrieve_level(
                    semantic_search,
                    keyword_search,
                    query,
                    course_id,
                    level='detail',
                    top_k=top_k_config.get('detail', SearchConfig.LEVEL3_TOP_K)
                )
                
                # Rerank detail chunks
                detail_chunks_reranked = self._rerank_chunks(query, detail_chunks)
                
                # Track stage 3
                self.retrieval_trace.append({
                    'stage': 3,
                    'level': 'detail',
                    'query': query,
                    'chunks_retrieved': len(detail_chunks_reranked),
                    'max_relevance': detail_chunks_reranked[0]['relevance_score'] if detail_chunks_reranked else 0.0,
                    'expansion_trigger': 'low_relevance' if max_relevance_l2 < SearchConfig.RELEVANCE_THRESHOLD else 'followup'
                })
                
                # Add detail chunks
                for chunk in detail_chunks_reranked:
                    if chunk['chunk_id'] not in chunk_ids_seen:
                        all_chunks.append(chunk)
                        chunk_ids_seen.add(chunk['chunk_id'])
        
        # Calculate final stats
        max_relevance = max([c['relevance_score'] for c in all_chunks], default=0.0)
        
        return {
            'chunks': all_chunks,
            'retrieval_path': self.retrieval_trace,
            'max_relevance': max_relevance,
            'total_chunks_retrieved': len(all_chunks)
        }
    
    def _retrieve_level(
        self,
        semantic_search,
        keyword_search,
        query: str,
        course_id: str,
        level: str,
        top_k: int
    ) -> List[Dict]:
        """Retrieve chunks at specific level using hybrid search."""
        chunks = self.hybrid_search.hybrid_search_with_chunks(
            semantic_search=semantic_search,
            keyword_search=keyword_search,
            query=query,
            top_k=top_k,
            course_id=course_id,
            chunk_level=level
        )
        return chunks
    
    def _rerank_chunks(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """
        Rerank chunks using reranker model.
        
        Args:
            query: Search query
            chunks: List of chunk dicts
        
        Returns:
            Reranked chunks with updated relevance_score
        """
        if not chunks:
            return []
        
        # Extract documents for reranking
        documents = [chunk['content'] for chunk in chunks]
        
        # Rerank
        reranked_results = self.reranker.rerank(query, documents)
        
        # Update chunks with reranked scores
        reranked_chunks = []
        for result in reranked_results:
            chunk = chunks[result['index']].copy()
            chunk['relevance_score'] = result['relevance_score']
            reranked_chunks.append(chunk)
        
        return reranked_chunks
    
    def _should_expand(
        self,
        current_level: int,
        max_depth: int,
        max_relevance: float,
        is_followup: bool
    ) -> bool:
        """
        Determine if retrieval should expand to next level.
        
        Args:
            current_level: Current level (1, 2, or 3)
            max_depth: Maximum allowed depth
            max_relevance: Highest relevance score at current level
            is_followup: Whether this is a follow-up question
        
        Returns:
            True if should expand
        """
        # Can't expand beyond max depth
        if current_level >= max_depth:
            return False
        
        # Always expand for follow-up questions (if depth allows)
        if is_followup:
            return True
        
        # Expand if relevance is below threshold
        if max_relevance < SearchConfig.RELEVANCE_THRESHOLD:
            return True
        
        return False
    
    def get_retrieval_trace(self) -> List[Dict]:
        """Get detailed retrieval path for debugging/logging."""
        return self.retrieval_trace
    
    def get_children(self, parent_chunk_id: str, database) -> List[Dict]:
        """
        Get child chunks for a parent (for manual expansion).
        
        Args:
            parent_chunk_id: Parent chunk ID
            database: LocalDB instance
        
        Returns:
            List of child chunks
        """
        # Extract course_id from parent_chunk_id (assumes format: chunk_xxx)
        # This is a simplified implementation
        # In practice, you'd query the database directly
        course_id = parent_chunk_id.split('_')[0]  # Simplified
        return database.get_child_chunks(course_id, parent_chunk_id)
