"""
Semantic Vector Search
Uses ChromaDB and embeddings for semantic similarity search.
"""
from typing import List, Dict, Tuple, Optional
from vector_store import VectorStore
from llm_client import EmbeddingClient


class SemanticSearch:
    """Semantic search using vector embeddings."""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.embedding_client = EmbeddingClient()
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        course_id: Optional[str] = None,
        chunk_level: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Search for semantically similar chunks.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            course_id: Optional course filter
            chunk_level: Optional level filter (description/header/detail)
        
        Returns:
            List of (chunk_id, cosine_similarity) tuples
        """
        # Generate query embedding
        query_embedding = self.embedding_client.embed_text(query)
        
        # Build metadata filter (ChromaDB requires $and operator for multiple conditions)
        where_filter = None
        if course_id and chunk_level:
            where_filter = {
                "$and": [
                    {"course_id": course_id},
                    {"chunk_level": chunk_level}
                ]
            }
        elif course_id:
            where_filter = {"course_id": course_id}
        elif chunk_level:
            where_filter = {"chunk_level": chunk_level}
        
        # Query ChromaDB
        results = self.vector_store.query_by_vector(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where_filter
        )
        
        # Convert distances to similarity scores (ChromaDB returns cosine distance)
        # Cosine similarity = 1 - cosine distance
        chunk_scores = [
            (chunk_id, 1.0 - distance)
            for chunk_id, distance in zip(results['ids'], results['distances'])
        ]
        
        return chunk_scores
    
    def get_chunks_with_metadata(
        self,
        query: str,
        top_k: int = 10,
        course_id: Optional[str] = None,
        chunk_level: Optional[str] = None
    ) -> List[Dict]:
        """
        Search and return full chunk data including metadata and content.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            course_id: Optional course filter
            chunk_level: Optional level filter
        
        Returns:
            List of dicts with keys: chunk_id, similarity, metadata, content
        """
        # Generate query embedding
        query_embedding = self.embedding_client.embed_text(query)
        
        # Build metadata filter (ChromaDB requires $and operator for multiple conditions)
        where_filter = None
        if course_id and chunk_level:
            where_filter = {
                "$and": [
                    {"course_id": course_id},
                    {"chunk_level": chunk_level}
                ]
            }
        elif course_id:
            where_filter = {"course_id": course_id}
        elif chunk_level:
            where_filter = {"chunk_level": chunk_level}
        
        # Query ChromaDB
        results = self.vector_store.query_by_vector(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where_filter
        )
        
        # Combine all data
        chunks = []
        for i in range(len(results['ids'])):
            chunks.append({
                'chunk_id': results['ids'][i],
                'similarity': 1.0 - results['distances'][i],
                'metadata': results['metadatas'][i],
                'content': results['documents'][i]
            })
        
        return chunks
