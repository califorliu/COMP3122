"""
ChromaDB Vector Store
Manages vector embeddings for knowledge chunks with metadata filtering.
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Tuple
from config import ChromaDBConfig


class VectorStore:
    def __init__(self):
        """Initialize ChromaDB client with persistent storage."""
        self.client = chromadb.PersistentClient(
            path=ChromaDBConfig.PERSIST_DIR,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=ChromaDBConfig.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_embeddings(
        self,
        embeddings: List[List[float]],
        chunk_ids: List[str],
        documents: List[str],
        metadatas: List[Dict]
    ) -> bool:
        """
        Add embeddings to ChromaDB collection.
        
        Args:
            embeddings: List of embedding vectors
            chunk_ids: List of unique chunk IDs
            documents: List of text content
            metadatas: List of metadata dicts (course_id, chunk_level, etc.)
        
        Returns:
            True if successful
        """
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=chunk_ids
            )
            return True
        except Exception as e:
            print(f"Error adding embeddings to ChromaDB: {e}")
            return False
    
    def query_by_vector(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> Dict:
        """
        Query ChromaDB by vector similarity.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            where: Metadata filter (e.g., {"course_id": "cs101", "chunk_level": "header"})
            where_document: Document content filter
        
        Returns:
            Dict with keys: ids, distances, metadatas, documents
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                where_document=where_document
            )
            
            # Flatten results (ChromaDB returns lists of lists)
            return {
                'ids': results['ids'][0] if results['ids'] else [],
                'distances': results['distances'][0] if results['distances'] else [],
                'metadatas': results['metadatas'][0] if results['metadatas'] else [],
                'documents': results['documents'][0] if results['documents'] else []
            }
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return {'ids': [], 'distances': [], 'metadatas': [], 'documents': []}
    
    def query_by_text(
        self,
        query_text: str,
        top_k: int = 10,
        where: Optional[Dict] = None
    ) -> List[Tuple[str, float]]:
        """
        Query by text (ChromaDB will embed it automatically if embedding function is set).
        Note: Requires embedding function to be configured on collection.
        
        Returns:
            List of (chunk_id, similarity_score) tuples
        """
        # This method is not used in our pipeline since we use external embedding service
        # Kept for potential future use
        raise NotImplementedError("Use query_by_vector with external embeddings instead")
    
    def delete_by_course(self, course_id: str) -> bool:
        """
        Delete all embeddings for a specific course.
        
        Args:
            course_id: Course identifier
        
        Returns:
            True if successful
        """
        try:
            # Get all IDs for this course
            results = self.collection.get(
                where={"course_id": course_id}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
            
            return True
        except Exception as e:
            print(f"Error deleting course embeddings: {e}")
            return False
    
    def delete_by_ids(self, chunk_ids: List[str]) -> bool:
        """
        Delete specific chunks by IDs.
        
        Args:
            chunk_ids: List of chunk IDs to delete
        
        Returns:
            True if successful
        """
        try:
            self.collection.delete(ids=chunk_ids)
            return True
        except Exception as e:
            print(f"Error deleting chunks: {e}")
            return False
    
    def get_by_ids(self, chunk_ids: List[str]) -> Dict:
        """
        Retrieve chunks by IDs.
        
        Args:
            chunk_ids: List of chunk IDs
        
        Returns:
            Dict with keys: ids, metadatas, documents
        """
        try:
            results = self.collection.get(ids=chunk_ids)
            return results
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return {'ids': [], 'metadatas': [], 'documents': []}
    
    def count_chunks(self, course_id: Optional[str] = None) -> int:
        """
        Count total chunks in collection, optionally filtered by course.
        
        Args:
            course_id: Optional course filter
        
        Returns:
            Number of chunks
        """
        try:
            if course_id:
                results = self.collection.get(where={"course_id": course_id})
                return len(results['ids'])
            else:
                return self.collection.count()
        except Exception as e:
            print(f"Error counting chunks: {e}")
            return 0
    
    def reset_collection(self) -> bool:
        """
        Delete and recreate the collection (WARNING: destroys all data).
        Use only for testing or migrations.
        
        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(ChromaDBConfig.COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=ChromaDBConfig.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            return True
        except Exception as e:
            print(f"Error resetting collection: {e}")
            return False
