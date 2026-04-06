"""
Context Optimizer
Deduplicates, reorders, and truncates context for generation.
"""
import tiktoken
from typing import List, Dict
import numpy as np
from config import GenerationConfig
from llm_client import EmbeddingClient


class ContextOptimizer:
    """Optimizes retrieved context for LLM generation."""
    
    def __init__(self):
        self.embedding_client = EmbeddingClient()
        try:
            # Use tiktoken for token counting (kimi-k2.5 uses similar encoding to GPT)
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoding = None
    
    def deduplicate(self, chunks: List[Dict], similarity_threshold: float = 0.95) -> List[Dict]:
        """
        Remove redundant chunks based on content similarity.
        
        Args:
            chunks: List of chunk dicts
            similarity_threshold: Cosine similarity threshold for duplicates
        
        Returns:
            Deduplicated list of chunks
        """
        if len(chunks) <= 1:
            return chunks
        
        # Get embeddings for all chunks
        contents = [chunk['content'] for chunk in chunks]
        embeddings = self.embedding_client.embed_batch(contents)
        
        # Convert to numpy arrays for efficient computation
        embeddings_array = np.array(embeddings)
        
        # Calculate cosine similarity matrix
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        normalized = embeddings_array / norms
        similarity_matrix = np.dot(normalized, normalized.T)
        
        # Keep track of chunks to keep
        keep_indices = []
        removed_indices = set()
        
        for i in range(len(chunks)):
            if i in removed_indices:
                continue
            
            keep_indices.append(i)
            
            # Mark similar chunks as removed
            for j in range(i + 1, len(chunks)):
                if j not in removed_indices and similarity_matrix[i][j] >= similarity_threshold:
                    removed_indices.add(j)
        
        # Return deduplicated chunks
        return [chunks[i] for i in keep_indices]
    
    def reorder_by_prerequisites(self, chunks: List[Dict]) -> List[Dict]:
        """
        Reorder chunks so prerequisites come first.
        
        Order: description → header → detail
        Within each level: sort by heading_path (breadth-first traversal)
        
        Args:
            chunks: List of chunk dicts
        
        Returns:
            Reordered list of chunks
        """
        # Group by chunk level
        by_level = {
            'description': [],
            'header': [],
            'detail': []
        }
        
        for chunk in chunks:
            level = chunk['metadata'].get('chunk_level', 'detail')
            if level in by_level:
                by_level[level].append(chunk)
        
        # Sort within each level by heading_path
        for level in by_level:
            by_level[level].sort(key=lambda x: x['metadata'].get('heading_path', ''))
        
        # Combine: description → header → detail
        reordered = []
        reordered.extend(by_level['description'])
        reordered.extend(by_level['header'])
        reordered.extend(by_level['detail'])
        
        return reordered
    
    def truncate_to_budget(
        self,
        chunks: List[Dict],
        max_tokens: int = None,
        reserve_for_response: int = 2000
    ) -> List[Dict]:
        """
        Truncate chunks to fit within token budget.
        
        Args:
            chunks: List of chunk dicts (should be ordered by priority)
            max_tokens: Maximum tokens (default from config)
            reserve_for_response: Tokens to reserve for model response
        
        Returns:
            Truncated list of chunks that fit within budget
        """
        if max_tokens is None:
            max_tokens = GenerationConfig.MAX_CONTEXT_TOKENS
        
        if not self.encoding:
            # Fallback: estimate 4 chars per token
            return self._truncate_by_chars(chunks, max_tokens * 4)
        
        available_tokens = max_tokens - reserve_for_response
        current_tokens = 0
        truncated_chunks = []
        
        for chunk in chunks:
            content = chunk['content']
            chunk_tokens = len(self.encoding.encode(content))
            
            if current_tokens + chunk_tokens <= available_tokens:
                truncated_chunks.append(chunk)
                current_tokens += chunk_tokens
            else:
                # Try to fit a truncated version of this chunk
                remaining_tokens = available_tokens - current_tokens
                if remaining_tokens > 100:  # Only if meaningful space left
                    truncated_content = self._truncate_text(content, remaining_tokens)
                    truncated_chunk = chunk.copy()
                    truncated_chunk['content'] = truncated_content
                    truncated_chunk['metadata']['truncated'] = True
                    truncated_chunks.append(truncated_chunk)
                break
        
        return truncated_chunks
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        if not self.encoding:
            return text[:max_tokens * 4]
        
        tokens = self.encoding.encode(text)
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens) + "..."
    
    def _truncate_by_chars(self, chunks: List[Dict], max_chars: int) -> List[Dict]:
        """Fallback truncation by character count."""
        current_chars = 0
        truncated_chunks = []
        
        for chunk in chunks:
            content = chunk['content']
            if current_chars + len(content) <= max_chars:
                truncated_chunks.append(chunk)
                current_chars += len(content)
            else:
                break
        
        return truncated_chunks
    
    def inject_context(
        self,
        chunks: List[Dict],
        course_name: str = "Unknown Course",
        learning_objectives: str = "",
        current_topic: str = ""
    ) -> str:
        """
        Format chunks with course context for generation.
        
        Args:
            chunks: List of optimized chunks
            course_name: Name of the course
            learning_objectives: Course learning objectives
            current_topic: Current topic being discussed
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Add course header
        context_parts.append(f"=== COURSE: {course_name} ===")
        if learning_objectives:
            context_parts.append(f"Learning Objectives: {learning_objectives}")
        if current_topic:
            context_parts.append(f"Current Topic: {current_topic}")
        context_parts.append("")
        
        # Add chunks with section markers
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk['metadata']
            heading_path = metadata.get('heading_path', 'Unknown Section')
            level = metadata.get('chunk_level', 'detail')
            
            context_parts.append(f"--- Section {i} ({level}): {heading_path} ---")
            context_parts.append(chunk['content'])
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def optimize_context(
        self,
        chunks: List[Dict],
        course_name: str = "Unknown Course",
        learning_objectives: str = "",
        current_topic: str = "",
        max_tokens: int = None
    ) -> str:
        """
        Full optimization pipeline: deduplicate → reorder → truncate → format.
        
        Args:
            chunks: Retrieved chunks
            course_name: Course name
            learning_objectives: Learning objectives
            current_topic: Current topic
            max_tokens: Token budget
        
        Returns:
            Optimized context string ready for generation
        """
        # Step 1: Deduplicate
        deduped = self.deduplicate(chunks)
        
        # Step 2: Reorder by prerequisites
        reordered = self.reorder_by_prerequisites(deduped)
        
        # Step 3: Truncate to budget
        truncated = self.truncate_to_budget(reordered, max_tokens)
        
        # Step 4: Format with context
        context = self.inject_context(
            truncated,
            course_name,
            learning_objectives,
            current_topic
        )
        
        return context
