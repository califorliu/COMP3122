import httpx
import json
import re
import time
from typing import List, Union, Dict
from config import (
    MOONSHOT_API_KEY, MOONSHOT_BASE_URL, MODEL_NAME,
    EmbeddingConfig, RerankerConfig, LLMConfig
)

def call_moonshot_json(system_prompt, user_prompt):
    """
    Call Moonshot API and parse JSON response.
    Returns dict if successful, None if failed.
    """
    url = f"{MOONSHOT_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {MOONSHOT_API_KEY}"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Debug: print raw content
            if not content or len(content.strip()) == 0:
                print(f"[WARNING] Empty response from LLM")
                print(f"Full API response: {result}")
                return None
            
            # Try to clean and parse JSON
            # Remove markdown code blocks
            clean_content = re.sub(r'^```(?:json)?\s*', '', content.strip(), flags=re.MULTILINE)
            clean_content = re.sub(r'```\s*$', '', clean_content.strip(), flags=re.MULTILINE)
            clean_content = clean_content.strip()
            
            # Try parsing
            try:
                return json.loads(clean_content)
            except json.JSONDecodeError as je:
                # If JSON parsing fails, try to extract JSON from text
                print(f"[WARNING] JSON parsing failed: {je}")
                print(f"Raw content: {content[:200]}...")
                print(f"Cleaned content: {clean_content[:200]}...")
                
                # Try to find JSON object in the text
                json_match = re.search(r'\{.*\}', clean_content, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group(0))
                    except:
                        pass
                
                # If still fails, return None
                return None
                
    except httpx.HTTPStatusError as e:
        print(f"AI 调用失败 (HTTP {e.response.status_code}): {e}")
        print(f"Response body: {e.response.text[:500]}")
        return None
    except Exception as e:
        print(f"AI 调用失败: {e}")
        import traceback
        traceback.print_exc()
        return None


class EmbeddingClient:
    """Client for generating text embeddings using Qwen3-Embedding-8B."""
    
    def __init__(self):
        self.api_url = EmbeddingConfig.API_URL
        self.api_key = EmbeddingConfig.API_KEY
        self.model_name = EmbeddingConfig.MODEL_NAME
        self.batch_size = EmbeddingConfig.BATCH_SIZE
        self.max_retries = EmbeddingConfig.MAX_RETRIES
        self.timeout = EmbeddingConfig.TIMEOUT
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
        
        Returns:
            Embedding vector as list of floats
        """
        return self.embed_batch([text])[0]
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts with retry logic.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch_with_retry(batch)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def _embed_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        Internal method to embed a batch with exponential backoff retry.
        
        Args:
            texts: Batch of texts (size <= batch_size)
        
        Returns:
            List of embedding vectors
        """
        for attempt in range(self.max_retries):
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model_name,
                    "input": texts
                }
                
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(self.api_url, headers=headers, json=payload)
                    response.raise_for_status()
                    result = response.json()
                    
                    # Extract embeddings from response
                    # Expected format: {"data": [{"embedding": [...]}, ...]}
                    embeddings = [item['embedding'] for item in result['data']]
                    return embeddings
                    
            except Exception as e:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"Embedding API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
                else:
                    print(f"Failed to generate embeddings after {self.max_retries} attempts")
                    # Return zero vectors as fallback
                    return [[0.0] * 768 for _ in texts]  # Qwen3-Embedding-8B outputs 768-dim vectors
        
        return [[0.0] * 768 for _ in texts]


class RerankerClient:
    """Client for reranking documents using Qwen3-Reranker-8B."""
    
    def __init__(self):
        self.api_url = RerankerConfig.API_URL
        self.api_key = RerankerConfig.API_KEY
        self.model_name = RerankerConfig.MODEL_NAME
        self.max_documents = RerankerConfig.MAX_DOCUMENTS
        self.threshold = RerankerConfig.THRESHOLD
        self.timeout = RerankerConfig.TIMEOUT
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = None
    ) -> List[Dict]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: Search query
            documents: List of document texts
            top_n: Optional limit on number of results
        
        Returns:
            List of dicts with keys: index, relevance_score, text
            Sorted by relevance_score descending
        """
        if not documents:
            return []
        
        # Limit to max documents per API call
        if len(documents) > self.max_documents:
            print(f"Warning: Truncating {len(documents)} documents to {self.max_documents}")
            documents = documents[:self.max_documents]
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model_name,
                "query": query,
                "documents": documents,
                "top_n": top_n if top_n else len(documents)
            }
            
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                
                # Expected format: {"results": [{"index": 0, "relevance_score": 0.95}, ...]}
                reranked = result.get('results', [])
                
                # Add document text to results
                for item in reranked:
                    item['text'] = documents[item['index']]
                
                return reranked
                
        except Exception as e:
            print(f"Reranker API error: {e}")
            # Fallback: return documents in original order with neutral scores
            return [
                {'index': i, 'relevance_score': 0.5, 'text': doc}
                for i, doc in enumerate(documents)
            ]
    
    def rerank_batch(
        self,
        query: str,
        documents: List[str],
        batch_size: int = None
    ) -> List[Dict]:
        """
        Rerank large document sets by batching.
        
        Args:
            query: Search query
            documents: List of document texts
            batch_size: Documents per batch (default: max_documents)
        
        Returns:
            Combined reranked results
        """
        if batch_size is None:
            batch_size = self.max_documents
        
        all_results = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_results = self.rerank(query, batch)
            
            # Adjust indices to global positions
            for item in batch_results:
                item['index'] += i
            
            all_results.extend(batch_results)
        
        # Re-sort all results by relevance score
        all_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return all_results
    
    def filter_by_score(self, reranked_results: List[Dict], threshold: float = None) -> List[Dict]:
        """
        Filter reranked results by minimum relevance score.
        
        Args:
            reranked_results: Output from rerank() or rerank_batch()
            threshold: Minimum score (default: from config)
        
        Returns:
            Filtered results
        """
        if threshold is None:
            threshold = self.threshold
        
        return [r for r in reranked_results if r['relevance_score'] >= threshold]