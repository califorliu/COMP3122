"""
Generation Pipeline
Route-specific generation with citations and "Learn More" suggestions.
"""
import httpx
import json
import re
from typing import List, Dict, Optional
from config import LLMConfig, GenerationConfig


class GenerationPipeline:
    """Generate responses for different route types with citations."""
    
    def __init__(self):
        self.api_url = LLMConfig.API_URL
        self.api_key = LLMConfig.API_KEY
        self.model_name = LLMConfig.MODEL_NAME
        self.temperature = LLMConfig.TEMPERATURE
        self.timeout = LLMConfig.TIMEOUT
    
    def generate_response(
        self,
        route_type: str,
        question: str,
        context: str,
        chunks: List[Dict],
        conversation_history: List[Dict] = None
    ) -> Dict:
        """
        Generate response based on route type.
        
        Args:
            route_type: One of: quick-answer, tutorial, deep-dive, mock-interview
            question: User question
            context: Optimized context from ContextOptimizer
            chunks: Retrieved chunks (for citations)
            conversation_history: Optional conversation history
        
        Returns:
            Dict with keys: response, citations, learn_more_suggestions
        """
        # Select route handler
        handlers = {
            'quick-answer': self._generate_quick_answer,
            'tutorial': self._generate_tutorial,
            'deep-dive': self._generate_deep_dive,
            'mock-interview': self._generate_mock_interview
        }
        
        handler = handlers.get(route_type, self._generate_deep_dive)
        
        # Generate response
        return handler(question, context, chunks, conversation_history)
    
    def _generate_quick_answer(
        self,
        question: str,
        context: str,
        chunks: List[Dict],
        conversation_history: List[Dict]
    ) -> Dict:
        """Generate concise 2-3 sentence answer."""
        # Limit chunks for quick answer
        limited_chunks = chunks[:GenerationConfig.QUICK_ANSWER_MAX_CHUNKS]
        limited_context = self._format_chunks_for_context(limited_chunks)
        
        system_prompt = """You are a concise educational assistant. Provide a brief, accurate answer in 2-3 sentences.
Focus on the core concept. Do not elaborate unless absolutely necessary."""

        user_prompt = f"""Question: {question}

Course Content:
{limited_context}

Provide a concise 2-3 sentence answer."""

        response = self._call_llm(system_prompt, user_prompt)
        
        return {
            'response': response,
            'citations': self.format_citations(limited_chunks),
            'learn_more_suggestions': self._suggest_next_topics(chunks, limited_chunks)
        }
    
    def _generate_tutorial(
        self,
        question: str,
        context: str,
        chunks: List[Dict],
        conversation_history: List[Dict]
    ) -> Dict:
        """Generate step-by-step tutorial."""
        limited_chunks = chunks[:GenerationConfig.TUTORIAL_MAX_CHUNKS]
        limited_context = self._format_chunks_for_context(limited_chunks)
        
        system_prompt = """You are an educational tutor. Create a step-by-step tutorial that helps students learn by doing.

Structure your response as:
1. **Prerequisites**: What the student should know first
2. **Steps**: Numbered steps with clear explanations
3. **Practice Exercise**: A simple exercise to reinforce learning
4. **Key Takeaways**: Brief summary of main points

Use clear, beginner-friendly language."""

        user_prompt = f"""Question: {question}

Course Content:
{limited_context}

Provide a step-by-step tutorial following the structure above."""

        response = self._call_llm(system_prompt, user_prompt)
        
        return {
            'response': response,
            'citations': self.format_citations(limited_chunks),
            'learn_more_suggestions': self._suggest_next_topics(chunks, limited_chunks)
        }
    
    def _generate_deep_dive(
        self,
        question: str,
        context: str,
        chunks: List[Dict],
        conversation_history: List[Dict]
    ) -> Dict:
        """Generate comprehensive explanation."""
        limited_chunks = chunks[:GenerationConfig.DEEPDIVE_MAX_CHUNKS]
        limited_context = self._format_chunks_for_context(limited_chunks)
        
        system_prompt = """You are an expert educational assistant. Provide a comprehensive, in-depth explanation.

Structure your response as:
1. **Overview**: High-level summary of the concept
2. **Theory**: Detailed theoretical explanation
3. **Examples**: Concrete examples with code/diagrams if relevant
4. **Edge Cases**: Common pitfalls or special cases
5. **Further Reading**: Suggest related topics

Be thorough but maintain clarity. Use technical terms appropriately."""

        user_prompt = f"""Question: {question}

Course Content:
{limited_context}

Provide a comprehensive explanation following the structure above."""

        response = self._call_llm(system_prompt, user_prompt)
        
        return {
            'response': response,
            'citations': self.format_citations(limited_chunks),
            'learn_more_suggestions': self._suggest_next_topics(chunks, limited_chunks)
        }
    
    def _generate_mock_interview(
        self,
        question: str,
        context: str,
        chunks: List[Dict],
        conversation_history: List[Dict]
    ) -> Dict:
        """Generate interview questions based on content."""
        limited_chunks = chunks[:GenerationConfig.DEEPDIVE_MAX_CHUNKS]
        limited_context = self._format_chunks_for_context(limited_chunks)
        
        system_prompt = """You are an interview preparation coach. Generate realistic interview questions based on the course content.

For each question provide:
1. The question itself
2. Expected answer with key points
3. A follow-up question to test deeper understanding

Generate 3-5 questions ranging from basic to advanced."""

        user_prompt = f"""Based on this course content, generate interview questions:

{limited_context}

Student's focus: {question}

Generate 3-5 interview Q&A pairs with follow-ups."""

        response = self._call_llm(system_prompt, user_prompt)
        
        return {
            'response': response,
            'citations': self.format_citations(limited_chunks),
            'learn_more_suggestions': self._suggest_next_topics(chunks, limited_chunks)
        }
    
    def _call_llm(self, system_prompt: str, user_prompt: str, max_retries: int = 2) -> str:
        """
        Call LLM API with retry logic for timeout errors.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            max_retries: Maximum number of retry attempts
        
        Returns:
            Generated response text
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": self.temperature
                }
                
                if attempt > 0:
                    print(f"[INFO] Retrying LLM call (attempt {attempt + 1}/{max_retries + 1})...")
                
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(self.api_url, headers=headers, json=payload)
                    response.raise_for_status()
                    result = response.json()
                    return result['choices'][0]['message']['content']
                    
            except httpx.ReadTimeout as e:
                last_error = f"Request timed out after {self.timeout} seconds"
                print(f"[WARNING] {last_error}")
                if attempt < max_retries:
                    print(f"[INFO] Will retry...")
                continue
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                print(f"[ERROR] LLM API error: {last_error}")
                break  # Don't retry HTTP errors
            except Exception as e:
                last_error = str(e)
                print(f"[ERROR] LLM generation error: {e}")
                break
        
        # All retries failed
        return f"I apologize, but I'm having trouble generating a response right now. The system timed out after multiple attempts. Please try:\n\n1. Asking a more specific question\n2. Trying again in a moment\n3. Checking your network connection\n\nTechnical details: {last_error}"
    
    def _format_chunks_for_context(self, chunks: List[Dict]) -> str:
        """Format chunks into context string."""
        parts = []
        for chunk in chunks:
            metadata = chunk['metadata']
            heading = metadata.get('heading_path', 'Unknown')
            parts.append(f"[{heading}]\n{chunk['content']}\n")
        return "\n".join(parts)
    
    def format_citations(self, chunks: List[Dict]) -> List[str]:
        """
        Generate inline citations from chunks.
        
        Args:
            chunks: List of chunks used in generation
        
        Returns:
            List of citation strings
        """
        citations = []
        seen_paths = set()
        
        for chunk in chunks:
            metadata = chunk['metadata']
            course_id = metadata.get('course_id', 'Unknown Course')
            heading_path = metadata.get('heading_path', 'Unknown Section')
            
            # Avoid duplicate citations
            citation_key = f"{course_id}::{heading_path}"
            if citation_key not in seen_paths:
                citations.append(f"[Source: {course_id}, Section: {heading_path}]")
                seen_paths.add(citation_key)
        
        return citations
    
    def _suggest_next_topics(
        self,
        all_chunks: List[Dict],
        used_chunks: List[Dict]
    ) -> List[str]:
        """
        Suggest unexplored topics with high relevance.
        
        Args:
            all_chunks: All retrieved chunks
            used_chunks: Chunks used in generation
        
        Returns:
            List of suggested topic headings
        """
        used_ids = {c['chunk_id'] for c in used_chunks}
        unused_chunks = [c for c in all_chunks if c['chunk_id'] not in used_ids]
        
        # Sort by relevance score
        unused_chunks.sort(
            key=lambda x: x.get('relevance_score', 0.0),
            reverse=True
        )
        
        # Extract unique topics (top 3-5)
        suggestions = []
        seen_paths = set()
        
        for chunk in unused_chunks[:10]:  # Check top 10 unused
            heading = chunk['metadata'].get('heading_path', '')
            if heading and heading not in seen_paths:
                suggestions.append(heading)
                seen_paths.add(heading)
            
            if len(suggestions) >= 5:
                break
        
        return suggestions
