"""
Search Router with Intent Classification
Routes queries to appropriate generation strategies based on LLM analysis.
"""
from typing import Dict
from llm_client import call_moonshot_json
from config import SearchConfig


class SearchRouter:
    """LLM-based intent analyzer and search strategy router."""
    
    ROUTE_TYPES = ['quick-answer', 'tutorial', 'deep-dive', 'mock-interview']
    
    def __init__(self):
        pass
    
    def analyze_intent(self, question: str, conversation_history: list = None) -> Dict:
        """
        Analyze user question to determine search strategy.
        
        Args:
            question: User's question
            conversation_history: Optional list of previous turns
        
        Returns:
            Dict with keys:
                - route_type: str (quick-answer/tutorial/deep-dive/mock-interview)
                - complexity_level: str (low/medium/high)
                - prior_knowledge_level: str (beginner/intermediate/advanced)
                - search_depth_limit: int (max retrieval level: 1, 2, or 3)
                - chunk_levels_to_query: list of str
                - top_k_per_level: dict {level: top_k}
        """
        # Build context from conversation history
        history_context = ""
        if conversation_history:
            recent = conversation_history[-3:]  # Last 3 turns
            history_context = "\n".join([
                f"Q: {turn.get('question', '')}\nA: {turn.get('response', '')[:200]}..."
                for turn in recent
            ])
        
        system_prompt = """You are an educational intent analyzer. Analyze the student's question and determine:

1. **route_type** (choose one):
   - "quick-answer": Simple factual question, needs 2-3 sentence answer
   - "tutorial": Needs step-by-step guidance or how-to explanation
   - "deep-dive": Complex topic requiring comprehensive explanation with theory and examples
   - "mock-interview": Student wants practice questions or interview preparation

2. **complexity_level**: "low", "medium", or "high"

3. **prior_knowledge_level**: "beginner", "intermediate", or "advanced" (infer from question phrasing)

4. **search_depth_limit**: 1, 2, or 3
   - 1: Only search description-level summaries (for broad overview questions)
   - 2: Search descriptions + headers (for specific topic questions)
   - 3: Search all levels including details (for complex/technical questions)

5. **chunk_levels_to_query**: List of levels to search: ["description"], ["description", "header"], or ["description", "header", "detail"]

Return valid JSON format with these exact keys."""

        user_prompt = f"""Question: "{question}"

{f"Recent conversation context:\n{history_context}" if history_context else ""}

Analyze this question and return JSON with: route_type, complexity_level, prior_knowledge_level, search_depth_limit, chunk_levels_to_query."""

        result = call_moonshot_json(system_prompt, user_prompt)
        
        if not result:
            # Fallback to safe defaults
            return self._get_default_config()
        
        # Validate and normalize result
        return self._validate_intent(result)
    
    def _validate_intent(self, intent: Dict) -> Dict:
        """Validate and normalize intent analysis result."""
        validated = {}
        
        # Route type
        route_type = intent.get('route_type', 'deep-dive')
        validated['route_type'] = route_type if route_type in self.ROUTE_TYPES else 'deep-dive'
        
        # Complexity level
        complexity = intent.get('complexity_level', 'medium')
        validated['complexity_level'] = complexity if complexity in ['low', 'medium', 'high'] else 'medium'
        
        # Prior knowledge
        knowledge = intent.get('prior_knowledge_level', 'intermediate')
        validated['prior_knowledge_level'] = knowledge if knowledge in ['beginner', 'intermediate', 'advanced'] else 'intermediate'
        
        # Search depth
        depth = intent.get('search_depth_limit', 2)
        validated['search_depth_limit'] = depth if depth in [1, 2, 3] else 2
        
        # Chunk levels
        levels = intent.get('chunk_levels_to_query', ['description', 'header'])
        valid_levels = [l for l in levels if l in ['description', 'header', 'detail']]
        validated['chunk_levels_to_query'] = valid_levels if valid_levels else ['description', 'header']
        
        # Determine top_k per level based on route type
        validated['top_k_per_level'] = self._get_top_k_config(validated['route_type'])
        
        return validated
    
    def _get_top_k_config(self, route_type: str) -> Dict[str, int]:
        """Get top-k configuration for each level based on route type."""
        configs = {
            'quick-answer': {
                'description': SearchConfig.LEVEL1_TOP_K,
                'header': 5,
                'detail': 3
            },
            'tutorial': {
                'description': SearchConfig.LEVEL1_TOP_K,
                'header': SearchConfig.LEVEL2_TOP_K,
                'detail': 10
            },
            'deep-dive': {
                'description': SearchConfig.LEVEL1_TOP_K,
                'header': SearchConfig.LEVEL2_TOP_K,
                'detail': SearchConfig.LEVEL3_TOP_K
            },
            'mock-interview': {
                'description': SearchConfig.LEVEL1_TOP_K,
                'header': 15,
                'detail': SearchConfig.LEVEL3_TOP_K
            }
        }
        return configs.get(route_type, configs['deep-dive'])
    
    def _get_default_config(self) -> Dict:
        """Return safe default configuration if LLM analysis fails."""
        return {
            'route_type': 'deep-dive',
            'complexity_level': 'medium',
            'prior_knowledge_level': 'intermediate',
            'search_depth_limit': 2,
            'chunk_levels_to_query': ['description', 'header'],
            'top_k_per_level': self._get_top_k_config('deep-dive')
        }
    
    def should_expand_search(self, current_level: int, max_relevance: float, is_followup: bool) -> bool:
        """
        Determine if search should expand to next level.
        
        Args:
            current_level: Current retrieval level (1, 2, or 3)
            max_relevance: Highest relevance score from current level
            is_followup: Whether this is a follow-up question
        
        Returns:
            True if should expand to next level
        """
        # Always expand if it's a follow-up question
        if is_followup:
            return current_level < 3
        
        # Expand if relevance is below threshold
        if max_relevance < SearchConfig.RELEVANCE_THRESHOLD:
            return current_level < 3
        
        return False
