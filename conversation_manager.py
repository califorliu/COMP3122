"""
Conversation Memory Manager
Manages multi-turn conversation context with session state tracking.
"""
import time
from typing import List, Dict, Optional
from config import GenerationConfig


class ConversationManager:
    """Manages conversation history and context for multi-turn interactions."""
    
    def __init__(self):
        self.sessions = {}  # {session_id: {student_id, course_id, turns, context_tokens}}
    
    def create_session(self, session_id: str, student_id: str, course_id: str) -> Dict:
        """
        Create a new conversation session.
        
        Args:
            session_id: Unique session identifier
            student_id: Student identifier
            course_id: Course identifier
        
        Returns:
            Session dict
        """
        self.sessions[session_id] = {
            'session_id': session_id,
            'student_id': student_id,
            'course_id': course_id,
            'turns': [],
            'context_tokens': 0,
            'created_at': time.time()
        }
        return self.sessions[session_id]
    
    def add_turn(
        self,
        session_id: str,
        question: str,
        retrieved_chunks: List[Dict],
        response: str,
        route_type: str = 'deep-dive'
    ) -> Dict:
        """
        Add a conversation turn to session.
        
        Args:
            session_id: Session identifier
            question: User question
            retrieved_chunks: Chunks retrieved for this turn
            response: Generated response
            route_type: Route type used
        
        Returns:
            Turn dict
        """
        if session_id not in self.sessions:
            # Auto-create session if doesn't exist
            self.sessions[session_id] = {
                'session_id': session_id,
                'student_id': 'unknown',
                'course_id': 'unknown',
                'turns': [],
                'context_tokens': 0,
                'created_at': time.time()
            }
        
        session = self.sessions[session_id]
        
        turn = {
            'turn_id': len(session['turns']) + 1,
            'question': question,
            'retrieved_chunks': [
                {
                    'chunk_id': c['chunk_id'],
                    'heading_path': c['metadata'].get('heading_path', ''),
                    'relevance_score': c.get('relevance_score', 0.0)
                }
                for c in retrieved_chunks
            ],
            'response': response,
            'route_type': route_type,
            'timestamp': time.time()
        }
        
        session['turns'].append(turn)
        
        # Keep only last N turns
        max_turns = GenerationConfig.CONVERSATION_HISTORY_TURNS
        if len(session['turns']) > max_turns:
            session['turns'] = session['turns'][-max_turns:]
        
        return turn
    
    def get_context(self, session_id: str) -> Dict:
        """
        Get conversation context for a session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Dict with recent turns and metadata
        """
        if session_id not in self.sessions:
            return {'turns': [], 'session_exists': False}
        
        session = self.sessions[session_id]
        return {
            'turns': session['turns'],
            'session_exists': True,
            'student_id': session['student_id'],
            'course_id': session['course_id']
        }
    
    def detect_followup(
        self,
        session_id: str,
        current_question: str
    ) -> Dict:
        """
        Detect if current question is a follow-up to previous conversation.
        
        Args:
            session_id: Session identifier
            current_question: Current question text
        
        Returns:
            Dict with keys: is_followup, previous_chunks, context_summary
        """
        if session_id not in self.sessions:
            return {
                'is_followup': False,
                'previous_chunks': [],
                'context_summary': ''
            }
        
        session = self.sessions[session_id]
        turns = session['turns']
        
        if not turns:
            return {
                'is_followup': False,
                'previous_chunks': [],
                'context_summary': ''
            }
        
        # Check for follow-up indicators
        followup_indicators = [
            'more', 'explain', 'elaborate', 'detail', 'what about',
            'can you', 'tell me more', 'expand', 'continue',
            'also', 'and', 'but', 'however', 'what if'
        ]
        
        question_lower = current_question.lower()
        is_followup = any(indicator in question_lower for indicator in followup_indicators)
        
        # Get chunks from last turn
        previous_chunks = []
        if turns:
            last_turn = turns[-1]
            previous_chunks = last_turn['retrieved_chunks']
        
        # Create context summary from recent turns
        context_parts = []
        for turn in turns[-3:]:  # Last 3 turns
            context_parts.append(f"Q: {turn['question']}")
            context_parts.append(f"A: {turn['response'][:200]}...")  # Truncate response
        
        context_summary = "\n".join(context_parts)
        
        return {
            'is_followup': is_followup,
            'previous_chunks': previous_chunks,
            'context_summary': context_summary
        }
    
    def get_previous_chunks(self, session_id: str, last_n_turns: int = 1) -> List[Dict]:
        """
        Get chunks from previous turns (to avoid re-retrieval).
        
        Args:
            session_id: Session identifier
            last_n_turns: Number of previous turns to include
        
        Returns:
            List of chunk metadata from previous turns
        """
        if session_id not in self.sessions:
            return []
        
        session = self.sessions[session_id]
        turns = session['turns']
        
        if not turns:
            return []
        
        # Collect chunks from last N turns
        all_chunks = []
        for turn in turns[-last_n_turns:]:
            all_chunks.extend(turn['retrieved_chunks'])
        
        # Deduplicate by chunk_id
        seen_ids = set()
        unique_chunks = []
        for chunk in all_chunks:
            if chunk['chunk_id'] not in seen_ids:
                unique_chunks.append(chunk)
                seen_ids.add(chunk['chunk_id'])
        
        return unique_chunks
    
    def should_reuse_context(
        self,
        session_id: str,
        current_question: str,
        relevance_threshold: float = 0.7
    ) -> bool:
        """
        Determine if previous context can answer current question.
        
        Args:
            session_id: Session identifier
            current_question: Current question
            relevance_threshold: Minimum relevance to reuse
        
        Returns:
            True if should reuse previous context
        """
        followup_info = self.detect_followup(session_id, current_question)
        
        if not followup_info['is_followup']:
            return False
        
        # Check if previous chunks have high relevance
        previous_chunks = followup_info['previous_chunks']
        if not previous_chunks:
            return False
        
        # If any previous chunk has high relevance, can potentially reuse
        max_relevance = max([c.get('relevance_score', 0.0) for c in previous_chunks], default=0.0)
        return max_relevance >= relevance_threshold
    
    def get_session_stats(self, session_id: str) -> Dict:
        """
        Get statistics for a session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Dict with session stats
        """
        if session_id not in self.sessions:
            return {'exists': False}
        
        session = self.sessions[session_id]
        
        return {
            'exists': True,
            'turn_count': len(session['turns']),
            'created_at': session['created_at'],
            'duration_seconds': time.time() - session['created_at'],
            'route_types_used': [t['route_type'] for t in session['turns']]
        }
    
    def clear_session(self, session_id: str):
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
