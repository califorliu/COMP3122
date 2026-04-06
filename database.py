import json
import os
import time
from typing import List, Dict, Optional
from collections import defaultdict, Counter
from config import COURSE_CONTENT_DIR, QUESTION_DIR

class LocalDB:
    @staticmethod
    def _read(path):
        if not os.path.exists(path): return {}
        with open(path, 'r', encoding='utf-8') as f:
            try: return json.load(f)
            except: return {}

    @staticmethod
    def _write(path, data):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ==================== LEGACY: 课件术语存储 (Backward Compatibility) ====================
    @classmethod
    def save_course_terms(cls, filename, terms_dict):
        path = os.path.join(COURSE_CONTENT_DIR, f"{filename}.json")
        cls._write(path, terms_dict)

    @classmethod
    def get_course_terms(cls, filename):
        path = os.path.join(COURSE_CONTENT_DIR, f"{filename}.json")
        return cls._read(path)

    @classmethod
    def get_all_verified_terms(cls):
        """汇总所有课件中已验证的术语"""
        all_verified = {}
        for file in os.listdir(COURSE_CONTENT_DIR):
            if file.endswith(".json") and not file.startswith("chunks_"):
                data = cls._read(os.path.join(COURSE_CONTENT_DIR, file))
                verified = {k: v for k, v in data.items() if v.get("status") == "verified"}
                all_verified.update(verified)
        return all_verified

    # 学生对话存储 (按学生ID分类)
    @classmethod
    def save_student_session(cls, student_id, session_data):
        path = os.path.join(QUESTION_DIR, f"{student_id}.json")
        records = cls._read(path)
        if not isinstance(records, list): records = []
        records.append(session_data)
        cls._write(path, records)
    
    # ==================== NEW: Knowledge Chunks Storage ====================
    @classmethod
    def save_knowledge_chunks(cls, course_id: str, chunks: Dict[str, List[Dict]]):
        """
        Save hierarchical chunks for a course.
        chunks = {'description': [...], 'header': [...], 'detail': [...]}
        """
        path = os.path.join(COURSE_CONTENT_DIR, f"chunks_{course_id}.json")
        cls._write(path, chunks)
    
    @classmethod
    def get_knowledge_chunks(cls, course_id: str) -> Dict[str, List[Dict]]:
        """Retrieve all chunks for a course."""
        path = os.path.join(COURSE_CONTENT_DIR, f"chunks_{course_id}.json")
        data = cls._read(path)
        if not data:
            return {'description': [], 'header': [], 'detail': []}
        return data
    
    @classmethod
    def get_chunks_by_level(cls, course_id: str, level: str) -> List[Dict]:
        """Get chunks at specific level (description/header/detail)."""
        chunks = cls.get_knowledge_chunks(course_id)
        return chunks.get(level, [])
    
    @classmethod
    def get_child_chunks(cls, course_id: str, parent_chunk_id: str) -> List[Dict]:
        """Get all chunks that have this parent_chunk_id."""
        chunks = cls.get_knowledge_chunks(course_id)
        children = []
        for level in ['header', 'detail']:
            for chunk in chunks.get(level, []):
                if chunk.get('parent_chunk_id') == parent_chunk_id:
                    children.append(chunk)
        return children
    
    @classmethod
    def get_chunks_by_course(cls, course_id: str) -> List[Dict]:
        """Get all chunks (flattened) for a course."""
        chunks = cls.get_knowledge_chunks(course_id)
        all_chunks = []
        for level in ['description', 'header', 'detail']:
            all_chunks.extend(chunks.get(level, []))
        return all_chunks
    
    # ==================== NEW: Student Questions & Analytics ====================
    @classmethod
    def save_question(cls, question_data: Dict):
        """
        Save student question with metadata.
        question_data = {
            'question_id': str,
            'student_id': str,
            'course_id': str,
            'question_text': str,
            'timestamp': float,
            'route_type': str,
            'retrieved_chunks': List[str],
            'response_quality': Optional[int]
        }
        """
        path = os.path.join(QUESTION_DIR, "all_questions.json")
        questions = cls._read(path)
        if not isinstance(questions, list):
            questions = []
        questions.append(question_data)
        cls._write(path, questions)
    
    @classmethod
    def get_all_questions(cls, course_id: Optional[str] = None) -> List[Dict]:
        """Retrieve all questions, optionally filtered by course."""
        path = os.path.join(QUESTION_DIR, "all_questions.json")
        questions = cls._read(path)
        if not isinstance(questions, list):
            return []
        
        if course_id:
            return [q for q in questions if q.get('course_id') == course_id]
        return questions
    
    @classmethod
    def get_hot_topics(cls, course_id: str, time_window: Optional[int] = None, top_n: int = 10) -> List[tuple]:
        """
        Get most frequently asked topics in a course.
        time_window: seconds from now (e.g., 86400 for last 24 hours)
        Returns: [(topic, count), ...]
        """
        questions = cls.get_all_questions(course_id)
        
        # Filter by time window if specified
        if time_window:
            cutoff = time.time() - time_window
            questions = [q for q in questions if q.get('timestamp', 0) >= cutoff]
        
        # Count topic frequency from heading_path in retrieved_chunks
        topic_counter = Counter()
        for q in questions:
            # Extract topics from retrieved chunks metadata
            chunks = q.get('retrieved_chunks', [])
            for chunk_id in chunks:
                # Extract topic from chunk metadata (simplified - would normally load from chunks)
                topic_counter[chunk_id] += 1
        
        return topic_counter.most_common(top_n)
    
    @classmethod
    def generate_wordcloud_data(cls, course_id: str) -> Dict[str, int]:
        """
        Generate word frequency data for word cloud visualization.
        Returns: {word: frequency}
        """
        import re
        questions = cls.get_all_questions(course_id)
        
        word_counter = Counter()
        for q in questions:
            question_text = q.get('question_text', '')
            # Simple tokenization (remove punctuation, lowercase, split)
            words = re.findall(r'\b\w+\b', question_text.lower())
            # Filter out common stop words
            stop_words = {'what', 'is', 'the', 'how', 'do', 'i', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
            filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
            word_counter.update(filtered_words)
        
        return dict(word_counter)
    
    @classmethod
    def get_common_questions(cls, course_id: str, top_n: int = 10) -> List[Dict]:
        """
        Get most frequently asked similar questions.
        Returns: [{question_text, count, route_type}, ...]
        """
        questions = cls.get_all_questions(course_id)
        
        # Group by normalized question text (simple approach)
        question_groups = defaultdict(list)
        for q in questions:
            # Normalize: lowercase, remove punctuation
            normalized = q.get('question_text', '').lower().strip()
            question_groups[normalized].append(q)
        
        # Count and sort
        common = [
            {
                'question_text': text,
                'count': len(group),
                'route_type': group[0].get('route_type', 'unknown')
            }
            for text, group in question_groups.items()
        ]
        common.sort(key=lambda x: x['count'], reverse=True)
        
        return common[:top_n]
    
    @classmethod
    def get_knowledge_gaps(cls, course_id: str, threshold: float = 0.5) -> List[Dict]:
        """
        Identify questions with low retrieval quality (knowledge gaps).
        Returns: [{question_text, timestamp, retrieved_chunks_count}, ...]
        """
        questions = cls.get_all_questions(course_id)
        
        gaps = []
        for q in questions:
            chunks = q.get('retrieved_chunks', [])
            quality = q.get('response_quality')
            
            # Flag as gap if: no chunks retrieved OR low quality rating
            if len(chunks) == 0 or (quality is not None and quality < threshold * 5):
                gaps.append({
                    'question_text': q.get('question_text'),
                    'timestamp': q.get('timestamp'),
                    'retrieved_chunks_count': len(chunks),
                    'response_quality': quality
                })
        
        return gaps