"""
Knowledge Base System (Refactored Core System)
Main orchestrator for the Progressive Disclosure Search & Generation System.
Replaces NoteSummarizerAI with full RAG pipeline.
"""
import time
import uuid
import os
from typing import Dict, List, Optional
from config import UPLOAD_DIR
from file_processor import FileProcessor
from database import LocalDB
from hierarchical_chunker import HierarchicalChunker
from vector_store import VectorStore
from llm_client import EmbeddingClient, RerankerClient
from keyword_search import KeywordSearch
from semantic_search import SemanticSearch
from hybrid_search import HybridSearch
from search_router import SearchRouter
from progressive_retrieval import ProgressiveRetrieval
from context_optimizer import ContextOptimizer
from generation_pipeline import GenerationPipeline
from conversation_manager import ConversationManager
from analytics import Analytics


class KnowledgeBaseSystem:
    """
    Main system orchestrating the full RAG pipeline with progressive disclosure.
    """
    
    def __init__(self):
        # Initialize components
        self.chunker = HierarchicalChunker()
        self.vector_store = VectorStore()
        self.embedding_client = EmbeddingClient()
        self.reranker_client = RerankerClient()
        self.keyword_search = KeywordSearch()
        self.semantic_search = SemanticSearch()
        self.hybrid_search = HybridSearch()
        self.search_router = SearchRouter()
        self.context_optimizer = ContextOptimizer()
        self.generation_pipeline = GenerationPipeline()
        self.conversation_manager = ConversationManager()
        self.analytics = Analytics()
        self.db = LocalDB()
        
        # Cache for course metadata
        self.course_metadata = {}
    
    def index_course_material(
        self,
        filename: str,
        course_id: str,
        course_name: str = None,
        learning_objectives: str = ""
    ) -> Dict:
        """
        Process and index course material.
        
        Pipeline:
        1. Read file
        2. Parse into hierarchical chunks
        3. Generate embeddings
        4. Store in ChromaDB and LocalDB
        5. Build BM25 index
        
        Args:
            filename: File name in upload directory
            course_id: Unique course identifier
            course_name: Human-readable course name
            learning_objectives: Course learning objectives
        
        Returns:
            Dict with indexing stats
        """
        print(f"[INFO] Indexing course material: {filename} (course_id: {course_id})")
        start_time = time.time()
        
        # Read file
        file_path = os.path.join(UPLOAD_DIR, filename)
        content = FileProcessor.process_file(file_path)
        
        if not content:
            return {'success': False, 'error': 'Failed to read file or file is empty'}
        
        if not content.strip():
            return {'success': False, 'error': 'File content is empty or contains no text'}
        
        # Parse hierarchical chunks
        print("[INFO] Parsing hierarchical structure...")
        chunks_dict = self.chunker.process_document(
            text=content,
            course_id=course_id,
            course_name=course_name or filename
        )
        
        total_chunks = sum(len(chunks_dict[level]) for level in chunks_dict)
        print(f"[INFO] Generated {total_chunks} chunks across 3 levels")
        
        if total_chunks == 0:
            return {'success': False, 'error': 'No chunks generated from document. Document may not have proper structure or extractable text.'}
        
        # Save chunks to LocalDB
        self.db.save_knowledge_chunks(course_id, chunks_dict)
        
        # Flatten chunks for embedding
        all_chunks = []
        for level in ['description', 'header', 'detail']:
            all_chunks.extend(chunks_dict[level])
        
        # Generate embeddings
        print("[INFO] Generating embeddings...")
        contents = [chunk['content'] for chunk in all_chunks]
        embeddings = self.embedding_client.embed_batch(contents)
        
        # Store in ChromaDB
        print("[INFO] Storing in ChromaDB...")
        chunk_ids = [chunk['chunk_id'] for chunk in all_chunks]
        metadatas = [chunk['metadata'] for chunk in all_chunks]
        
        # Add required metadata fields
        for i, chunk in enumerate(all_chunks):
            metadatas[i]['course_id'] = course_id
            metadatas[i]['chunk_level'] = chunk['chunk_level']
            metadatas[i]['heading_path'] = chunk['heading_path']
        
        success = self.vector_store.add_embeddings(
            embeddings=embeddings,
            chunk_ids=chunk_ids,
            documents=contents,
            metadatas=metadatas
        )
        
        # Build BM25 keyword index
        print("[INFO] Building BM25 index...")
        self.keyword_search.build_index(all_chunks)
        
        # Save course metadata
        self.course_metadata[course_id] = {
            'course_name': course_name or filename,
            'learning_objectives': learning_objectives,
            'filename': filename,
            'indexed_at': time.time(),
            'total_chunks': total_chunks
        }
        
        elapsed = time.time() - start_time
        print(f"[INFO] Indexing complete in {elapsed:.2f}s")
        
        return {
            'success': success,
            'course_id': course_id,
            'total_chunks': total_chunks,
            'chunks_by_level': {
                'description': len(chunks_dict['description']),
                'header': len(chunks_dict['header']),
                'detail': len(chunks_dict['detail'])
            },
            'elapsed_time': elapsed
        }
    
    def ask_question(
        self,
        student_id: str,
        question: str,
        course_id: str,
        session_id: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Answer student question using full RAG pipeline.
        
        Pipeline:
        1. Analyze intent (route classification)
        2. Progressive retrieval (L1→L2→L3)
        3. Reranking
        4. Context optimization
        5. Generation with citations
        6. Track question for analytics
        
        Args:
            student_id: Student identifier
            question: Student's question
            course_id: Course identifier
            session_id: Optional session ID for multi-turn
            conversation_history: Optional conversation history
        
        Returns:
            Dict with response, citations, retrieval_path, etc.
        """
        print(f"\n[QUESTION] {question}")
        start_time = time.time()
        
        # Create session if needed
        if not session_id:
            session_id = f"session_{uuid.uuid4().hex[:8]}"
            self.conversation_manager.create_session(session_id, student_id, course_id)
        
        # Detect follow-up
        followup_info = self.conversation_manager.detect_followup(session_id, question)
        is_followup = followup_info['is_followup']
        
        # Step 1: Analyze intent
        print("[STEP 1] Analyzing intent...")
        search_config = self.search_router.analyze_intent(
            question=question,
            conversation_history=conversation_history or []
        )
        route_type = search_config['route_type']
        print(f"  Route: {route_type}, Depth: {search_config['search_depth_limit']}")
        
        # Step 2: Progressive retrieval
        print("[STEP 2] Progressive retrieval...")
        progressive_retrieval = ProgressiveRetrieval(
            hybrid_search=self.hybrid_search,
            reranker=self.reranker_client
        )
        
        retrieval_result = progressive_retrieval.retrieve_progressive(
            semantic_search=self.semantic_search,
            keyword_search=self.keyword_search,
            query=question,
            course_id=course_id,
            search_config=search_config,
            is_followup=is_followup
        )
        
        chunks = retrieval_result['chunks']
        retrieval_path = retrieval_result['retrieval_path']
        print(f"  Retrieved {len(chunks)} chunks across {len(retrieval_path)} stages")
        
        # Step 3: Context optimization
        print("[STEP 3] Optimizing context...")
        course_meta = self.course_metadata.get(course_id, {})
        optimized_context = self.context_optimizer.optimize_context(
            chunks=chunks,
            course_name=course_meta.get('course_name', course_id),
            learning_objectives=course_meta.get('learning_objectives', ''),
            current_topic=chunks[0]['metadata'].get('heading_path', '') if chunks else ''
        )
        
        # Step 4: Generate response
        print(f"[STEP 4] Generating {route_type} response...")
        generation_result = self.generation_pipeline.generate_response(
            route_type=route_type,
            question=question,
            context=optimized_context,
            chunks=chunks,
            conversation_history=conversation_history
        )
        
        response = generation_result['response']
        citations = generation_result['citations']
        learn_more = generation_result['learn_more_suggestions']
        
        # Step 5: Track question for analytics
        question_id = f"q_{uuid.uuid4().hex[:8]}"
        self.db.save_question({
            'question_id': question_id,
            'student_id': student_id,
            'course_id': course_id,
            'question_text': question,
            'timestamp': time.time(),
            'route_type': route_type,
            'retrieved_chunks': [c['chunk_id'] for c in chunks],
            'response_quality': None  # Can be set later via feedback
        })
        
        # Step 6: Update conversation
        self.conversation_manager.add_turn(
            session_id=session_id,
            question=question,
            retrieved_chunks=chunks,
            response=response,
            route_type=route_type
        )
        
        elapsed = time.time() - start_time
        print(f"[COMPLETE] Response generated in {elapsed:.2f}s\n")
        
        return {
            'question_id': question_id,
            'session_id': session_id,
            'response': response,
            'citations': citations,
            'learn_more_suggestions': learn_more,
            'route_type': route_type,
            'retrieval_path': retrieval_path,
            'chunks_retrieved': len(chunks),
            'is_followup': is_followup,
            'elapsed_time': elapsed
        }
    
    def analyze_course_questions(self, course_id: str) -> Dict:
        """
        Generate analytics for a course.
        
        Args:
            course_id: Course identifier
        
        Returns:
            Dict with analytics data
        """
        return {
            'course_id': course_id,
            'topic_distribution': self.analytics.get_topic_distribution(course_id),
            'wordcloud_data': self.analytics.get_wordcloud_data(course_id),
            'time_series': self.analytics.get_time_series(course_id, days=7),
            'student_engagement': self.analytics.get_student_engagement(course_id),
            'route_distribution': self.analytics.get_route_distribution(course_id),
            'knowledge_gaps': self.analytics.get_knowledge_gaps_report(course_id),
            'common_questions': self.db.get_common_questions(course_id)
        }
    
    def export_analytics(
        self,
        course_id: str,
        output_dir: str = "./data/analytics"
    ):
        """
        Export analytics to files.
        
        Args:
            course_id: Course identifier
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # JSON export
        json_path = os.path.join(output_dir, f"{course_id}_analytics.json")
        self.analytics.export_analytics_json(course_id, json_path)
        
        # CSV exports
        questions_csv = os.path.join(output_dir, f"{course_id}_questions.csv")
        engagement_csv = os.path.join(output_dir, f"{course_id}_engagement.csv")
        gaps_csv = os.path.join(output_dir, f"{course_id}_gaps.csv")
        
        self.analytics.export_csv(course_id, questions_csv, 'questions')
        self.analytics.export_csv(course_id, engagement_csv, 'engagement')
        self.analytics.export_csv(course_id, gaps_csv, 'gaps')
        
        print(f"[INFO] Analytics exported to {output_dir}")
