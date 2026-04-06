"""
Streamlit Web UI for Progressive Disclosure Search & Generation System
Run with: streamlit run app.py
"""
import streamlit as st
import os
import json
import time
from datetime import datetime
from knowledge_base_system import KnowledgeBaseSystem
from database import LocalDB
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'kb_system' not in st.session_state:
    st.session_state.kb_system = KnowledgeBaseSystem()
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

kb_system = st.session_state.kb_system
db = LocalDB()

# Sidebar
with st.sidebar:
    st.title("🎓 AI Learning Assistant")
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["📚 Course Management", "💬 Ask Questions", "📊 Analytics", "🐛 Debug View"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Course selector
    st.subheader("Select Course")
    
    # Get available courses
    available_courses = []
    if os.path.exists("./data/course_content"):
        for file in os.listdir("./data/course_content"):
            if file.startswith("chunks_") and file.endswith(".json"):
                course_id = file.replace("chunks_", "").replace(".json", "")
                available_courses.append(course_id)
    
    if available_courses:
        selected_course = st.selectbox(
            "Course",
            available_courses,
            key="selected_course"
        )
    else:
        st.info("No courses indexed yet. Upload a course in Course Management.")
        selected_course = None
    
    st.markdown("---")
    st.caption("Progressive Disclosure RAG System v1.0")

# Main content area
if page == "📚 Course Management":
    st.title("📚 Course Management")
    st.markdown("Upload and index course materials for the knowledge base.")
    
    tab1, tab2, tab3 = st.tabs(["Upload New Course", "Add to Existing Course", "View Courses"])
    
    with tab1:
        st.subheader("Upload New Course")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            course_name = st.text_input("Course Name", placeholder="e.g., Introduction to Python")
            course_id = st.text_input("Course ID", placeholder="e.g., python101")
        
        with col2:
            learning_objectives = st.text_area(
                "Learning Objectives (optional)",
                placeholder="What students will learn...",
                height=100
            )
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['txt', 'md', 'pdf', 'docx'],
            help="Supported formats: TXT, MD, PDF, DOCX"
        )
        
        if st.button("📤 Index Course", type="primary", disabled=not uploaded_file or not course_id):
            if uploaded_file and course_id:
                # Save uploaded file
                upload_path = os.path.join("./upload_file", uploaded_file.name)
                with open(upload_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with st.spinner("Indexing course material... This may take a minute."):
                    try:
                        result = kb_system.index_course_material(
                            filename=uploaded_file.name,
                            course_id=course_id,
                            course_name=course_name or uploaded_file.name,
                            learning_objectives=learning_objectives
                        )
                        
                        st.success(f"✅ Course indexed successfully!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Chunks", result['total_chunks'])
                        col2.metric("Descriptions", result['chunks_by_level']['description'])
                        col3.metric("Headers", result['chunks_by_level']['header'])
                        col4.metric("Details", result['chunks_by_level']['detail'])
                        
                        st.info(f"⏱️ Indexing completed in {result['elapsed_time']:.2f} seconds")
                        
                    except Exception as e:
                        st.error(f"❌ Indexing failed: {str(e)}")
                        with st.expander("Error Details"):
                            st.exception(e)
    
    with tab2:
        st.subheader("Add to Existing Course")
        st.info("🚧 Feature coming soon: Add additional materials to existing courses")
    
    with tab3:
        st.subheader("Indexed Courses")
        
        if available_courses:
            for course_id in available_courses:
                with st.expander(f"📖 {course_id}"):
                    chunks = db.get_knowledge_chunks(course_id)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Description Chunks", len(chunks.get('description', [])))
                    col2.metric("Header Chunks", len(chunks.get('header', [])))
                    col3.metric("Detail Chunks", len(chunks.get('detail', [])))
                    
                    # Course metadata
                    if course_id in kb_system.course_metadata:
                        meta = kb_system.course_metadata[course_id]
                        st.write(f"**Course Name:** {meta.get('course_name', 'N/A')}")
                        st.write(f"**Indexed:** {datetime.fromtimestamp(meta.get('indexed_at', 0)).strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        if meta.get('learning_objectives'):
                            st.write(f"**Learning Objectives:** {meta['learning_objectives']}")
        else:
            st.info("No courses indexed yet.")

elif page == "💬 Ask Questions":
    st.title("💬 Ask Questions")
    
    if not selected_course:
        st.warning("⚠️ Please select a course from the sidebar first.")
    else:
        st.markdown(f"**Current Course:** {selected_course}")
        
        # Display conversation history
        if st.session_state.conversation_history:
            st.subheader("Conversation History")
            for i, turn in enumerate(st.session_state.conversation_history):
                with st.chat_message("user"):
                    st.write(turn['question'])
                
                with st.chat_message("assistant"):
                    st.markdown(turn['response'])
                    
                    # Show route type badge
                    route_color = {
                        'quick-answer': '🟢',
                        'tutorial': '🔵',
                        'deep-dive': '🟣',
                        'mock-interview': '🟡'
                    }
                    st.caption(f"{route_color.get(turn.get('route_type', ''), '⚪')} {turn.get('route_type', 'unknown')}")
                    
                    # Citations in expander
                    if turn.get('citations'):
                        with st.expander("📚 Sources"):
                            for citation in turn['citations']:
                                st.caption(citation)
        
        # Question input
        st.markdown("---")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            question = st.text_input(
                "Your Question",
                placeholder="What would you like to know?",
                label_visibility="collapsed",
                key="question_input"
            )
        
        with col2:
            student_id = st.text_input("Student ID", value="student_001", key="student_id")
        
        if st.button("🚀 Ask", type="primary", disabled=not question):
            if question:
                with st.spinner("Thinking... 🤔"):
                    try:
                        response = kb_system.ask_question(
                            student_id=student_id,
                            question=question,
                            course_id=selected_course,
                            session_id=st.session_state.current_session_id
                        )
                        
                        # Update session
                        if not st.session_state.current_session_id:
                            st.session_state.current_session_id = response['session_id']
                        
                        # Add to history
                        st.session_state.conversation_history.append({
                            'question': question,
                            'response': response['response'],
                            'route_type': response['route_type'],
                            'citations': response.get('citations', []),
                            'learn_more': response.get('learn_more_suggestions', [])
                        })
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        with st.expander("Error Details"):
                            st.exception(e)
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.conversation_history = []
            st.session_state.current_session_id = None
            st.rerun()

elif page == "📊 Analytics":
    st.title("📊 Course Analytics")
    
    if not selected_course:
        st.warning("⚠️ Please select a course from the sidebar first.")
    else:
        try:
            analytics = kb_system.analyze_course_questions(selected_course)
            questions = db.get_all_questions(selected_course)
            
            # Overview metrics
            st.subheader("Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Total Questions", len(questions))
            
            route_dist = analytics.get('route_distribution', {})
            most_common_route = max(route_dist.items(), key=lambda x: x[1])[0] if route_dist else "N/A"
            col2.metric("Most Used Route", most_common_route)
            
            wordcloud_data = analytics.get('wordcloud_data', {})
            col3.metric("Unique Keywords", len(wordcloud_data))
            
            engagement = analytics.get('student_engagement', [])
            col4.metric("Active Students", len(engagement))
            
            st.markdown("---")
            
            # Tabs for different analytics
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Route Distribution", "☁️ Word Cloud", "👥 Student Engagement", "🚨 Knowledge Gaps"])
            
            with tab1:
                st.subheader("Route Distribution")
                
                if route_dist:
                    # Create bar chart
                    df = pd.DataFrame(list(route_dist.items()), columns=['Route', 'Count'])
                    st.bar_chart(df.set_index('Route'))
                    
                    # Show percentages
                    total = sum(route_dist.values())
                    for route, count in route_dist.items():
                        pct = (count / total * 100) if total > 0 else 0
                        st.write(f"**{route}:** {count} questions ({pct:.1f}%)")
                else:
                    st.info("No route data available yet.")
            
            with tab2:
                st.subheader("Question Keywords Word Cloud")
                
                if wordcloud_data:
                    try:
                        from wordcloud import WordCloud
                        import matplotlib.pyplot as plt
                        
                        # Generate word cloud
                        wordcloud = WordCloud(
                            width=800,
                            height=400,
                            background_color='white',
                            colormap='viridis'
                        ).generate_from_frequencies(wordcloud_data)
                        
                        # Display
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                        
                        # Top keywords
                        st.subheader("Top Keywords")
                        top_words = sorted(wordcloud_data.items(), key=lambda x: x[1], reverse=True)[:20]
                        
                        col1, col2 = st.columns(2)
                        for i, (word, freq) in enumerate(top_words):
                            if i < 10:
                                col1.write(f"{i+1}. **{word}** ({freq})")
                            else:
                                col2.write(f"{i+1}. **{word}** ({freq})")
                        
                    except ImportError:
                        st.warning("Install wordcloud and matplotlib to see visualization: `pip install wordcloud matplotlib`")
                        
                        # Show as table instead
                        top_words = sorted(wordcloud_data.items(), key=lambda x: x[1], reverse=True)[:30]
                        df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
                        st.dataframe(df, use_container_width=True)
                else:
                    st.info("No word cloud data available yet.")
            
            with tab3:
                st.subheader("Student Engagement")
                
                if engagement:
                    df = pd.DataFrame(engagement)
                    df = df[['student_id', 'question_count', 'avg_quality']]
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No engagement data available yet.")
            
            with tab4:
                st.subheader("Knowledge Gaps")
                
                gaps = analytics.get('knowledge_gaps', [])
                if gaps:
                    for gap in gaps:
                        with st.expander(f"❓ {gap['question_sample'][:80]}..."):
                            st.write(f"**Occurrences:** {gap['occurrences']}")
                            st.write(f"**Avg Chunks Retrieved:** {gap['avg_chunks_retrieved']:.2f}")
                            st.caption("This question pattern had low retrieval quality - consider adding content.")
                else:
                    st.success("No knowledge gaps detected!")
            
            # Export button
            st.markdown("---")
            if st.button("📥 Export Analytics"):
                with st.spinner("Exporting..."):
                    output_dir = "./data/analytics"
                    kb_system.export_analytics(selected_course, output_dir)
                    st.success(f"✅ Analytics exported to {output_dir}/")
        
        except Exception as e:
            st.error(f"❌ Error loading analytics: {str(e)}")
            with st.expander("Error Details"):
                st.exception(e)

elif page == "🐛 Debug View":
    st.title("🐛 Debug View")
    
    if not selected_course:
        st.warning("⚠️ Please select a course from the sidebar first.")
    else:
        tab1, tab2, tab3 = st.tabs(["Raw Chunks", "Recent Questions", "System Info"])
        
        with tab1:
            st.subheader("Raw Chunk Data")
            
            chunks = db.get_knowledge_chunks(selected_course)
            
            level = st.selectbox("Chunk Level", ["description", "header", "detail"])
            
            if chunks.get(level):
                chunk_list = chunks[level]
                st.write(f"**Total {level} chunks:** {len(chunk_list)}")
                
                # Show first few chunks
                for i, chunk in enumerate(chunk_list[:5]):
                    with st.expander(f"Chunk {i+1}: {chunk.get('chunk_id', 'N/A')}"):
                        st.json(chunk)
            else:
                st.info(f"No {level} chunks found.")
        
        with tab2:
            st.subheader("Recent Questions")
            
            questions = db.get_all_questions(selected_course)
            
            if questions:
                # Sort by timestamp
                questions.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
                
                for q in questions[:10]:
                    with st.expander(f"Q: {q.get('question_text', 'N/A')[:80]}..."):
                        st.write(f"**Question ID:** {q.get('question_id')}")
                        st.write(f"**Student ID:** {q.get('student_id')}")
                        st.write(f"**Route Type:** {q.get('route_type')}")
                        st.write(f"**Timestamp:** {datetime.fromtimestamp(q.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')}")
                        st.write(f"**Chunks Retrieved:** {len(q.get('retrieved_chunks', []))}")
                        
                        if q.get('retrieved_chunks'):
                            st.write("**Retrieved Chunk IDs:**")
                            st.code(", ".join(q['retrieved_chunks'][:10]))
            else:
                st.info("No questions recorded yet.")
        
        with tab3:
            st.subheader("System Information")
            
            st.write("**Configuration:**")
            from config import SearchConfig, GenerationConfig, EmbeddingConfig, RerankerConfig, LLMConfig
            
            config_info = {
                "Vector Weight": SearchConfig.VECTOR_WEIGHT,
                "BM25 Weight": SearchConfig.BM25_WEIGHT,
                "RRF K": SearchConfig.RRF_K,
                "Relevance Threshold": SearchConfig.RELEVANCE_THRESHOLD,
                "Max Context Tokens": GenerationConfig.MAX_CONTEXT_TOKENS,
                "Conversation History Turns": GenerationConfig.CONVERSATION_HISTORY_TURNS,
                "Embedding Model": EmbeddingConfig.MODEL_NAME,
                "Reranker Model": RerankerConfig.MODEL_NAME,
                "LLM Model": LLMConfig.MODEL_NAME
            }
            
            for key, value in config_info.items():
                st.write(f"**{key}:** `{value}`")
            
            st.markdown("---")
            st.write("**ChromaDB Stats:**")
            try:
                count = kb_system.vector_store.count_chunks(selected_course)
                st.write(f"**Total Vectors:** {count}")
            except Exception as e:
                st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.caption("Progressive Disclosure Search & Generation System | Built with Streamlit")
