"""
Analytics Module
Data aggregation for question tracking, topic distribution, and word clouds.
"""
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from database import LocalDB


class Analytics:
    """Analytics for student questions and course engagement."""
    
    def __init__(self):
        self.db = LocalDB()
    
    def get_topic_distribution(
        self,
        course_id: str,
        time_window_hours: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Get topic distribution for pie chart visualization.
        
        Args:
            course_id: Course identifier
            time_window_hours: Optional time window in hours
        
        Returns:
            Dict {topic: question_count}
        """
        questions = self.db.get_all_questions(course_id)
        
        # Filter by time window if specified
        if time_window_hours:
            cutoff = datetime.now().timestamp() - (time_window_hours * 3600)
            questions = [q for q in questions if q.get('timestamp', 0) >= cutoff]
        
        # Count questions by topic (extracted from retrieved chunks)
        topic_counts = {}
        
        for question in questions:
            chunks = question.get('retrieved_chunks', [])
            
            # Extract topics from chunk IDs (simplified - would normally load chunk metadata)
            for chunk_id in chunks:
                # Use route_type as proxy for topic
                route = question.get('route_type', 'general')
                topic_counts[route] = topic_counts.get(route, 0) + 1
        
        return topic_counts
    
    def get_wordcloud_data(
        self,
        course_id: str,
        min_word_length: int = 3,
        top_n: int = 100
    ) -> Dict[str, int]:
        """
        Get word frequency data for word cloud.
        
        Args:
            course_id: Course identifier
            min_word_length: Minimum word length to include
            top_n: Number of top words to return
        
        Returns:
            Dict {word: frequency}
        """
        word_data = self.db.generate_wordcloud_data(course_id)
        
        # Filter by minimum length
        filtered = {
            word: count
            for word, count in word_data.items()
            if len(word) >= min_word_length
        }
        
        # Sort and take top N
        sorted_words = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_words[:top_n])
    
    def get_time_series(
        self,
        course_id: str,
        days: int = 7,
        interval_hours: int = 24
    ) -> List[Dict]:
        """
        Get question volume over time.
        
        Args:
            course_id: Course identifier
            days: Number of days to analyze
            interval_hours: Time bucket size in hours
        
        Returns:
            List of {timestamp, count, interval_start, interval_end}
        """
        questions = self.db.get_all_questions(course_id)
        
        # Calculate time buckets
        now = datetime.now()
        cutoff = (now - timedelta(days=days)).timestamp()
        interval_seconds = interval_hours * 3600
        
        # Filter questions within time window
        recent_questions = [q for q in questions if q.get('timestamp', 0) >= cutoff]
        
        # Create time buckets
        buckets = {}
        start_time = cutoff
        
        while start_time < now.timestamp():
            end_time = start_time + interval_seconds
            bucket_key = int(start_time)
            buckets[bucket_key] = {
                'interval_start': start_time,
                'interval_end': end_time,
                'count': 0,
                'timestamp': start_time
            }
            start_time = end_time
        
        # Count questions per bucket
        for question in recent_questions:
            q_time = question.get('timestamp', 0)
            bucket_key = int((q_time - cutoff) // interval_seconds) * interval_seconds + int(cutoff)
            if bucket_key in buckets:
                buckets[bucket_key]['count'] += 1
        
        return sorted(buckets.values(), key=lambda x: x['timestamp'])
    
    def get_student_engagement(
        self,
        course_id: str,
        top_n: int = 10
    ) -> List[Dict]:
        """
        Get student engagement ranking by question count.
        
        Args:
            course_id: Course identifier
            top_n: Number of top students to return
        
        Returns:
            List of {student_id, question_count, avg_quality}
        """
        questions = self.db.get_all_questions(course_id)
        
        # Group by student
        student_stats = {}
        for question in questions:
            student_id = question.get('student_id', 'unknown')
            
            if student_id not in student_stats:
                student_stats[student_id] = {
                    'student_id': student_id,
                    'question_count': 0,
                    'total_quality': 0,
                    'quality_ratings': 0
                }
            
            student_stats[student_id]['question_count'] += 1
            
            # Track quality ratings if available
            quality = question.get('response_quality')
            if quality is not None:
                student_stats[student_id]['total_quality'] += quality
                student_stats[student_id]['quality_ratings'] += 1
        
        # Calculate average quality
        for stats in student_stats.values():
            if stats['quality_ratings'] > 0:
                stats['avg_quality'] = stats['total_quality'] / stats['quality_ratings']
            else:
                stats['avg_quality'] = None
        
        # Sort by question count
        ranked = sorted(
            student_stats.values(),
            key=lambda x: x['question_count'],
            reverse=True
        )
        
        return ranked[:top_n]
    
    def get_route_distribution(self, course_id: str) -> Dict[str, int]:
        """
        Get distribution of route types used.
        
        Args:
            course_id: Course identifier
        
        Returns:
            Dict {route_type: count}
        """
        questions = self.db.get_all_questions(course_id)
        
        route_counts = {}
        for question in questions:
            route = question.get('route_type', 'unknown')
            route_counts[route] = route_counts.get(route, 0) + 1
        
        return route_counts
    
    def get_knowledge_gaps_report(
        self,
        course_id: str,
        min_occurrences: int = 2
    ) -> List[Dict]:
        """
        Identify knowledge gaps (questions with low retrieval quality).
        
        Args:
            course_id: Course identifier
            min_occurrences: Minimum times a gap must occur to report
        
        Returns:
            List of gap reports
        """
        gaps = self.db.get_knowledge_gaps(course_id)
        
        # Group similar questions
        gap_groups = {}
        for gap in gaps:
            # Simple grouping by first 50 chars (would use semantic similarity in production)
            key = gap['question_text'][:50].lower()
            
            if key not in gap_groups:
                gap_groups[key] = {
                    'question_sample': gap['question_text'],
                    'occurrences': 0,
                    'avg_chunks_retrieved': 0,
                    'timestamps': []
                }
            
            gap_groups[key]['occurrences'] += 1
            gap_groups[key]['avg_chunks_retrieved'] += gap['retrieved_chunks_count']
            gap_groups[key]['timestamps'].append(gap['timestamp'])
        
        # Calculate averages and filter
        gap_reports = []
        for group_data in gap_groups.values():
            if group_data['occurrences'] >= min_occurrences:
                group_data['avg_chunks_retrieved'] /= group_data['occurrences']
                gap_reports.append(group_data)
        
        # Sort by occurrence frequency
        gap_reports.sort(key=lambda x: x['occurrences'], reverse=True)
        
        return gap_reports
    
    def export_analytics_json(
        self,
        course_id: str,
        output_path: str
    ):
        """
        Export comprehensive analytics to JSON file.
        
        Args:
            course_id: Course identifier
            output_path: Path to output JSON file
        """
        analytics_data = {
            'course_id': course_id,
            'generated_at': datetime.now().isoformat(),
            'topic_distribution': self.get_topic_distribution(course_id),
            'wordcloud_data': self.get_wordcloud_data(course_id),
            'time_series_7d': self.get_time_series(course_id, days=7),
            'student_engagement': self.get_student_engagement(course_id),
            'route_distribution': self.get_route_distribution(course_id),
            'knowledge_gaps': self.get_knowledge_gaps_report(course_id),
            'common_questions': self.db.get_common_questions(course_id)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analytics_data, f, ensure_ascii=False, indent=2)
    
    def export_csv(
        self,
        course_id: str,
        output_path: str,
        data_type: str = 'questions'
    ):
        """
        Export analytics data to CSV format.
        
        Args:
            course_id: Course identifier
            output_path: Path to output CSV file
            data_type: Type of data ('questions', 'engagement', 'gaps')
        """
        import csv
        
        if data_type == 'questions':
            questions = self.db.get_all_questions(course_id)
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'question_id', 'student_id', 'question_text',
                    'route_type', 'timestamp', 'chunks_count'
                ])
                writer.writeheader()
                
                # Write only the specified fields
                for q in questions:
                    writer.writerow({
                        'question_id': q.get('question_id', ''),
                        'student_id': q.get('student_id', ''),
                        'question_text': q.get('question_text', ''),
                        'route_type': q.get('route_type', ''),
                        'timestamp': q.get('timestamp', ''),
                        'chunks_count': len(q.get('retrieved_chunks', []))
                    })
        
        elif data_type == 'engagement':
            engagement = self.get_student_engagement(course_id, top_n=100)
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'student_id', 'question_count', 'avg_quality'
                ])
                writer.writeheader()
                
                # Write only the fields we need (exclude internal fields)
                for row in engagement:
                    writer.writerow({
                        'student_id': row['student_id'],
                        'question_count': row['question_count'],
                        'avg_quality': row['avg_quality']
                    })
        
        elif data_type == 'gaps':
            gaps = self.get_knowledge_gaps_report(course_id)
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'question_sample', 'occurrences', 'avg_chunks_retrieved'
                ])
                writer.writeheader()
                
                # Write only the fields we need (exclude timestamps array)
                for gap in gaps:
                    writer.writerow({
                        'question_sample': gap['question_sample'],
                        'occurrences': gap['occurrences'],
                        'avg_chunks_retrieved': gap['avg_chunks_retrieved']
                    })
