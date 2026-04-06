"""
Quick test to verify CSV export fix
"""
import os
import sys

print("Testing CSV Export Fix...")
print("-" * 50)

# Test the analytics CSV export
from analytics import Analytics
from database import LocalDB

# Create a test question
db = LocalDB()
test_question = {
    'question_id': 'test_q1',
    'student_id': 'test_student',
    'course_id': 'test_course',
    'question_text': 'What is Python?',
    'timestamp': 1234567890.0,
    'route_type': 'quick-answer',
    'retrieved_chunks': ['chunk1', 'chunk2'],
    'response_quality': 4.5
}

db.save_question(test_question)

# Create analytics instance
analytics = Analytics()

# Test CSV exports
test_dir = "./test_csv_output"
os.makedirs(test_dir, exist_ok=True)

try:
    # Test 1: Questions CSV
    print("\n[1] Testing questions CSV export...")
    questions_csv = os.path.join(test_dir, "test_questions.csv")
    analytics.export_csv('test_course', questions_csv, 'questions')
    print(f"  ✓ Created: {questions_csv}")
    
    # Verify content
    with open(questions_csv, 'r', encoding='utf-8') as f:
        content = f.read()
        print(f"  Content preview:\n{content[:200]}")
    
    # Test 2: Engagement CSV
    print("\n[2] Testing engagement CSV export...")
    engagement_csv = os.path.join(test_dir, "test_engagement.csv")
    analytics.export_csv('test_course', engagement_csv, 'engagement')
    print(f"  ✓ Created: {engagement_csv}")
    
    # Verify content
    with open(engagement_csv, 'r', encoding='utf-8') as f:
        content = f.read()
        print(f"  Content preview:\n{content[:200]}")
    
    # Test 3: Gaps CSV
    print("\n[3] Testing gaps CSV export...")
    gaps_csv = os.path.join(test_dir, "test_gaps.csv")
    analytics.export_csv('test_course', gaps_csv, 'gaps')
    print(f"  ✓ Created: {gaps_csv}")
    
    # Verify content
    with open(gaps_csv, 'r', encoding='utf-8') as f:
        content = f.read()
        print(f"  Content preview:\n{content[:200]}")
    
    print("\n" + "=" * 50)
    print("✓ All CSV exports working correctly!")
    print("=" * 50)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Cleanup
import shutil
shutil.rmtree(test_dir)
print("\nCleanup complete.")
