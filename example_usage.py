"""
Example Usage of the Progressive Disclosure Search & Generation System

This demonstrates how to use the new KnowledgeBaseSystem.
"""

from knowledge_base_system import KnowledgeBaseSystem

def main():
    print("=" * 70)
    print("Progressive Disclosure Search & Generation System - Demo")
    print("=" * 70)
    
    # Initialize the system
    print("\n[1] Initializing Knowledge Base System...")
    kb_system = KnowledgeBaseSystem()
    print("✓ System initialized successfully")
    
    # Example 1: Index course material
    print("\n[2] Indexing Course Material...")
    print("-" * 70)
    
    # Create a sample markdown file for testing
    sample_content = """# Introduction to Python Programming

Python is a high-level, interpreted programming language known for its simplicity and readability.

## Variables and Data Types

### Variables
Variables are containers for storing data values. In Python, you don't need to declare variable types explicitly.

Example:
```python
name = "Alice"
age = 25
is_student = True
```

### Data Types
Python has several built-in data types:
- int: Integer numbers
- float: Decimal numbers
- str: Text strings
- bool: True/False values
- list: Ordered collections
- dict: Key-value pairs

## Functions

Functions are reusable blocks of code that perform specific tasks.

### Defining Functions
Use the `def` keyword to define a function:

```python
def greet(name):
    return f"Hello, {name}!"
```

### Function Parameters
Functions can accept multiple parameters and return values.

## Control Flow

### If Statements
Conditional execution based on conditions:

```python
if age >= 18:
    print("Adult")
else:
    print("Minor")
```

### Loops
Python supports for and while loops for iteration.

#### For Loops
Iterate over sequences:

```python
for i in range(5):
    print(i)
```

#### While Loops
Repeat while a condition is true:

```python
count = 0
while count < 5:
    count += 1
```
"""
    
    # Save sample file
    import os
    from config import UPLOAD_DIR
    sample_file = os.path.join(UPLOAD_DIR, "python_basics.md")
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(sample_content)
    
    # Index the material
    result = kb_system.index_course_material(
        filename="python_basics.md",
        course_id="python101",
        course_name="Python Programming Basics",
        learning_objectives="Learn fundamental Python concepts including variables, functions, and control flow"
    )
    
    print(f"✓ Indexed {result['total_chunks']} chunks")
    print(f"  - Description level: {result['chunks_by_level']['description']} chunks")
    print(f"  - Header level: {result['chunks_by_level']['header']} chunks")
    print(f"  - Detail level: {result['chunks_by_level']['detail']} chunks")
    print(f"  - Time: {result['elapsed_time']:.2f}s")
    
    # Example 2: Ask questions with different routes
    print("\n[3] Asking Questions...")
    print("-" * 70)
    
    # Question 1: Quick Answer
    print("\nQ1: What is Python? (Expected: Quick Answer)")
    response1 = kb_system.ask_question(
        student_id="student_demo",
        question="What is Python?",
        course_id="python101"
    )
    print(f"Route: {response1['route_type']}")
    print(f"Chunks retrieved: {response1['chunks_retrieved']}")
    print(f"Response preview: {response1['response'][:200]}...")
    
    # Question 2: Tutorial
    print("\n\nQ2: How do I define a function? (Expected: Tutorial)")
    response2 = kb_system.ask_question(
        student_id="student_demo",
        question="How do I define a function in Python?",
        course_id="python101",
        session_id=response1['session_id']  # Same session for multi-turn
    )
    print(f"Route: {response2['route_type']}")
    print(f"Chunks retrieved: {response2['chunks_retrieved']}")
    print(f"Is follow-up: {response2['is_followup']}")
    print(f"Response preview: {response2['response'][:200]}...")
    
    # Question 3: Deep Dive
    print("\n\nQ3: Explain control flow in detail (Expected: Deep Dive)")
    response3 = kb_system.ask_question(
        student_id="student_demo",
        question="Explain control flow structures in Python with examples",
        course_id="python101"
    )
    print(f"Route: {response3['route_type']}")
    print(f"Chunks retrieved: {response3['chunks_retrieved']}")
    print(f"Retrieval stages: {len(response3['retrieval_path'])}")
    print(f"Response preview: {response3['response'][:200]}...")
    
    # Show citations
    if response3['citations']:
        print(f"\nCitations:")
        for citation in response3['citations'][:3]:
            print(f"  - {citation}")
    
    # Show learn more suggestions
    if response3['learn_more_suggestions']:
        print(f"\nLearn More:")
        for suggestion in response3['learn_more_suggestions'][:3]:
            print(f"  - {suggestion}")
    
    # Example 3: Analytics
    print("\n\n[4] Analytics Dashboard...")
    print("-" * 70)
    
    analytics = kb_system.analyze_course_questions("python101")
    
    print(f"\nTotal questions tracked: {len(kb_system.db.get_all_questions('python101'))}")
    print(f"\nRoute distribution:")
    for route, count in analytics['route_distribution'].items():
        print(f"  - {route}: {count}")
    
    print(f"\nWord cloud top words:")
    for word, freq in list(analytics['wordcloud_data'].items())[:10]:
        print(f"  - {word}: {freq}")
    
    # Example 4: Export analytics
    print("\n\n[5] Exporting Analytics...")
    print("-" * 70)
    
    try:
        kb_system.export_analytics(
            course_id="python101",
            output_dir="./data/analytics"
        )
        print("✓ Analytics exported to ./data/analytics/")
        print("  - python101_analytics.json")
        print("  - python101_questions.csv")
        print("  - python101_engagement.csv")
        print("  - python101_gaps.csv")
    except Exception as e:
        print(f"✗ Analytics export failed: {e}")
        print("  (This is OK - you can still use the system)")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Check ./data/analytics/ for exported analytics")
    print("2. Upload your own course materials to ./upload_file/")
    print("3. Use kb_system.index_course_material() to index them")
    print("4. Start asking questions with kb_system.ask_question()")
    print("\nFor more info, see IMPLEMENTATION_COMPLETE.md")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\nCheck DEBUGGING.md for troubleshooting help.")
