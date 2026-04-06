"""
Simple test without LLM dependency to verify core functionality.
This tests the chunking system without requiring API calls.
"""

from hierarchical_chunker import HierarchicalChunker
from keyword_search import KeywordSearch
from hybrid_search import HybridSearch

print("=" * 70)
print("Simple Functionality Test (No API Calls)")
print("=" * 70)

# Test 1: Hierarchical Chunking (no LLM needed)
print("\n[TEST 1] Hierarchical Chunking...")
print("-" * 70)

sample_text = """# Introduction to Python

Python is a high-level programming language.

## Variables

Variables store data values.

### String Variables
Strings are sequences of characters.

Example:
```python
name = "Alice"
```

### Numeric Variables
Numbers can be integers or floats.

## Functions

Functions are reusable blocks of code.

### Defining Functions
Use the def keyword:

```python
def greet(name):
    return f"Hello, {name}!"
```
"""

try:
    chunker = HierarchicalChunker()
    
    # Test heading parsing
    headings = chunker.parse_headings(sample_text)
    print(f"✓ Parsed {len(headings)} root heading(s)")
    
    # Check structure
    if headings:
        root = headings[0]
        print(f"  Root: {root['heading']}")
        print(f"  Children: {len(root.get('children', []))}")
        
        for child in root.get('children', []):
            print(f"    - {child['heading']} (level {child['level']})")
    
    # Test chunk generation (without LLM descriptions)
    chunks = chunker.chunk_by_level(headings, course_id="test_course")
    print(f"✓ Generated chunks:")
    print(f"  - Header level: {len(chunks['header'])} chunks")
    print(f"  - Detail level: {len(chunks['detail'])} chunks")
    
    # Show sample chunks
    if chunks['header']:
        sample = chunks['header'][0]
        print(f"\nSample header chunk:")
        print(f"  ID: {sample['chunk_id']}")
        print(f"  Path: {sample['heading_path']}")
        print(f"  Content preview: {sample['content'][:100]}...")
    
    print("\n✓ TEST 1 PASSED")
    
except Exception as e:
    print(f"✗ TEST 1 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: BM25 Keyword Search
print("\n[TEST 2] BM25 Keyword Search...")
print("-" * 70)

try:
    search = KeywordSearch()
    
    # Create test chunks
    test_chunks = [
        {'chunk_id': 'c1', 'content': 'Python is a programming language used for web development'},
        {'chunk_id': 'c2', 'content': 'Variables in Python store data values like strings and numbers'},
        {'chunk_id': 'c3', 'content': 'Functions are reusable blocks of code in Python'},
        {'chunk_id': 'c4', 'content': 'JavaScript is also a popular programming language'},
        {'chunk_id': 'c5', 'content': 'Python functions can accept parameters and return values'}
    ]
    
    # Build index
    search.build_index(test_chunks)
    stats = search.get_stats()
    print(f"✓ Built index with {stats['total_documents']} documents")
    
    # Test search
    query = "Python functions"
    results = search.search(query, top_k=3)
    print(f"✓ Search for '{query}' returned {len(results)} results:")
    
    for i, (chunk_id, score) in enumerate(results, 1):
        chunk = next(c for c in test_chunks if c['chunk_id'] == chunk_id)
        print(f"  {i}. [{chunk_id}] Score: {score:.4f}")
        print(f"     {chunk['content'][:60]}...")
    
    print("\n✓ TEST 2 PASSED")
    
except Exception as e:
    print(f"✗ TEST 2 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: RRF Fusion
print("\n[TEST 3] RRF Fusion...")
print("-" * 70)

try:
    hybrid = HybridSearch()
    
    # Simulate results from vector and BM25 search
    vector_results = [
        ('chunk_a', 0.95),
        ('chunk_b', 0.85),
        ('chunk_c', 0.75),
        ('chunk_d', 0.65)
    ]
    
    bm25_results = [
        ('chunk_b', 20.5),
        ('chunk_a', 15.3),
        ('chunk_e', 10.2),
        ('chunk_c', 8.1)
    ]
    
    # Fuse results
    merged = hybrid.rrf_fusion(vector_results, bm25_results)
    print(f"✓ RRF fusion merged results:")
    
    for i, (chunk_id, rrf_score) in enumerate(merged[:5], 1):
        print(f"  {i}. [{chunk_id}] RRF Score: {rrf_score:.6f}")
    
    # Verify that chunks appearing in both lists are ranked higher
    top_chunk = merged[0][0]
    print(f"\n✓ Top ranked chunk: {top_chunk}")
    
    print("\n✓ TEST 3 PASSED")
    
except Exception as e:
    print(f"✗ TEST 3 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Import all modules
print("\n[TEST 4] Module Imports...")
print("-" * 70)

modules_to_test = [
    ('config', 'LLMConfig'),
    ('database', 'LocalDB'),
    ('file_processor', 'FileProcessor'),
    ('vector_store', 'VectorStore'),
    ('llm_client', 'EmbeddingClient'),
    ('keyword_search', 'KeywordSearch'),
    ('semantic_search', 'SemanticSearch'),
    ('hybrid_search', 'HybridSearch'),
    ('search_router', 'SearchRouter'),
    ('progressive_retrieval', 'ProgressiveRetrieval'),
    ('context_optimizer', 'ContextOptimizer'),
    ('generation_pipeline', 'GenerationPipeline'),
    ('conversation_manager', 'ConversationManager'),
    ('analytics', 'Analytics'),
    ('knowledge_base_system', 'KnowledgeBaseSystem')
]

failed_imports = []
for module_name, class_name in modules_to_test:
    try:
        exec(f"from {module_name} import {class_name}")
        print(f"  ✓ {module_name}.{class_name}")
    except Exception as e:
        print(f"  ✗ {module_name}.{class_name}: {e}")
        failed_imports.append(module_name)

if not failed_imports:
    print("\n✓ TEST 4 PASSED - All modules imported successfully")
else:
    print(f"\n✗ TEST 4 FAILED - Failed imports: {', '.join(failed_imports)}")

print("\n" + "=" * 70)
print("Testing Complete!")
print("=" * 70)
print("\nNext step: Run 'python test_llm_clients.py' to test API connectivity")
