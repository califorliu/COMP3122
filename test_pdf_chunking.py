"""
Test script to verify PDF chunking works correctly
"""
from hierarchical_chunker import HierarchicalChunker

# Test with plain text (simulating PDF content)
test_text = """
This is the first paragraph of the document. It contains some important information about the topic.
This paragraph continues with more details about the subject matter.

Here is a second paragraph with additional content. It provides more context and explanation.
The information here is crucial for understanding the overall concept.

Third paragraph introduces new ideas and concepts. These are important for the learning objectives.
We want to make sure all content is properly indexed and searchable.

Fourth paragraph contains more detailed information. This helps students understand complex topics.
The content should be broken into manageable chunks for better retrieval.

Fifth paragraph adds even more content to ensure we have enough text for testing.
We need to verify that the chunking algorithm works correctly with plain text.

Sixth paragraph continues the discussion with more examples and explanations.
This ensures comprehensive coverage of the topic.

Seventh paragraph provides additional context and details.
The chunking should handle this content appropriately.

Eighth paragraph demonstrates how longer documents are processed.
Each section should be searchable and retrievable.

Ninth paragraph adds more test content for verification.
The system needs to handle various document structures.

Tenth paragraph concludes the test content.
This should be split into multiple chunks for better organization.
""".strip()

print("=" * 60)
print("Testing Plain Text Chunking (PDF simulation)")
print("=" * 60)

chunker = HierarchicalChunker()
chunks = chunker.process_document(
    text=test_text,
    course_id="test_course",
    course_name="Test PDF Course"
)

print(f"\n📊 Chunking Results:")
print(f"  Description chunks: {len(chunks['description'])}")
print(f"  Header chunks: {len(chunks['header'])}")
print(f"  Detail chunks: {len(chunks['detail'])}")
print(f"  Total chunks: {sum(len(chunks[level]) for level in chunks)}")

print(f"\n📝 Description Chunks:")
for i, chunk in enumerate(chunks['description'], 1):
    print(f"\n  {i}. {chunk['heading_path']}")
    print(f"     Content preview: {chunk['content'][:100]}...")

print(f"\n📋 Header Chunks:")
for i, chunk in enumerate(chunks['header'], 1):
    print(f"\n  {i}. {chunk['heading_path']}")
    print(f"     Content preview: {chunk['content'][:80]}...")

print(f"\n📄 Detail Chunks:")
for i, chunk in enumerate(chunks['detail'], 1):
    print(f"\n  {i}. {chunk['heading_path']}")
    print(f"     Length: {len(chunk['content'])} chars")

# Test with markdown
test_markdown = """
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence. It focuses on building systems that learn from data.

## Supervised Learning

Supervised learning uses labeled data to train models. The algorithm learns patterns from input-output pairs.

### Classification

Classification assigns inputs to discrete categories. Examples include spam detection and image recognition.

### Regression

Regression predicts continuous values. It's used for price prediction and trend analysis.

## Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data. It discovers hidden structures automatically.
""".strip()

print("\n\n" + "=" * 60)
print("Testing Markdown Chunking")
print("=" * 60)

chunks_md = chunker.process_document(
    text=test_markdown,
    course_id="test_course_md",
    course_name="Test Markdown Course"
)

print(f"\n📊 Chunking Results:")
print(f"  Description chunks: {len(chunks_md['description'])}")
print(f"  Header chunks: {len(chunks_md['header'])}")
print(f"  Detail chunks: {len(chunks_md['detail'])}")
print(f"  Total chunks: {sum(len(chunks_md[level]) for level in chunks_md)}")

print(f"\n📝 Heading Paths:")
for chunk in chunks_md['header']:
    print(f"  - {chunk['heading_path']}")

print("\n✅ Test completed!")
