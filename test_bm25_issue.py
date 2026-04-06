"""Test script to reproduce the division by zero error with BM25"""
from rank_bm25 import BM25Okapi
import traceback

print("Testing BM25Okapi with various inputs...")

# Test 1: Empty document list
print("\n1. Testing with empty list of documents:")
try:
    bm25 = BM25Okapi([])
    print("   ✓ Empty list OK")
except Exception as e:
    print(f"   ✗ Error: {e}")
    traceback.print_exc()

# Test 2: Single empty document
print("\n2. Testing with single empty document:")
try:
    bm25 = BM25Okapi([[]])
    print("   ✓ Single empty doc OK")
except Exception as e:
    print(f"   ✗ Error: {e}")
    traceback.print_exc()

# Test 3: Multiple empty documents
print("\n3. Testing with multiple empty documents:")
try:
    bm25 = BM25Okapi([[], [], []])
    print("   ✓ Multiple empty docs OK")
except Exception as e:
    print(f"   ✗ Error: {e}")
    traceback.print_exc()

# Test 4: Normal documents
print("\n4. Testing with normal documents:")
try:
    bm25 = BM25Okapi([['hello', 'world'], ['test', 'document']])
    print("   ✓ Normal docs OK")
except Exception as e:
    print(f"   ✗ Error: {e}")
    traceback.print_exc()

print("\nDone!")
