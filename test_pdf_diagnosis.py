"""
Diagnostic script to check PDF text extraction and chunking
Run this to see how your PDF is being processed
"""
import os
import sys
from file_processor import FileProcessor
from hierarchical_chunker import HierarchicalChunker

def diagnose_pdf(pdf_filename):
    """Diagnose a PDF file in the upload_file directory"""
    
    file_path = os.path.join("./upload_file", pdf_filename)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        print(f"   Available files in upload_file/:")
        if os.path.exists("./upload_file"):
            for f in os.listdir("./upload_file"):
                print(f"   - {f}")
        return
    
    print("=" * 70)
    print(f"📄 Diagnosing: {pdf_filename}")
    print("=" * 70)
    
    # Step 1: Extract text
    print("\n[1] Extracting text from PDF...")
    content = FileProcessor.process_file(file_path)
    
    if not content:
        print("   ❌ No content extracted!")
        return
    
    print(f"   ✅ Extracted {len(content)} characters")
    print(f"   First 300 chars:")
    print(f"   {'-' * 70}")
    print(f"   {content[:300]}")
    print(f"   {'-' * 70}")
    
    # Step 2: Check for markdown headings
    print("\n[2] Checking for markdown headings...")
    heading_count = content.count('\n#')
    print(f"   Found {heading_count} potential markdown headings")
    
    if heading_count > 0:
        print("   Document will be processed as markdown")
    else:
        print("   Document will be processed as plain text")
    
    # Step 3: Test chunking
    print("\n[3] Testing chunking...")
    chunker = HierarchicalChunker()
    chunks = chunker.process_document(
        text=content,
        course_id="diagnostic_test",
        course_name=pdf_filename
    )
    
    # Step 4: Show results
    print("\n[4] Chunking Results:")
    print(f"   📊 Total chunks: {sum(len(chunks[level]) for level in chunks)}")
    print(f"   📝 Description chunks: {len(chunks['description'])}")
    print(f"   📋 Header chunks: {len(chunks['header'])}")
    print(f"   📄 Detail chunks: {len(chunks['detail'])}")
    
    if len(chunks['description']) > 0:
        print(f"\n   Description chunk examples:")
        for i, chunk in enumerate(chunks['description'][:3], 1):
            print(f"\n   {i}. {chunk['heading_path']}")
            print(f"      Preview: {chunk['content'][:150]}...")
    
    if len(chunks['header']) > 0:
        print(f"\n   Header chunk examples:")
        for i, chunk in enumerate(chunks['header'][:5], 1):
            print(f"\n   {i}. {chunk['heading_path']}")
            print(f"      Preview: {chunk['content'][:100]}...")
    
    # Step 5: Recommendations
    print("\n[5] Recommendations:")
    total = sum(len(chunks[level]) for level in chunks)
    if total < 5:
        print("   ⚠️  Very few chunks created. Possible issues:")
        print("      - PDF may have very little text")
        print("      - Text extraction may not be working properly")
        print("      - Consider checking if PDF is scanned (image-based)")
    elif total < 10:
        print("   ⚠️  Few chunks created. This is OK for small documents.")
    else:
        print("   ✅ Good number of chunks created!")
    
    print("\n" + "=" * 70)
    print("Diagnosis complete!")
    print("=" * 70)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_filename = sys.argv[1]
    else:
        # List available files
        print("Usage: python test_pdf_diagnosis.py <pdf_filename>")
        print("\nAvailable files in upload_file/:")
        if os.path.exists("./upload_file"):
            files = [f for f in os.listdir("./upload_file") if f.endswith('.pdf')]
            if files:
                for f in files:
                    print(f"  - {f}")
                print(f"\nExample: python test_pdf_diagnosis.py {files[0]}")
            else:
                print("  No PDF files found")
        else:
            print("  upload_file/ directory not found")
        sys.exit(0)
    
    diagnose_pdf(pdf_filename)
