import os
import pdfplumber
from docx import Document

class FileProcessor:
    @staticmethod
    def process_file(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif ext == '.pdf':
                text = ""
                with pdfplumber.open(file_path) as pdf:
                    num_pages = len(pdf.pages)
                    if num_pages == 0:
                        print(f"Warning: PDF {file_path} has no pages")
                        return None
                    print(f"[INFO] Processing PDF with {num_pages} pages...")
                    for page_num, page in enumerate(pdf.pages, 1):
                        content = page.extract_text()
                        if content: 
                            text += content + "\n"
                            if page_num <= 2:  # Log first 2 pages
                                print(f"  Page {page_num}: {len(content)} characters extracted")
                if not text.strip():
                    print(f"Warning: No text extracted from PDF {file_path}")
                    return None
                print(f"[INFO] Total text extracted: {len(text)} characters")
                return text
            elif ext == '.docx':
                doc = Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])
            return None
        except Exception as e:
            print(f"解析文件 {file_path} 出错: {e}")
            return None