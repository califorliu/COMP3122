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
                    if len(pdf.pages) == 0:
                        print(f"Warning: PDF {file_path} has no pages")
                        return None
                    for page in pdf.pages:
                        content = page.extract_text()
                        if content: 
                            text += content + "\n"
                if not text.strip():
                    print(f"Warning: No text extracted from PDF {file_path}")
                    return None
                return text
            elif ext == '.docx':
                doc = Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])
            return None
        except Exception as e:
            print(f"解析文件 {file_path} 出错: {e}")
            return None