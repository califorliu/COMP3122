"""
Hierarchical Document Chunker
Parses documents by heading structure and generates 3-level chunks:
- Description level: Auto-generated summaries (50-200 words)
- Header level: Heading text + first paragraph
- Detail level: Full content under each heading
"""
import re
import uuid
from typing import List, Dict, Optional
from llm_client import call_moonshot_json


class HierarchicalChunker:
    def __init__(self):
        self.chunk_counter = 0
    
    def parse_headings(self, text: str) -> List[Dict]:
        """
        Parse markdown text into hierarchical structure by headings.
        Returns list of heading nodes with metadata.
        """
        lines = text.split('\n')
        headings = []
        current_content = []
        current_heading = None
        
        for line in lines:
            # Match markdown headings (# H1, ## H2, ### H3, etc.)
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            
            if heading_match:
                # Save previous heading's content
                if current_heading:
                    current_heading['content'] = '\n'.join(current_content).strip()
                    headings.append(current_heading)
                
                # Start new heading
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()
                
                current_heading = {
                    'level': level,
                    'heading': heading_text,
                    'content': '',
                    'line_number': len(headings)
                }
                current_content = []
            else:
                # Accumulate content under current heading
                if line.strip():
                    current_content.append(line)
        
        # Don't forget the last heading
        if current_heading:
            current_heading['content'] = '\n'.join(current_content).strip()
            headings.append(current_heading)
        
        return self._build_hierarchy(headings)
    
    def _build_hierarchy(self, headings: List[Dict]) -> List[Dict]:
        """
        Build parent-child relationships between headings.
        """
        if not headings:
            return []
        
        root_nodes = []
        stack = []  # Track parent headings at each level
        
        for heading in headings:
            level = heading['level']
            
            # Pop stack until we find the parent level
            while stack and stack[-1]['level'] >= level:
                stack.pop()
            
            # Set parent if exists
            if stack:
                parent = stack[-1]
                heading['parent_id'] = parent.get('chunk_id')
                heading['heading_path'] = f"{parent.get('heading_path', '')} > {heading['heading']}"
                if 'children' not in parent:
                    parent['children'] = []
                parent['children'].append(heading)
            else:
                heading['parent_id'] = None
                heading['heading_path'] = heading['heading']
                root_nodes.append(heading)
            
            # Generate unique chunk_id
            heading['chunk_id'] = f"chunk_{uuid.uuid4().hex[:8]}"
            stack.append(heading)
        
        return root_nodes
    
    def chunk_by_level(self, heading_tree: List[Dict], course_id: str) -> Dict[str, List[Dict]]:
        """
        Generate 3-level chunks from heading tree:
        - description: Course/module summaries
        - header: Heading + first paragraph
        - detail: Full content
        """
        chunks = {
            'description': [],
            'header': [],
            'detail': []
        }
        
        def process_node(node, parent_chunk_id=None):
            chunk_id = node['chunk_id']
            heading = node['heading']
            content = node['content']
            heading_path = node['heading_path']
            level = node['level']
            
            # Extract first paragraph for header level
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            first_paragraph = paragraphs[0] if paragraphs else content[:200]
            
            # Header-level chunk: heading + first paragraph
            header_chunk = {
                'chunk_id': chunk_id + '_header',
                'course_id': course_id,
                'chunk_level': 'header',
                'parent_chunk_id': parent_chunk_id,
                'heading_path': heading_path,
                'content': f"{heading}\n\n{first_paragraph}",
                'metadata': {
                    'heading_level': level,
                    'has_children': 'children' in node and len(node.get('children', [])) > 0
                }
            }
            chunks['header'].append(header_chunk)
            
            # Detail-level chunk: full content
            if content:
                detail_chunk = {
                    'chunk_id': chunk_id + '_detail',
                    'course_id': course_id,
                    'chunk_level': 'detail',
                    'parent_chunk_id': chunk_id + '_header',
                    'heading_path': heading_path,
                    'content': f"{heading}\n\n{content}",
                    'metadata': {
                        'heading_level': level,
                        'content_length': len(content)
                    }
                }
                chunks['detail'].append(detail_chunk)
            
            # Recursively process children
            for child in node.get('children', []):
                process_node(child, parent_chunk_id=chunk_id + '_header')
        
        # Process all root nodes
        for root in heading_tree:
            process_node(root)
        
        return chunks
    
    def generate_descriptions(self, chunks: Dict[str, List[Dict]], course_name: str = "Unknown Course") -> List[Dict]:
        """
        Generate description-level summaries using LLM.
        Groups content by top-level sections (H1 or H2).
        """
        descriptions = []
        
        # Group header chunks by top-level headings (H1/H2)
        top_level_groups = {}
        for header_chunk in chunks['header']:
            if header_chunk['metadata']['heading_level'] in [1, 2]:
                heading_path = header_chunk['heading_path']
                if heading_path not in top_level_groups:
                    top_level_groups[heading_path] = []
                top_level_groups[heading_path].append(header_chunk)
        
        # Generate description for each top-level group
        for heading_path, group_chunks in top_level_groups.items():
            # Collect content from this section
            section_content = '\n\n'.join([c['content'] for c in group_chunks[:5]])  # Limit to first 5 chunks
            
            # Generate summary using LLM
            system_prompt = (
                "You are a technical summarizer. Generate a concise summary (50-200 words) "
                "that captures the key concepts and learning objectives of this course section. "
                "Return ONLY a JSON object with a 'summary' field containing the summary text. "
                "Example: {\"summary\": \"Your summary here\"}"
            )
            user_prompt = f"Course: {course_name}\nSection: {heading_path}\n\nContent:\n{section_content[:2000]}\n\nReturn JSON with 'summary' field:"
            
            summary = ""
            try:
                result = call_moonshot_json(system_prompt, user_prompt)
                
                if result and isinstance(result, dict):
                    summary = result.get('summary', '')
                    
                # If no summary from LLM, use fallback
                if not summary:
                    print(f"[INFO] Using fallback summary for: {heading_path}")
                    # Fallback: extract first 200 chars
                    summary = section_content[:200].strip()
                    if len(section_content) > 200:
                        summary += "..."
                        
            except Exception as e:
                print(f"[WARNING] Failed to generate description for {heading_path}: {e}")
                # Fallback: extract first 200 chars
                summary = section_content[:200].strip()
                if len(section_content) > 200:
                    summary += "..."
            
            # Create description chunk
            desc_chunk = {
                'chunk_id': f"desc_{uuid.uuid4().hex[:8]}",
                'course_id': group_chunks[0]['course_id'],
                'chunk_level': 'description',
                'parent_chunk_id': None,
                'heading_path': heading_path,
                'content': f"{heading_path}\n\n{summary}",
                'metadata': {
                    'is_summary': True,
                    'covers_chunks': len(group_chunks)
                }
            }
            descriptions.append(desc_chunk)
        
        return descriptions
    
    def process_document(self, text: str, course_id: str, course_name: str = "Unknown Course") -> Dict[str, List[Dict]]:
        """
        Complete pipeline: parse → chunk → generate descriptions.
        Returns all three levels of chunks.
        """
        # Parse headings
        heading_tree = self.parse_headings(text)
        
        # Handle documents without headings (e.g., plain text PDFs)
        if not heading_tree:
            print("[INFO] No headings found. Creating chunks from plain text...")
            return self._process_plain_text(text, course_id, course_name)
        
        # Generate header and detail chunks
        chunks = self.chunk_by_level(heading_tree, course_id)
        
        # Generate description chunks
        descriptions = self.generate_descriptions(chunks, course_name)
        chunks['description'] = descriptions
        
        return chunks
    
    def _process_plain_text(self, text: str, course_id: str, course_name: str) -> Dict[str, List[Dict]]:
        """
        Process documents without headings by splitting into paragraphs.
        """
        # Split by paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            # Try splitting by newlines if no double-newlines found
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        if not paragraphs:
            print("[WARNING] No content found in document")
            return {'description': [], 'header': [], 'detail': []}
        
        chunks = {'description': [], 'header': [], 'detail': []}
        
        # Group paragraphs into chunks (e.g., 3-5 paragraphs per chunk)
        chunk_size = 5
        for i in range(0, len(paragraphs), chunk_size):
            chunk_paras = paragraphs[i:i+chunk_size]
            chunk_content = '\n\n'.join(chunk_paras)
            chunk_id = f"chunk_{uuid.uuid4().hex[:8]}"
            
            # First paragraph as header
            first_para = chunk_paras[0][:200] + ('...' if len(chunk_paras[0]) > 200 else '')
            
            # Header chunk
            header_chunk = {
                'chunk_id': chunk_id + '_header',
                'course_id': course_id,
                'chunk_level': 'header',
                'parent_chunk_id': None,
                'heading_path': f"Section {i//chunk_size + 1}",
                'content': first_para,
                'metadata': {
                    'heading_level': 1,
                    'has_children': True,
                    'is_plain_text': True
                }
            }
            chunks['header'].append(header_chunk)
            
            # Detail chunk
            detail_chunk = {
                'chunk_id': chunk_id + '_detail',
                'course_id': course_id,
                'chunk_level': 'detail',
                'parent_chunk_id': chunk_id + '_header',
                'heading_path': f"Section {i//chunk_size + 1}",
                'content': chunk_content,
                'metadata': {
                    'heading_level': 1,
                    'content_length': len(chunk_content),
                    'is_plain_text': True
                }
            }
            chunks['detail'].append(detail_chunk)
        
        # Create a single description for the whole document
        desc_content = text[:500] + ('...' if len(text) > 500 else '')
        desc_chunk = {
            'chunk_id': f"desc_{uuid.uuid4().hex[:8]}",
            'course_id': course_id,
            'chunk_level': 'description',
            'parent_chunk_id': None,
            'heading_path': course_name,
            'content': f"{course_name}\n\n{desc_content}",
            'metadata': {
                'is_summary': True,
                'covers_chunks': len(chunks['header']),
                'is_plain_text': True
            }
        }
        chunks['description'].append(desc_chunk)
        
        return chunks
