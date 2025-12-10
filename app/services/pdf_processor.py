"""PDF and document processing service."""

import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import PyPDF2

try:
    import docx
except ImportError:
    docx = None


class PDFProcessor:
    """Process PDF and DOCX files to extract text."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize PDF processor with optional cache directory."""
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                
                # Extract text from all pages
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page_text
                
                return text.strip()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return f"ERROR: Could not extract text from {pdf_path}"
    
    def extract_text_from_docx(self, docx_path: str) -> str:
        """
        Extract text from a DOCX file.
        
        Args:
            docx_path: Path to the DOCX file
            
        Returns:
            Extracted text as string
        """
        if docx is None:
            return "ERROR: python-docx not installed"
        
        try:
            doc = docx.Document(docx_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from {docx_path}: {e}")
            return f"ERROR: Could not extract text from {docx_path}"
    
    def extract_text_from_txt(self, txt_path: str) -> str:
        """
        Extract text from a TXT file.
        
        Args:
            txt_path: Path to the TXT file
            
        Returns:
            Extracted text as string
        """
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            print(f"Error reading {txt_path}: {e}")
            return f"ERROR: Could not read {txt_path}"
    
    def process_document(self, file_path: str) -> str:
        """
        Process a document and extract text based on file extension.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Extracted text as string
        """
        ext = Path(file_path).suffix.lower()
        
        if ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif ext == '.docx':
            return self.extract_text_from_docx(file_path)
        elif ext == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            return f"ERROR: Unsupported file type: {ext}"
    
    def process_directory(self, directory: str) -> List[Dict[str, str]]:
        """
        Process all PDF, DOCX, and TXT files in a directory.
        
        Args:
            directory: Path to directory containing documents
            
        Returns:
            List of dictionaries with filename and extracted text
        """
        directory_path = Path(directory)
        if not directory_path.exists():
            print(f"Error: Directory {directory} does not exist")
            return []
        
        documents = []
        supported_extensions = ['.pdf', '.docx', '.txt']
        
        # Get all supported files
        files = [f for f in directory_path.iterdir() 
                if f.is_file() and f.suffix.lower() in supported_extensions]
        
        print(f"Found {len(files)} document(s) in {directory}")
        
        for file_path in files:
            print(f"\nProcessing: {file_path.name}")
            text = self.process_document(str(file_path))
            
            documents.append({
                'filename': file_path.name,
                'filepath': str(file_path),
                'text': text,
                'text_length': len(text),
                'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            print(f"  Extracted {len(text)} characters")
        
        return documents
    
    def chunk_text(self, text: str, max_chunk_size: int = 2000) -> List[str]:
        """
        Split long text into smaller chunks.
        
        Args:
            text: Text to chunk
            max_chunk_size: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_chunk_size:
            return [text]
        
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1  # +1 for space
            if current_size + word_size > max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
