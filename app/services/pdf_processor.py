# app/services/pdf_processor.py
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

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None


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
    
    # ==================== NEW METHODS (For Vector RAG) ====================
    
    def extract_text_with_pages(self, pdf_path: str) -> List[Dict]:
        """
        Extract text from PDF with page-level metadata.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dicts with text and page metadata
        """
        pages_data = []
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(reader.pages, start=1):
                    page_text = page.extract_text()
                    
                    if page_text.strip():  # Only include non-empty pages
                        pages_data.append({
                            "text": page_text,
                            "page": page_num,
                            "source": Path(pdf_path).name,
                            "total_pages": len(reader.pages)
                        })
        
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return []
        
        return pages_data
    
    def chunk_text_with_metadata(
        self, 
        text: str, 
        page_num: int, 
        source: str,
        chunk_size: int = 1500,  # Increased for better context
        chunk_overlap: int = 300  # Increased overlap
    ) -> List[Dict]:
        """
        Chunk text with metadata preservation.
        
        IMPROVED: Splits by paragraph boundaries first to avoid cutting mid-sentence.
        
        Args:
            text: Text to chunk
            page_num: Page number
            source: Source document name
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of chunks with metadata
        """
        # First, split by paragraph boundaries (double newline or single newline)
        paragraphs = []
        for para in text.split('\n\n'):
            if para.strip():
                paragraphs.append(para.strip())
        
        # If no paragraphs found, split by single newline
        if len(paragraphs) <= 1:
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        # Now combine paragraphs into chunks, respecting size limits
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            # If single paragraph exceeds chunk_size, we have to split it
            if para_size > chunk_size:
                # Save current chunk first
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph by sentences (look for . followed by space or newline)
                import re
                sentences = re.split(r'(?<=[.!?])\s+', para)
                sentence_chunk = []
                sentence_size = 0
                
                for sentence in sentences:
                    if sentence_size + len(sentence) > chunk_size and sentence_chunk:
                        chunks.append(' '.join(sentence_chunk))
                        sentence_chunk = [sentence]
                        sentence_size = len(sentence)
                    else:
                        sentence_chunk.append(sentence)
                        sentence_size += len(sentence)
                
                if sentence_chunk:
                    chunks.append(' '.join(sentence_chunk))
            
            # If adding this paragraph exceeds the limit, save current chunk
            elif current_size + para_size > chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        # Create chunk metadata
        chunk_results = []
        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_results.append({
                "text": chunk_text,
                "page": page_num,
                "paragraph_number": chunk_idx + 1,  # Track which paragraph/chunk on this page
                "chunk_id": f"{source}_p{page_num}_c{chunk_idx}",
                "source": source
            })
        
        return chunk_results
    
    def process_pdf_with_chunks(self, pdf_path: str) -> List[Dict]:
        """
        Process PDF and create chunks with page metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of chunks with metadata
        """
        print(f"📄 Processing: {Path(pdf_path).name}")
        
        # Extract text with page info
        pages_data = self.extract_text_with_pages(pdf_path)
        
        if not pages_data:
            print(f"⚠️  No text extracted from {Path(pdf_path).name}")
            return []
        
        print(f"   ✅ Extracted {len(pages_data)} pages")
        
        # Chunk each page
        all_chunks = []
        for page_data in pages_data:
            chunks = self.chunk_text_with_metadata(
                text=page_data["text"],
                page_num=page_data["page"],
                source=page_data["source"]
            )
            all_chunks.extend(chunks)
        
        print(f"   ✅ Created {len(all_chunks)} chunks")
        
        return all_chunks
    
    def process_directory_with_chunks(self, directory: str) -> List[Dict]:
        """
        Process all PDFs in directory and create chunks with metadata.
        
        Args:
            directory: Path to directory containing PDFs
            
        Returns:
            List of all chunks from all PDFs with metadata
        """
        directory_path = Path(directory)
        if not directory_path.exists():
            print(f"Error: Directory {directory} does not exist")
            return []
        
        all_chunks = []
        pdf_files = list(directory_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f"⚠️  No PDF files found in {directory}")
            return []
        
        print(f"\n📚 Found {len(pdf_files)} PDF file(s) in {directory_path.name}/")
        print("="*70)
        
        for pdf_path in pdf_files:
            chunks = self.process_pdf_with_chunks(str(pdf_path))
            all_chunks.extend(chunks)
        
        print("="*70)
        print(f"✅ Total chunks created: {len(all_chunks)}\n")
        
        return all_chunks
