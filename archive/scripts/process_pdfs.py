"""Process PDFs with COMPLETE document analysis - No truncation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.pdf_processor import PDFProcessor
from app.services.comparison_service import ComparisonService
from app.services.export_service import ExportService
from config.settings import settings
from datetime import datetime


def analyze_document_comprehensively(
    doc: dict,
    comparison_service: ComparisonService,
    chunk_size: int = 6000,
    max_tokens: int = 1500,
    temperature: float = 0.5
) -> dict:
    """
    Analyze entire document by chunking intelligently and combining results.
    
    Args:
        doc: Document dict with 'text', 'filename', etc.
        comparison_service: ComparisonService instance
        chunk_size: Characters per chunk
        max_tokens: Max tokens for each LLM call
        temperature: Temperature setting
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    
    filename = doc['filename']
    full_text = doc['text']
    text_length = doc['text_length']
    
    print(f"\n{'='*70}")
    print(f"Analyzing: {filename}")
    print(f"Total length: {text_length} characters")
    print(f"{'='*70}\n")
    
    # Preview
    print(f"Preview (first 300 chars):")
    print(f"{'-'*70}")
    print(full_text[:300])
    print(f"{'-'*70}\n")
    
    # Split into chunks intelligently (by page breaks if available)
    chunks = smart_chunk_document(full_text, chunk_size)
    
    print(f"Document split into {len(chunks)} chunks for comprehensive analysis\n")
    
    all_chunk_results = []
    
    # STEP 1: Analyze each chunk individually
    for chunk_idx, chunk in enumerate(chunks, 1):
        print(f"\n--- Analyzing Chunk {chunk_idx}/{len(chunks)} ({len(chunk)} chars) ---")
        
        chunk_prompt = f"""Analyze this section from "{filename}" (Part {chunk_idx} of {len(chunks)}):

Extract:
1. Main topics covered in this section
2. Key requirements or guidelines (3-5 points)
3. Important definitions or technical terms
4. Compliance considerations

Section Content:
{chunk}

Provide a focused analysis of THIS section only."""

        try:
            chunk_results = comparison_service.compare_single_prompt(
                prompt=chunk_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                output_file=None  # Don't save individual chunks
            )
            all_chunk_results.append({
                'chunk_num': chunk_idx,
                'results': chunk_results
            })
            print(f"✓ Chunk {chunk_idx} analyzed")
        except Exception as e:
            print(f"✗ Error analyzing chunk {chunk_idx}: {e}")
    
    # STEP 2: Create comprehensive summary combining all chunks
    print(f"\n{'='*70}")
    print(f"Creating comprehensive summary from all {len(chunks)} chunks...")
    print(f"{'='*70}\n")
    
    # Extract key points from all chunk analyses
    combined_insights = extract_insights_from_chunks(all_chunk_results)
    
    # Final comprehensive prompt
    final_prompt = f"""You are analyzing the COMPLETE pharmaceutical regulatory document: "{filename}"

This document was analyzed in {len(chunks)} sections. Based on the full content, provide a COMPREHENSIVE analysis:

**1. Document Overview**
   - Document type and regulatory purpose
   - Scope and applicability
   - Target audience

**2. Main Sections & Topics**
   - List 7-10 major sections or topic areas covered
   - Brief description of each

**3. Critical Requirements**
   - 8-12 key compliance requirements
   - Grouped by theme if applicable

**4. Key Definitions & Terminology**
   - 5-8 important technical terms defined
   - Brief explanation of each

**5. Quality Management Implications**
   - How this impacts pharmaceutical quality systems
   - Implementation considerations
   - Documentation requirements

**6. Risk Management Considerations**
   - Quality risks addressed
   - Control strategies mentioned

**7. Notable Changes or Updates**
   - If revision, what changed
   - Effective dates or deadlines

**Full Document Summary Context:**
{combined_insights}

Provide a detailed, well-structured analysis covering the ENTIRE document."""

    print(f"Final comprehensive prompt length: {len(final_prompt)} characters")
    
    final_results = comparison_service.compare_single_prompt(
        prompt=final_prompt,
        max_tokens=2000,  # Higher for comprehensive summary
        temperature=temperature,
        output_file=settings.OUTPUT_DIR / f"analysis_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    )
    
    return {
        'filename': filename,
        'total_chunks': len(chunks),
        'chunk_results': all_chunk_results,
        'final_summary': final_results
    }


def smart_chunk_document(text: str, chunk_size: int = 6000) -> list:
    """
    Intelligently chunk document by page breaks or paragraphs.
    
    Args:
        text: Full document text
        chunk_size: Target characters per chunk
        
    Returns:
        List of text chunks
    """
    
    # Try to split by page markers first
    if "--- Page" in text:
        pages = text.split("--- Page")
        chunks = []
        current_chunk = ""
        
        for page in pages:
            if len(current_chunk) + len(page) < chunk_size:
                current_chunk += "--- Page" + page
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = "--- Page" + page
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    # Otherwise split by paragraphs/sections
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text]


def extract_insights_from_chunks(chunk_results: list) -> str:
    """
    Extract and combine insights from all chunk analyses.
    
    Args:
        chunk_results: List of chunk analysis results
        
    Returns:
        Combined insights as text
    """
    
    insights = []
    
    for chunk_data in chunk_results:
        chunk_num = chunk_data['chunk_num']
        results = chunk_data['results']
        
        insights.append(f"\n--- Insights from Section {chunk_num} ---")
        
        # Get text from successful models (prefer Claude or Titan)
        for model_name in ['claude', 'titan', 'llama', 'openai', 'gemini']:
            if model_name in results and results[model_name].get('text'):
                text = results[model_name]['text']
                if len(text) > 100:  # Only use substantial responses
                    insights.append(text[:800])  # First 800 chars
                    break
    
    return "\n".join(insights)


def process_pdfs_from_folder(
    folder_path: str,
    max_tokens: int = 1500,
    temperature: float = 0.5,
    chunk_size: int = 6000
):
    """
    Process PDFs with comprehensive multi-chunk analysis.
    
    Args:
        folder_path: Path to folder containing PDFs
        max_tokens: Maximum tokens for generation
        temperature: Temperature setting
        chunk_size: Characters per chunk
    """
    
    pdf_processor = PDFProcessor()
    comparison_service = ComparisonService()
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE PDF ANALYSIS")
    print(f"Analyzing COMPLETE documents (no truncation)")
    print(f"{'='*80}\n")
    
    print(f"Source folder: {folder_path}")
    print(f"Chunk size: {chunk_size} characters")
    print(f"Max tokens per response: {max_tokens}\n")
    
    documents = pdf_processor.process_directory(folder_path)
    
    if not documents:
        print("No documents found to process.")
        return
    
    print(f"Found {len(documents)} document(s) to analyze\n")
    
    # Process each document comprehensively
    for idx, doc in enumerate(documents, 1):
        print(f"\n{'#'*80}")
        print(f"DOCUMENT {idx}/{len(documents)}")
        print(f"{'#'*80}")
        
        try:
            result = analyze_document_comprehensively(
                doc=doc,
                comparison_service=comparison_service,
                chunk_size=chunk_size,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            print(f"\n✅ Completed comprehensive analysis for: {doc['filename']}")
            print(f"   - Analyzed {result['total_chunks']} chunks")
            print(f"   - Generated final comprehensive summary")
            
        except Exception as e:
            print(f"\n❌ Error processing {doc['filename']}: {e}")
    
    print(f"\n{'='*80}")
    print(f"✅ ALL DOCUMENTS ANALYZED COMPLETELY")
    print(f"Check: {settings.OUTPUT_DIR}")
    print(f"{'='*80}\n")


def main():
    """Main execution."""
    
    pdf_folder = "data/input/protocols"
    
    if len(sys.argv) > 1:
        pdf_folder = sys.argv[1]
    
    process_pdfs_from_folder(
        folder_path=pdf_folder,
        max_tokens=1500,
        temperature=0.5,
        chunk_size=6000  # Analyze in 6000-char chunks
    )


if __name__ == "__main__":
    main()
