#!/usr/bin/env python3
"""
Medical Protocol RAG Query Tool

A command-line interface for querying medical protocol documents.

Usage:
    python run_medical_query.py "What are the HbA1c requirements?"
    python run_medical_query.py "What are the HbA1c requirements?" --model llama
    python run_medical_query.py "What are the HbA1c requirements?" --top-k 20

Examples:
    # Query with default settings (Claude, top 10 sources)
    python run_medical_query.py "What are the HbA1c measurement requirements?"
    
    # Query with specific model
    python run_medical_query.py "What are the HbA1c requirements?" --model gpt-oss
    
    # Get more sources
    python run_medical_query.py "What are the HbA1c requirements?" --top-k 20
    
    # Show all sources in detail
    python run_medical_query.py "What are the HbA1c requirements?" --verbose
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Query medical protocol documents using RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "What are the HbA1c requirements?"
  %(prog)s "What are the HbA1c requirements?" --model llama
  %(prog)s "What are the HbA1c requirements?" --top-k 20 --verbose
        """
    )
    
    parser.add_argument(
        "query",
        type=str,
        help="The question to ask about the medical protocols"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="claude",
        choices=["claude", "gpt-oss", "mistral", "llama", "titan"],
        help="LLM model to use for answer generation (default: claude)"
    )
    
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=10,
        help="Number of source documents to retrieve (default: 10)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show all sources in detail"
    )
    
    parser.add_argument(
        "--sources-only",
        action="store_true",
        help="Only show sources, skip answer generation"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Save results to a text file (e.g., --output results.txt)"
    )
    
    args = parser.parse_args()
    
    # Import RAG system
    from app.medical_rag_system import MedicalRAGSystem
    
    # Initialize
    print("\n" + "="*70)
    print("MEDICAL PROTOCOL RAG SYSTEM")
    print("="*70)
    
    rag = MedicalRAGSystem()
    
    # Prepare output (both for terminal and file)
    output_lines = []
    
    def add_output(text):
        """Add to both terminal and file output."""
        print(text)
        output_lines.append(text)
    
    # Run query
    if args.sources_only:
        # Just search and rerank, no LLM
        candidates = rag._search(args.query, top_k=50)
        reranked = rag._rerank(args.query, candidates, top_k=args.top_k)
        sources = rag._format_sources(reranked)
        
        add_output("\n" + "="*70)
        add_output(f"QUERY: {args.query}")
        add_output("="*70)
        add_output(f"\nTOP {len(sources)} SOURCES:\n")
        
        for source in sources:
            add_output(f"\n[{source.source_id}] {source.document} - Page {source.page}")
            add_output(f"    Chunk ID: {source.chunk_id}")
            add_output(f"    PARAGRAPH TEXT:")
            add_output("-" * 60)
            add_output(source.paragraph_text)
            add_output("-" * 60)
    else:
        # Full query with LLM
        result = rag.query(
            args.query,
            model=args.model,
            top_k=args.top_k,
        )
        
        # Output answer
        add_output("\n" + "="*70)
        add_output("ANSWER:")
        add_output("="*70)
        add_output(result.answer)
        
        # Output sources
        add_output("\n" + "="*70)
        add_output(f"SOURCES ({len(result.sources)} documents):")
        add_output("="*70)
        
        for source in result.sources:
            add_output(f"\n[{source.source_id}] {source.document} - Page {source.page}")
            add_output(f"    Chunk ID: {source.chunk_id}")
            add_output(f"    PARAGRAPH TEXT:")
            add_output("-" * 60)
            add_output(source.paragraph_text)
            add_output("-" * 60)
        
        add_output("\n" + "="*70)
        add_output(f"Model used: {result.model_used}")
        add_output("="*70 + "\n")
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_lines))
        print(f"\n✅ Results saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
