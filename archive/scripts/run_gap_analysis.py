#!/usr/bin/env python3
"""
Run Gap Analysis with Multi-Agent RAG

Uses the 'clinical-rules' Pinecone index containing FDA guidelines and drug label rules.
Follows the same pattern as run_multi_agent_rag.py for consistency.

Usage:
    # Query against clinical rules (Pinecone)
    python scripts/run_gap_analysis.py "Does the protocol address thyroid cancer exclusion criteria?"
    
    # With different LLM
    python scripts/run_gap_analysis.py "Check HbA1c requirements" --model llama
    
    # With custom top-k
    python scripts/run_gap_analysis.py "Pancreatitis warnings" --top-k 15

Results are automatically saved to: data/output/gap_analysis_TIMESTAMP.txt
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env file
from dotenv import load_dotenv
load_dotenv(project_root / ".env")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Gap Analysis using Multi-Agent RAG with Clinical Rules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check protocol against FDA guidelines
  %(prog)s "Does the protocol include thyroid cancer exclusion criteria?"
  
  # Check specific drug label requirement
  %(prog)s "What are the pancreatitis monitoring requirements?"
  
  # Check inclusion/exclusion criteria compliance
  %(prog)s "Are there gaps in the exclusion criteria for hypersensitivity reactions?"
  
  # With different LLM
  %(prog)s "HbA1c requirements" --model llama
  
Results are automatically saved to: data/output/gap_analysis_TIMESTAMP.txt
        """
    )
    
    parser.add_argument(
        "query",
        type=str,
        help="The gap analysis question to check against FDA guidelines and drug label"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="claude",
        choices=["claude", "gpt-oss", "mistral", "llama", "titan"],
        help="LLM model to use (default: claude)"
    )
    
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=10,
        help="Number of sources after reranking (default: 10)"
    )
    
    parser.add_argument(
        "--initial-candidates", "-c",
        type=int,
        default=50,
        help="Number of initial candidates before reranking (default: 50)"
    )
    
    parser.add_argument(
        "--index-name",
        type=str,
        default="clinical-rules",
        help="Pinecone index name (default: clinical-rules)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("GAP ANALYSIS - MULTI-AGENT RAG")
    print("="*80)
    print(f"\n📋 Query: {args.query}")
    print(f"🔍 Index: {args.index_name} (Pinecone)")
    print(f"🤖 Model: {args.model}")
    print(f"📚 Top-K: {args.top_k}")
    
    # Import and run
    from app.agents.multi_agent_rag import MultiAgentRAG
    
    print("\n" + "-"*80)
    print("Initializing Multi-Agent RAG System...")
    print("-"*80)
    
    rag = MultiAgentRAG(
        backend="pinecone",
        pinecone_index=args.index_name,
        llm_model=args.model,
        top_k=args.top_k,
        initial_candidates=args.initial_candidates
    )
    
    print("\n" + "-"*80)
    print("Running 4-Agent Pipeline...")
    print("-"*80)
    
    result = rag.evaluate(args.query)
    
    # Print final response
    print("\n" + "="*80)
    print("📝 GAP ANALYSIS RESULT:")
    print("="*80 + "\n")
    print(result.get('generated_response', 'No response generated'))
    
    # Print chairman analysis
    if result.get('chairman_analysis'):
        print("\n" + "-"*80)
        print("🎖️ CHAIRMAN ASSESSMENT:")
        print("-"*80 + "\n")
        print(result['chairman_analysis'])
    
    # Print sources used
    if result.get('sources'):
        print("\n" + "-"*80)
        print(f"📚 SOURCES USED ({len(result['sources'])} rules):")
        print("-"*80)
        for i, src in enumerate(result['sources'][:5], 1):
            document = src.get('document', 'Unknown')
            page = src.get('page', 'N/A')
            chunk_id = src.get('chunk_id', 'N/A')
            text = src.get('paragraph_text', '')[:200]
            print(f"\n[{i}] {document} - Page {page}")
            print(f"    Chunk ID: {chunk_id}")
            print(f"    {text}...")
    
    print("\n" + "="*80)
    print(f"✅ Results saved to: data/output/")
    print("="*80 + "\n")
    
    return result


if __name__ == "__main__":
    main()
