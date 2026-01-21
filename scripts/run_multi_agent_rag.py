#!/usr/bin/env python3
"""
Run Multi-Agent RAG Evaluation

Supports two backends:
- FAISS (local, default)
- Pinecone (cloud)

Usage:
    # FAISS (local)
    python run_multi_agent_rag.py "What are the HbA1c requirements?"
    
    # Pinecone (cloud)
    python run_multi_agent_rag.py "What are the HbA1c requirements?" --backend pinecone
    
    # Different LLM
    python run_multi_agent_rag.py "What are the HbA1c requirements?" --model llama

Results are automatically saved to: data/output/rag_evaluation_TIMESTAMP.txt
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env file
from dotenv import load_dotenv
load_dotenv(project_root / ".env")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-Agent RAG Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # FAISS (local, default)
  %(prog)s "What are the HbA1c requirements?"
  
  # Pinecone (cloud)
  %(prog)s "What are the HbA1c requirements?" --backend pinecone
  
  # Different LLM
  %(prog)s "What are the HbA1c requirements?" --model llama
  
Results are automatically saved to: data/output/
        """
    )
    
    parser.add_argument(
        "query",
        type=str,
        help="The question to answer using medical protocols"
    )
    
    parser.add_argument(
        "--backend", "-b",
        type=str,
        default="faiss",
        choices=["faiss", "pinecone"],
        help="Vector store backend (default: faiss)"
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
        "--index-name",
        type=str,
        default="medical-protocols",
        help="Pinecone index name (for pinecone backend)"
    )
    
    args = parser.parse_args()
    
    # Import and run
    from app.agents.multi_agent_rag import MultiAgentRAG
    
    rag = MultiAgentRAG(
        backend=args.backend,
        pinecone_index=args.index_name,
        llm_model=args.model,
        top_k=args.top_k
    )
    result = rag.evaluate(args.query)
    
    # Print final response
    print("\n" + "="*70)
    print("📝 FINAL RESPONSE:")
    print("="*70 + "\n")
    print(result['generated_response'])
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
