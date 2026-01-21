#!/usr/bin/env python3
"""
Run Multi-Agent RAG Evaluation

Usage:
    python run_multi_agent_rag.py "What are the HbA1c requirements?"
    python run_multi_agent_rag.py "What are the HbA1c requirements?" --model llama
    python run_multi_agent_rag.py "What are the HbA1c requirements?" --top-k 15

Results are automatically saved to: data/output/rag_evaluation_TIMESTAMP.txt
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-Agent RAG Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "What are the HbA1c requirements for diabetes trials?"
  %(prog)s "What are the HbA1c requirements?" --model llama
  %(prog)s "What are the HbA1c requirements?" --top-k 15
  
Results are automatically saved to: data/output/
        """
    )
    
    parser.add_argument(
        "query",
        type=str,
        help="The question to answer using medical protocols"
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
    
    args = parser.parse_args()
    
    # Import and run
    from app.agents.multi_agent_rag import MultiAgentRAG
    
    rag = MultiAgentRAG(llm_model=args.model, top_k=args.top_k)
    result = rag.evaluate(args.query)
    
    # Print final response
    print("\n" + "="*70)
    print("📝 FINAL RESPONSE:")
    print("="*70 + "\n")
    print(result['generated_response'])
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
