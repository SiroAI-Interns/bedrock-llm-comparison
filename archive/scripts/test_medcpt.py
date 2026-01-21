# scripts/test_medcpt.py
"""
Test script to build MedCPT vector database and run evaluation.
Uses medical-specific embeddings optimized for clinical guidelines.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.agents import MultiAgentEvaluator


def main():
    """Build MedCPT index and test with medical query."""
    
    print("="*70)
    print("🏥 MedCPT VECTOR DATABASE BUILDER")
    print("="*70)
    print("This will build a medical-optimized vector database using:")
    print("  • ncbi/MedCPT-Article-Encoder (for documents)")
    print("  • ncbi/MedCPT-Query-Encoder (for queries)")
    print()
    print("📥 First run: Downloads ~840MB of models")
    print("⏱️  Subsequent runs: Uses cached models")
    print("="*70)
    print()
    
    # Define models to evaluate - UPDATED with GPT-OSS and Gemma
    models = {
        "Claude": ("anthropic.claude-3-5-sonnet-20240620-v1:0", "us-east-1"),
        "Llama": ("meta.llama3-70b-instruct-v1:0", "us-east-1"),
        "Mistral": ("mistral.mistral-7b-instruct-v0:2", "us-east-1"),
        "DeepSeek": ("us.deepseek.r1-v1:0", "us-west-2"),
        "GPT-OSS": ("openai.gpt-oss-20b-1:0", "us-east-1"),        # ← ADDED
        "Gemma": ("google.gemma-3-27b-it", "us-east-1"),           # ← ADDED
        "Titan": ("amazon.titan-text-express-v1", "us-east-1"),   # ← ADDED (bonus)
    }
    
    # Build vector database with MedCPT
    print(f"🔨 Building vector database from PDFs in data/input/protocols/...")
    print(f"📊 Will evaluate with {len(models)} models\n")
    
    evaluator = MultiAgentEvaluator(models, rebuild_index=True)
    
    # Test queries related to FDA guidelines
    test_queries = [
        "What are the HbA1c measurement requirements for diabetes device clinical trials?",
        "What sample size does FDA recommend for early feasibility studies?",
        "What are the safety endpoint requirements for T2DM device trials?",
    ]
    
    print("\n" + "="*70)
    print("🧪 TESTING MEDCPT RETRIEVAL")
    print("="*70)
    print(f"Running {len(test_queries)} test queries...\n")
    
    # Run first query with full evaluation
    print("="*70)
    print(f"📝 Query 1: {test_queries[0]}")
    print("="*70)
    
    result = evaluator.evaluate(test_queries[0])
    
    # Save result
    output_dir = project_root / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "medcpt_evaluation_result.txt"
    with open(output_file, "w") as f:
        f.write(result)
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("="*70)
    print(f"📄 Full result saved to: {output_file}")
    print()
    
    # Quick test for other queries (just show retrieval)
    print("="*70)
    print("🔍 QUICK RETRIEVAL TEST FOR OTHER QUERIES")
    print("="*70)
    
    for idx, query in enumerate(test_queries[1:], start=2):
        print(f"\n📝 Query {idx}: {query}")
        print("-"*70)
        
        # Just test retrieval (not full evaluation)
        relevant_chunks = evaluator.research_agent.retrieve_relevant_chunks(query, top_k=3)
        
        if relevant_chunks:
            print(f"✅ Found {len(relevant_chunks)} relevant chunks:")
            for chunk in relevant_chunks[:3]:
                print(f"   • {chunk['source']} - Page {chunk['page']} (Score: {chunk['score']:.4f})")
        else:
            print("⚠️  No relevant chunks found")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETE!")
    print("="*70)
    print(f"Vector database saved to: data/vectordb/")
    print(f"Full evaluation saved to: {output_file}")
    print(f"\n📊 Models evaluated: {', '.join(models.keys())}")
    print()


if __name__ == "__main__":
    main()
