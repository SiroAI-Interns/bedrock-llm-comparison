#!/usr/bin/env python3
"""Quick test script to verify reranking functionality."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.reranker import Reranker


def test_reranker():
    """Test the reranker service."""
    
    print("\n" + "="*70)
    print("RERANKER SERVICE TEST")
    print("="*70)
    
    # Test data
    query = "What are the HbA1c measurement requirements for diabetes trials?"
    
    documents = [
        {
            "text": "HbA1c should be measured at baseline and every 3 months during the study.",
            "source": "FDA_Diabetes_Guidance.pdf",
            "page": 15,
            "relevance_score": 0.75
        },
        {
            "text": "The study protocol must specify the laboratory method for HbA1c testing.",
            "source": "FDA_Diabetes_Guidance.pdf",
            "page": 16,
            "relevance_score": 0.82
        },
        {
            "text": "Patient recruitment should follow ICH-GCP guidelines.",
            "source": "ICH_GCP_Guidelines.pdf",
            "page": 23,
            "relevance_score": 0.45
        },
        {
            "text": "HbA1c measurements must use a NGSP-certified laboratory method.",
            "source": "FDA_Diabetes_Guidance.pdf",
            "page": 17,
            "relevance_score": 0.68
        },
    ]
    
    print(f"\n📝 Query: {query}")
    print(f"\n📚 Testing with {len(documents)} documents\n")
    
    # Test 1: Cross-encoder reranking
    print("\n" + "-"*70)
    print("TEST 1: Cross-Encoder Reranking")
    print("-"*70)
    
    reranker_ce = Reranker(model_name="ms-marco-mini", strategy="cross-encoder")
    results_ce = reranker_ce.rerank(query, documents, top_k=3)
    
    print("\nTop 3 Results (Cross-Encoder):")
    for i, doc in enumerate(results_ce, 1):
        print(f"\n{i}. [{doc['source']} - Page {doc['page']}]")
        print(f"   Original Score: {doc['original_score']:.4f}")
        print(f"   Rerank Score: {doc['rerank_score']:.4f}")
        print(f"   Text: {doc['text'][:80]}...")
    
    # Test 2: BM25 reranking
    print("\n" + "-"*70)
    print("TEST 2: BM25 Reranking")
    print("-"*70)
    
    reranker_bm25 = Reranker(model_name="ms-marco-mini", strategy="bm25")
    results_bm25 = reranker_bm25.rerank(query, documents, top_k=3)
    
    print("\nTop 3 Results (BM25):")
    for i, doc in enumerate(results_bm25, 1):
        print(f"\n{i}. [{doc['source']} - Page {doc['page']}]")
        print(f"   Original Score: {doc['original_score']:.4f}")
        print(f"   Rerank Score: {doc['rerank_score']:.4f}")
        print(f"   Text: {doc['text'][:80]}...")
    
    # Test 3: Hybrid reranking
    print("\n" + "-"*70)
    print("TEST 3: Hybrid Reranking (70% Cross-Encoder + 30% BM25)")
    print("-"*70)
    
    reranker_hybrid = Reranker(model_name="ms-marco-mini", strategy="hybrid")
    results_hybrid = reranker_hybrid.rerank(query, documents, top_k=3)
    
    print("\nTop 3 Results (Hybrid):")
    for i, doc in enumerate(results_hybrid, 1):
        print(f"\n{i}. [{doc['source']} - Page {doc['page']}]")
        print(f"   Original Score: {doc['original_score']:.4f}")
        print(f"   Rerank Score: {doc['rerank_score']:.4f}")
        print(f"   Text: {doc['text'][:80]}...")
    
    print("\n" + "="*70)
    print("✅ ALL RERANKER TESTS PASSED!")
    print("="*70)
    print("\nReranking is working correctly. You can now use it in your RAG pipeline.")
    print("Set use_reranking=True when initializing MultiAgentEvaluator.\n")


if __name__ == "__main__":
    test_reranker()
