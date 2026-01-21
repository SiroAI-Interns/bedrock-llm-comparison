#!/usr/bin/env python3
"""
Test script to compare retrieval quality WITH vs WITHOUT reranking.
This demonstrates the real-world impact of reranking on search precision.
"""

import sys
from pathlib import Path
from typing import List, Dict
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.vector_store import VectorStore
from app.services.pdf_processor import PDFProcessor


def print_results(title: str, results: List[Dict], show_rerank: bool = False):
    """Pretty print search results."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. [{result['source']} - Page {result['page']}]")
        print(f"   Relevance Score: {result.get('relevance_score', 0):.4f}")
        
        if show_rerank and 'rerank_score' in result:
            print(f"   Rerank Score: {result['rerank_score']:.4f}")
            print(f"   Score Change: {result['rerank_score'] - result.get('original_score', 0):+.4f}")
        
        # Show first 150 chars of text
        text = result['text'][:150].replace('\n', ' ')
        print(f"   Text: {text}...")


def compare_retrieval(query: str, vector_store_basic: VectorStore, vector_store_rerank: VectorStore, top_k: int = 5):
    """Compare retrieval with and without reranking for a single query."""
    
    print(f"\n{'#'*80}")
    print(f"QUERY: {query}")
    print(f"{'#'*80}")
    
    # Test 1: Basic retrieval (no reranking)
    print("\n🔍 Searching WITHOUT reranking...")
    start_time = time.time()
    results_basic = vector_store_basic.search(query, top_k=top_k)
    time_basic = time.time() - start_time
    
    print_results(
        f"📊 BASIC RETRIEVAL (No Reranking) - {time_basic*1000:.0f}ms",
        results_basic,
        show_rerank=False
    )
    
    # Test 2: With reranking
    print("\n🔍 Searching WITH reranking...")
    start_time = time.time()
    results_rerank = vector_store_rerank.search(query, top_k=top_k)
    time_rerank = time.time() - start_time
    
    print_results(
        f"🎯 WITH RERANKING (Cross-Encoder) - {time_rerank*1000:.0f}ms",
        results_rerank,
        show_rerank=True
    )
    
    # Analysis
    print(f"\n{'='*80}")
    print("📈 ANALYSIS")
    print(f"{'='*80}")
    print(f"⏱️  Time overhead: +{(time_rerank - time_basic)*1000:.0f}ms ({time_rerank/time_basic:.1f}x)")
    
    # Check if top results changed
    basic_top_sources = [(r['source'], r['page']) for r in results_basic[:3]]
    rerank_top_sources = [(r['source'], r['page']) for r in results_rerank[:3]]
    
    if basic_top_sources != rerank_top_sources:
        print("🔄 Top-3 results CHANGED after reranking:")
        print(f"   Before: {basic_top_sources}")
        print(f"   After:  {rerank_top_sources}")
    else:
        print("✅ Top-3 results remained the same (reranking confirmed initial ranking)")
    
    # Check score improvements
    if results_rerank and 'rerank_score' in results_rerank[0]:
        avg_score_change = sum(
            r['rerank_score'] - r.get('original_score', 0) 
            for r in results_rerank
        ) / len(results_rerank)
        print(f"📊 Average score change: {avg_score_change:+.4f}")


def main():
    """Run comprehensive reranking comparison test."""
    
    print("\n" + "="*80)
    print("RERANKING IMPACT TEST")
    print("Comparing retrieval quality WITH vs WITHOUT reranking")
    print("="*80)
    
    # Setup paths
    input_dir = project_root / "data" / "input" / "protocols"
    vector_db_basic = project_root / "data" / "vectordb_test_basic"
    vector_db_rerank = project_root / "data" / "vectordb_test_rerank"
    
    # Initialize PDF processor
    pdf_processor = PDFProcessor()
    
    # Process PDFs (shared for both tests)
    print("\n📚 Processing PDF documents...")
    chunks = pdf_processor.process_directory_with_chunks(str(input_dir))
    
    if not chunks:
        print("❌ No PDF chunks found. Please add PDFs to data/input/protocols/")
        return
    
    print(f"✅ Loaded {len(chunks)} document chunks\n")
    
    # Initialize vector stores
    print("🔧 Setting up vector stores...")
    
    # 1. Basic retrieval (no reranking)
    print("\n1️⃣  Creating BASIC vector store (no reranking)...")
    vs_basic = VectorStore(
        embedding_model="medcpt",
        vector_db_path=vector_db_basic,
        use_reranking=False
    )
    vs_basic.build_index(chunks)
    
    # 2. With reranking
    print("\n2️⃣  Creating RERANKING vector store (cross-encoder)...")
    vs_rerank = VectorStore(
        embedding_model="medcpt",
        vector_db_path=vector_db_rerank,
        use_reranking=True,
        reranker_model="ms-marco-mini",
        reranking_strategy="cross-encoder"
    )
    vs_rerank.build_index(chunks)
    
    # Test queries (regulatory/medical domain)
    test_queries = [
        "What are the HbA1c measurement requirements for diabetes device trials?",
        "What sample size is required for feasibility studies?",
        "What are the FDA requirements for adverse event reporting?",
        "How should primary endpoints be defined in device trials?",
        "What are the inclusion and exclusion criteria requirements?",
    ]
    
    print("\n" + "="*80)
    print(f"RUNNING {len(test_queries)} TEST QUERIES")
    print("="*80)
    
    # Run comparisons
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'█'*80}")
        print(f"TEST {i}/{len(test_queries)}")
        print(f"{'█'*80}")
        
        compare_retrieval(
            query=query,
            vector_store_basic=vs_basic,
            vector_store_rerank=vs_rerank,
            top_k=5
        )
        
        if i < len(test_queries):
            print("\n" + "─"*80)
            input("Press Enter to continue to next query...")
    
    # Final summary
    print("\n\n" + "="*80)
    print("✅ RERANKING IMPACT TEST COMPLETE")
    print("="*80)
    print("\n📋 Summary:")
    print("   • Reranking adds ~100-400ms latency per query")
    print("   • Cross-encoder models understand query-document relationships better")
    print("   • Particularly useful for complex regulatory/medical queries")
    print("   • Helps surface the most relevant guideline sections")
    print("\n💡 Recommendation:")
    print("   Use reranking for:")
    print("   ✓ High-stakes queries (regulatory compliance)")
    print("   ✓ When precision is critical")
    print("   ✓ Complex multi-part questions")
    print("\n   Skip reranking for:")
    print("   ✗ Simple keyword searches")
    print("   ✗ When speed is critical (< 100ms)")
    print("   ✗ Exploratory browsing\n")


if __name__ == "__main__":
    main()
