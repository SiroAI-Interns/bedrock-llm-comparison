#!/usr/bin/env python3
"""
Test script to demonstrate high-recall retrieval vs standard retrieval.
Shows how high-recall approach prevents missing relevant documents.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.vector_store import VectorStore
from app.services.high_recall_retriever import create_high_recall_retriever
from app.services.pdf_processor import PDFProcessor


def compare_retrieval_strategies(query: str):
    """Compare standard top-k vs high-recall retrieval."""
    
    print("\n" + "="*80)
    print(f"QUERY: {query}")
    print("="*80)
    
    # Setup
    input_dir = project_root / "data" / "input" / "protocols"
    vector_db_path = project_root / "data" / "vectordb"
    
    # Initialize vector store
    print("\n📚 Loading vector store...")
    vector_store = VectorStore(
        embedding_model="medcpt",
        vector_db_path=vector_db_path,
        use_reranking=False  # We'll handle reranking separately
    )
    
    # Load or build index
    if not vector_store.load():
        print("Building index from PDFs...")
        pdf_processor = PDFProcessor()
        chunks = pdf_processor.process_directory_with_chunks(str(input_dir))
        vector_store.build_index(chunks)
        vector_store.save()
    
    # Test 1: Standard retrieval (top-5)
    print("\n" + "─"*80)
    print("📊 STANDARD RETRIEVAL (top-5)")
    print("─"*80)
    
    standard_results = vector_store.search(query, top_k=5)
    
    print(f"\nRetrieved {len(standard_results)} documents:")
    for i, doc in enumerate(standard_results, 1):
        print(f"\n{i}. [{doc['source']} - Page {doc['page']}]")
        print(f"   Score: {doc['relevance_score']:.4f}")
        print(f"   {doc['text'][:120]}...")
    
    # Test 2: High-recall retrieval
    print("\n\n" + "─"*80)
    print("🎯 HIGH-RECALL RETRIEVAL")
    print("─"*80)
    
    high_recall_retriever = create_high_recall_retriever(
        vector_store=vector_store,
        use_reranking=True,
        reranker_model="ms-marco-mini",
        reranking_strategy="cross-encoder"
    )
    
    high_recall_results = high_recall_retriever.retrieve(query, verbose=True)
    
    print(f"\nTop {min(10, len(high_recall_results))} results:")
    for i, doc in enumerate(high_recall_results[:10], 1):
        print(f"\n{i}. [{doc['source']} - Page {doc['page']}]")
        print(f"   Rerank Score: {doc.get('rerank_score', 0):.4f}")
        print(f"   Original Score: {doc.get('original_score', doc.get('relevance_score', 0)):.4f}")
        print(f"   {doc['text'][:120]}...")
    
    # Analysis
    print("\n\n" + "="*80)
    print("📈 COMPARISON")
    print("="*80)
    
    # Check what high-recall found that standard missed
    standard_doc_ids = {(d['source'], d['page']) for d in standard_results}
    high_recall_doc_ids = {(d['source'], d['page']) for d in high_recall_results[:10]}
    
    missed_by_standard = high_recall_doc_ids - standard_doc_ids
    
    print(f"\n📊 Standard retrieval: {len(standard_results)} docs")
    print(f"📊 High-recall retrieval: {len(high_recall_results)} docs total, showing top-10")
    
    if missed_by_standard:
        print(f"\n⚠️  Standard retrieval MISSED {len(missed_by_standard)} relevant docs:")
        for source, page in missed_by_standard:
            # Find the doc in high_recall_results
            doc = next(d for d in high_recall_results if d['source'] == source and d['page'] == page)
            print(f"   • {source} - Page {page} (rerank score: {doc.get('rerank_score', 0):.4f})")
    else:
        print("\n✅ Standard retrieval captured all top-10 high-recall results")
    
    print("\n💡 Key Insight:")
    print("   High-recall retrieval casts a wider net (100+ candidates)")
    print("   Then uses reranking to surface the most relevant")
    print("   This prevents missing critical FDA requirements\n")


def main():
    """Run comparison tests."""
    
    print("\n" + "="*80)
    print("HIGH-RECALL RETRIEVAL TEST")
    print("Comparing standard top-k vs high-recall approach")
    print("="*80)
    
    # Test queries
    test_queries = [
        "What are the HbA1c measurement requirements for diabetes device trials?",
        "What sample size is required for feasibility studies?",
        "What are the FDA requirements for adverse event reporting?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'█'*80}")
        print(f"TEST {i}/{len(test_queries)}")
        print(f"{'█'*80}")
        
        compare_retrieval_strategies(query)
        
        if i < len(test_queries):
            input("\nPress Enter to continue to next query...")
    
    print("\n\n" + "="*80)
    print("✅ HIGH-RECALL RETRIEVAL TEST COMPLETE")
    print("="*80)
    print("\n📋 Summary:")
    print("   • High-recall retrieval prevents missing relevant documents")
    print("   • Retrieves 100+ candidates, filters to 20-30 high-confidence docs")
    print("   • Uses reranking to surface most relevant")
    print("   • Critical for regulatory compliance where missing a requirement is unacceptable\n")


if __name__ == "__main__":
    main()
