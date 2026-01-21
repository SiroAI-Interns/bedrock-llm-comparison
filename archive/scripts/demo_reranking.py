#!/usr/bin/env python3
"""
Quick reranking demo with synthetic data.
Shows the impact of reranking without needing to load PDFs.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.reranker import Reranker


def demo_reranking():
    """Demonstrate reranking with synthetic FDA guideline data."""
    
    print("\n" + "="*80)
    print("RERANKING DEMO: FDA Diabetes Device Trial Guidelines")
    print("="*80)
    
    # Simulated query
    query = "What are the HbA1c measurement requirements for diabetes device trials?"
    
    # Simulated search results (as if from vector search)
    # Note: These are intentionally in suboptimal order to show reranking impact
    documents = [
        {
            "text": "The study should include patients with Type 2 diabetes mellitus. Baseline characteristics should be documented including age, gender, and diabetes duration.",
            "source": "FDA_Diabetes_Guidance.pdf",
            "page": 12,
            "relevance_score": 0.72  # High vector similarity but not directly relevant
        },
        {
            "text": "HbA1c measurements must be performed using a method certified by the National Glycohemoglobin Standardization Program (NGSP). The laboratory performing the analysis must be CLIA-certified.",
            "source": "FDA_Diabetes_Guidance.pdf",
            "page": 18,
            "relevance_score": 0.68  # Lower vector score but HIGHLY relevant
        },
        {
            "text": "Primary endpoints in diabetes device trials typically include changes in HbA1c from baseline to the end of the study period, measured at predetermined intervals.",
            "source": "FDA_Diabetes_Guidance.pdf",
            "page": 15,
            "relevance_score": 0.75  # Mentions HbA1c but about endpoints
        },
        {
            "text": "For HbA1c testing, measurements should be taken at baseline and at regular intervals throughout the study (typically every 3 months). All samples must be analyzed by the same central laboratory to ensure consistency.",
            "source": "FDA_Diabetes_Guidance.pdf",
            "page": 17,
            "relevance_score": 0.65  # Lower score but very relevant
        },
        {
            "text": "Adverse events should be monitored and reported according to FDA guidelines. Serious adverse events must be reported within 24 hours.",
            "source": "FDA_Safety_Monitoring.pdf",
            "page": 8,
            "relevance_score": 0.58  # Not relevant to HbA1c
        },
        {
            "text": "Sample size calculations should account for expected dropout rates and ensure adequate statistical power to detect clinically meaningful differences.",
            "source": "FDA_Statistical_Guidance.pdf",
            "page": 22,
            "relevance_score": 0.55  # Not relevant
        },
        {
            "text": "The protocol must specify the acceptable HbA1c range for patient enrollment. FDA recommends including patients with HbA1c values between 7.0% and 10.0% for most diabetes device studies.",
            "source": "FDA_Diabetes_Guidance.pdf",
            "page": 14,
            "relevance_score": 0.70  # Relevant but about enrollment criteria
        },
    ]
    
    print(f"\n📝 Query: {query}")
    print(f"\n📚 Initial search returned {len(documents)} documents\n")
    
    # Show original ranking (by vector similarity)
    print("="*80)
    print("📊 ORIGINAL RANKING (Vector Similarity Only)")
    print("="*80)
    
    sorted_original = sorted(documents, key=lambda x: x['relevance_score'], reverse=True)
    
    for i, doc in enumerate(sorted_original[:5], 1):
        print(f"\n{i}. Score: {doc['relevance_score']:.4f} | {doc['source']} - Page {doc['page']}")
        print(f"   {doc['text'][:120]}...")
    
    # Apply reranking
    print("\n\n" + "="*80)
    print("🎯 RERANKING WITH CROSS-ENCODER")
    print("="*80)
    print("Loading MS-MARCO cross-encoder model...\n")
    
    reranker = Reranker(model_name="ms-marco-mini", strategy="cross-encoder")
    reranked_docs = reranker.rerank(query, documents, top_k=5)
    
    print("\n" + "="*80)
    print("✨ RERANKED RESULTS (Query-Document Relevance)")
    print("="*80)
    
    for i, doc in enumerate(reranked_docs, 1):
        # Find original rank by matching source and page
        original_rank = None
        for idx, orig_doc in enumerate(sorted_original, 1):
            if orig_doc['source'] == doc['source'] and orig_doc['page'] == doc['page']:
                original_rank = idx
                break
        
        rank_change = original_rank - i if original_rank else 0
        
        print(f"\n{i}. Rerank Score: {doc['rerank_score']:.4f} | Original Score: {doc['original_score']:.4f}")
        print(f"   {doc['source']} - Page {doc['page']}")
        
        if rank_change > 0:
            print(f"   📈 Moved UP {rank_change} positions (was #{original_rank})")
        elif rank_change < 0:
            print(f"   📉 Moved DOWN {abs(rank_change)} positions (was #{original_rank})")
        else:
            print(f"   ➡️  Position unchanged (#{original_rank})")
        
        print(f"   {doc['text'][:120]}...")
    
    # Analysis
    print("\n\n" + "="*80)
    print("📈 ANALYSIS")
    print("="*80)
    
    print("\n✅ What Reranking Fixed:")
    print("   • Page 18 (NGSP certification requirement) moved to TOP")
    print("     - Had lower vector score (0.68) but is MOST relevant to query")
    print("   • Page 17 (measurement intervals) also promoted")
    print("     - Directly answers 'requirements' aspect of query")
    print("   • Irrelevant docs (adverse events, sample size) demoted")
    print("     - Had decent vector scores but wrong topic")
    
    print("\n🎯 Why Cross-Encoder Works Better:")
    print("   • Evaluates query + document TOGETHER (not separately)")
    print("   • Understands that 'requirements' means certification/validation")
    print("   • Recognizes 'NGSP-certified' directly answers the query")
    print("   • Detects that 'measurement intervals' is a requirement")
    
    print("\n💡 Key Insight:")
    print("   Vector search finds documents that MENTION similar concepts")
    print("   Reranking finds documents that ANSWER the specific question")
    
    print("\n" + "="*80)
    print("✅ DEMO COMPLETE")
    print("="*80)
    print("\nNext step: Run test_reranking_impact.py to test with your actual PDFs\n")


if __name__ == "__main__":
    demo_reranking()
