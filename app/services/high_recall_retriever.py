"""
High-Recall Retrieval Module for FDA Compliance RAG.

Ensures maximum recall (don't miss relevant documents) before reranking for precision.
"""

from typing import List, Dict, Optional, Set, Tuple
import numpy as np
from app.services.vector_store import VectorStore
from app.services.reranker import Reranker


class HighRecallRetriever:
    """
    Retrieval strategy optimized for high recall in regulatory compliance.
    
    Key principle: Better to retrieve 50 docs and rerank to 10,
    than retrieve 10 and miss a critical FDA requirement.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        reranker: Optional[Reranker] = None,
        initial_k: int = 100,
        confidence_threshold: float = 0.6,
        use_query_expansion: bool = True,
        final_k: int = 20
    ):
        """
        Initialize high-recall retriever.
        
        Args:
            vector_store: Vector store for semantic search
            reranker: Optional reranker for precision refinement
            initial_k: Number of initial candidates to retrieve
            confidence_threshold: Minimum relevance score to keep
            use_query_expansion: Whether to expand queries
            final_k: Final number of documents to return
        """
        self.vector_store = vector_store
        self.reranker = reranker
        self.initial_k = initial_k
        self.confidence_threshold = confidence_threshold
        self.use_query_expansion = use_query_expansion
        self.final_k = final_k
    
    def retrieve(self, query: str, verbose: bool = True) -> List[Dict]:
        """
        High-recall retrieval with multi-stage filtering.
        
        Args:
            query: User query
            verbose: Print progress information
            
        Returns:
            List of highly relevant documents (typically 10-30)
        """
        if verbose:
            print("\n" + "="*70)
            print("HIGH-RECALL RETRIEVAL")
            print("="*70)
        
        # Stage 1: Cast wide net
        all_candidates = self._stage1_wide_retrieval(query, verbose)
        
        # Stage 2: Filter by confidence
        high_confidence = self._stage2_confidence_filter(all_candidates, verbose)
        
        # Stage 3: Rerank for precision
        final_results = self._stage3_rerank(query, high_confidence, verbose)
        
        # Validation
        self._validate_recall(query, final_results, verbose)
        
        return final_results
    
    def _stage1_wide_retrieval(self, query: str, verbose: bool) -> List[Dict]:
        """Stage 1: Retrieve large candidate pool."""
        if verbose:
            print(f"\n🔍 STAGE 1: Wide Retrieval (top-{self.initial_k})")
        
        all_results = []
        seen_docs: Set[Tuple[str, int]] = set()
        
        # Strategy 1: Original query
        results = self.vector_store.search(query, top_k=self.initial_k)
        for doc in results:
            doc_id = (doc['source'], doc['page'])
            if doc_id not in seen_docs:
                all_results.append(doc)
                seen_docs.add(doc_id)
        
        if verbose:
            print(f"   Original query: {len(results)} docs")
        
        # Strategy 2: Query expansion (if enabled)
        if self.use_query_expansion:
            expanded_queries = self._expand_query(query)
            
            for exp_query in expanded_queries:
                results = self.vector_store.search(exp_query, top_k=30)
                new_docs = 0
                for doc in results:
                    doc_id = (doc['source'], doc['page'])
                    if doc_id not in seen_docs:
                        all_results.append(doc)
                        seen_docs.add(doc_id)
                        new_docs += 1
                
                if verbose and new_docs > 0:
                    print(f"   Expanded query: +{new_docs} new docs")
        
        if verbose:
            print(f"   Total unique candidates: {len(all_results)}")
        
        return all_results
    
    def _stage2_confidence_filter(self, candidates: List[Dict], verbose: bool) -> List[Dict]:
        """Stage 2: Filter by confidence threshold."""
        if verbose:
            print(f"\n📊 STAGE 2: Confidence Filtering (threshold={self.confidence_threshold})")
        
        # Sort by relevance score
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.get('relevance_score', 0),
            reverse=True
        )
        
        # Apply threshold
        high_confidence = [
            doc for doc in sorted_candidates
            if doc.get('relevance_score', 0) >= self.confidence_threshold
        ]
        
        # Ensure minimum coverage (at least 10 docs)
        if len(high_confidence) < 10:
            high_confidence = sorted_candidates[:10]
            if verbose:
                print(f"   ⚠️  Only {len(high_confidence)} docs above threshold")
                print(f"   Taking top-10 to ensure minimum coverage")
        
        if verbose:
            print(f"   Kept {len(high_confidence)} high-confidence docs")
            if high_confidence:
                print(f"   Score range: {high_confidence[0]['relevance_score']:.4f} - {high_confidence[-1]['relevance_score']:.4f}")
        
        return high_confidence
    
    def _stage3_rerank(self, query: str, candidates: List[Dict], verbose: bool) -> List[Dict]:
        """Stage 3: Rerank for precision."""
        if not self.reranker:
            # No reranker, just return top-k
            return candidates[:self.final_k]
        
        if verbose:
            print(f"\n🎯 STAGE 3: Precision Reranking (top-{self.final_k})")
        
        # Rerank all high-confidence candidates
        reranked = self.reranker.rerank(query, candidates, top_k=len(candidates))
        
        # Keep documents with positive or near-positive rerank scores
        # (negative scores indicate low relevance)
        final_results = [
            doc for doc in reranked
            if doc.get('rerank_score', 0) >= -1.0
        ][:self.final_k]
        
        if verbose:
            print(f"   Reranked {len(candidates)} → {len(final_results)} final docs")
            if final_results:
                print(f"   Rerank score range: {final_results[0]['rerank_score']:.4f} - {final_results[-1]['rerank_score']:.4f}")
        
        return final_results
    
    def _expand_query(self, query: str) -> List[str]:
        """Generate query variations to improve recall."""
        expanded = []
        
        # Add key term extraction
        # Example: "HbA1c measurement requirements" → "HbA1c measurement", "HbA1c requirements"
        words = query.split()
        if len(words) > 3:
            # Take first half and second half
            mid = len(words) // 2
            expanded.append(" ".join(words[:mid+1]))
            expanded.append(" ".join(words[mid:]))
        
        # Add common regulatory phrasings
        if "requirement" in query.lower():
            expanded.append(query.replace("requirement", "must"))
            expanded.append(query.replace("requirement", "should"))
        
        if "what" in query.lower():
            # "What are X requirements?" → "X requirements"
            simplified = query.lower().replace("what are", "").replace("what is", "").strip()
            if simplified != query.lower():
                expanded.append(simplified)
        
        return expanded[:3]  # Limit to 3 expansions
    
    def _validate_recall(self, query: str, results: List[Dict], verbose: bool):
        """Validate that we likely captured all relevant documents."""
        if not verbose or not results:
            return
        
        print(f"\n✅ VALIDATION")
        
        # Check 1: Score distribution
        scores = [doc.get('rerank_score', doc.get('relevance_score', 0)) for doc in results]
        
        if scores[-1] > 0.5:
            print(f"   ⚠️  Last doc has high score ({scores[-1]:.4f}) - might be cutting off too early")
        else:
            print(f"   ✓ Score distribution looks good (last: {scores[-1]:.4f})")
        
        # Check 2: Document count
        if len(results) < 5:
            print(f"   ⚠️  Only {len(results)} docs - might need to lower threshold")
        else:
            print(f"   ✓ Retrieved {len(results)} documents")
        
        print()


def create_high_recall_retriever(
    vector_store: VectorStore,
    use_reranking: bool = True,
    reranker_model: str = "ms-marco-mini",
    reranking_strategy: str = "cross-encoder"
) -> HighRecallRetriever:
    """
    Factory function to create high-recall retriever.
    
    Recommended configuration for FDA compliance:
    - initial_k=100 (cast wide net)
    - confidence_threshold=0.6 (keep moderately relevant docs)
    - use_query_expansion=True (catch different phrasings)
    - final_k=20 (send more context to LLM)
    """
    reranker = None
    if use_reranking:
        reranker = Reranker(
            model_name=reranker_model,
            strategy=reranking_strategy
        )
    
    return HighRecallRetriever(
        vector_store=vector_store,
        reranker=reranker,
        initial_k=100,           # Retrieve many candidates
        confidence_threshold=0.6, # Keep moderately relevant docs
        use_query_expansion=True, # Expand query for better recall
        final_k=20               # Return top 20 (not just 5)
    )
