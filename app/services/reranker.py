# app/services/reranker.py
"""Reranking service for improving RAG retrieval quality."""

from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi


class Reranker:
    """Reranks search results using cross-encoder models or hybrid approaches."""
    
    RERANKER_MODELS = {
        "ms-marco-mini": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "ms-marco-medium": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "ms-marco-base": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
    }
    
    def __init__(
        self,
        model_name: str = "ms-marco-mini",
        strategy: str = "cross-encoder"
    ):
        """
        Initialize reranker.
        
        Args:
            model_name: Reranker model to use (ms-marco-mini, ms-marco-medium, ms-marco-base)
            strategy: Reranking strategy ("cross-encoder", "bm25", "hybrid")
        """
        self.model_name = model_name
        self.strategy = strategy
        
        # Load cross-encoder model if needed
        if strategy in ["cross-encoder", "hybrid"]:
            model_id = self.RERANKER_MODELS.get(model_name, model_name)
            print(f"🔄 Loading reranker model: {model_name}")
            self.cross_encoder = CrossEncoder(model_id)
            print(f"   ✅ Reranker loaded: {model_id}\n")
        else:
            self.cross_encoder = None
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Rerank documents based on query relevance.
        
        Args:
            query: Search query
            documents: List of document chunks with metadata
            top_k: Number of top results to return
            
        Returns:
            Reranked list of documents with updated relevance scores
        """
        if not documents:
            return []
        
        if self.strategy == "cross-encoder":
            return self._cross_encoder_rerank(query, documents, top_k)
        elif self.strategy == "bm25":
            return self._bm25_rerank(query, documents, top_k)
        elif self.strategy == "hybrid":
            return self._hybrid_rerank(query, documents, top_k)
        else:
            raise ValueError(f"Unknown reranking strategy: {self.strategy}")
    
    def _cross_encoder_rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """Rerank using cross-encoder model."""
        # Prepare query-document pairs
        pairs = [[query, doc["text"]] for doc in documents]
        
        # Score all pairs
        scores = self.cross_encoder.predict(pairs)
        
        # Sort by score (descending)
        scored_docs = []
        for doc, score in zip(documents, scores):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(score)
            doc_copy["original_score"] = doc.get("relevance_score", 0.0)
            scored_docs.append(doc_copy)
        
        # Sort by rerank score
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Return top-k
        return scored_docs[:top_k]
    
    def _bm25_rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """Rerank using BM25 algorithm."""
        # Tokenize documents
        tokenized_docs = [doc["text"].lower().split() for doc in documents]
        
        # Build BM25 index
        bm25 = BM25Okapi(tokenized_docs)
        
        # Score query
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        
        # Sort by score
        scored_docs = []
        for doc, score in zip(documents, scores):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(score)
            doc_copy["original_score"] = doc.get("relevance_score", 0.0)
            scored_docs.append(doc_copy)
        
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return scored_docs[:top_k]
    
    def _hybrid_rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        Hybrid reranking combining cross-encoder and BM25.
        
        Uses weighted combination:
        - Cross-encoder: 70%
        - BM25: 30%
        """
        # Get cross-encoder scores
        ce_docs = self._cross_encoder_rerank(query, documents, len(documents))
        
        # Get BM25 scores
        bm25_docs = self._bm25_rerank(query, documents, len(documents))
        
        # Normalize scores to [0, 1]
        ce_scores = np.array([doc["rerank_score"] for doc in ce_docs])
        bm25_scores = np.array([doc["rerank_score"] for doc in bm25_docs])
        
        # Min-max normalization
        if ce_scores.max() > ce_scores.min():
            ce_scores = (ce_scores - ce_scores.min()) / (ce_scores.max() - ce_scores.min())
        else:
            ce_scores = np.ones_like(ce_scores)
        
        if bm25_scores.max() > bm25_scores.min():
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
        else:
            bm25_scores = np.ones_like(bm25_scores)
        
        # Combine scores (70% cross-encoder, 30% BM25)
        combined_scores = 0.7 * ce_scores + 0.3 * bm25_scores
        
        # Create final scored documents
        scored_docs = []
        for doc, score in zip(documents, combined_scores):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(score)
            doc_copy["original_score"] = doc.get("relevance_score", 0.0)
            scored_docs.append(doc_copy)
        
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return scored_docs[:top_k]
