# app/core/vector_store.py
"""Vector store with multiple embedding model support."""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from app.services.reranker import Reranker


class VectorStore:
    """Manages document embeddings and similarity search."""
    
    EMBEDDING_MODELS = {
        "medcpt": "ncbi/MedCPT-Query-Encoder",
        "biobert": "dmis-lab/biobert-base-cased-v1.1",
        "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "scibert": "allenai/scibert_scivocab_uncased",
        "all-mpnet": "sentence-transformers/all-mpnet-base-v2",  # General baseline
    }
    
    def __init__(
        self,
        embedding_model: str = "medcpt",
        vector_db_path: Path = None,
        use_hybrid: bool = False,
        hybrid_models: Optional[List[str]] = None,
        use_reranking: bool = False,
        reranker_model: str = "ms-marco-mini",
        reranking_strategy: str = "cross-encoder"
    ):
        """
        Initialize vector store.
        
        Args:
            embedding_model: Primary embedding model to use
            vector_db_path: Path to save/load vector database
            use_hybrid: If True, combine multiple embeddings
            hybrid_models: List of models to combine (e.g., ["medcpt", "biobert"])
            use_reranking: If True, rerank search results
            reranker_model: Reranker model to use (ms-marco-mini, ms-marco-medium, ms-marco-base)
            reranking_strategy: Reranking strategy (cross-encoder, bm25, hybrid)
        """
        self.embedding_model_name = embedding_model
        self.vector_db_path = vector_db_path or Path("data/vector_db")
        self.use_hybrid = use_hybrid
        self.hybrid_models = hybrid_models or [embedding_model]
        self.use_reranking = use_reranking
        self.reranker_model = reranker_model
        self.reranking_strategy = reranking_strategy
        
        # Initialize embedding model(s)
        if self.use_hybrid:
            print(f"\n🔬 Loading hybrid embeddings: {' × '.join(self.hybrid_models)}")
            self.embedding_models = {}
            for model_name in self.hybrid_models:
                model_id = self.EMBEDDING_MODELS.get(model_name, model_name)
                print(f"   Loading {model_name}...")
                self.embedding_models[model_name] = SentenceTransformer(model_id)
            self.embedding_dim = sum(
                model.get_sentence_embedding_dimension() 
                for model in self.embedding_models.values()
            )
        else:
            print(f"\n🔬 Loading embedding model: {embedding_model}")
            model_id = self.EMBEDDING_MODELS.get(embedding_model, embedding_model)
            self.model = SentenceTransformer(model_id)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize reranker if enabled
        if self.use_reranking:
            self.reranker = Reranker(
                model_name=reranker_model,
                strategy=reranking_strategy
            )
        else:
            self.reranker = None
        
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict] = []
        
        print(f"   Embedding dimension: {self.embedding_dim}")
        print(f"   Vector DB path: {self.vector_db_path}")
        if self.use_reranking:
            print(f"   🔄 Reranking: {reranking_strategy} ({reranker_model})")
        print()

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using selected strategy."""
        if self.use_hybrid:
            # Combine embeddings from multiple models
            embeddings = []
            for model_name, model in self.embedding_models.items():
                emb = model.encode(text, convert_to_numpy=True)
                embeddings.append(emb)
            
            # Concatenate embeddings
            combined = np.concatenate(embeddings)
            
            # ✅ Normalize with safety check
            norm = np.linalg.norm(combined)
            if norm < 1e-10:  # Avoid division by zero
                print(f"⚠️ WARNING: Zero norm embedding for text: {text[:50]}...")
                return combined  # Return unnormalized
            
            return combined / norm
        else:
            # Single model embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            # ✅ Normalize with safety check
            norm = np.linalg.norm(embedding)
            if norm < 1e-10:
                print(f"⚠️ WARNING: Zero norm embedding for text: {text[:50]}...")
                return embedding
            
            return embedding / norm

    def _get_hybrid_embedding_multiplicative(self, text: str) -> np.ndarray:
        """
        Get hybrid embedding using element-wise multiplication.
        
        This approach captures interactions between embeddings.
        Embeddings must be same dimension or padded.
        """
        embeddings = []
        max_dim = 0
        
        # Get all embeddings and find max dimension
        for model_name, model in self.embedding_models.items():
            emb = model.encode(text, convert_to_numpy=True)
            embeddings.append(emb)
            max_dim = max(max_dim, len(emb))
        
        # Pad embeddings to same dimension
        padded_embeddings = []
        for emb in embeddings:
            if len(emb) < max_dim:
                # Pad with zeros
                padded = np.pad(emb, (0, max_dim - len(emb)), mode='constant')
            else:
                padded = emb
            padded_embeddings.append(padded)
        
        # Element-wise multiplication
        result = np.ones(max_dim)
        for emb in padded_embeddings:
            result = result * (emb + 1)  # Add 1 to avoid zeros killing the product
        
        # Normalize with safety check
        norm = np.linalg.norm(result)
        if norm < 1e-10:
            print(f"⚠️ WARNING: Zero norm multiplicative embedding")
            return result
        
        return result / norm

    def build_index(self, chunks: List[Dict]) -> None:
        """Build FAISS index from document chunks."""
        print(f"Building FAISS index with {len(chunks)} chunks...")
        
        self.chunks = chunks
        
        # Create embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = []
        
        print("Generating embeddings...")
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"  {i}/{len(texts)}", end="\r")
            emb = self._get_embedding(text)
            embeddings.append(emb)
        
        print(f"  {len(texts)}/{len(texts)} ✅")
        
        # Convert to numpy array
        embeddings_matrix = np.vstack(embeddings).astype('float32')
        
        # ✅ Debug: Check embedding stats
        print(f"   Embedding matrix shape: {embeddings_matrix.shape}")
        print(f"   Embedding matrix stats: min={embeddings_matrix.min():.4f}, max={embeddings_matrix.max():.4f}, mean={embeddings_matrix.mean():.4f}")
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)
        self.index.add(embeddings_matrix)
        
        print(f"✅ Index built with {self.index.ntotal} vectors\n")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar chunks with optional reranking."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # ✅ HIGH-RECALL STRATEGY: Fetch MORE candidates when reranking
        # This ensures we don't miss relevant documents that might be ranked lower initially
        if self.use_reranking:
            # Fetch 6x more candidates for reranking (was 4x, increased for better recall)
            fetch_k = min(top_k * 6, self.index.ntotal)
            print(f"   📊 Fetching {fetch_k} candidates for reranking (target: top-{top_k})")
        else:
            fetch_k = min(top_k, self.index.ntotal)
        
        # Get query embedding
        query_embedding = self._get_embedding(query).reshape(1, -1).astype('float32')
        
        # ✅ Debug: Check query embedding
        print(f"   Query embedding stats: min={query_embedding.min():.4f}, max={query_embedding.max():.4f}, norm={np.linalg.norm(query_embedding):.4f}")
        
        # Search
        distances, indices = self.index.search(query_embedding, fetch_k)
        
        # ✅ Debug: Check raw distances
        print(f"   Raw FAISS distances (top-{fetch_k}): {distances[0][:5]}...")
        
        # Collect initial results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                result = self.chunks[idx].copy()
                result["relevance_score"] = float(distance)
                results.append(result)
        
        # Apply reranking if enabled
        if self.use_reranking and self.reranker and results:
            print(f"   🔄 Reranking {len(results)} candidates -> top-{top_k}")
            results = self.reranker.rerank(query, results, top_k)
            # Format scores without nested f-strings
            top_scores = [r.get("rerank_score", 0) for r in results[:3]]
            scores_str = ", ".join([f"{s:.4f}" for s in top_scores])
            print(f"   ✅ Reranked scores: [{scores_str}]...")
        else:
            # Just take top-k if no reranking
            results = results[:top_k]
        
        return results

    def save(self) -> None:
        """Save index and chunks to disk."""
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = self.vector_db_path / "faiss.index"
        faiss.write_index(self.index, str(index_file))
        
        # Save chunks and metadata
        metadata_file = self.vector_db_path / "metadata.pkl"
        metadata = {
            "chunks": self.chunks,
            "embedding_model": self.embedding_model_name,
            "embedding_dim": self.embedding_dim,
            "use_hybrid": self.use_hybrid,
            "hybrid_models": self.hybrid_models,
        }
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✅ Vector database saved to {self.vector_db_path}")

    def load(self) -> bool:
        """Load index and chunks from disk."""
        index_file = self.vector_db_path / "faiss.index"
        metadata_file = self.vector_db_path / "metadata.pkl"
        
        if not index_file.exists() or not metadata_file.exists():
            return False
        
        # Load metadata first to check compatibility
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        # ✅ Check if embedding configuration matches
        config_changed = False
        
        if metadata.get("embedding_model") != self.embedding_model_name:
            print(f"⚠️ WARNING: Index was built with '{metadata.get('embedding_model')}' but current model is '{self.embedding_model_name}'")
            config_changed = True
        
        if metadata.get("use_hybrid") != self.use_hybrid:
            print(f"⚠️ WARNING: Index hybrid mode mismatch (stored: {metadata.get('use_hybrid')}, current: {self.use_hybrid})")
            config_changed = True
        
        if metadata.get("hybrid_models") != self.hybrid_models:
            print(f"⚠️ WARNING: Hybrid models mismatch (stored: {metadata.get('hybrid_models')}, current: {self.hybrid_models})")
            config_changed = True
        
        if metadata.get("embedding_dim") != self.embedding_dim:
            print(f"⚠️ WARNING: Embedding dimension mismatch (stored: {metadata.get('embedding_dim')}, current: {self.embedding_dim})")
            config_changed = True
        
        if config_changed:
            print("❌ Index configuration doesn't match. Need to rebuild index!")
            return False
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_file))
        self.chunks = metadata["chunks"]
        
        print(f"✅ Loaded vector database from {self.vector_db_path}")
        print(f"   Model: {metadata['embedding_model']}")
        print(f"   Chunks: {len(self.chunks)}")
        print(f"   Dimension: {metadata['embedding_dim']}\n")
        
        return True
