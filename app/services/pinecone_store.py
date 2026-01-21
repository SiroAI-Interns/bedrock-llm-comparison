# app/services/pinecone_store.py
"""
Pinecone Vector Store for Medical Protocol RAG

Features:
- Cloud-based vector storage (no local files)
- Multi-document support with metadata filtering
- Query across all documents or specific subsets
- Automatic text storage in metadata

Usage:
    from app.services.pinecone_store import PineconeStore
    
    store = PineconeStore(index_name="medical-protocols")
    store.upsert_chunks(chunks)
    results = store.search("HbA1c requirements", top_k=50)
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    raise ImportError("Please install pinecone-client: pip install pinecone-client>=3.0.0")


class PineconeStore:
    """
    Pinecone Vector Store for Medical Protocols
    
    Stores document chunks with rich metadata for multi-PDF retrieval.
    Each chunk includes: document name, page, paragraph text, etc.
    """
    
    EMBEDDING_MODEL = "ncbi/MedCPT-Query-Encoder"
    EMBEDDING_DIM = 768
    
    def __init__(
        self,
        index_name: str = "medical-protocols",
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
    ):
        """
        Initialize Pinecone store.
        
        Args:
            index_name: Name of the Pinecone index
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            environment: Pinecone environment (defaults to PINECONE_ENVIRONMENT env var)
        """
        self.index_name = index_name
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.environment = environment or os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        
        if not self.api_key:
            raise ValueError(
                "Pinecone API key required. Set PINECONE_API_KEY env var or pass api_key parameter."
            )
        
        print(f"🔌 Connecting to Pinecone...")
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)
        
        # Create or connect to index
        self._init_index()
        
        # Load embedding model
        print(f"📚 Loading MedCPT embedding model...")
        self.encoder = SentenceTransformer(self.EMBEDDING_MODEL)
        
        print(f"✅ Pinecone store ready (index: {self.index_name})")
    
    def _init_index(self):
        """Create index if it doesn't exist, or connect to existing."""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"📦 Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=self.environment
                )
            )
        else:
            print(f"📦 Connected to existing index: {self.index_name}")
        
        self.index = self.pc.Index(self.index_name)
    
    def embed_text(self, text: str) -> List[float]:
        """Embed text using MedCPT."""
        embedding = self.encoder.encode(text, convert_to_numpy=True)
        return embedding.astype('float32').tolist()
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Embed multiple texts in batches."""
        embeddings = self.encoder.encode(texts, batch_size=batch_size, convert_to_numpy=True)
        return embeddings.astype('float32').tolist()
    
    def upsert_chunks(
        self,
        chunks: List[Dict],
        namespace: str = "",
        batch_size: int = 100,
    ) -> int:
        """
        Upload chunks to Pinecone with metadata.
        
        Args:
            chunks: List of chunk dicts with text, page, source, etc.
            namespace: Optional namespace for organization
            batch_size: Number of vectors per upsert batch
            
        Returns:
            Number of chunks uploaded
        """
        print(f"\n📤 Uploading {len(chunks)} chunks to Pinecone...")
        
        # Prepare vectors
        vectors = []
        for i, chunk in enumerate(chunks):
            # Generate unique ID
            doc_name = Path(chunk.get("source", f"doc_{i}")).stem
            page = chunk.get("page", 0)
            chunk_idx = chunk.get("chunk_id", f"c{i}")
            vector_id = f"{doc_name}_p{page}_{chunk_idx}"
            
            # Get text and create embedding
            text = chunk.get("text", "")
            
            # Prepare metadata (Pinecone stores this with the vector)
            metadata = {
                "document": str(chunk.get("source", "")),
                "page": int(chunk.get("page", 0)),
                "paragraph_number": int(chunk.get("paragraph_number", 0)),
                "chunk_id": str(chunk.get("chunk_id", "")),
                "text": text[:10000],  # Pinecone metadata limit
                "upload_date": datetime.now().isoformat(),
            }
            
            vectors.append({
                "id": vector_id,
                "values": None,  # Will be filled with embedding
                "metadata": metadata,
                "text_for_embedding": text,  # Temporary, for batch embedding
            })
        
        # Batch embed all texts
        print(f"   🔄 Embedding {len(vectors)} texts...")
        texts = [v.pop("text_for_embedding") for v in vectors]
        embeddings = self.embed_batch(texts)
        
        # Add embeddings to vectors
        for v, emb in zip(vectors, embeddings):
            v["values"] = emb
        
        # Upsert in batches
        print(f"   📤 Uploading to Pinecone in batches of {batch_size}...")
        total_uploaded = 0
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            # Format for pinecone upsert
            upsert_batch = [(v["id"], v["values"], v["metadata"]) for v in batch]
            self.index.upsert(vectors=upsert_batch, namespace=namespace)
            total_uploaded += len(batch)
            print(f"   ✅ Uploaded {total_uploaded}/{len(vectors)}")
        
        print(f"\n✅ Successfully uploaded {total_uploaded} chunks")
        return total_uploaded
    
    def search(
        self,
        query: str,
        top_k: int = 50,
        filter: Optional[Dict] = None,
        namespace: str = "",
    ) -> List[Dict]:
        """
        Search for relevant chunks across all documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter: Optional metadata filter (e.g., {"document": "FDA.pdf"})
            namespace: Optional namespace
            
        Returns:
            List of chunks with text and metadata
        """
        # Embed query
        query_embedding = self.embed_text(query)
        
        # Search Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter,
            namespace=namespace,
        )
        
        # Format results
        chunks = []
        for match in results.matches:
            chunk = {
                "text": match.metadata.get("text", ""),
                "source": match.metadata.get("document", ""),
                "page": match.metadata.get("page", 0),
                "paragraph_number": match.metadata.get("paragraph_number", 0),
                "chunk_id": match.metadata.get("chunk_id", ""),
                "pinecone_score": match.score,
                "vector_id": match.id,
            }
            chunks.append(chunk)
        
        return chunks
    
    def delete_document(self, document_name: str, namespace: str = ""):
        """
        Delete all chunks from a specific document.
        
        Args:
            document_name: Name of the document to delete
            namespace: Optional namespace
        """
        print(f"🗑️ Deleting chunks for: {document_name}")
        
        # Pinecone requires IDs, so we search first
        # Using metadata filter and delete by ID
        self.index.delete(
            filter={"document": {"$eq": document_name}},
            namespace=namespace,
        )
        
        print(f"✅ Deleted all chunks for {document_name}")
    
    def list_documents(self, namespace: str = "") -> List[str]:
        """
        List all unique documents in the index.
        
        Note: This is an approximation using a sample query.
        """
        # Pinecone doesn't have a native "list unique metadata values" 
        # So we do a broad search and extract unique docs
        stats = self.index.describe_index_stats()
        print(f"📊 Index stats: {stats.total_vector_count} total vectors")
        
        # Return stats for now
        return {
            "total_vectors": stats.total_vector_count,
            "namespaces": dict(stats.namespaces) if stats.namespaces else {},
        }
    
    def clear_index(self, namespace: str = ""):
        """Delete all vectors in the index (or namespace)."""
        print(f"⚠️ Clearing all vectors from index...")
        self.index.delete(delete_all=True, namespace=namespace)
        print(f"✅ Index cleared")


# ==============================================================================
# CONVENIENCE FUNCTION
# ==============================================================================

def create_pinecone_store(**kwargs) -> PineconeStore:
    """Factory function to create PineconeStore with env defaults."""
    return PineconeStore(**kwargs)
