# app/services/vector_store_pinecone.py
"""Pinecone-based vector store using MedCPT embeddings."""

from typing import List, Dict, Optional
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone


class PineconeVectorStore:
    """Vector store backed by Pinecone."""

    def __init__(
        self,
        index_name: str,
        embedding_model: str = "ncbi/MedCPT-Article-Encoder",
        query_encoder: Optional[str] = "ncbi/MedCPT-Query-Encoder",
        env_var_name: str = "PINECONE_API_KEY",
    ):
        api_key = os.getenv(env_var_name)
        if not api_key:
            raise ValueError(f"{env_var_name} not set in environment/.env")

        # Init Pinecone client
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)

        # Load encoders
        print(f"📥 Loading document encoder: {embedding_model}")
        self.doc_encoder = SentenceTransformer(embedding_model)

        if query_encoder and query_encoder != embedding_model:
            print(f"📥 Loading query encoder: {query_encoder}")
            self.query_encoder = SentenceTransformer(query_encoder)
        else:
            self.query_encoder = self.doc_encoder

        # Cache chunks locally (for metadata)
        self.chunks: List[Dict] = []

    def _embed_docs(self, texts: List[str]) -> np.ndarray:
        return self.doc_encoder.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

    def _embed_query(self, text: str) -> np.ndarray:
        return self.query_encoder.encode(
            [text],
            batch_size=1,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

    def build_index(self, chunks: List[Dict]):
        """Upload all chunks + embeddings into Pinecone."""
        if not chunks:
            print("⚠️ No chunks to index")
            return

        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        print(f"🧮 Creating embeddings for {len(texts)} chunks...")
        embeddings = self._embed_docs(texts)

        print("☁️ Upserting vectors into Pinecone...")
        vectors = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                "id": f"chunk-{i}",
                "values": emb.tolist(),
                "metadata": {
                    "chunk_index": i,
                    "text": chunk["text"][:950],
                    "page": chunk.get("page", 0),
                    "source": chunk.get("source", "unknown"),
                },
            })

        # batched upsert
        batch_size = 100
        for start in range(0, len(vectors), batch_size):
            batch = vectors[start:start+batch_size]
            self.index.upsert(vectors=batch)


        print(f"✅ Pinecone index updated with {len(vectors)} vectors")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search Pinecone with a query encoded by MedCPT query encoder."""
        q_emb = self._embed_query(query)

        res = self.index.query(
            vector=q_emb.tolist(),
            top_k=top_k,
            include_metadata=True,
        )

        results: List[Dict] = []
        for rank, match in enumerate(res.get("matches", []), start=1):
            meta = match["metadata"]
            idx = meta["chunk_index"]
            base = self.chunks[idx].copy()
            base["score"] = float(match["score"])
            base["rank"] = rank
            results.append(base)

        return results
