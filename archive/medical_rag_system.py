# app/medical_rag_system.py
"""
Medical Protocol RAG System

Supports FDA guidelines, clinical trial protocols, regulatory standards, and more.

Features:
- MedCPT embeddings optimized for medical/regulatory text
- BioMedical cross-encoder reranking for precision
- Returns exact paragraph text from source PDFs
- Supports multiple LLM backends (Claude, GPT-OSS, Mistral, Llama, Titan)

Usage:
    from app.medical_rag_system import MedicalRAGSystem
    
    rag = MedicalRAGSystem()
    result = rag.query("What are the HbA1c requirements?", model="claude")
    print(result.answer)
    for source in result.sources:
        print(f"[{source.document} - Page {source.page}] {source.paragraph_text}")
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class Source:
    """A source document retrieved from the knowledge base."""
    source_id: int
    document: str
    page: int
    paragraph_number: int
    chunk_id: str
    paragraph_text: str


@dataclass
class RAGResponse:
    """Response from the RAG system."""
    query: str
    answer: str
    sources: List[Source]
    model_used: str


# ==============================================================================
# LLM MODEL CONFIGURATIONS
# ==============================================================================

LLM_MODELS = {
    "claude": {
        "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "region": "us-east-1",
        "max_tokens": 2000,
    },
    "gpt-oss": {
        "model_id": "openai.gpt-oss-20b-1:0",
        "region": "us-east-1",
        "max_tokens": 4000,
    },
    "mistral": {
        "model_id": "mistral.mistral-7b-instruct-v0:2",
        "region": "us-east-1",
        "max_tokens": 2000,
    },
    "llama": {
        "model_id": "meta.llama3-70b-instruct-v1:0",
        "region": "us-east-1",
        "max_tokens": 2000,
    },
    "titan": {
        "model_id": "amazon.titan-text-express-v1",
        "region": "us-east-1",
        "max_tokens": 2000,
    },
}


# ==============================================================================
# PROMPT TEMPLATE
# ==============================================================================

ANSWER_TEMPLATE = """You are a medical protocol expert. Answer the following question based ONLY on the provided source documents.

QUESTION:
{query}

SOURCE DOCUMENTS:
{sources}

INSTRUCTIONS:
1. Provide a clear, accurate answer based ONLY on the source documents above.
2. When citing sources, use this format: (Document Name, Page X) - NOT just "Source 1".
   Example: "HbA1c should be measured at baseline (FDA_Standards.pdf, Page 14)."
3. If the sources don't contain enough information, say so.
4. Be specific and reference the document name and page number.
5. Format your answer in clear paragraphs.

IMPORTANT: Never write just "Source 1" or "Source 7" - always include the document name and page number so readers can verify.

ANSWER:"""


# ==============================================================================
# MEDICAL RAG SYSTEM
# ==============================================================================

class MedicalRAGSystem:
    """
    Medical Protocol RAG System
    
    Works with any medical domain documents:
    - FDA guidelines
    - Clinical trial protocols
    - Regulatory standards
    - Healthcare policy documents
    
    Features:
    - MedCPT embeddings optimized for medical/regulatory text
    - BioMedical cross-encoder reranking for precision
    - Returns exact paragraph text from source PDFs
    """
    
    # Medical-specific cross-encoder for reranking
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Alternative: "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    
    def __init__(
        self,
        vector_db_path: Optional[Path] = None,
        reranker_model: Optional[str] = None,
    ):
        """
        Initialize the Medical RAG System.
        
        Args:
            vector_db_path: Path to the vector database directory.
                           Defaults to data/vectordb_faiss_medcpt/
            reranker_model: Cross-encoder model for reranking.
                           Defaults to MS-MARCO MiniLM.
        """
        # Set paths
        if vector_db_path is None:
            vector_db_path = project_root / "data" / "vectordb_faiss_medcpt"
        self.vector_db_path = Path(vector_db_path)
        
        print("="*70)
        print("MEDICAL PROTOCOL RAG SYSTEM")
        print("="*70)
        
        # Load embedding model (MedCPT)
        print("\n📚 Loading MedCPT embedding model...")
        self.query_encoder = SentenceTransformer("ncbi/MedCPT-Query-Encoder")
        self.embedding_dim = 768
        print(f"   ✅ Embedding dimension: {self.embedding_dim}")
        
        # Load reranker
        reranker_model = reranker_model or self.RERANKER_MODEL
        print(f"\n🔄 Loading cross-encoder reranker: {reranker_model}")
        self.reranker = CrossEncoder(reranker_model)
        print(f"   ✅ Reranker loaded")
        
        # Load vector database
        print(f"\n💾 Loading vector database from: {self.vector_db_path}")
        self._load_vector_db()
        
        # Initialize LLM client (lazy loading)
        self._llm_clients = {}
        
        print("\n" + "="*70)
        print("✅ System ready")
        print("="*70 + "\n")
    
    def _load_vector_db(self):
        """Load the FAISS index and document chunks."""
        index_file = self.vector_db_path / "faiss.index"
        metadata_file = self.vector_db_path / "metadata.pkl"
        
        if not index_file.exists():
            raise FileNotFoundError(
                f"Vector database not found at {self.vector_db_path}. "
                "Please build the index first using scripts/build_index.py"
            )
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_file))
        
        # Load chunks/metadata
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            self.chunks = metadata.get("chunks", [])
        else:
            # Fallback to chunks.pkl
            chunks_file = self.vector_db_path / "chunks.pkl"
            with open(chunks_file, 'rb') as f:
                self.chunks = pickle.load(f)
        
        print(f"   ✅ Loaded {len(self.chunks)} document chunks")
        print(f"   ✅ FAISS index with {self.index.ntotal} vectors")
    
    def _get_llm_client(self, model: str):
        """Get or create an LLM client for the specified model."""
        if model not in self._llm_clients:
            if model not in LLM_MODELS:
                raise ValueError(
                    f"Unknown model: {model}. "
                    f"Available models: {list(LLM_MODELS.keys())}"
                )
            
            from app.core.unified_bedrock_client import UnifiedBedrockClient
            
            config = LLM_MODELS[model]
            self._llm_clients[model] = UnifiedBedrockClient(
                model_id=config["model_id"],
                region_name=config["region"],
                max_tokens=config["max_tokens"],
                temperature=0.3,  # Low temperature for factual responses
            )
        
        return self._llm_clients[model]
    
    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query using MedCPT."""
        embedding = self.query_encoder.encode(query, convert_to_numpy=True)
        return embedding.astype('float32').reshape(1, -1)
    
    def _search(self, query: str, top_k: int = 50) -> List[Dict]:
        """
        Search the vector database.
        
        Args:
            query: The search query
            top_k: Number of candidates to retrieve (before reranking)
        
        Returns:
            List of document chunks with relevance scores
        """
        # Embed query
        query_embedding = self._embed_query(query)
        
        # Search FAISS
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Collect results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks) and idx >= 0:
                chunk = self.chunks[idx].copy()
                chunk["faiss_distance"] = float(distance)
                chunk["chunk_index"] = int(idx)
                results.append(chunk)
        
        return results
    
    def _rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Rerank candidates using cross-encoder.
        
        Args:
            query: The search query
            candidates: List of candidate documents
            top_k: Number of top results to return
        
        Returns:
            Reranked list of documents
        """
        if not candidates:
            return []
        
        # Create query-document pairs
        pairs = [[query, doc.get("text", "")] for doc in candidates]
        
        # Get reranking scores
        scores = self.reranker.predict(pairs)
        
        # Add scores to documents
        for doc, score in zip(candidates, scores):
            doc["rerank_score"] = float(score)
        
        # Sort by rerank score (descending)
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        
        return reranked[:top_k]
    
    def _format_sources(self, chunks: List[Dict]) -> List[Source]:
        """Convert chunks to Source objects with paragraph info."""
        sources = []
        
        for i, chunk in enumerate(chunks, 1):
            # Extract page number from chunk metadata
            page = chunk.get("page", 0)
            if isinstance(page, str):
                try:
                    page = int(page)
                except:
                    page = 0
            
            # Get document name
            document = chunk.get("source", "Unknown")
            if isinstance(document, Path):
                document = document.name
            
            # Get the paragraph text (this is the actual text from the PDF)
            paragraph_text = chunk.get("text", "")
            
            # Generate unique chunk ID
            chunk_index = chunk.get("chunk_index", i)
            chunk_id = f"{Path(document).stem}_p{page}_c{chunk_index}"
            
            # Estimate paragraph number based on chunk index within the page
            # This is an approximation - for exact paragraph numbers,
            # we would need to track this during PDF processing
            paragraph_number = chunk.get("paragraph_number", i)
            
            source = Source(
                source_id=i,
                document=str(document),
                page=page,
                paragraph_number=paragraph_number,
                chunk_id=chunk_id,
                paragraph_text=paragraph_text,
            )
            sources.append(source)
        
        return sources
    
    def _generate_answer(self, query: str, sources: List[Source], model: str) -> str:
        """Generate an answer using the specified LLM."""
        # Format sources for the prompt
        sources_text = ""
        for source in sources:
            sources_text += f"\n[Source {source.source_id}: {source.document} - Page {source.page}]\n"
            sources_text += f"{source.paragraph_text}\n"
            sources_text += "-" * 40 + "\n"
        
        # Build the prompt
        prompt = ANSWER_TEMPLATE.format(query=query, sources=sources_text)
        
        # Get the LLM client and generate
        client = self._get_llm_client(model)
        answer = client.generate(prompt)
        
        return answer
    
    def query(
        self,
        question: str,
        top_k: int = 10,
        model: str = "claude",
        initial_candidates: int = 50,
    ) -> RAGResponse:
        """
        Answer a query using medical guideline documents.
        
        Args:
            question: The query to answer
            top_k: Number of sources to return
            model: LLM to use ("claude", "gpt-oss", "mistral", "llama", "titan")
            initial_candidates: Number of candidates to fetch before reranking
        
        Returns:
            RAGResponse with answer and sources
        """
        print(f"\n{'='*70}")
        print(f"QUERY: {question}")
        print(f"{'='*70}")
        
        # Step 1: Search for candidates
        print(f"\n🔍 Searching for relevant documents...")
        candidates = self._search(question, top_k=initial_candidates)
        print(f"   Found {len(candidates)} initial candidates")
        
        # Step 2: Rerank with cross-encoder
        print(f"\n🔄 Reranking with cross-encoder...")
        reranked = self._rerank(question, candidates, top_k=top_k)
        print(f"   Selected top {len(reranked)} documents")
        
        # Step 3: Format sources with exact paragraph text
        sources = self._format_sources(reranked)
        
        # Step 4: Generate answer using LLM
        print(f"\n🤖 Generating answer using {model}...")
        answer = self._generate_answer(question, sources, model)
        print(f"   ✅ Answer generated")
        
        # Step 5: Print sources
        print(f"\n📚 SOURCES RETRIEVED:")
        print("-" * 70)
        for source in sources[:5]:  # Show first 5
            print(f"   [{source.source_id}] {source.document} - Page {source.page}")
            preview = source.paragraph_text[:100].replace('\n', ' ')
            print(f"       {preview}...")
        if len(sources) > 5:
            print(f"   ... and {len(sources) - 5} more sources")
        
        return RAGResponse(
            query=question,
            answer=answer,
            sources=sources,
            model_used=model,
        )
    
    def get_available_models(self) -> List[str]:
        """Return list of available LLM models."""
        return list(LLM_MODELS.keys())


# ==============================================================================
# CONVENIENCE FUNCTION
# ==============================================================================

def quick_query(
    question: str,
    model: str = "claude",
    top_k: int = 10,
) -> RAGResponse:
    """
    Quick function to query the RAG system.
    
    Example:
        result = quick_query("What are HbA1c requirements?")
        print(result.answer)
    """
    rag = MedicalRAGSystem()
    return rag.query(question, model=model, top_k=top_k)


# ==============================================================================
# MAIN (for testing)
# ==============================================================================

if __name__ == "__main__":
    # Test query
    rag = MedicalRAGSystem()
    
    result = rag.query(
        "What are the HbA1c measurement requirements for diabetes clinical trials?",
        model="claude",
        top_k=10,
    )
    
    print("\n" + "="*70)
    print("ANSWER:")
    print("="*70)
    print(result.answer)
    
    print("\n" + "="*70)
    print("ALL SOURCES:")
    print("="*70)
    for source in result.sources:
        print(f"\n[{source.source_id}] {source.document} - Page {source.page}")
        print(f"    Paragraph: {source.paragraph_number}")
        print(f"    Chunk ID: {source.chunk_id}")
        print(f"    Text: {source.paragraph_text[:200]}...")
