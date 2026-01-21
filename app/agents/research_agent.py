# app/agents/research_agent.py
"""Agent 0: Research Agent - Retrieves relevant information from PDF documents."""

from typing import List, Dict, Optional
from pathlib import Path
from app.core.unified_bedrock_client import UnifiedBedrockClient
from app.agents.templates import RESEARCH_SYNTHESIS_TEMPLATE
from app.services.pdf_processor import PDFProcessor  # ← Your existing service
from app.services.vector_store import VectorStore    # ← New service


# app/agents/research_agent.py

class ResearchAgent:
    """Performs research using PDF documents and vector search."""

    def __init__(
        self,
        research_client: UnifiedBedrockClient,
        input_dir: Path = None,
        vector_db_path: Path = None,
        rebuild_index: bool = False,
        embedding_model: str = "medcpt",           # ✅ NEW
        use_hybrid: bool = False,                  # ✅ NEW
        hybrid_models: Optional[List[str]] = None, # ✅ NEW
        use_reranking: bool = False,               # ✅ NEW: Reranking flag
        reranker_model: str = "ms-marco-mini",     # ✅ NEW: Reranker model
        reranking_strategy: str = "cross-encoder", # ✅ NEW: Reranking strategy
    ):
        """
        Initialize Research Agent with PDF-based RAG.
        
        Args:
            research_client: LLM client for synthesis
            input_dir: Directory containing PDF files
            vector_db_path: Path to save/load vector database
            rebuild_index: Force rebuild of vector index
            embedding_model: Embedding model to use ("medcpt", "biobert", etc.)
            use_hybrid: If True, combine multiple embedding models
            hybrid_models: List of models to combine (e.g., ["medcpt", "biobert"])
            use_reranking: If True, rerank search results for better precision
            reranker_model: Reranker model to use (ms-marco-mini, ms-marco-medium, ms-marco-base)
            reranking_strategy: Reranking strategy (cross-encoder, bm25, hybrid)
        """
        self.research_client = research_client
        
        # Set default paths
        project_root = Path(__file__).parent.parent.parent
        self.input_dir = input_dir or (project_root / "data" / "input" / "protocols")
        self.vector_db_path = vector_db_path or (project_root / "data" / "vectordb")
        
        # Store embedding configuration
        self.embedding_model = embedding_model
        self.use_hybrid = use_hybrid
        self.hybrid_models = hybrid_models or [embedding_model]
        self.use_reranking = use_reranking
        self.reranker_model = reranker_model
        self.reranking_strategy = reranking_strategy
        
        # Initialize PDF processor
        self.pdf_processor = PDFProcessor()
        
        # ✅ Initialize vector store with embedding and reranking configuration
        self.vector_store = VectorStore(
            embedding_model=embedding_model,
            vector_db_path=self.vector_db_path,
            use_hybrid=use_hybrid,
            hybrid_models=hybrid_models,
            use_reranking=use_reranking,
            reranker_model=reranker_model,
            reranking_strategy=reranking_strategy,
        )
        
        # Load or build vector index
        self._initialize_vector_store(rebuild_index)


    def _initialize_vector_store(self, rebuild: bool = False):
        """
        Load existing vector store or build new one from PDFs.
        
        Args:
            rebuild: Force rebuild even if saved index exists
        """
        # Try to load existing index
        if not rebuild and self.vector_store.load():
            print("✅ Using existing vector database\n")
            return
        
        # Build new index from PDFs
        print("\n🔄 Building new vector database from PDFs...")
        
        # Ensure input directory exists
        self.input_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all PDFs using your existing processor
        chunks = self.pdf_processor.process_directory_with_chunks(str(self.input_dir))
        
        if not chunks:
            print("⚠️  No PDF content to index. Add PDFs to data/input/")
            return
        
        # Build vector index
        self.vector_store.build_index(chunks)
        
        # Save for future use
        self.vector_store.save()

    def retrieve_relevant_chunks(
        self,
        query: str,
        top_k: int = 20  # ✅ CHANGED: Increased from 5 to 20 for better coverage
    ) -> List[Dict]:
        """
        Retrieve most relevant document chunks for a query.
        
        Uses high-recall strategy: fetches many candidates, then reranks to find
        the most relevant. This prevents missing critical FDA requirements.
        
        Args:
            query: User's search query
            top_k: Number of final chunks to return (default: 20)
        
        Returns:
            List of relevant chunks with metadata, ranked by relevance
        """
        # ✅ HIGH-RECALL STRATEGY: Fetch MORE candidates before reranking
        # This ensures we don't miss relevant documents at position k+1, k+2, etc.
        if self.use_reranking:
            # Fetch 3x more candidates, then rerank to top_k
            fetch_k = top_k * 3  # e.g., fetch 60, rerank to top-20
            print(f"   🔍 High-recall mode: Fetching {fetch_k} candidates, will rerank to top-{top_k}")
        else:
            # No reranking, just fetch what we need
            fetch_k = top_k
        
        # Search with expanded candidate pool
        results = self.vector_store.search(query, top_k=fetch_k)
        
        # Return top_k (reranking already happened in vector_store.search if enabled)
        return results[:top_k]

    def run(
        self,
        prompt: str,
        synthesis_template: Optional[str] = None,
        top_k: int = 5
    ) -> Dict:
        """
        Run research agent to retrieve and synthesize PDF-based information.
        
        Args:
            prompt: User's query
            synthesis_template: Custom template for synthesis
            top_k: Number of document chunks to retrieve
        
        Returns:
            Dictionary with research context and sources
        """
        print("\n" + "="*70)
        print("AGENT 0: RESEARCH AGENT (PDF-BASED)")
        print("="*70)
        print(f"📚 Researching: {prompt[:100]}...\n")

        # Use default template if not provided
        if synthesis_template is None:
            synthesis_template = RESEARCH_SYNTHESIS_TEMPLATE

        # Retrieve relevant chunks
        print(f"🔍 Searching vector database for relevant information...")
        relevant_chunks = self.retrieve_relevant_chunks(prompt, top_k=top_k)

        if not relevant_chunks:
            print("⚠️  No relevant information found in PDFs")
            return {
                "prompt": prompt,
                "research_context": "No relevant information found in guideline documents.",
                "sources": [],
                "raw_search_results": "",
            }

        print(f"✅ Found {len(relevant_chunks)} relevant chunks\n")

        # Format chunks for synthesis
        search_text = ""
        sources = []
        seen_sources = set()  # Track unique sources
        
        for idx, chunk in enumerate(relevant_chunks, 1):
            source_name = chunk["source"]
            page_num = chunk["page"]
            text = chunk["text"]
            score = chunk.get("relevance_score", 0)  # ✅ FIXED: Use "relevance_score" not "score"
            
            search_text += f"\n[Source {idx}] {source_name} - Page {page_num}\n"
            search_text += f"Relevance Score: {score:.4f}\n"
            search_text += f"Content: {text}\n"
            search_text += "-" * 60 + "\n"
            
            # Track unique sources for citation
            source_key = f"{source_name}_p{page_num}"
            if source_key not in seen_sources:
                sources.append({
                    "index": idx,
                    "document": source_name,
                    "page": page_num,
                    "relevance_score": score  # ✅ FIXED: This was already correct
                })
                seen_sources.add(source_key)
            
            # Print source for visibility
            print(f"  [{idx}] {source_name} - Page {page_num}")
            print(f"      Relevance: {score:.4f}")  # ✅ This will now show correct score!

        # Build research prompt
        research_prompt = synthesis_template.format(
            query=prompt,
            search_results=search_text,
        )

        print("\n🤖 Synthesizing research findings from guidelines...")
        research_context = self.research_client.generate(research_prompt)
        print(f"✅ Research complete - {len(sources)} source(s) cited\n")

        return {
            "prompt": prompt,
            "research_context": research_context,
            "sources": sources,
            "raw_search_results": search_text,
        }
