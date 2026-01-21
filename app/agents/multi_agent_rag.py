# app/agents/multi_agent_rag.py
"""
Multi-Agent RAG System with MedCPT Embeddings and Cross-Encoder Reranking

This is the complete multi-agent RAG flow with:
- MedCPT embeddings (single model, optimized for medical text)
- Cross-encoder reranking for precision
- 4-agent pipeline: Research → Generator → Reviewer → Chairman
- Auto-saves results to data/output/ folder
- Prints top 10 reranked documents in terminal

Usage:
    from app.agents.multi_agent_rag import MultiAgentRAG
    
    rag = MultiAgentRAG()
    result = rag.evaluate("What are the HbA1c requirements?")
    # Results auto-saved to data/output/
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.core.unified_bedrock_client import UnifiedBedrockClient


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
class AgentResponse:
    """Response from an agent."""
    agent_name: str
    content: str


# ==============================================================================
# LLM MODEL CONFIGURATIONS
# ==============================================================================

LLM_MODELS = {
    "claude": ("anthropic.claude-3-5-sonnet-20240620-v1:0", "us-east-1"),
    "gpt-oss": ("openai.gpt-oss-20b-1:0", "us-east-1"),
    "mistral": ("mistral.mistral-7b-instruct-v0:2", "us-east-1"),
    "llama": ("meta.llama3-70b-instruct-v1:0", "us-east-1"),
    "titan": ("amazon.titan-text-express-v1", "us-east-1"),
}


# ==============================================================================
# PROMPT TEMPLATES
# ==============================================================================

RESEARCH_SYNTHESIS_TEMPLATE = """You are a medical protocol research assistant. Synthesize the following source documents into a clear, factual summary.

QUERY: {query}

SOURCE DOCUMENTS:
{sources}

INSTRUCTIONS:
1. Synthesize the information into 3-5 key bullet points
2. When citing sources, use format: (Document Name, Page X)
3. Be specific and accurate - don't add information not in sources
4. Focus on directly answering the query

RESEARCH SUMMARY:"""


GENERATOR_TEMPLATE = """Answer the following medical protocol query using the research context provided.

QUERY: {query}

RESEARCH CONTEXT:
{research_context}

SOURCES AVAILABLE:
{sources}

INSTRUCTIONS:
1. Provide a clear, accurate response (2-3 paragraphs)
2. Cite sources using format: (Document Name, Page X)
3. NEVER write just "Source 1" - always include document name and page
4. If information is incomplete, say so

RESPONSE:"""


REVIEWER_TEMPLATE = """You are an expert medical/regulatory evaluator. Review and score this response.

ORIGINAL QUERY: {query}

RESPONSE TO REVIEW:
{response}

SOURCES USED:
{sources}

Evaluate on these criteria (1-10 each):
1. Factual Accuracy - Does it match the sources?
2. Completeness - Does it fully answer the query?
3. Citation Quality - Are sources properly cited with document names and pages?
4. Clarity - Is it well-organized and clear?

Return your evaluation as JSON:
{{
    "score": <overall 1-10>,
    "accuracy": <1-10>,
    "completeness": <1-10>,
    "citations": <1-10>,
    "clarity": <1-10>,
    "reasoning": "<brief explanation>"
}}"""


CHAIRMAN_TEMPLATE = """You are the Chairman reviewing the full evaluation pipeline.

ORIGINAL QUERY:
{query}

RESEARCH CONTEXT:
{research_context}

SOURCES USED (with exact paragraph text):
{sources}

GENERATED RESPONSE:
{response}

REVIEWER EVALUATION:
{review}

Provide a final analysis covering:
1. **Answer Quality**: Is the response accurate and complete?
2. **Source Usage**: Were sources properly cited with document names and page numbers?
3. **Key Findings**: What are the main takeaways from the sources?
4. **Recommendations**: Any improvements needed?

FINAL ANALYSIS:"""


# ==============================================================================
# MULTI-AGENT RAG SYSTEM
# ==============================================================================

class MultiAgentRAG:
    """
    Multi-Agent RAG System
    
    4-Agent Pipeline:
    1. Research Agent - Retrieves and synthesizes documents with MedCPT + reranking
    2. Generator Agent - Creates response using research context
    3. Reviewer Agent - Evaluates response quality
    4. Chairman Agent - Final analysis and report
    
    Features:
    - MedCPT embeddings (768 dim, medical-optimized)
    - Cross-encoder reranking for precision
    - Auto-saves results to data/output/
    - Prints top 10 reranked docs in terminal
    """
    
    def __init__(
        self,
        vector_db_path: Optional[Path] = None,
        llm_model: str = "claude",
        top_k: int = 10,
        initial_candidates: int = 50,
    ):
        """
        Initialize the Multi-Agent RAG System.
        
        Args:
            vector_db_path: Path to MedCPT vector database
            llm_model: LLM to use (claude, gpt-oss, mistral, llama, titan)
            top_k: Number of sources to use after reranking
            initial_candidates: Number of candidates to fetch before reranking
        """
        self.top_k = top_k
        self.initial_candidates = initial_candidates
        self.llm_model = llm_model
        
        # Set paths
        if vector_db_path is None:
            vector_db_path = project_root / "data" / "vectordb_faiss_medcpt"
        self.vector_db_path = Path(vector_db_path)
        self.output_dir = project_root / "data" / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print("MULTI-AGENT RAG SYSTEM")
        print("="*70)
        
        # Load MedCPT embedding model
        print("\n📚 Loading MedCPT embedding model...")
        self.query_encoder = SentenceTransformer("ncbi/MedCPT-Query-Encoder")
        self.embedding_dim = 768
        print(f"   ✅ Embedding dimension: {self.embedding_dim}")
        
        # Load cross-encoder reranker
        print("\n🔄 Loading cross-encoder reranker...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("   ✅ Reranker loaded")
        
        # Load vector database
        print(f"\n💾 Loading vector database...")
        self._load_vector_db()
        
        # Initialize LLM clients
        print(f"\n🤖 Initializing LLM client: {llm_model}")
        self._init_llm_clients()
        
        print("\n" + "="*70)
        print("✅ System ready")
        print("="*70 + "\n")
    
    def _load_vector_db(self):
        """Load the FAISS index and document chunks."""
        index_file = self.vector_db_path / "faiss.index"
        metadata_file = self.vector_db_path / "metadata.pkl"
        
        if not index_file.exists():
            raise FileNotFoundError(f"Vector database not found at {self.vector_db_path}")
        
        self.index = faiss.read_index(str(index_file))
        
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            self.chunks = metadata.get("chunks", [])
        else:
            chunks_file = self.vector_db_path / "chunks.pkl"
            with open(chunks_file, 'rb') as f:
                self.chunks = pickle.load(f)
        
        print(f"   ✅ Loaded {len(self.chunks)} chunks, {self.index.ntotal} vectors")
    
    def _init_llm_clients(self):
        """Initialize LLM clients for each agent."""
        model_id, region = LLM_MODELS[self.llm_model]
        
        # Research agent - lower temperature for factual synthesis
        self.research_client = UnifiedBedrockClient(
            model_id=model_id, region_name=region,
            max_tokens=2000, temperature=0.3
        )
        
        # Generator agent - slightly more creative
        self.generator_client = UnifiedBedrockClient(
            model_id=model_id, region_name=region,
            max_tokens=2000, temperature=0.5
        )
        
        # Reviewer agent - strict evaluation
        self.reviewer_client = UnifiedBedrockClient(
            model_id=model_id, region_name=region,
            max_tokens=1000, temperature=0.2
        )
        
        # Chairman agent - balanced analysis
        self.chairman_client = UnifiedBedrockClient(
            model_id=model_id, region_name=region,
            max_tokens=2000, temperature=0.3
        )
    
    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query using MedCPT."""
        embedding = self.query_encoder.encode(query, convert_to_numpy=True)
        return embedding.astype('float32').reshape(1, -1)
    
    def _search(self, query: str, top_k: int) -> List[Dict]:
        """Search the vector database."""
        query_embedding = self._embed_query(query)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks) and idx >= 0:
                chunk = self.chunks[idx].copy()
                chunk["faiss_distance"] = float(distance)
                chunk["chunk_index"] = int(idx)
                results.append(chunk)
        
        return results
    
    def _rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """Rerank candidates using cross-encoder."""
        if not candidates:
            return []
        
        pairs = [[query, doc.get("text", "")] for doc in candidates]
        scores = self.reranker.predict(pairs)
        
        for doc, score in zip(candidates, scores):
            doc["rerank_score"] = float(score)
        
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
    
    def _format_sources(self, chunks: List[Dict]) -> List[Source]:
        """Convert chunks to Source objects."""
        sources = []
        for i, chunk in enumerate(chunks, 1):
            page = chunk.get("page", 0)
            if isinstance(page, str):
                try:
                    page = int(page)
                except:
                    page = 0
            
            document = chunk.get("source", "Unknown")
            if isinstance(document, Path):
                document = document.name
            
            source = Source(
                source_id=i,
                document=str(document),
                page=page,
                paragraph_number=chunk.get("paragraph_number", i),
                chunk_id=chunk.get("chunk_id", f"chunk_{i}"),
                paragraph_text=chunk.get("text", ""),
            )
            sources.append(source)
        
        return sources
    
    def _format_sources_for_prompt(self, sources: List[Source]) -> str:
        """Format sources for LLM prompts."""
        text = ""
        for s in sources:
            text += f"\n[Source {s.source_id}: {s.document} - Page {s.page}]\n"
            text += f"{s.paragraph_text}\n"
            text += "-" * 40 + "\n"
        return text
    
    def _print_top_docs(self, sources: List[Source]):
        """Print top 10 reranked documents in terminal."""
        print("\n" + "="*70)
        print("📚 TOP 10 RERANKED DOCUMENTS")
        print("="*70)
        
        for s in sources[:10]:
            print(f"\n[{s.source_id}] {s.document} - Page {s.page}")
            print(f"    Paragraph: {s.paragraph_number} | Chunk ID: {s.chunk_id}")
            preview = s.paragraph_text[:150].replace('\n', ' ')
            print(f"    Text: {preview}...")
        
        print("="*70 + "\n")
    
    def evaluate(self, query: str) -> Dict:
        """
        Run the full 4-agent evaluation pipeline.
        
        Args:
            query: The question to answer
            
        Returns:
            Dictionary with full evaluation results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "="*70)
        print("STARTING MULTI-AGENT EVALUATION")
        print("="*70)
        print(f"\n📝 Query: {query}\n")
        
        # ==================== AGENT 0: RESEARCH ====================
        print("\n" + "-"*70)
        print("AGENT 0: RESEARCH (MedCPT + Reranking)")
        print("-"*70)
        
        # Step 1: Search for candidates
        print(f"🔍 Searching for {self.initial_candidates} candidates...")
        candidates = self._search(query, top_k=self.initial_candidates)
        print(f"   Found {len(candidates)} candidates")
        
        # Step 2: Rerank with cross-encoder
        print(f"🔄 Reranking to top {self.top_k}...")
        reranked = self._rerank(query, candidates, top_k=self.top_k)
        sources = self._format_sources(reranked)
        
        # Print top 10 docs in terminal
        self._print_top_docs(sources)
        
        # Step 3: Synthesize research
        sources_text = self._format_sources_for_prompt(sources)
        research_prompt = RESEARCH_SYNTHESIS_TEMPLATE.format(
            query=query, sources=sources_text
        )
        
        print("📖 Synthesizing research...")
        research_context = self.research_client.generate(research_prompt)
        print("   ✅ Research complete")
        
        # ==================== AGENT 1: GENERATOR ====================
        print("\n" + "-"*70)
        print("AGENT 1: GENERATOR")
        print("-"*70)
        
        generator_prompt = GENERATOR_TEMPLATE.format(
            query=query,
            research_context=research_context,
            sources=sources_text
        )
        
        print("✍️ Generating response...")
        generated_response = self.generator_client.generate(generator_prompt)
        print("   ✅ Response generated")
        
        # ==================== AGENT 2: REVIEWER ====================
        print("\n" + "-"*70)
        print("AGENT 2: REVIEWER")
        print("-"*70)
        
        reviewer_prompt = REVIEWER_TEMPLATE.format(
            query=query,
            response=generated_response,
            sources=sources_text
        )
        
        print("⚖️ Reviewing response...")
        review_result = self.reviewer_client.generate(reviewer_prompt)
        print("   ✅ Review complete")
        
        # ==================== AGENT 3: CHAIRMAN ====================
        print("\n" + "-"*70)
        print("AGENT 3: CHAIRMAN")
        print("-"*70)
        
        chairman_prompt = CHAIRMAN_TEMPLATE.format(
            query=query,
            research_context=research_context,
            sources=sources_text,
            response=generated_response,
            review=review_result
        )
        
        print("👔 Chairman analysis...")
        chairman_analysis = self.chairman_client.generate(chairman_prompt)
        print("   ✅ Analysis complete")
        
        # ==================== SAVE RESULTS ====================
        result = {
            "query": query,
            "timestamp": timestamp,
            "model_used": self.llm_model,
            "sources": [
                {
                    "source_id": s.source_id,
                    "document": s.document,
                    "page": s.page,
                    "paragraph_number": s.paragraph_number,
                    "chunk_id": s.chunk_id,
                    "paragraph_text": s.paragraph_text
                }
                for s in sources
            ],
            "research_context": research_context,
            "generated_response": generated_response,
            "reviewer_evaluation": review_result,
            "chairman_analysis": chairman_analysis,
        }
        
        # Save to file
        output_path = self._save_results(result)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        print(f"\n📁 Results saved to: {output_path}\n")
        
        return result
    
    def _save_results(self, result: Dict) -> Path:
        """Save results to output folder."""
        timestamp = result["timestamp"]
        
        # Create readable text file
        output_path = self.output_dir / f"rag_evaluation_{timestamp}.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("MULTI-AGENT RAG EVALUATION RESULTS\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Query: {result['query']}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {result['model_used']}\n\n")
            
            f.write("="*70 + "\n")
            f.write("SOURCES (Exact Paragraph Text)\n")
            f.write("="*70 + "\n\n")
            
            for s in result['sources']:
                f.write(f"[{s['source_id']}] {s['document']} - Page {s['page']}\n")
                f.write(f"    Paragraph: {s['paragraph_number']}\n")
                f.write(f"    Chunk ID: {s['chunk_id']}\n")
                f.write("-"*60 + "\n")
                f.write(s['paragraph_text'] + "\n")
                f.write("-"*60 + "\n\n")
            
            f.write("="*70 + "\n")
            f.write("RESEARCH SYNTHESIS\n")
            f.write("="*70 + "\n\n")
            f.write(result['research_context'] + "\n\n")
            
            f.write("="*70 + "\n")
            f.write("GENERATED RESPONSE\n")
            f.write("="*70 + "\n\n")
            f.write(result['generated_response'] + "\n\n")
            
            f.write("="*70 + "\n")
            f.write("REVIEWER EVALUATION\n")
            f.write("="*70 + "\n\n")
            f.write(result['reviewer_evaluation'] + "\n\n")
            
            f.write("="*70 + "\n")
            f.write("CHAIRMAN ANALYSIS\n")
            f.write("="*70 + "\n\n")
            f.write(result['chairman_analysis'] + "\n")
        
        return output_path


# ==============================================================================
# CLI SCRIPT
# ==============================================================================

def main():
    """Command-line interface for Multi-Agent RAG."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Agent RAG Evaluation")
    parser.add_argument("query", type=str, help="The question to answer")
    parser.add_argument("--model", "-m", type=str, default="claude",
                       choices=["claude", "gpt-oss", "mistral", "llama", "titan"],
                       help="LLM model to use")
    parser.add_argument("--top-k", "-k", type=int, default=10,
                       help="Number of sources after reranking")
    
    args = parser.parse_args()
    
    rag = MultiAgentRAG(llm_model=args.model, top_k=args.top_k)
    result = rag.evaluate(args.query)
    
    print("\n📝 FINAL RESPONSE:\n")
    print(result['generated_response'])


if __name__ == "__main__":
    main()
