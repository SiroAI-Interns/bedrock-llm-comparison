#!/usr/bin/env python3
"""
Rebuild Vector Index

Supports two backends:
- FAISS (local, default)
- Pinecone (cloud, requires API key)

Usage:
    # FAISS (local, default)
    python rebuild_index.py
    python rebuild_index.py --backend faiss
    
    # Pinecone (cloud)
    python rebuild_index.py --backend pinecone
    
    # Clear Pinecone index before uploading
    python rebuild_index.py --backend pinecone --clear
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env file
from dotenv import load_dotenv
load_dotenv(project_root / ".env")


def main():
    parser = argparse.ArgumentParser(description="Rebuild vector index from PDFs")
    parser.add_argument(
        "--backend", "-b",
        type=str,
        default="faiss",
        choices=["faiss", "pinecone"],
        help="Vector store backend (default: faiss)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing index before uploading (Pinecone only)"
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default="medical-protocols",
        help="Pinecone index name (default: medical-protocols)"
    )
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(f"REBUILDING VECTOR INDEX ({args.backend.upper()})")
    print("="*80)
    
    # Load PDF processor
    print("\n1️⃣  Loading modules...")
    from app.services.pdf_processor import PDFProcessor
    print("✅ Modules loaded")
    
    # Paths
    input_dir = project_root / "data" / "input" / "protocols"
    
    # List PDFs
    pdf_files = list(input_dir.glob("*.pdf"))
    print(f"\n2️⃣  Found {len(pdf_files)} PDF(s) in {input_dir}")
    for pdf in pdf_files:
        print(f"   📄 {pdf.name}")
    
    if not pdf_files:
        print("❌ No PDFs found. Add PDFs to data/input/protocols/")
        sys.exit(1)
    
    # Process PDFs
    print("\n3️⃣  Processing PDFs...")
    pdf_processor = PDFProcessor()
    chunks = pdf_processor.process_directory_with_chunks(str(input_dir))
    
    if not chunks:
        print("❌ No chunks extracted from PDFs")
        sys.exit(1)
    
    print(f"✅ Extracted {len(chunks)} chunks from {len(pdf_files)} PDF(s)")
    
    # Build index based on backend
    if args.backend == "pinecone":
        _build_pinecone(chunks, args)
    else:
        _build_faiss(chunks)
    
    print("\n" + "="*80)
    print("✅ INDEX REBUILT SUCCESSFULLY")
    print("="*80)
    print("\nYou can now run: python scripts/run_multi_agent_rag.py \"Your query\"")
    print("="*80 + "\n")


def _build_faiss(chunks):
    """Build local FAISS index."""
    from app.services.vector_store import VectorStore
    
    vector_db_path = project_root / "data" / "vectordb_faiss_medcpt"
    
    print(f"\n4️⃣  Initializing FAISS vector store...")
    print(f"   Path: {vector_db_path}")
    
    vector_store = VectorStore(
        embedding_model="medcpt",
        vector_db_path=vector_db_path,
        use_reranking=False,
        use_hybrid=False
    )
    
    print("\n5️⃣  Building FAISS index...")
    vector_store.build_index(chunks)
    
    print("\n6️⃣  Saving index...")
    vector_store.save()
    
    print(f"\n📊 Index stats:")
    print(f"   Total vectors: {vector_store.index.ntotal}")
    print(f"   Embedding dim: 768 (MedCPT)")
    print(f"   Location: {vector_db_path}")


def _build_pinecone(chunks, args):
    """Build Pinecone cloud index."""
    from app.services.pinecone_store import PineconeStore
    
    print(f"\n4️⃣  Connecting to Pinecone...")
    print(f"   Index: {args.index_name}")
    
    store = PineconeStore(index_name=args.index_name)
    
    if args.clear:
        print("\n⚠️  Clearing existing index...")
        store.clear_index()
    
    print("\n5️⃣  Uploading chunks to Pinecone...")
    uploaded = store.upsert_chunks(chunks)
    
    print(f"\n📊 Index stats:")
    stats = store.list_documents()
    print(f"   Total vectors: {stats.get('total_vectors', uploaded)}")
    print(f"   Embedding dim: 768 (MedCPT)")
    print(f"   Index: {args.index_name}")


if __name__ == "__main__":
    main()
