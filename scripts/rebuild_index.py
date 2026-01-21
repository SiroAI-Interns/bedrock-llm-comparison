#!/usr/bin/env python3
"""
Rebuild vector index using only MedCPT embeddings (single model, not hybrid).
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("\n" + "="*80)
print("REBUILDING VECTOR INDEX (MedCPT Only)")
print("="*80)

print("\n1️⃣  Loading modules...")
from app.services.vector_store import VectorStore
from app.services.pdf_processor import PDFProcessor

print("✅ Modules loaded")

# Paths
input_dir = project_root / "data" / "input" / "protocols"
vector_db_path = project_root / "data" / "vectordb_faiss_medcpt"  # ← FIXED: Match multi_agent_rag.py

print(f"\n2️⃣  Input PDFs: {input_dir}")
print(f"   Vector DB: {vector_db_path}")

# Initialize vector store with single MedCPT embedding (no hybrid)
print("\n3️⃣  Initializing vector store (MedCPT only, no hybrid)...")
vector_store = VectorStore(
    embedding_model="medcpt",
    vector_db_path=vector_db_path,
    use_reranking=False,
    use_hybrid=False  # Single embedding, not hybrid
)

# Process PDFs
print("\n4️⃣  Processing PDFs...")
pdf_processor = PDFProcessor()
chunks = pdf_processor.process_directory_with_chunks(str(input_dir))

if not chunks:
    print("❌ No PDFs found in", input_dir)
    sys.exit(1)

print(f"✅ Found {len(chunks)} chunks from PDFs")

# Build index
print("\n5️⃣  Building FAISS index...")
vector_store.build_index(chunks)

# Save index
print("\n6️⃣  Saving index...")
vector_store.save()

print("\n" + "="*80)
print("✅ INDEX REBUILT SUCCESSFULLY")
print("="*80)
print(f"\nIndex saved to: {vector_db_path}")
print(f"Total vectors: {vector_store.index.ntotal}")
print(f"Embedding dimension: 768 (MedCPT)")
print("\nYou can now run: python demo_for_sme.py")
print("="*80 + "\n")
