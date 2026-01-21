#!/usr/bin/env python3
"""
Quick test of high-recall retrieval with progress output.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("="*80)
print("HIGH-RECALL RETRIEVAL - QUICK TEST")
print("="*80)

print("\n1️⃣  Importing modules...")
from app.services.vector_store import VectorStore
from app.services.high_recall_retriever import create_high_recall_retriever
from app.services.pdf_processor import PDFProcessor

print("✅ Imports successful")

print("\n2️⃣  Setting up paths...")
input_dir = project_root / "data" / "input" / "protocols"
vector_db_path = project_root / "data" / "vectordb"
print(f"   Input: {input_dir}")
print(f"   Vector DB: {vector_db_path}")

print("\n3️⃣  Loading vector store (this may take 2-3 minutes on first run)...")
print("   Loading MedCPT embedding model...")

vector_store = VectorStore(
    embedding_model="medcpt",
    vector_db_path=vector_db_path,
    use_reranking=False
)

print("\n4️⃣  Loading or building index...")
if vector_store.load():
    print("✅ Loaded existing index")
else:
    print("⚠️  No existing index found, building from PDFs...")
    pdf_processor = PDFProcessor()
    chunks = pdf_processor.process_directory_with_chunks(str(input_dir))
    
    if not chunks:
        print("❌ No PDFs found. Please add PDFs to data/input/protocols/")
        sys.exit(1)
    
    print(f"   Processing {len(chunks)} chunks...")
    vector_store.build_index(chunks)
    vector_store.save()
    print("✅ Index built and saved")

print("\n5️⃣  Creating high-recall retriever...")
retriever = create_high_recall_retriever(
    vector_store=vector_store,
    use_reranking=True,
    reranker_model="ms-marco-mini",
    reranking_strategy="cross-encoder"
)
print("✅ Retriever ready")

print("\n6️⃣  Testing with sample query...")
query = "What are the HbA1c measurement requirements for diabetes device trials?"
print(f"   Query: {query}")

print("\n" + "─"*80)
results = retriever.retrieve(query, verbose=True)
print("─"*80)

print(f"\n✅ SUCCESS! Retrieved {len(results)} documents")
print("\nTop 5 results:")
for i, doc in enumerate(results[:5], 1):
    print(f"\n{i}. [{doc['source']} - Page {doc['page']}]")
    print(f"   Rerank Score: {doc.get('rerank_score', 0):.4f}")
    print(f"   {doc['text'][:100]}...")

print("\n" + "="*80)
print("✅ HIGH-RECALL RETRIEVAL TEST COMPLETE")
print("="*80)
print("\nKey takeaway: Retrieved more candidates, then reranked to find most relevant.")
print("This prevents missing critical FDA requirements.\n")
