# scripts/compare_embeddings.py
"""
Compare MedCPT vs PubMedBERT vs BioBERT retrieval quality on FDA guidelines.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.vector_store import VectorStore
from app.services.pdf_processor import PDFProcessor


def test_embedding_model(model_name: str, query_encoder: str = None):
    """Test retrieval with a specific embedding model."""
    
    print("\n" + "="*70)
    print(f"🧪 TESTING: {model_name}")
    print("="*70)
    
    # Paths
    input_dir = project_root / "data" / "input" / "protocols"
    vector_db_path = project_root / "data" / f"vectordb_{model_name.replace('/', '_')}"
    
    # Initialize PDF processor
    pdf_processor = PDFProcessor()
    
    # Process PDFs using the built-in method
    print(f"\n📚 Processing PDFs from: {input_dir}\n")
    all_chunks = pdf_processor.process_directory_with_chunks(str(input_dir))
    
    if not all_chunks:
        print("❌ No chunks created. Exiting.")
        return None
    
    # Initialize vector store with specified model
    print(f"\n📥 Loading embedding model: {model_name}")
    vector_store = VectorStore(
        embedding_model=model_name,
        query_encoder=query_encoder,
        vector_db_path=vector_db_path
    )
    
    # Build index
    vector_store.build_index(all_chunks)
    vector_store.save()
    
    # Test queries
    test_queries = [
        "What are the HbA1c measurement requirements for diabetes device clinical trials?",
        "What sample size does FDA recommend for early feasibility studies?",
        "What are the safety endpoint requirements for T2DM device trials?",
    ]
    
    print("\n" + "="*70)
    print("🔍 RETRIEVAL RESULTS")
    print("="*70)
    
    for idx, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"📝 Query {idx}: {query}")
        print(f"{'='*70}")
        
        results = vector_store.search(query, top_k=5)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"\n[{i}] {result['source']} - Page {result['page']}")
                print(f"    Distance: {result['score']:.4f}")
                print(f"    Preview: {result['text'][:150]}...")
        else:
            print("⚠️  No results found")
    
    return vector_store


def main():
    """Compare MedCPT, PubMedBERT, and BioBERT side by side."""
    
    print("="*70)
    print("🆚 MEDICAL EMBEDDING MODEL COMPARISON")
    print("="*70)
    print("Comparing retrieval quality on FDA medical device guidelines")
    print()
    print("Models being tested:")
    print("  1. MedCPT         - Dual encoder, PubMed contrastive learning")
    print("  2. PubMedBERT     - Single encoder, PubMed masked LM")
    print("  3. BioBERT        - Single encoder, PubMed + PMC masked LM")
    print()
    
    # Test 1: MedCPT (Dual Encoder)
    print("\n" + "🏥"*35)
    print("TEST 1: MedCPT (Medical-Specific Dual Encoder)")
    print("🏥"*35)
    medcpt_store = test_embedding_model(
        model_name="ncbi/MedCPT-Article-Encoder",
        query_encoder="ncbi/MedCPT-Query-Encoder"
    )
    
    if medcpt_store is None:
        print("\n❌ MedCPT test failed. Exiting.")
        return
    
    # Test 2: PubMedBERT (Single Encoder)
    print("\n" + "📚"*35)
    print("TEST 2: PubMedBERT (Single Encoder)")
    print("📚"*35)
    pubmed_store = test_embedding_model(
        model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    )
    
    # Test 3: BioBERT (Single Encoder)
    print("\n" + "🧬"*35)
    print("TEST 3: BioBERT (Single Encoder)")
    print("🧬"*35)
    biobert_store = test_embedding_model(
        model_name="dmis-lab/biobert-base-cased-v1.1"
    )
    
    print("\n" + "="*70)
    print("✅ COMPARISON COMPLETE!")
    print("="*70)
    print("\n📊 COMPARISON SUMMARY")
    print("="*70)
    print("\nVector databases saved to:")
    print(f"  • MedCPT:      data/vectordb_ncbi_MedCPT-Article-Encoder/")
    print(f"  • PubMedBERT:  data/vectordb_microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract/")
    print(f"  • BioBERT:     data/vectordb_dmis-lab_biobert-base-cased-v1.1/")
    print()
    print("📊 KEY OBSERVATIONS:")
    print("   • Lower distance scores = Better relevance")
    print("   • Check which pages are retrieved for each query")
    print("   • Compare ranking order (is most relevant page #1?)")
    print()
    print("💡 EXPECTED RESULTS:")
    print("   • MedCPT:      Distances 30-40 (Best - dual encoder)")
    print("   • PubMedBERT:  Distances 40-60 (Good - single encoder)")
    print("   • BioBERT:     Distances 40-65 (Good - single encoder)")
    print()
    print("🎯 WHY COMPARE THESE THREE?")
    print("   • MedCPT:      Trained specifically for retrieval (contrastive)")
    print("   • PubMedBERT:  Trained on PubMed abstracts only")
    print("   • BioBERT:     Trained on PubMed + PMC full-text articles")
    print()


if __name__ == "__main__":
    main()
