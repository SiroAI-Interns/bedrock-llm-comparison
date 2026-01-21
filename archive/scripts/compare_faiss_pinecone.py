# scripts/compare_faiss_vs_pinecone.py

from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.pdf_processor import PDFProcessor
from app.services.vector_store import VectorStore          # FAISS
from app.services.vector_store_pinecone import PineconeVectorStore  # Pinecone


QUERY = "What are the HbA1c measurement requirements for diabetes device clinical trials?"


def build_chunks():
    pdf_dir = project_root / "data" / "input" / "protocols"
    proc = PDFProcessor()
    return proc.process_directory_with_chunks(str(pdf_dir))


def test_faiss(chunks):
    print("\n=== FAISS (local) ===")
    vs = VectorStore(
        embedding_model="ncbi/MedCPT-Article-Encoder",
        query_encoder="ncbi/MedCPT-Query-Encoder",
        vector_db_path=project_root / "data" / "vectordb_faiss_medcpt",
    )
    vs.build_index(chunks)
    vs.save()
    res = vs.search(QUERY, top_k=5)
    for r in res:
        print(f"Rank {r['rank']}: page {r['page']} | score {r['score']:.4f}")
    return res


def test_pinecone(chunks):
    print("\n=== Pinecone (cloud) ===")
    vs = PineconeVectorStore(
        index_name="siro",  # same name you created in UI
        embedding_model="ncbi/MedCPT-Article-Encoder",
        query_encoder="ncbi/MedCPT-Query-Encoder",
    )
    vs.build_index(chunks)
    res = vs.search(QUERY, top_k=5)
    for r in res:
        print(f"Rank {r['rank']}: page {r['page']} | score {r['score']:.4f}")
    return res


def main():
    chunks = build_chunks()
    faiss_res = test_faiss(chunks)
    pine_res = test_pinecone(chunks)

    print("\n=== Quick sanity check ===")
    print("FAISS pages:", [r["page"] for r in faiss_res])
    print("Pinecone pages:", [r["page"] for r in pine_res])


if __name__ == "__main__":
    main()
