#!/usr/bin/env python3
"""
Index FDA Guidelines and Drug Label Rules to Pinecone

This script:
1. Reads FDA_diabetes.json and drug_label.json
2. Converts rules to embeddings using MedCPT
3. Uploads to a NEW Pinecone index: "clinical-rules"

Usage:
    python scripts/index_rules_to_pinecone.py

Requirements:
    - PINECONE_API_KEY in .env
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer


# Pinecone index name for clinical rules
PINECONE_INDEX_NAME = "clinical-rules"
PINECONE_DIMENSION = 768  # MedCPT dimension
MEDCPT_MODEL = "ncbi/MedCPT-Article-Encoder"


def load_json_files():
    """Load JSON files from input directory."""
    input_dir = project_root / "data" / "input"
    
    data = {}
    files = {
        "fda_guidelines": input_dir / "FDA_diabetes.json",
        "drug_label": input_dir / "drug_label.json"
    }
    
    for name, path in files.items():
        if path.exists():
            with open(path, 'r') as f:
                data[name] = json.load(f)
            print(f"✅ Loaded {name}: {path.name}")
        else:
            print(f"❌ Missing: {path}")
    
    return data


def create_rule_chunks(fda_data, drug_label_data):
    """
    Convert JSON rules to chunks for embedding.
    
    Returns list of dicts with:
        - id: unique chunk ID
        - text: rule text for embedding
        - metadata: source info, domain, modality
    """
    chunks = []
    chunk_id = 0
    
    # Process FDA rules
    print("\n📋 Processing FDA Guidelines...")
    for rule in fda_data.get("validation_ruleset", []):
        usdm_tags = rule.get("usdm_tags") or {}
        rule_text = rule.get("rule", "")
        pretext = rule.get("pretext", "")
        
        if not rule_text:
            continue
        
        # Create full text for embedding
        full_text = f"FDA GUIDELINE: {rule_text}"
        if pretext:
            full_text += f"\n\nContext: {pretext}"
        
        chunks.append({
            "id": f"fda_rule_{chunk_id}",
            "text": full_text,
            "metadata": {
                "text": full_text[:1000],  # Main text field for retrieval
                "document": "FDA_diabetes.json",  # For PineconeStore compatibility
                "source": "FDA_diabetes.json",
                "source_type": "fda_guideline",
                "rule_text": rule_text[:1000],
                "pretext": (pretext or "")[:500],
                "rule_id": str(rule.get("rule_id", "")),
                "page": int(rule.get("page_number", 0)),  # 'page' not 'page_number'
                "paragraph_number": 0,
                "domain": usdm_tags.get("domain", ""),
                "modality": usdm_tags.get("modality", ""),
                "chunk_id": f"fda_rule_{chunk_id}"
            }
        })
        chunk_id += 1
    
    print(f"   ✅ FDA rules: {chunk_id}")
    
    # Process Drug Label rules
    print("\n📋 Processing Drug Label...")
    drug_start = chunk_id
    for rule in drug_label_data.get("validation_ruleset", []):
        usdm_tags = rule.get("usdm_tags") or {}
        rule_text = rule.get("rule", "")
        pretext = rule.get("pretext", "")
        
        if not rule_text:
            continue
        
        # Create full text for embedding
        full_text = f"DRUG LABEL (TANZEUM): {rule_text}"
        if pretext:
            full_text += f"\n\nContext: {pretext}"
        
        chunks.append({
            "id": f"drug_label_{chunk_id}",
            "text": full_text,
            "metadata": {
                "text": full_text[:1000],  # Main text field for retrieval
                "document": "drug_label.json",  # For PineconeStore compatibility
                "source": "drug_label.json",
                "source_type": "drug_label",
                "rule_text": rule_text[:1000],
                "pretext": (pretext or "")[:500],
                "rule_id": str(rule.get("rule_id", "")),
                "page": int(rule.get("page_number", 0)),  # 'page' not 'page_number'
                "paragraph_number": 0,
                "domain": usdm_tags.get("domain", ""),
                "modality": usdm_tags.get("modality", ""),
                "chunk_id": f"drug_label_{chunk_id}"
            }
        })
        chunk_id += 1
    
    print(f"   ✅ Drug Label rules: {chunk_id - drug_start}")
    print(f"\n📊 Total chunks: {len(chunks)}")
    
    return chunks


def create_pinecone_index():
    """Create Pinecone index if it doesn't exist."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment")
    
    pc = Pinecone(api_key=api_key)
    
    # Check if index exists
    existing_indexes = pc.list_indexes().names()
    
    if PINECONE_INDEX_NAME in existing_indexes:
        print(f"\n🗑️  Deleting existing index: {PINECONE_INDEX_NAME}")
        pc.delete_index(PINECONE_INDEX_NAME)
        import time
        time.sleep(10)  # Wait for deletion
    
    print(f"\n📦 Creating new Pinecone index: {PINECONE_INDEX_NAME}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=PINECONE_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    
    # Wait for index to be ready
    print("   Waiting for index to be ready...")
    import time
    time.sleep(30)
    
    return pc.Index(PINECONE_INDEX_NAME)


def upload_to_pinecone(chunks, index, model):
    """Generate embeddings and upload to Pinecone."""
    
    print(f"\n🔄 Generating embeddings for {len(chunks)} chunks...")
    
    # Get all texts
    texts = [c["text"] for c in chunks]
    
    # Generate embeddings in batches using SentenceTransformer
    batch_size = 50
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(batch_embeddings.tolist())
        print(f"   Embedded {min(i+batch_size, len(texts))}/{len(texts)}")
    
    print(f"   ✅ Generated {len(all_embeddings)} embeddings")
    
    # Prepare vectors for upsert
    print("\n📤 Uploading to Pinecone...")
    vectors = []
    for i, chunk in enumerate(chunks):
        vectors.append({
            "id": chunk["id"],
            "values": all_embeddings[i],
            "metadata": chunk["metadata"]
        })
    
    # Upsert in batches
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)
        print(f"   Upserted {min(i+batch_size, len(vectors))}/{len(vectors)}")
    
    print(f"\n✅ Successfully uploaded {len(vectors)} vectors to Pinecone")
    
    # Verify
    import time
    time.sleep(5)  # Wait for index to update
    stats = index.describe_index_stats()
    print(f"   Index stats: {stats['total_vector_count']} vectors")
    
    return len(vectors)


def main():
    print("\n" + "="*80)
    print("INDEX CLINICAL RULES TO PINECONE")
    print("="*80)
    
    # 1. Load JSON files
    print("\n1️⃣  Loading JSON files...")
    data = load_json_files()
    
    if "fda_guidelines" not in data or "drug_label" not in data:
        print("❌ Missing required JSON files!")
        return
    
    # 2. Create rule chunks
    print("\n2️⃣  Creating rule chunks...")
    chunks = create_rule_chunks(data["fda_guidelines"], data["drug_label"])
    
    # 3. Initialize embedding model
    print("\n3️⃣  Loading MedCPT embedding model...")
    print(f"   Loading: {MEDCPT_MODEL}")
    model = SentenceTransformer(MEDCPT_MODEL)
    print(f"   ✅ Model loaded: {MEDCPT_MODEL}")
    
    # 4. Create Pinecone index
    print("\n4️⃣  Setting up Pinecone index...")
    index = create_pinecone_index()
    
    # 5. Upload embeddings
    print("\n5️⃣  Uploading embeddings to Pinecone...")
    count = upload_to_pinecone(chunks, index, model)
    
    # Summary
    print("\n" + "="*80)
    print("✅ INDEXING COMPLETE")
    print("="*80)
    print(f"\n   Index Name: {PINECONE_INDEX_NAME}")
    print(f"   Vectors: {count}")
    print(f"   FDA Rules: {len(data['fda_guidelines'].get('validation_ruleset', []))}")
    print(f"   Drug Label Rules: {len(data['drug_label'].get('validation_ruleset', []))}")
    print(f"\n   Use this index with: --index-name {PINECONE_INDEX_NAME}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
