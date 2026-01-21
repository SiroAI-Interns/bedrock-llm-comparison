# Medical Protocol RAG System

A multi-agent RAG (Retrieval-Augmented Generation) system for querying medical protocols and clinical trial guidelines.

## Features

- **MedCPT Embeddings**: Medical-optimized text embeddings (768 dimensions)
- **Cross-Encoder Reranking**: Improved precision with semantic re-scoring
- **4-Agent Pipeline**: Research → Generator → Reviewer → Chairman
- **Multiple LLM Support**: Claude, Llama, Mistral, GPT-OSS, Titan
- **Auto-Save Results**: All queries saved to `data/output/`

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up AWS Credentials

```bash
cp .env.example .env
# Edit .env with your AWS credentials
```

```env
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1
```

### 3. Add Your PDFs

Place your medical protocol PDFs in:
```
data/input/protocols/
```

### 4. Build the Vector Index

```bash
python scripts/rebuild_index.py
```

This will:
- Process all PDFs in `data/input/protocols/`
- Create MedCPT embeddings
- Save index to `data/vectordb_faiss_medcpt/`

### 5. Run a Query

```bash
python scripts/run_multi_agent_rag.py "What are the HbA1c requirements?"
```

**Options:**
```bash
# Use different LLM
python scripts/run_multi_agent_rag.py "..." --model llama

# Get more sources
python scripts/run_multi_agent_rag.py "..." --top-k 15
```

**Available Models:** `claude`, `llama`, `mistral`, `gpt-oss`, `titan`

---

## Output

Results are automatically saved to:
```
data/output/rag_evaluation_YYYYMMDD_HHMMSS.txt
```

Each file contains:
- Query
- Top 10 source documents (with exact paragraph text)
- Research synthesis
- Generated response with citations
- Reviewer evaluation
- Chairman analysis

---

## Project Structure

```
bedrock-llm-comparison/
├── app/
│   ├── agents/
│   │   └── multi_agent_rag.py    # Main 4-agent RAG system
│   ├── services/
│   │   ├── pdf_processor.py      # PDF extraction & chunking
│   │   ├── vector_store.py       # FAISS vector database
│   │   └── reranker.py           # Cross-encoder reranking
│   └── core/
│       └── unified_bedrock_client.py  # AWS Bedrock LLM
│
├── scripts/
│   ├── run_multi_agent_rag.py    # CLI entry point
│   └── rebuild_index.py          # Build vector index
│
├── data/
│   ├── input/protocols/          # Your PDF files
│   ├── output/                   # Query results
│   └── vectordb_faiss_medcpt/    # Vector index
│
└── config/
    └── settings.py               # Configuration
```

---

## How It Works

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│ 1. MedCPT Embedding                 │
│    Query → 768-dim vector           │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2. FAISS Search                     │
│    Find top 50 similar chunks       │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 3. Cross-Encoder Reranking          │
│    Select top 10 most relevant      │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 4. Multi-Agent Pipeline             │
│    Research → Generator →           │
│    Reviewer → Chairman              │
└─────────────────────────────────────┘
    │
    ▼
Answer + Sources (saved to file)
```

---

## Requirements

- Python 3.10+
- AWS Account with Bedrock access
- ~2GB disk space for models

---

## Troubleshooting

### "Vector database not found"
Run `python scripts/rebuild_index.py` first.

### "No PDFs found"
Place PDF files in `data/input/protocols/`.

### Slow import times
If using iCloud, ensure files are downloaded locally.

---

## License

MIT License

---

## Archived Scripts (Legacy)

The following scripts have been moved to `archive/scripts/` for organization. They are still available for team members:

| Script | Purpose |
|--------|---------|
| `multi_agent_evaluator.py` | Compare multiple LLMs (Claude, Llama, Mistral, etc.) |
| `run_all_models.py` | Run all LLM models in parallel |
| `compare_embeddings*.py` | Embedding model comparisons |
| `demo_*.py` | Demo scripts for presentations |
| `test_*.py` | Testing and benchmarking scripts |

**To use archived scripts:**
```bash
python archive/scripts/run_all_models.py
```
