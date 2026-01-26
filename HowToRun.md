# How to Run: Automated Protocol Gap Analysis System

## 1. Executive Summary

This system automates the compliance review of clinical trial protocols against FDA guidelines and specific drug label requirements (e.g., TANZEUM).

It replaces manual "Ctrl+F" review with a semantic AI engine that:
1.  **Understands** the intent of FDA rules (not just keywords).
2.  **Retrieves** precise rules relevant to each protocol section.
3.  **Analyzes** the protocol holistically to ensure logic isn't missed across different sections ("De-Siloing").

---

## 2. System Architecture

The core of this system is a **Multi-Agent RAG (Retrieval Augmented Generation)** pipeline. It uses four specialized AI agents to ensure high accuracy.

### The 4-Agent Pipeline
| Agent | Role | Responsibility |
|-------|------|----------------|
| **1. Research Agent** | The Librarian | Uses **MedCPT** (specialized medical embeddings) to find relevant FDA rules. Reranks results using a Cross-Encoder for maximum precision. |
| **2. Generator Agent** | The Analyst | Drafts the initial gap analysis by comparing the protocol section against the retrieved rules. |
| **3. Reviewer Agent** | The Auditor | Scores the analysis for accuracy, completeness, and citations. Assigns a quality score (1-10). |
| **4. Chairman Agent** | The Decision Maker | Reviews the entire chain (Research + Draft + Review) and produces the final, authoritative assessment. |

---

## 3. Key Feature: The "De-Siloing" Strategy

A common failure in AI analysis is "Blindness" — analyzing a section (e.g., *Inclusion Criteria*) in isolation and flagging a gap because the proof is actually in *Statistical Methods*.

**Our Solution: Double-Pass Analysis**
For every section (e.g., "Inclusion Criteria"), the system runs two passes:
1.  **Focused Retrieval Pass**: It searches the vector database using *only* the section topic (e.g., "FDA rules for Inclusion Criteria"). This ensures we get highly relevant rules without noise.
2.  **Holistic Analysis Pass**: The AI checks those rules against the **ENTIRE Protocol Text** (all sections stitched together). This allows it to see that a requirement missing from Section 5 is actually satisfied in Section 9.

---

## 4. Setup & Installation

### Prerequisites
- Python 3.9+
- AWS Credentials (for Bedrock access)
- Pinecone API Key (for vector database)

### Step 1: Clone & Install
```bash
git clone <repo-url>
cd bedrock-llm-comparison
pip install -r requirements.txt
```

### Step 2: Configure Environment
Create a `.env` file in the root directory:
```env
# AWS Bedrock Access
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# Pinecone Vector DB
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENV=us-east-1
```

### Step 3: Index Clinical Rules to Pinecone (One-Time Setup)
Before running gap analysis, you must index the FDA guidelines and drug label rules into Pinecone. This is a **one-time step**.

```bash
python scripts/index_rules_to_pinecone.py
```

**What this does:**
1. Reads `data/input/FDA_diabetes.json` (59 FDA rules)
2. Reads `data/input/drug_label.json` (362 TANZEUM label rules)
3. Generates MedCPT embeddings for each rule
4. Uploads to Pinecone index: `clinical-rules`

**Expected Output:**
```
Loading JSON files...
  ✅ FDA guidelines: 59 rules
  ✅ Drug label: 362 rules
Creating Pinecone index: clinical-rules...
  ✅ Index ready
Uploading 421 chunks to Pinecone...
  ✅ Upload complete
```

> **Note:** You only need to run this once. The index persists in Pinecone cloud.

---

## 5. How to Run

### Run Full Gap Analysis (Recommended)
This command analyzes **all 9 key sections** of the protocol. It is the standard compliance check.
```bash
python scripts/analyze_protocol_gaps.py
```

### Run Specific Section (For Quick Checks)
If you only updated one section (e.g., "Sample Size") and want a quick re-check:
```bash
python scripts/analyze_protocol_gaps.py --sections "Sample Size"
```

### Advanced Options
| Option | Description | Example |
|--------|-------------|---------|
| `--model` | Switch LLM (default: claude) | `--model llama` |
| `--top-k` | Retrieval depth (default: 10) | `--top-k 15` |
| `--sections` | Analyze specific sections only | `--sections "Inclusion Criteria"` |

---

## 6. Understanding the Output

The system generates a comprehensive text report in `data/output/`.
**File Format**: `protocol_gap_analysis_YYYYMMDD_HHMMSS.txt`

### Report Structure
For every section, you will see:
1.  **Protocol Excerpt**: The text that was analyzed.
2.  **Gap Analysis**: The AI's detailed findings of what is missing.
3.  **Chairman Assessment**: Final verdict/summary.
4.  **Sources Consulted**: Exact FDA rule or Drug Label text cited (with page numbers).

### Example Output Snippet
```text
SECTION 3: Primary Objective
================================================================================

🔍 GAP ANALYSIS:
----------------------------------------
The protocol is MISSING the requirement to specify "Central Laboratory" for A1C testing.
FDA Guidelines (Page 6) state: "A1C should generally be measured in a central laboratory..."
...
```

---

## 7. Project Structure

| File | Purpose |
|------|---------|
| `app/agents/multi_agent_rag.py` | Core 4-Agent RAG engine |
| `scripts/analyze_protocol_gaps.py` | Main gap analysis workflow |
| `scripts/index_rules_to_pinecone.py` | Index FDA/Label rules to Pinecone |
| `scripts/run_multi_agent_rag.py` | Ad-hoc query CLI |
| `data/input/DataExtraction_protocol.json` | Protocol data |
| `data/input/FDA_diabetes.json` | FDA guidelines (59 rules) |
| `data/input/drug_label.json` | TANZEUM label (362 rules) |
| `data/output/` | Generated reports |
