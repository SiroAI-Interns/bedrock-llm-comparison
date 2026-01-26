#!/usr/bin/env python3
"""
Gap Analysis with Multi-Agent RAG

Uses the existing multi-agent RAG system to analyze protocol data
against FDA guidelines and drug label rules.

Flow:
1. Extract protocol elements → Generate queries
2. Use Multi-Agent RAG to search FDA/Drug Label rules
3. 4-Agent pipeline analyzes gaps
4. Save gap report

Usage:
    python scripts/test_gap_analysis_rag.py
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


def load_json_files():
    """Load all JSON files from input directory."""
    input_dir = project_root / "data" / "input"
    
    files = {
        "protocol": input_dir / "DataExtraction_protocol.json",
        "fda_guidelines": input_dir / "FDA_diabetes.json", 
        "drug_label": input_dir / "drug_label.json"
    }
    
    data = {}
    for name, path in files.items():
        if path.exists():
            with open(path, 'r') as f:
                data[name] = json.load(f)
            print(f"✅ Loaded {name}: {path.name}")
        else:
            print(f"❌ Missing: {path}")
            
    return data


def extract_protocol_elements(protocol_data):
    """
    Extract key elements from protocol to check against guidelines.
    """
    elements = []
    mapped = protocol_data.get("mappedFields", {})
    
    fields_to_check = [
        ("Inclusion Criteria", "inclusionCriteria"),
        ("Exclusion Criteria", "exclusionCriteria"),
        ("Primary Objective", "primaryObjective"),
        ("Secondary Objective", "secondaryObjective"),
        ("Sample Size", "sampleSizeMin"),
        ("Safety Assessment", "safetyAssessment"),
        ("Outcome Measures", "outcomeMeasures"),
    ]
    
    for category, field_name in fields_to_check:
        field_data = mapped.get(field_name, {})
        section_text = field_data.get("section_text", "")
        header = field_data.get("header", "")
        
        if section_text and section_text.strip():
            elements.append({
                "category": category,
                "header": header,
                "text": section_text,
            })
    
    return elements


def create_rules_index(fda_data, drug_label_data):
    """
    Convert JSON rules to chunks that can be indexed by the RAG system.
    Returns list of chunks in the format expected by vector store.
    """
    chunks = []
    
    # Process FDA rules
    for rule in fda_data.get("validation_ruleset", []):
        usdm_tags = rule.get("usdm_tags") or {}
        chunk_text = f"FDA GUIDELINE: {rule.get('rule', '')}"
        if rule.get('pretext'):
            chunk_text += f"\n\nContext: {rule.get('pretext', '')}"
        
        chunks.append({
            "text": chunk_text,
            "source": "FDA_diabetes.json",
            "page": rule.get("page_number", "N/A"),
            "paragraph_number": 0,
            "rule_id": rule.get("rule_id", ""),
            "domain": usdm_tags.get("domain", ""),
            "modality": usdm_tags.get("modality", ""),
        })
    
    # Process Drug Label rules
    for rule in drug_label_data.get("validation_ruleset", []):
        usdm_tags = rule.get("usdm_tags") or {}
        chunk_text = f"DRUG LABEL (TANZEUM): {rule.get('rule', '')}"
        if rule.get('pretext'):
            chunk_text += f"\n\nContext: {rule.get('pretext', '')}"
        
        chunks.append({
            "text": chunk_text,
            "source": "drug_label.json",
            "page": rule.get("page_number", "N/A"),
            "paragraph_number": 0,
            "rule_id": rule.get("rule_id", ""),
            "domain": usdm_tags.get("domain", ""),
            "modality": usdm_tags.get("modality", ""),
        })
    
    return chunks


def run_gap_analysis_with_rag(protocol_elements, rules_chunks, llm_model="claude"):
    """
    Use the Multi-Agent RAG system to analyze gaps.
    """
    from app.agents.multi_agent_rag import MultiAgentRAG
    from app.services.vector_store import VectorStore
    
    print("\n" + "="*80)
    print("INITIALIZING MULTI-AGENT RAG SYSTEM")
    print("="*80)
    
    # Create a temporary vector store with rules
    print("\n📦 Building temporary vector index from rules...")
    vector_store = VectorStore()
    vector_store.build_index(rules_chunks)
    print(f"   ✅ Indexed {len(rules_chunks)} rules")
    
    # Initialize Multi-Agent RAG with the vector store
    rag = MultiAgentRAG(
        backend="faiss",  # Using local FAISS
        llm_model=llm_model,
        top_k=10,
        initial_candidates=30,
    )
    
    # Override the vector store with our rules index
    rag.vector_store = vector_store
    
    results = []
    
    print("\n" + "="*80)
    print("RUNNING GAP ANALYSIS WITH 4-AGENT PIPELINE")
    print("="*80)
    
    for i, element in enumerate(protocol_elements, 1):
        category = element["category"]
        protocol_text = element["text"][:1500]  # Limit text length
        
        print(f"\n{'='*60}")
        print(f"[{i}/{len(protocol_elements)}] Analyzing: {category}")
        print("="*60)
        
        # Create query for gap analysis
        query = f"""
        Analyze the following clinical trial protocol section for compliance with FDA guidelines and drug label requirements.
        
        PROTOCOL SECTION: {category}
        ---
        {protocol_text}
        ---
        
        Question: What are the FDA requirements and drug label requirements for {category.lower()}? 
        Are there any gaps or missing elements in this protocol section compared to the requirements?
        """
        
        print(f"🔍 Searching relevant rules...")
        
        try:
            # Run through multi-agent RAG pipeline
            result = rag.evaluate(query)
            
            print(f"✅ Analysis complete")
            print(f"   Sources used: {len(result.get('sources', []))}")
            
            # Collect results
            results.append({
                "category": category,
                "protocol_excerpt": protocol_text[:500],
                "sources_found": len(result.get("sources", [])),
                "research_synthesis": result.get("research_context", "")[:1000],
                "generated_response": result.get("response", "")[:2000],
                "reviewer_feedback": result.get("reviewer_feedback", {}),
                "chairman_analysis": result.get("chairman_analysis", "")[:1000],
                "sources": [
                    {
                        "text": s.get("text", "")[:300],
                        "source": s.get("source", ""),
                        "score": s.get("score", 0)
                    }
                    for s in result.get("sources", [])[:5]
                ]
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                "category": category,
                "error": str(e)
            })
    
    return results


def save_rag_gap_report(results, output_path):
    """Save the RAG-based gap analysis report."""
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MULTI-AGENT RAG GAP ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("="*80 + "\n\n")
        
        f.write("This report uses the 4-agent pipeline (Research → Generator → Reviewer → Chairman)\n")
        f.write("to analyze protocol compliance with FDA guidelines and drug label requirements.\n\n")
        
        f.write(f"Total Protocol Elements Analyzed: {len(results)}\n")
        f.write("-"*80 + "\n\n")
        
        for i, result in enumerate(results, 1):
            f.write("\n" + "="*80 + "\n")
            f.write(f"SECTION {i}: {result.get('category', 'Unknown')}\n")
            f.write("="*80 + "\n\n")
            
            if result.get("error"):
                f.write(f"❌ ERROR: {result['error']}\n")
                continue
            
            # Protocol excerpt
            f.write("📋 PROTOCOL EXCERPT:\n")
            f.write("-"*40 + "\n")
            f.write(result.get("protocol_excerpt", "N/A")[:500] + "...\n\n")
            
            # Sources used
            f.write(f"📚 SOURCES FOUND: {result.get('sources_found', 0)}\n")
            f.write("-"*40 + "\n")
            for j, src in enumerate(result.get("sources", []), 1):
                f.write(f"  [{j}] {src.get('source', 'N/A')} (Score: {src.get('score', 0):.3f})\n")
                f.write(f"      {src.get('text', '')[:100]}...\n\n")
            
            # Research synthesis
            f.write("\n🔬 RESEARCH SYNTHESIS:\n")
            f.write("-"*40 + "\n")
            f.write(result.get("research_synthesis", "N/A")[:800] + "\n\n")
            
            # Generated response (gap analysis)
            f.write("\n📝 GAP ANALYSIS:\n")
            f.write("-"*40 + "\n")
            f.write(result.get("generated_response", "N/A") + "\n\n")
            
            # Reviewer feedback
            reviewer = result.get("reviewer_feedback", {})
            if reviewer:
                f.write("\n👀 REVIEWER EVALUATION:\n")
                f.write("-"*40 + "\n")
                f.write(f"  Accuracy: {reviewer.get('accuracy_score', 'N/A')}/10\n")
                f.write(f"  Completeness: {reviewer.get('completeness_score', 'N/A')}/10\n")
                f.write(f"  Citations: {reviewer.get('citation_score', 'N/A')}/10\n\n")
            
            # Chairman analysis
            f.write("\n🎖️ CHAIRMAN ANALYSIS:\n")
            f.write("-"*40 + "\n")
            f.write(result.get("chairman_analysis", "N/A")[:800] + "\n")
            
            f.write("\n")
    
    print(f"\n✅ Report saved to: {output_path}")
    return output_path


def main():
    print("\n" + "="*80)
    print("MULTI-AGENT RAG GAP ANALYSIS")
    print("Protocol vs FDA Guidelines + Drug Label")
    print("="*80)
    
    # 1. Load JSON files
    print("\n1️⃣  Loading JSON files...")
    data = load_json_files()
    
    if not all(k in data for k in ["protocol", "fda_guidelines", "drug_label"]):
        print("❌ Missing required JSON files!")
        return
    
    # 2. Extract protocol elements
    print("\n2️⃣  Extracting protocol elements...")
    protocol_elements = extract_protocol_elements(data["protocol"])
    print(f"   Found {len(protocol_elements)} elements to analyze:")
    for elem in protocol_elements:
        print(f"   - {elem['category']}")
    
    # 3. Create rules chunks for indexing
    print("\n3️⃣  Creating rules index...")
    rules_chunks = create_rules_index(data["fda_guidelines"], data["drug_label"])
    print(f"   Total rules: {len(rules_chunks)}")
    print(f"   - FDA: {len(data['fda_guidelines'].get('validation_ruleset', []))}")
    print(f"   - Drug Label: {len(data['drug_label'].get('validation_ruleset', []))}")
    
    # 4. Run gap analysis with Multi-Agent RAG
    print("\n4️⃣  Running Multi-Agent RAG gap analysis...")
    print("   This uses the 4-agent pipeline: Research → Generator → Reviewer → Chairman")
    
    # Analyze first 3 elements for testing (remove [:3] for full analysis)
    results = run_gap_analysis_with_rag(
        protocol_elements[:3],  # Testing with first 3
        rules_chunks,
        llm_model="claude"
    )
    
    # 5. Save report
    output_dir = project_root / "data" / "output"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"rag_gap_analysis_{timestamp}.txt"
    save_rag_gap_report(results, report_path)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n✅ Multi-Agent RAG Gap Analysis Complete!")
    print(f"   Protocol Elements Analyzed: {len(results)}")
    print(f"   Rules Searched: {len(rules_chunks)}")
    print(f"\n📁 Report: {report_path}")
    print("="*80)


if __name__ == "__main__":
    main()
