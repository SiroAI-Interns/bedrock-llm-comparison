#!/usr/bin/env python3
"""
Gap Analysis Test Script

Tests the protocol data against FDA guidelines and drug label rules
to identify compliance gaps.

Usage:
    python scripts/test_gap_analysis.py
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


def extract_protocol_queries(protocol_data):
    """
    Extract key elements from protocol to form queries.
    Returns list of (category, query_text) tuples.
    """
    queries = []
    mapped = protocol_data.get("mappedFields", {})
    
    # Key fields to check against guidelines
    fields_to_check = [
        ("Inclusion Criteria", "inclusionCriteria"),
        ("Exclusion Criteria", "exclusionCriteria"),
        ("Primary Objective", "primaryObjective"),
        ("Secondary Objective", "secondaryObjective"),
        ("Sample Size", "sampleSizeMin"),
        ("Study Design", "studyDesign"),
        ("Safety Assessment", "safetyAssessment"),
        ("Outcome Measures", "outcomeMeasures"),
    ]
    
    for category, field_name in fields_to_check:
        field_data = mapped.get(field_name, {})
        section_text = field_data.get("section_text", "")
        
        if section_text and section_text.strip():
            # Create queries based on protocol content
            queries.append({
                "category": category,
                "protocol_text": section_text[:2000],  # Limit length
                "query": f"What are the FDA requirements for {category.lower()} in diabetes trials?"
            })
    
    return queries


def extract_rules(data, source_name):
    """Extract rules from FDA guidelines or drug label."""
    rules = []
    ruleset = data.get("validation_ruleset", [])
    
    for rule_item in ruleset:
        usdm_tags = rule_item.get("usdm_tags") or {}
        rules.append({
            "source": source_name,
            "rule": rule_item.get("rule", ""),
            "pretext": rule_item.get("pretext", ""),
            "rule_id": rule_item.get("rule_id", ""),
            "domain": usdm_tags.get("domain", ""),
            "modality": usdm_tags.get("modality", ""),
        })
    
    return rules


def check_protocol_against_rule(protocol_text, rule):
    """
    Simple keyword-based check if protocol addresses a rule.
    Returns True if there might be a gap (rule not addressed).
    """
    rule_text = rule["rule"].lower()
    protocol_lower = protocol_text.lower()
    
    # Extract key terms from rule
    key_terms = []
    
    # Medical terms to check
    medical_terms = [
        "hba1c", "a1c", "hypoglycemia", "hyoglycemic",
        "pancreatitis", "thyroid", "mtc", "men-2", "men 2",
        "renal", "kidney", "egfr", "cardiovascular",
        "pregnancy", "pregnant", "lactation", "breastfeeding",
        "insulin", "sulfonylurea", "metformin",
        "type 1 diabetes", "type 2 diabetes",
        "ketoacidosis", "gastrointestinal",
        "hepatotoxicity", "liver", "alt", "bilirubin",
        "hypersensitivity", "allergy", "allergic",
    ]
    
    for term in medical_terms:
        if term in rule_text:
            key_terms.append(term)
    
    # If we found key terms, check if protocol addresses them
    if key_terms:
        missing_terms = [t for t in key_terms if t not in protocol_lower]
        if missing_terms:
            return True, missing_terms
            
    return False, []


def analyze_gaps_simple(protocol_data, fda_rules, drug_rules):
    """
    Simple gap analysis without using RAG.
    Checks if protocol addresses key rules.
    """
    gaps = []
    mapped = protocol_data.get("mappedFields", {})
    
    # Combine all protocol text
    all_protocol_text = ""
    for field_name, field_data in mapped.items():
        section_text = field_data.get("section_text", "")
        if section_text:
            all_protocol_text += f"\n{section_text}"
    
    # Check FDA rules
    print("\n" + "="*80)
    print("CHECKING FDA GUIDELINES")
    print("="*80)
    
    for rule in fda_rules[:30]:  # Check first 30 rules
        if rule["modality"] == "Mandatory":
            has_gap, missing = check_protocol_against_rule(all_protocol_text, rule)
            if has_gap:
                gaps.append({
                    "source": "FDA Guidelines",
                    "rule": rule["rule"],
                    "domain": rule["domain"],
                    "modality": rule["modality"],
                    "missing_terms": missing,
                    "severity": "HIGH" if rule["modality"] == "Mandatory" else "MEDIUM"
                })
    
    # Check Drug Label rules
    print("\n" + "="*80)
    print("CHECKING DRUG LABEL")
    print("="*80)
    
    for rule in drug_rules[:30]:
        if rule["modality"] == "Mandatory":
            has_gap, missing = check_protocol_against_rule(all_protocol_text, rule)
            if has_gap:
                gaps.append({
                    "source": "Drug Label (TANZEUM)",
                    "rule": rule["rule"],
                    "domain": rule["domain"],
                    "modality": rule["modality"],
                    "missing_terms": missing,
                    "severity": "HIGH" if rule["modality"] == "Mandatory" else "MEDIUM"
                })
    
    return gaps


def analyze_gaps_with_rag(protocol_data, fda_data, drug_data):
    """
    Use RAG system to analyze gaps with LLM intelligence.
    """
    from app.agents.multi_agent_rag import MultiAgentRAG
    
    print("\n" + "="*80)
    print("USING RAG SYSTEM FOR GAP ANALYSIS")
    print("="*80)
    
    # Initialize RAG with FAISS (local) for now
    # Switch to pinecone for production
    rag = MultiAgentRAG(
        backend="faiss",
        llm_model="claude",
        top_k=5,
        initial_candidates=20
    )
    
    gaps = []
    queries = extract_protocol_queries(protocol_data)
    
    for q in queries[:3]:  # Test with first 3 queries
        print(f"\n📝 Checking: {q['category']}")
        print(f"   Query: {q['query'][:100]}...")
        
        # Run query through RAG
        try:
            result = rag.process_query(q['query'])
            
            # Store result for analysis
            gaps.append({
                "category": q["category"],
                "query": q["query"],
                "protocol_excerpt": q["protocol_text"][:500],
                "rag_response": result.get("response", "")[:1000],
                "sources_used": len(result.get("sources", [])),
            })
            
            print(f"   ✅ Got response with {len(result.get('sources', []))} sources")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            gaps.append({
                "category": q["category"],
                "query": q["query"],
                "error": str(e)
            })
    
    return gaps


def save_gap_report(gaps, output_path):
    """Save gap analysis report to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GAP ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Potential Gaps Found: {len(gaps)}\n\n")
        
        high_severity = [g for g in gaps if g.get("severity") == "HIGH"]
        medium_severity = [g for g in gaps if g.get("severity") == "MEDIUM"]
        
        f.write(f"HIGH Severity: {len(high_severity)}\n")
        f.write(f"MEDIUM Severity: {len(medium_severity)}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("DETAILED GAPS\n")
        f.write("-"*80 + "\n\n")
        
        for i, gap in enumerate(gaps, 1):
            f.write(f"\n[GAP {i}] - {gap.get('severity', 'N/A')}\n")
            f.write(f"Source: {gap.get('source', 'N/A')}\n")
            f.write(f"Domain: {gap.get('domain', 'N/A')}\n")
            f.write(f"Rule: {gap.get('rule', 'N/A')}\n")
            if gap.get('missing_terms'):
                f.write(f"Missing Terms: {', '.join(gap['missing_terms'])}\n")
            f.write("-"*40 + "\n")
    
    print(f"\n✅ Report saved to: {output_path}")
    return output_path


def main():
    print("\n" + "="*80)
    print("PROTOCOL GAP ANALYSIS TEST")
    print("="*80)
    
    # 1. Load JSON files
    print("\n1️⃣  Loading JSON files...")
    data = load_json_files()
    
    if not all(k in data for k in ["protocol", "fda_guidelines", "drug_label"]):
        print("❌ Missing required JSON files!")
        return
    
    # 2. Extract rules
    print("\n2️⃣  Extracting rules...")
    fda_rules = extract_rules(data["fda_guidelines"], "FDA Guidelines")
    drug_rules = extract_rules(data["drug_label"], "Drug Label")
    print(f"   FDA Rules: {len(fda_rules)}")
    print(f"   Drug Label Rules: {len(drug_rules)}")
    
    # 3. Extract protocol queries
    print("\n3️⃣  Extracting protocol elements...")
    queries = extract_protocol_queries(data["protocol"])
    print(f"   Protocol Elements: {len(queries)}")
    for q in queries:
        print(f"   - {q['category']}")
    
    # 4. Simple gap analysis (keyword-based)
    print("\n4️⃣  Running simple gap analysis...")
    simple_gaps = analyze_gaps_simple(data["protocol"], fda_rules, drug_rules)
    print(f"   Found {len(simple_gaps)} potential gaps")
    
    # 5. Save report
    output_dir = project_root / "data" / "output"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"gap_analysis_{timestamp}.txt"
    save_gap_report(simple_gaps, report_path)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n📊 Analysis Complete!")
    print(f"   Protocol: DataExtraction_protocol.json")
    print(f"   FDA Rules Checked: {len(fda_rules)}")
    print(f"   Drug Label Rules Checked: {len(drug_rules)}")
    print(f"   Potential Gaps: {len(simple_gaps)}")
    print(f"\n📁 Report: {report_path}")
    
    # Show first few gaps
    if simple_gaps:
        print("\n" + "-"*80)
        print("TOP GAPS (First 5):")
        print("-"*80)
        for gap in simple_gaps[:5]:
            print(f"\n⚠️  [{gap['severity']}] {gap['source']}")
            print(f"   Rule: {gap['rule'][:100]}...")
            if gap.get('missing_terms'):
                print(f"   Missing: {', '.join(gap['missing_terms'][:5])}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
