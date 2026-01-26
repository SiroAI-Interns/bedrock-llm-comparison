#!/usr/bin/env python3
"""
Automated Protocol Gap Analysis Tool

This script performs a semantic gap analysis of clinical trial protocols against
FDA guidelines and specific drug label requirements (e.g., TANZEUM).

Methodology:
    1.  Extracts key sections from the protocol JSON.
    2.  Constructs a virtual "Full Protocol Context" to enable cross-referencing.
    3.  Uses a Multi-Agent RAG system to:
        a. Retrieve specific rules for each section (Focused Search).
        b. Analyze the section against those rules, using the full protocol context
           to verify if requirements are met elsewhere (Holistic Analysis).
    4.  Generates a comprehensive compliance report.

Usage:
    python scripts/analyze_protocol_gaps.py [--model claude] [--sections "Inclusion Criteria"]

"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from app.agents.multi_agent_rag import MultiAgentRAG

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Constants
PROTOCOL_PATH = project_root / "data" / "input" / "DataExtraction_protocol.json"
OUTPUT_DIR = project_root / "data" / "output"

SECTION_CONFIGS = [
    ("Inclusion Criteria", "inclusionCriteria"),
    ("Exclusion Criteria", "exclusionCriteria"),
    ("Primary Objective", "primaryObjective"),
    ("Secondary Objective", "secondaryObjective"),
    ("Sample Size", "sampleSizeMin"),
    ("Study Design", "studyDesign"),
    ("Safety Assessment", "safetyAssessment"),
    ("Outcome Measures", "outcomeMeasures"),
    ("Statistical Analysis", "statisticalPlan"),
]

SECTION_PRIORITY_ORDER = [
    "protocolTitle", "synopsis", "inclusionCriteria", "exclusionCriteria", 
    "primaryObjective", "secondaryObjective", "studyDesign", "studyPopulation",
    "sampleSizeMin", "safetyAssessment", "outcomeMeasures", "statisticalPlan"
]


def load_protocol_json() -> Dict[str, Any]:
    """
    Load the protocol JSON file.

    Returns:
        Dict containing the parsed JSON data.

    Raises:
        FileNotFoundError: If the protocol file does not exist.
    """
    if not PROTOCOL_PATH.exists():
        logger.error(f"Protocol file not found: {PROTOCOL_PATH}")
        raise FileNotFoundError(f"Protocol file not found: {PROTOCOL_PATH}")
    
    with open(PROTOCOL_PATH, 'r') as f:
        return json.load(f)


def extract_protocol_sections(protocol_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract key sections from protocol for gap analysis.

    Args:
        protocol_data: The full protocol JSON object.

    Returns:
        List of dictionaries containing section metadata and text.
    """
    sections = []
    mapped = protocol_data.get("mappedFields", {})
    
    for section_name, field_key in SECTION_CONFIGS:
        field_data = mapped.get(field_key, {})
        section_text = field_data.get("section_text", "")
        header = field_data.get("header", "")
        
        if section_text and section_text.strip():
            # Limit text length to prevent context window overflow if extremely large,
            # though current models handle 100k+ tokens easily.
            sections.append({
                "section_name": section_name,
                "field_key": field_key,
                "header": header,
                "text": section_text.strip(),
            })
    
    return sections


def extract_full_protocol_text(protocol_data: Dict[str, Any]) -> str:
    """
    Extract ALL text from the protocol for cross-referencing.
    
    This creates a flattened text representation of the entire protocol,
    allowing the LLM to 'see' information across different sections 
    (solving the 'Silo Problem').

    Args:
        protocol_data: The full protocol JSON object.

    Returns:
        A single string containing all available protocol text.
    """
    full_text = []
    mapped = protocol_data.get("mappedFields", {})
    
    # 1. Add priority sections first for logical flow
    processed_keys = set()
    for key in SECTION_PRIORITY_ORDER:
        if key in mapped and mapped[key].get("section_text"):
            header = mapped[key].get("header", key)
            text = mapped[key].get("section_text", "").strip()
            if text:
                full_text.append(f"=== SECTION: {header} ===\n{text}\n")
                processed_keys.add(key)
    
    # 2. Add any remaining sections
    for key, data in mapped.items():
        if key not in processed_keys and isinstance(data, dict) and data.get("section_text"):
            header = data.get("header", key)
            text = data.get("section_text", "").strip()
            if text:
                full_text.append(f"=== SECTION: {header} ===\n{text}\n")
    
    return "\n".join(full_text)


def analyze_section_gaps(
    section: Dict[str, str], 
    rag: MultiAgentRAG, 
    full_protocol_context: str, 
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Analyze gaps for a single protocol section using the decoupled retrieval strategy.

    Args:
        section: Dictionary containing section name and text.
        rag: Initialized MultiAgentRAG instance.
        full_protocol_context: Full text of the protocol for cross-referencing.
        verbose: Whether to log progress.

    Returns:
        Dictionary containing the analysis results.
    """
    if verbose:
        logger.info(f"Analyzing Section: {section['section_name']}")
    
    # Strategy:
    # 1. Search Query: Focused on the SPECIFIC section to retrieve relevant rules.
    # 2. Analysis Query: Contains FULL context to check compliance across the document.
    
    search_query = f"FDA guidelines and TANZEUM drug label requirements for {section['section_name']}"
    if section['text']:
        # Append context hint
        search_query += f". Key topics: {section['text'][:200]}..."

    analysis_query = f"""
Perform a critical compliance check for this specific protocol section against FDA guidelines and TANZEUM drug label requirements.

TARGET SECTION TO ANALYZE:
"{section['section_name']}"
---
{section['text']}
---

CRITICAL INSTRUCTIONS:
1. Identify requirements strictly relevant to this section.
2. BEFORE declaring a "GAP", you MUST check the "FULL PROTOCOL CONTEXT" provided below.
   - If a requirement is missing from this section but PRESENT elsewhere (e.g., in "Statistical Methods"), it is NOT a gap.
   - Explicitly note it as "Compliant (addressed in Section X)".
3. Only report a GAP if it is COMPLETELY MISSING from the entire protocol.

FULL PROTOCOL CONTEXT (Use this to verify false positives):
---
{full_protocol_context}
---

TASK:
1. List VALID GAPS (requirements completely missing from the ENTIRE protocol).
2. For each Valid Gap, cite the specific FDA/Label requirement.
3. If a requirement is found elsewhere, explicitly state: "Requirement X is met in Section Y".
"""

    # Temporarily disable file dump for cleaner execution
    original_save = rag._save_results
    rag._save_results = lambda x: None
    
    try:
        if verbose:
            logger.info("  - Retrieval & Analysis in progress...")
        result = rag.evaluate(analysis_query, search_query=search_query)
    finally:
        rag._save_results = original_save
    
    return {
        "section_name": section["section_name"],
        "protocol_text": section["text"][:500],
        "gap_analysis": result.get("generated_response", ""),
        "chairman_assessment": result.get("chairman_analysis", ""),
        "sources_used": len(result.get("sources", [])),
        "sources": [
            {
                "document": s.get("document", ""),
                "page": s.get("page", 0),
                "text": s.get("paragraph_text", "")[:200]
            }
            for s in result.get("sources", [])[:5]
        ]
    }


def save_gap_report(protocol_sections_analysis: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Save the comprehensive gap analysis report to a file.

    Args:
        protocol_sections_analysis: List of analysis results.
        output_path: Path to save the report.
    """
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PROTOCOL GAP ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("="*80 + "\n\n")
        
        f.write("Scope:\n")
        f.write("- FDA Guidelines for Diabetes Trials (59 rules)\n")
        f.write("- TANZEUM Drug Label Requirements (362 rules)\n\n")
        
        f.write(f"Total Sections Analyzed: {len(protocol_sections_analysis)}\n")
        f.write("-"*80 + "\n\n")
        
        for i, analysis in enumerate(protocol_sections_analysis, 1):
            f.write("\n" + "="*80 + "\n")
            f.write(f"SECTION {i}: {analysis['section_name']}\n")
            f.write("="*80 + "\n\n")
            
            f.write("📋 PROTOCOL EXCERPT:\n")
            f.write("-"*40 + "\n")
            f.write(analysis['protocol_text'] + "...\n\n")
            
            f.write("🔍 GAP ANALYSIS:\n")
            f.write("-"*40 + "\n")
            f.write(analysis['gap_analysis'] + "\n\n")
            
            if analysis.get('chairman_assessment'):
                f.write("🎖️ CHAIRMAN ASSESSMENT:\n")
                f.write("-"*40 + "\n")
                f.write(analysis['chairman_assessment'] + "\n\n")
            
            f.write(f"📚 SOURCES CONSULTED: {analysis['sources_used']}\n")
            f.write("-"*40 + "\n")
            for j, src in enumerate(analysis['sources'], 1):
                f.write(f"  [{j}] {src['document']} - Page {src['page']}\n")
                f.write(f"      {src['text']}...\n")
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Sections Analyzed: {len(protocol_sections_analysis)}\n")
        f.write(f"Total Requirements Checked: {sum(a['sources_used'] for a in protocol_sections_analysis)}\n")
        f.write("\nEnd of Report.\n")
    
    logger.info(f"Report report saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Automated Protocol Gap Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--model", "-m", type=str, default="claude", help="LLM model (claude, gpt-oss, etc.)")
    parser.add_argument("--top-k", "-k", type=int, default=10, help="Number of sources after reranking")
    parser.add_argument("--sections", "-s", type=str, nargs="+", help="Specific sections to analyze (default: all)")
    
    args = parser.parse_args()
    
    logger.info("Starting Automated Protocol Gap Analysis")
    
    # 1. Load Data
    try:
        protocol_data = load_protocol_json()
    except Exception as e:
        logger.error(f"Failed to load protocol: {e}")
        sys.exit(1)

    sections = extract_protocol_sections(protocol_data)
    logger.info(f"Found {len(sections)} analyzable sections.")
    
    if args.sections:
        sections = [s for s in sections if s['section_name'] in args.sections]
        logger.info(f"Filtered to {len(sections)} sections: {args.sections}")
    
    if not sections:
        logger.warning("No sections to analyze. Exiting.")
        sys.exit(0)

    # 2. Initialize RAG
    logger.info("Initializing Multi-Agent RAG System...")
    rag = MultiAgentRAG(
        backend="pinecone",
        pinecone_index="clinical-rules",
        llm_model=args.model,
        top_k=args.top_k,
        initial_candidates=50
    )
    
    # 3. Prepare Full Context
    logger.info("Extracting Full Protocol Context...")
    full_protocol_context = extract_full_protocol_text(protocol_data)
    logger.info(f"Context ready ({len(full_protocol_context)} chars).")
    
    # 4. Run Analysis
    logger.info("Running Analysis Pipeline...")
    results = []
    for i, section in enumerate(sections, 1):
        logger.info(f"[{i}/{len(sections)}] Analyzing {section['section_name']}...")
        analysis = analyze_section_gaps(section, rag, full_protocol_context, verbose=True)
        results.append(analysis)
    
    # 5. Save Results
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"protocol_gap_analysis_{timestamp}.txt"
    
    save_gap_report(results, output_path)
    logger.info("Analysis Complete successfully.")


if __name__ == "__main__":
    main()
