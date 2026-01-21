# scripts/evaluate_violations.py
"""
Evaluate LLMs on FDA compliance analysis.
Given a trial protocol, LLMs must search FDA guidelines and find violations.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.agents import MultiAgentEvaluator


TRIAL_PROTOCOL = """
Clinical Trial Protocol: T2DM-Device-001
GlucoControl Insulin Management System - Feasibility Study

STUDY DESIGN:
• Design: Single-arm, open-label
• Sample Size: 15 patients  
• Duration: 3 months
• Follow-up: End of study visit at Month 3

ENROLLMENT CRITERIA:
Inclusion:
• Age 18-65 years
• Type 2 Diabetes Mellitus
• HbA1c: 6.0% to 11.5%
• Any diabetes medications allowed

Exclusion:
• Type 1 diabetes
• Pregnancy
• Severe renal impairment

ENDPOINTS:
Primary: HbA1c change at Month 3
Secondary: Body weight, satisfaction survey, usability score

MEASUREMENTS:
• HbA1c: Local hospital lab, standard analyzer, baseline and Month 3
• Satisfaction: Custom 10-item survey (pilot tested with 5 patients)
• Usability: 5-point Likert scale (not validated)

SAFETY MONITORING:
• Adverse events: Patient-reported to coordinator
• Hypoglycemia: Patient diary entries
• Review: Monthly visits

DIABETES MANAGEMENT:
• Baseline: Any medications, no minimum failed therapies required
• Rescue therapy: At investigator discretion, no pre-specified criteria
"""


def main():
    print("="*70)
    print("🔬 FDA COMPLIANCE EVALUATION")
    print("="*70)
    print()
    print("Task: Given trial protocol, search FDA guidelines and find violations")
    print()
    
    models = {
        "Claude": ("anthropic.claude-3-5-sonnet-20240620-v1:0", "us-east-1"),
        "Llama": ("meta.llama3-70b-instruct-v1:0", "us-east-1"),
        "Mistral": ("mistral.mistral-7b-instruct-v0:2", "us-east-1"),
        "Titan": ("amazon.titan-text-express-v1", "us-east-1"),
        "DeepSeek": ("us.deepseek.r1-v1:0", "us-west-2"),
        "GPT-OSS": ("openai.gpt-oss-20b-1:0", "us-east-1"),
        "Gemma": ("google.gemma-3-27b-it", "us-east-1"),
    }
    
    print(f"📊 Evaluating {len(models)} models\n")
    
    evaluator = MultiAgentEvaluator(models, rebuild_index=False)
    
    query = f"""You are a regulatory expert. Analyze this diabetes device trial protocol and identify ALL FDA guideline violations.

TRIAL PROTOCOL:
{TRIAL_PROTOCOL}

Search FDA guidelines and for each violation specify:
1. What the protocol states
2. What FDA requires (cite page)
3. Why it's non-compliant
4. How to fix it

Be thorough and systematic."""

    print("🚀 Starting evaluation...\n")
    
    result = evaluator.evaluate(query)
    
    # Save
    output_dir = project_root / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "fda_compliance_evaluation.txt"
    
    with open(output_file, "w") as f:
        f.write(result)
    
    print(f"\n✅ Results saved: {output_file}\n")


if __name__ == "__main__":
    main()
