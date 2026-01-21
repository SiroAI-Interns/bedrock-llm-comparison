# scripts/compare_embeddings.py
"""Compare different embedding models and hybrid approaches."""

import sys
from pathlib import Path
import time
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.agents.multi_agent_evaluator import MultiAgentEvaluator


def test_embedding_model(
    embedding_model: str,
    use_hybrid: bool = False,
    hybrid_models=None,
    use_reranking: bool = False,
    reranker_model: str = "ms-marco-mini",
    reranking_strategy: str = "cross-encoder"
):
    """Run evaluation with specific embedding configuration."""
    
    print("\n" + "="*80)
    if use_hybrid:
        model_name = " × ".join(hybrid_models)
        config_str = f"HYBRID ({model_name})"
    else:
        config_str = embedding_model.upper()
    
    if use_reranking:
        config_str += f" + RERANKING ({reranking_strategy})"
    
    print(f"TESTING: {config_str}")
    print("="*80)
    
    # Model configuration (use fewer models for faster testing)
    models = {
        "Claude": ("anthropic.claude-3-5-sonnet-20240620-v1:0", "us-east-1"),
        "Llama": ("meta.llama3-70b-instruct-v1:0", "us-east-1"),
        "Mistral": ("mistral.mistral-7b-instruct-v0:2", "us-east-1"),
    }
    
    # ✅ Time the initialization
    start_time = time.time()
    
    # Initialize evaluator with custom embedding model
    evaluator = MultiAgentEvaluator(
        models=models,
        embedding_model=embedding_model,
        use_hybrid=use_hybrid,
        hybrid_models=hybrid_models,
        use_reranking=use_reranking,
        reranker_model=reranker_model,
        reranking_strategy=reranking_strategy,
        rebuild_index=True  # Force rebuild for each test
    )
    
    init_time = time.time() - start_time
    print(f"\n⏱️ Initialization time: {init_time:.2f}s\n")
    
    # Test prompt
    prompt = """You are a regulatory expert. Analyze this diabetes device trial protocol and identify ALL FDA guideline violations.

Protocol: T2DM-Device-001 - GlucoControl Insulin Management System Feasibility Study
- Study Design: Single-arm, open-label feasibility study
- Sample Size: 15 patients
- Duration: 3 months
- Primary Endpoint: HbA1c change at Month 3
- Inclusion Criteria: Type 2 diabetes, HbA1c 6.0%-11.5%, age 18-65
- Exclusion: Severe renal impairment
- Measurements: HbA1c (local hospital lab), satisfaction survey (custom 10-item, pilot tested with 5 patients), usability (5-point Likert scale, not validated)
- Safety Monitoring: Adverse events (patient-reported to coordinator), hypoglycemia diary
- Diabetes Management: Baseline any glucose-lowering medications allowed, rescue therapy at investigator discretion (no pre-specified criteria)

Search FDA guidelines and for each violation specify:
1. What the protocol states
2. What FDA requires (cite page)
3. Why it's non-compliant
4. How to fix it"""

    # ✅ Time the evaluation
    eval_start = time.time()
    result = evaluator.evaluate(prompt)
    eval_time = time.time() - eval_start
    
    print(f"\n⏱️ Evaluation time: {eval_time:.2f}s")
    print(f"⏱️ Total time: {init_time + eval_time:.2f}s\n")
    
    # Save results with timestamp and timing info
    output_dir = Path("outputs/embedding_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if use_hybrid:
        filename = f"{'_x_'.join(hybrid_models)}"
    else:
        filename = f"{embedding_model}"
    
    if use_reranking:
        filename += f"_rerank_{reranking_strategy}"
    
    filename += "_evaluation.txt"
    
    output_file = output_dir / filename
    
    # ✅ Add metadata header to output
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("EMBEDDING MODEL EVALUATION RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if use_hybrid:
            f.write(f"Model: Hybrid ({' × '.join(hybrid_models)})\n")
        else:
            f.write(f"Model: {embedding_model}\n")
        f.write(f"Initialization time: {init_time:.2f}s\n")
        f.write(f"Evaluation time: {eval_time:.2f}s\n")
        f.write(f"Total time: {init_time + eval_time:.2f}s\n")
        f.write("="*80 + "\n\n")
        f.write(result)
    
    print(f"✅ Results saved to: {output_file}\n")
    
    model_key = embedding_model if not use_hybrid else f"hybrid_{'_'.join(hybrid_models)}"
    if use_reranking:
        model_key += f"_rerank_{reranking_strategy}"
    
    return {
        "model": model_key,
        "init_time": init_time,
        "eval_time": eval_time,
        "total_time": init_time + eval_time,
        "output_file": output_file,
        "result": result,
    }


def main():
    """Run comparison tests."""
    
    print("\n" + "="*80)
    print("EMBEDDING MODEL + RERANKING COMPARISON STUDY")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results_summary = []
    
    # Test 1: MedCPT (baseline)
    print("\n🧪 TEST 1/6: MedCPT (Current Baseline)")
    result1 = test_embedding_model("medcpt")
    results_summary.append(result1)
    
    # Test 2: BioBERT
    print("\n🧪 TEST 2/6: BioBERT")
    result2 = test_embedding_model("biobert")
    results_summary.append(result2)
    
    # Test 3: PubMedBERT
    print("\n🧪 TEST 3/6: PubMedBERT")
    result3 = test_embedding_model("pubmedbert")
    results_summary.append(result3)
    
    # Test 4: Hybrid (MedCPT × BioBERT)
    print("\n🧪 TEST 4/6: Hybrid (MedCPT × BioBERT)")
    result4 = test_embedding_model(
        embedding_model="medcpt",
        use_hybrid=True,
        hybrid_models=["medcpt", "biobert"]
    )
    results_summary.append(result4)
    
    # Test 5: MedCPT + Reranking
    print("\n🧪 TEST 5/6: MedCPT + Reranking (Cross-Encoder)")
    result5 = test_embedding_model(
        embedding_model="medcpt",
        use_reranking=True,
        reranker_model="ms-marco-mini",
        reranking_strategy="cross-encoder"
    )
    results_summary.append(result5)
    
    # Test 6: Hybrid + Reranking
    print("\n🧪 TEST 6/6: Hybrid (MedCPT × BioBERT) + Reranking")
    result6 = test_embedding_model(
        embedding_model="medcpt",
        use_hybrid=True,
        hybrid_models=["medcpt", "biobert"],
        use_reranking=True,
        reranker_model="ms-marco-mini",
        reranking_strategy="cross-encoder"
    )
    results_summary.append(result6)
    
    # ✅ Print summary table
    print("\n" + "="*80)
    print("SUMMARY: TIMING COMPARISON")
    print("="*80)
    print(f"{'Model':<30} {'Init (s)':<12} {'Eval (s)':<12} {'Total (s)':<12}")
    print("-"*80)
    for result in results_summary:
        model_name = result['model']
        init_time = result['init_time']
        eval_time = result['eval_time']
        total_time = result['total_time']
        print(f"{model_name:<30} {init_time:<12.2f} {eval_time:<12.2f} {total_time:<12.2f}")
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETE!")
    print("="*80)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved in: outputs/embedding_comparison/")
    print("\nNext steps:")
    print("1. Compare winner scores across models")
    print("2. Analyze research quality and citation accuracy")
    print("3. Check relevance scores in output files")
    print("4. Compare reranking impact on retrieval quality")
    print("5. Choose best model for production\n")


if __name__ == "__main__":
    main()
