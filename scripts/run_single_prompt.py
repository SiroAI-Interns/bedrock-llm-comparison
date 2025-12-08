"""Run a single prompt across all 5 models."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.comparison_service import ComparisonService

def main():
    """Main execution."""
    service = ComparisonService()
    
    results = service.compare_single_prompt(
        prompt="Explain machine learning in 5 bullet points",
        max_tokens=256,
        temperature=0.7
    )
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    for model_name, result in results.items():
        if "error" in result:
            print(f"\n{model_name.upper()}: ERROR - {result.get('error', 'Unknown')}")
        else:
            text = result.get('text', '')
            preview = text[:200] + '...' if len(text) > 200 else text
            print(f"\n{model_name.upper()}:\n  {preview}")

if __name__ == "__main__":
    main()
