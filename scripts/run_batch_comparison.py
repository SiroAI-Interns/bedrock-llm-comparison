"""Run multiple prompts across all 5 models."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.comparison_service import ComparisonService

def main():
    """Main execution."""
    service = ComparisonService()
    
    prompts = [
        "Explain overfitting in 3 bullets",
        "What is gradient descent?",
        "Describe neural networks simply"
    ]
    
    service.compare_batch(
        prompts=prompts,
        max_tokens=200,
        temperature=0.8
    )
    
    print("\nBatch comparison complete!")

if __name__ == "__main__":
    main()

