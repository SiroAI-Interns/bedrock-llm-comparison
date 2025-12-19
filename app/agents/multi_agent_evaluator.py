"""
Multi-agent LLM evaluator with research capability and customizable prompts.
Main orchestrator that coordinates all agents.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from typing import Dict, Optional
from app.core.unified_bedrock_client import UnifiedBedrockClient
from app.agents.research_agent import ResearchAgent
from app.agents.generator_agent import GeneratorAgent
from app.agents.reviewer_agent import ReviewerAgent
from app.agents.chairman_agent import ChairmanAgent


class MultiAgentEvaluator:
    """
    Top-level orchestrator for multi-agent LLM evaluation.
    Coordinates research, generation, review, and analysis.
    """

    def __init__(self, models: Dict[str, tuple]):
        """
        Initialize MultiAgentEvaluator with model configurations.
        
        Args:
            models: Dict mapping model name -> (model_id, region)
                   Example: {"Claude": ("anthropic.claude-3...", "us-east-1")}
        """
        self.models = models
        self.generator_clients: Dict[str, UnifiedBedrockClient] = {}
        self.reviewer_clients: Dict[str, UnifiedBedrockClient] = {}

        print("="*70)
        print("MULTI-AGENT LLM EVALUATOR WITH RESEARCH")
        print("="*70)

        # Initialize generator clients (creative responses)
        for name, (model_id, region) in models.items():
            self.generator_clients[name] = UnifiedBedrockClient(
                model_id=model_id,
                region_name=region,
                max_tokens=500,
                temperature=0.7,
            )

            # Initialize reviewer clients (consistent scoring)
            self.reviewer_clients[name] = UnifiedBedrockClient(
                model_id=model_id,
                region_name=region,
                max_tokens=1500,
                temperature=0.3,
            )

        # Chairman uses Claude
        self.chairman_client = UnifiedBedrockClient(
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
            region_name="us-east-1",
            max_tokens=2000,
            temperature=0.3,
        )

        # Research client (for synthesizing web search results)
        self.research_client = UnifiedBedrockClient(
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
            region_name="us-east-1",
            max_tokens=1500,
            temperature=0.3,
        )

        # Instantiate agents
        self.research_agent = ResearchAgent(self.research_client)
        self.generator_agent = GeneratorAgent(self.generator_clients)
        self.reviewer_agent = ReviewerAgent(self.reviewer_clients)
        self.chairman_agent = ChairmanAgent(self.chairman_client)

        print(f"✅ Initialized {len(self.generator_clients)} models")
        print(f"✅ Research agent ready\n")

    def evaluate(
        self,
        prompt: str,
        synthesis_template: Optional[str] = None,
        generator_template: Optional[str] = None,
        reviewer_template: Optional[str] = None,
        chairman_template: Optional[str] = None,
    ) -> str:
        """
        Run complete 4-agent evaluation pipeline.
        
        Args:
            prompt: User's query to evaluate
            synthesis_template: Custom template for research synthesis (Agent 0)
            generator_template: Custom template for response generation (Agent 1)
            reviewer_template: Custom template for reviews (Agent 2)
            chairman_template: Custom template for final analysis (Agent 3)
            
            All templates default to values in templates.py if not provided.
        
        Returns:
            Final evaluation report as string
        """
        print("\n" + "="*70)
        print("STARTING EVALUATION PIPELINE")
        print("="*70)
        print(f"\n📝 User Prompt: {prompt}\n")

        # Agent 0: Research
        research_output = self.research_agent.run(
            prompt,
            synthesis_template=synthesis_template,
        )

        # Agent 1: Generate with research context
        generator_output = self.generator_agent.run(
            research_output,
            generator_template=generator_template,
        )

        # Agent 2: Review
        reviewer_output = self.reviewer_agent.run(
            generator_output,
            reviewer_template=reviewer_template,
        )

        # Agent 3: Chairman analysis
        final_report = self.chairman_agent.run(
            reviewer_output,
            chairman_template=chairman_template,
        )

        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)

        return final_report


# ==================== TEST FUNCTION ====================

def test_evaluation():
    """Test the evaluator with a single prompt."""
    
    models = {
        "Claude": ("anthropic.claude-3-5-sonnet-20240620-v1:0", "us-east-1"),
        "Llama": ("meta.llama3-70b-instruct-v1:0", "us-east-1"),
        "Mistral": ("mistral.mistral-7b-instruct-v0:2", "us-east-1"),
        "Titan": ("amazon.titan-text-express-v1", "us-east-1"),
        "DeepSeek": ("us.deepseek.r1-v1:0", "us-west-2"),
        "GPT-OSS": ("openai.gpt-oss-20b-1:0", "us-east-1"),
        "Gemma": ("google.gemma-3-27b-it", "us-east-1"),
    }

    evaluator = MultiAgentEvaluator(models)

    # Single prompt evaluation
    prompt = "Explain HbA1c and its importance in diabetes clinical trials. Answer in 2-3 sentences."

    result = evaluator.evaluate(prompt)

    print(result)


if __name__ == "__main__":
    test_evaluation()
