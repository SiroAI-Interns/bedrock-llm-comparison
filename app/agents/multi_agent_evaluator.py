# app/agents/multi_agent_evaluator.py
"""
Multi-agent LLM evaluator with research capability and customizable prompts.
Main orchestrator that coordinates all agents.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from typing import Dict, Optional, List  # ✅ Added List
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

    def __init__(
        self,
        models: Dict[str, tuple],
        rebuild_index: bool = False,
        embedding_model: str = "medcpt",           # ✅ NEW
        use_hybrid: bool = False,                  # ✅ NEW
        hybrid_models: Optional[List[str]] = None, # ✅ NEW
        use_reranking: bool = False,               # ✅ NEW: Reranking flag
        reranker_model: str = "ms-marco-mini",     # ✅ NEW: Reranker model
        reranking_strategy: str = "cross-encoder", # ✅ NEW: Reranking strategy
    ):
        """
        Initialize MultiAgentEvaluator with model configurations.
        
        Args:
            models: Dict mapping model name -> (model_id, region)
                   Example: {"Claude": ("anthropic.claude-3...", "us-east-1")}
            rebuild_index: Force rebuild of PDF vector index (default: False)
            embedding_model: Embedding model for document retrieval
                   Options: "medcpt", "biobert", "pubmedbert", "scibert", "all-mpnet"
            use_hybrid: If True, combine multiple embedding models
            hybrid_models: List of models to combine (e.g., ["medcpt", "biobert"])
            use_reranking: If True, rerank search results for better precision
            reranker_model: Reranker model to use (ms-marco-mini, ms-marco-medium, ms-marco-base)
            reranking_strategy: Reranking strategy (cross-encoder, bm25, hybrid)
        """
        self.models = models
        self.generator_clients: Dict[str, UnifiedBedrockClient] = {}
        self.reviewer_clients: Dict[str, UnifiedBedrockClient] = {}

        print("="*70)
        print("MULTI-AGENT LLM EVALUATOR WITH RESEARCH")
        print("="*70)

        # ========== GENERATOR CLIENTS (CREATIVE RESPONSES) ==========
        print("\n🎨 Initializing generator models...")
        for name, (model_id, region) in models.items():
            # ✅ AUTO-DETECT: Reasoning models need MORE tokens
            if self._is_reasoning_model(model_id):
                max_tokens = 8000  # High for reasoning chains (GPT-OSS, DeepSeek-R1)
                print(f"   🧠 {name}: {max_tokens} tokens (reasoning model)")
            else:
                max_tokens = 4000  # Standard for normal models
                print(f"   💬 {name}: {max_tokens} tokens")
            
            self.generator_clients[name] = UnifiedBedrockClient(
                model_id=model_id,
                region_name=region,
                max_tokens=max_tokens,
                temperature=0.7,  # Creative generation
            )

        # ========== REVIEWER CLIENTS (CONSISTENT SCORING) ==========
        print("\n⚖️ Initializing reviewer models...")
        for name, (model_id, region) in models.items():
            self.reviewer_clients[name] = UnifiedBedrockClient(
                model_id=model_id,
                region_name=region,
                max_tokens=2000,  # ✅ Increased from 1500 for thorough reviews
                temperature=0.3,   # Consistent evaluation
            )
            print(f"   📝 {name}: 2000 tokens")

        # ========== CHAIRMAN CLIENT (FINAL ANALYSIS) ==========
        print("\n👔 Initializing chairman...")
        self.chairman_client = UnifiedBedrockClient(
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
            region_name="us-east-1",
            max_tokens=2000,
            temperature=0.3,
        )
        print("   📊 Claude (Chairman): 2000 tokens")

        # ========== RESEARCH CLIENT (PDF SYNTHESIS) ==========
        print("\n🔬 Initializing research agent...")
        self.research_client = UnifiedBedrockClient(
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
            region_name="us-east-1",
            max_tokens=2000,  # ✅ Increased from 1500 for better synthesis
            temperature=0.3,
        )
        
        # ✅ Display embedding configuration
        if use_hybrid and hybrid_models:
            print(f"   🧬 Embeddings: Hybrid ({' × '.join(hybrid_models)})")
        else:
            print(f"   🧬 Embeddings: {embedding_model}")
        
        # ✅ Display reranking configuration
        if use_reranking:
            print(f"   🔄 Reranking: {reranking_strategy} ({reranker_model})")

        # ========== INSTANTIATE AGENTS ==========
        self.research_agent = ResearchAgent(
            self.research_client,
            rebuild_index=rebuild_index,
            embedding_model=embedding_model,      # ✅ Pass to research agent
            use_hybrid=use_hybrid,                # ✅ Pass hybrid flag
            hybrid_models=hybrid_models,          # ✅ Pass hybrid models list
            use_reranking=use_reranking,          # ✅ Pass reranking flag
            reranker_model=reranker_model,        # ✅ Pass reranker model
            reranking_strategy=reranking_strategy,# ✅ Pass reranking strategy
        )
        self.generator_agent = GeneratorAgent(self.generator_clients)
        self.reviewer_agent = ReviewerAgent(self.reviewer_clients)
        self.chairman_agent = ChairmanAgent(self.chairman_client)

        print(f"\n✅ Initialized {len(self.generator_clients)} models")
        print(f"✅ Research agent ready\n")

    def _is_reasoning_model(self, model_id: str) -> bool:
        """
        Check if model uses chain-of-thought reasoning (o1-style).
        These models need more tokens to complete reasoning + answer.
        """
        reasoning_keywords = [
            "gpt-oss",        # OpenAI o1-mini/o1-preview on Bedrock
            "o1-mini",
            "o1-preview",
            "openai.gpt",     # OpenAI models on Bedrock
            "deepseek.r1",    # DeepSeek R1 reasoning model
            "r1-distill",     # R1-distilled models
            "us.deepseek",    # DeepSeek models on Bedrock
        ]
        return any(keyword in model_id.lower() for keyword in reasoning_keywords)

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

    # ✅ Test with different embedding configurations
    
    # Test 1: Default (MedCPT)
    print("\n" + "="*70)
    print("TEST 1: MedCPT (Default)")
    print("="*70)
    evaluator1 = MultiAgentEvaluator(models, rebuild_index=False)
    
    # Test 2: BioBERT
    print("\n" + "="*70)
    print("TEST 2: BioBERT")
    print("="*70)
    evaluator2 = MultiAgentEvaluator(
        models,
        embedding_model="biobert",
        rebuild_index=True
    )
    
    # Test 3: Hybrid (MedCPT × BioBERT)
    print("\n" + "="*70)
    print("TEST 3: Hybrid (MedCPT × BioBERT)")
    print("="*70)
    evaluator3 = MultiAgentEvaluator(
        models,
        use_hybrid=True,
        hybrid_models=["medcpt", "biobert"],
        rebuild_index=True
    )

    # Single prompt evaluation
    prompt = "What are the HbA1c measurement requirements for diabetes clinical trials?"

    print("\n" + "="*70)
    print("RUNNING EVALUATION")
    print("="*70)
    result = evaluator1.evaluate(prompt)

    print(result)


if __name__ == "__main__":
    test_evaluation()
