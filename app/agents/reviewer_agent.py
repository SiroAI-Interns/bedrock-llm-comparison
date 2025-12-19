# app/agents/reviewer_agent.py
"""Agent 2: Reviewer Agent - Reviews and scores responses."""

from typing import Dict, Optional
import json
import re
from app.core.unified_bedrock_client import UnifiedBedrockClient
from app.agents.templates import REVIEWER_TEMPLATE


class ReviewerAgent:
    """Reviews all responses blindly and assigns scores."""

    def __init__(self, reviewer_clients: Dict[str, UnifiedBedrockClient]):
        """
        Initialize Reviewer Agent.
        
        Args:
            reviewer_clients: Dict of model name -> Bedrock client
        """
        self.reviewer_clients = reviewer_clients

    def run(
        self,
        generator_output: Dict,
        reviewer_template: Optional[str] = None
    ) -> Dict:
        """
        Review all responses from all models.
        
        Args:
            generator_output: Output from Generator Agent
            reviewer_template: Custom template for reviews
        
        Returns:
            Dictionary with all reviews
        """
        print("\n" + "="*70)
        print("AGENT 2: REVIEWER")
        print("="*70)

        # Use default template if not provided
        if reviewer_template is None:
            reviewer_template = REVIEWER_TEMPLATE

        prompt = generator_output["prompt"]
        responses = generator_output["responses"]
        mapping = generator_output["mapping"]

        # Format responses for display
        resp_text = "\n\n".join([f"=== {k} ===\n{v}" for k, v in responses.items()])

        print(f"Each of {len(self.reviewer_clients)} reviewers evaluating {len(responses)} responses...\n")

        all_reviews: Dict[str, Dict] = {}

        for reviewer_name, client in self.reviewer_clients.items():
            print(f"  ⏳ {reviewer_name} reviewing...", end=" ")

            # Build review prompt
            review_prompt = reviewer_template.format(
                query=prompt,
                responses=resp_text,
            )

            try:
                review_text = client.generate(review_prompt)

                # Try to extract JSON
                match = re.search(r'\{.*\}', review_text, re.DOTALL)
                parsed: Dict = {}

                if match:
                    review_data = json.loads(match.group(0))

                    # Parse scores safely
                    for label in responses.keys():
                        if label in review_data:
                            parsed[label] = {
                                "score": int(review_data[label].get("score", 5)),
                                "reasoning": review_data[label].get("reasoning", ""),
                            }
                        else:
                            parsed[label] = {"score": 5, "reasoning": "No review provided"}
                else:
                    # Parse failed
                    parsed = {
                        k: {"score": 5, "reasoning": "Parse failed"}
                        for k in responses.keys()
                    }

                all_reviews[reviewer_name] = parsed
                print("✅")

            except Exception as e:
                print(f"❌ Error: {e}")
                all_reviews[reviewer_name] = {
                    label: {"score": 5, "reasoning": "Error during review"}
                    for label in responses.keys()
                }

        print(f"\n✅ Collected reviews from {len(all_reviews)} reviewers\n")

        return {
            "reviews": all_reviews,
            "mapping": mapping,
            "responses": responses,
            "prompt": prompt,
            "research_context": generator_output.get("research_context", ""),
            "sources": generator_output.get("sources", []),
        }
