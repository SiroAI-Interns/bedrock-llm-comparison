# app/agents/chairman_agent.py
"""Agent 3: Chairman Agent - Final comprehensive analysis."""

from typing import Dict, Optional
from app.core.unified_bedrock_client import UnifiedBedrockClient
from app.agents.templates import CHAIRMAN_TEMPLATE


class ChairmanAgent:
    """Provides final comprehensive analysis of evaluation results."""

    def __init__(self, chairman_client: UnifiedBedrockClient):
        """
        Initialize Chairman Agent.
        
        Args:
            chairman_client: Bedrock client for final analysis
        """
        self.chairman_client = chairman_client

    def run(
        self,
        reviewer_output: Dict,
        chairman_template: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive final analysis.
        
        Args:
            reviewer_output: Output from Reviewer Agent
            chairman_template: Custom template for analysis
        
        Returns:
            Final evaluation report as string
        """
        print("\n" + "="*70)
        print("AGENT 3: CHAIRMAN")
        print("="*70)

        # Use default template if not provided
        if chairman_template is None:
            chairman_template = CHAIRMAN_TEMPLATE

        reviews = reviewer_output["reviews"]
        mapping = reviewer_output["mapping"]
        responses = reviewer_output["responses"]
        prompt = reviewer_output["prompt"]
        research_context = reviewer_output.get("research_context", "")
        sources = reviewer_output.get("sources", [])

        # Calculate average scores
        scores = {label: [] for label in responses.keys()}
        for reviewer, review_data in reviews.items():
            for label, review in review_data.items():
                if label in scores:
                    scores[label].append(review["score"])

        avg_scores = {
            label: (sum(v) / len(v) if v else 0)
            for label, v in scores.items()
        }

        winner_label = max(avg_scores, key=avg_scores.get)
        winner_model = mapping[winner_label]
        winner_score = avg_scores[winner_label]

        # Print scores
        print(f"\n{'='*60}")
        print("SCORE BREAKDOWN")
        print(f"{'='*60}")
        sorted_scores = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        scores_text = ""
        for label, score in sorted_scores:
            model = mapping[label]
            print(f"{label} ({model:12s}): {score:.2f}/10")
            scores_text += f"  {label} ({model}): {score:.2f}/10\n"
        print(f"{'='*60}")
        print(f"\n🏆 Winner: {winner_label} ({winner_model}) - {winner_score:.2f}/10")

        # Format all responses
        responses_text = "\n\n".join([
            f"=== {label} ===\n{text}"
            for label, text in responses.items()
        ])

        # Format individual reviews
        individual_reviews_text = ""
        for reviewer_name, review_data in reviews.items():
            individual_reviews_text += f"\n{reviewer_name}'s Reviews:\n"
            for label in sorted(review_data.keys()):
                review = review_data[label]
                individual_reviews_text += f"  • {label}: {review['score']}/10\n"
                individual_reviews_text += f"    Reasoning: {review['reasoning']}\n"

        # Format sources - UPDATED to handle both PDF and web sources
        if sources:
            sources_list = []
            for s in sources:
                # Check if it's a PDF source or web source
                if 'document' in s:
                    # PDF source format
                    source_line = f"[{s['index']}] {s['document']} - Page {s['page']}"
                    if 'relevance_score' in s:
                        source_line += f"\n    Relevance: {s['relevance_score']:.4f}"
                    sources_list.append(source_line)
                elif 'title' in s:
                    # Web source format (backward compatibility)
                    sources_list.append(f"[{s['index']}] {s['title']}\n    {s['url']}")
                else:
                    # Unknown format fallback
                    sources_list.append(f"[{s['index']}] Source {s['index']}")
            
            sources_text = "\n\n".join(sources_list)
        else:
            sources_text = "No sources available"

        # Build chairman prompt
        chairman_prompt = chairman_template.format(
            query=prompt,
            research_context=research_context,
            sources=sources_text,
            responses=responses_text,
            reviews=individual_reviews_text,
            scores=scores_text,
            winner_label=winner_label,
            winner_model=winner_model,
            winner_score=winner_score,
        )

        print("\nGenerating comprehensive chairman analysis...")
        analysis = self.chairman_client.generate(chairman_prompt)
        print("\n✅ Analysis complete\n")

        return f"""
{'='*70}
FINAL EVALUATION REPORT
{'='*70}

🏆 WINNER: {winner_label} ({winner_model})
📊 SCORE: {winner_score:.2f}/10

{'='*70}
SCORE BREAKDOWN (AVERAGE ACROSS ALL REVIEWERS)
{'='*70}
{scores_text}

{'='*70}
RESEARCH CONTEXT PROVIDED TO ALL MODELS
{'='*70}
{research_context}

{'='*70}
SOURCES (From FDA Guidelines)
{'='*70}
{sources_text}

{'='*70}
WINNING RESPONSE
{'='*70}
{responses[winner_label]}

{'='*70}
ALL RESPONSES (Blinded)
{'='*70}
{responses_text}

{'='*70}
CHAIRMAN'S COMPREHENSIVE ANALYSIS
{'='*70}
{analysis}

{'='*70}
EVALUATION STATISTICS
{'='*70}
✅ Research sources gathered: {len(sources)}
✅ Models evaluated: {len(responses)}
✅ Reviews per response: {len(reviews)}
✅ Total reviews: {len(reviews) * len(responses)}
✅ Analysis includes factual accuracy verification using FDA guidelines

{'='*70}
"""
