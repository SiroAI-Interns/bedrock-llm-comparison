# app/agents/generator_agent.py
"""Agent 1: Generator Agent - Creates responses using research context."""

from typing import Dict, Optional
import random
from app.core.unified_bedrock_client import UnifiedBedrockClient
from app.agents.templates import GENERATOR_TEMPLATE


class GeneratorAgent:
    """Generates responses from multiple models using research context."""

    def __init__(self, generator_clients: Dict[str, UnifiedBedrockClient]):
        """
        Initialize Generator Agent.
        
        Args:
            generator_clients: Dict of model name -> Bedrock client
        """
        self.generator_clients = generator_clients

    def run(
        self,
        research_output: Dict,
        generator_template: Optional[str] = None
    ) -> Dict:
        """
        Generate responses from all models using research context.
        
        Args:
            research_output: Output from Research Agent
            generator_template: Custom template for generation
        
        Returns:
            Dictionary with blinded responses and mapping
        """
        print("\n" + "="*70)
        print("AGENT 1: GENERATOR (WITH RESEARCH CONTEXT)")
        print("="*70)

        # Use default template if not provided
        if generator_template is None:
            generator_template = GENERATOR_TEMPLATE

        prompt = research_output["prompt"]
        research_context = research_output["research_context"]
        sources = research_output["sources"]

        # Format sources with URLs for citation
        if sources:
            sources_text = "Available sources with URLs:\n"
            for s in sources:
                sources_text += f"[{s['index']}] {s['title']}\n"
                sources_text += f"    URL: {s['url']}\n\n"
        else:
            sources_text = "No sources available"

        # Build enhanced prompt
        enhanced_prompt = generator_template.format(
            query=prompt,
            research_context=research_context,
            sources=sources_text,
        )

        print(f"Generating responses from {len(self.generator_clients)} models...\n")

        # Generate responses from all models
        raw_responses = {}
        for name, client in self.generator_clients.items():
            try:
                print(f"  ⏳ {name}...", end=" ")
                response = client.generate(enhanced_prompt)
                raw_responses[name] = response.strip() if response else ""
                print("✅")
            except Exception as e:
                print(f"❌ Error: {e}")
                raw_responses[name] = ""

        # Print all generated responses before blinding
        print("\n" + "="*70)
        print("GENERATED RESPONSES (Before Blinding)")
        print("="*70)
        for name, response in raw_responses.items():
            print(f"\n{'='*70}")
            print(f"MODEL: {name}")
            print(f"{'='*70}")
            print(response)
            print()

        # Shuffle and blind responses
        model_names = list(raw_responses.keys())
        random.shuffle(model_names)

        blind_labels = [f"resp_{chr(97+i)}" for i in range(len(model_names))]

        blinded = {}
        mapping = {}
        for label, model in zip(blind_labels, model_names):
            blinded[label] = raw_responses[model]
            mapping[label] = model

        print(f"\n✅ Generated and blinded {len(blinded)} responses\n")

        return {
            "prompt": prompt,
            "responses": blinded,
            "mapping": mapping,
            "research_context": research_context,
            "sources": sources,
        }
