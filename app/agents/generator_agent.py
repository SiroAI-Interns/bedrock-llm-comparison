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

    def _is_reasoning_model(self, model_id: str) -> bool:
        """Check if model is a reasoning model (o1-style)."""
        reasoning_keywords = ["gpt-oss", "o1", "openai.gpt"]
        return any(keyword in model_id.lower() for keyword in reasoning_keywords)

    def _build_prompt(self, model_id, user_prompt, research_context, sources_text, generator_template):
        if self._is_reasoning_model(model_id):
            # ✅ EXPLICIT: Don't show reasoning
            prompt = f"""You are a regulatory expert. Answer the following question directly without showing your reasoning process.

    QUESTION:
    {user_prompt}

    REGULATORY GUIDANCE:
    {research_context}

    SOURCES:
    {sources_text}

    Provide your systematic analysis now (no reasoning tags, just the answer) Also make sure dont give your outputs in the form of tables give me in the form of text paragraphs and bullets:"""
        else:
            # Full template for regular models
            prompt = generator_template.format(
                query=user_prompt,
                research_context=research_context,
                sources=sources_text,
            )
        
        return prompt

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
        print("AGENT 1: GENERATOR (WITH PDF RESEARCH CONTEXT)")
        print("="*70)

        # Use default template if not provided
        if generator_template is None:
            generator_template = GENERATOR_TEMPLATE

        prompt = research_output["prompt"]
        research_context = research_output["research_context"]
        sources = research_output["sources"]

        # Format sources with document names and page numbers
        if sources:
            sources_text = "Available sources from guideline documents:\n"
            for s in sources:
                if 'document' in s:
                    # PDF source format
                    sources_text += f"[{s['index']}] {s['document']} - Page {s['page']}\n"
                    if 'relevance_score' in s:
                        sources_text += f"    Relevance: {s['relevance_score']:.4f}\n\n"
                elif 'title' in s:
                    # Web source format
                    sources_text += f"[{s['index']}] {s['title']}\n"
                    sources_text += f"    URL: {s['url']}\n\n"
                else:
                    sources_text += f"[{s['index']}] Source {s['index']}\n\n"
        else:
            sources_text = "No sources available"

        print(f"Generating responses from {len(self.generator_clients)} models...\n")

        # Generate responses from all models
        raw_responses = {}
        for name, client in self.generator_clients.items():
            try:
                print(f"  ⏳ {name}...", end=" ")
                
                # Build model-specific prompt
                enhanced_prompt = self._build_prompt(
                    model_id=client.model_id,
                    user_prompt=prompt,
                    research_context=research_context,
                    sources_text=sources_text,
                    generator_template=generator_template
                )
                
                # Add debug info for reasoning models
                if self._is_reasoning_model(client.model_id):
                    print(f"[reasoning model, simplified prompt]", end=" ")
                
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
