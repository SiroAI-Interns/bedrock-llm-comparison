"""Service for orchestrating multi-model comparisons."""

from typing import List, Optional
from datetime import datetime
from pathlib import Path

from app.core.bedrock_client import BedrockClient
from app.core.openai_client import PaidModelsClient
from app.services.export_service import ExportService
from config.settings import settings


class ComparisonService:
    """Orchestrate LLM comparisons across multiple providers."""
    
    def __init__(self):
        settings.ensure_directories()
    
    def compare_single_prompt(
        self, 
        prompt: str, 
        max_tokens: int = 512, 
        temperature: float = 0.7,
        output_file: Optional[str] = None,
        region_titan: str = None,
        region_llama: str = None,
        region_claude: str = None
    ) -> dict:
        """Run a single prompt through all available models."""
        
        # Use settings defaults if not provided
        region_titan = region_titan or settings.BEDROCK_TITAN_REGION
        region_llama = region_llama or settings.BEDROCK_LLAMA_REGION
        region_claude = region_claude or settings.BEDROCK_CLAUDE_REGION
        
        # Create exporter
        if output_file is None:
            output_file = settings.OUTPUT_DIR / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        exporter = ExportService(str(output_file))
        results = {}
        
        print(f"\n{'='*70}")
        print(f"Testing prompt across all models: {prompt[:60]}...")
        print(f"{'='*70}\n")
        
        # Model configurations
        models = [
            ("titan", "amazon.titan-text-express-v1", "bedrock", region_titan, "Titan"),
            ("llama", "meta.llama3-8b-instruct-v1:0", "bedrock", region_llama, "LLaMA"),
            ("claude", "anthropic.claude-3-5-sonnet-20240620-v1:0", "bedrock", region_claude, "Claude 3.5"),
            ("openai", "gpt-4o-mini", "openai", None, "OpenAI GPT-4o-mini"),
            ("gemini", "gemini-2.0-flash-lite", "gemini", None, "Gemini 2.0 Flash"),
        ]
        
        for idx, (model_type, model_id, provider, region, display_name) in enumerate(models, 1):
            print(f"[{idx}/5] Testing {display_name}...")
            try:
                if provider == "bedrock":
                    client = BedrockClient(model_id=model_id, model_type=model_type, region=region)
                    result = client.generate(prompt, max_tokens=max_tokens, temperature=temperature)
                    results[model_type] = result
                    exporter.add_response(prompt, provider, model_id, result["text"], temperature, max_tokens)
                    print(f"  [OK] {display_name} completed: {len(result['text'])} chars\n")
                else:
                    paid = PaidModelsClient()
                    result = paid.generate(provider, model_id, prompt=prompt, 
                                         max_tokens=max_tokens, temperature=temperature)
                    results[model_type] = result
                    exporter.add_response(prompt, provider, model_id, result.get("text", ""), temperature, max_tokens)
                    print(f"  [OK] {display_name} completed: {len(result.get('text', ''))} chars\n")
            except Exception as e:
                print(f"  [FAIL] {display_name} failed: {e}\n")
                results[model_type] = {"error": str(e), "text": ""}
                exporter.add_response(prompt, provider, model_id, f"ERROR: {str(e)}", temperature, max_tokens)
        
        print(f"{'='*70}")
        print("Exporting to Excel...")
        print(f"{'='*70}\n")
        
        exporter.export_to_excel()
        return results
    
    def compare_batch(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        output_file: Optional[str] = None
    ) -> None:
        """Compare multiple prompts across all models."""
        
        if output_file is None:
            output_file = settings.OUTPUT_DIR / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        exporter = ExportService(str(output_file))
        
        for idx, prompt in enumerate(prompts, 1):
            print(f"\n--- Prompt {idx}/{len(prompts)} ---")
            self.compare_single_prompt(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                output_file=None  # Use exporter directly
            )
        
        print(f"\n{'='*70}")
        print(f"Batch complete! Results: {output_file}")
        print(f"{'='*70}")
