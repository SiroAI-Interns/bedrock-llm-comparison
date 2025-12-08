"""OpenAI and Gemini API clients."""

import os
from typing import Dict, Any, Optional, List

try:
    from openai import OpenAI as OpenAIClient
except Exception:
    OpenAIClient = None


def _norm_response(provider: str, model: str, text: str, raw: Any, usage: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"provider": provider, "model": model, "text": text or "", "raw": raw, "usage": usage or {}}


class PaidModelsClient:
    """OpenAI + Gemini client."""
    
    def __init__(self, openai_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        
        self._openai_client = None
        if OpenAIClient is not None and self.openai_api_key:
            try:
                self._openai_client = OpenAIClient(api_key=self.openai_api_key)
            except Exception:
                self._openai_client = None
    
    def generate(self, provider: str, model: str, *,
                 prompt: Optional[str] = None,
                 messages: Optional[List[Dict[str, str]]] = None,
                 max_tokens: Optional[int] = 512,
                 temperature: float = 0.0,
                 **kwargs) -> Dict[str, Any]:
        """Generate response from OpenAI or Gemini."""
        provider = provider.lower()
        
        if provider == "openai":
            if self._openai_client is None:
                return _norm_response("openai", model, "", {"error": "OpenAI client not configured"}, {})
            try:
                single = [{"role": "user", "content": prompt or ""}]
                resp = self._openai_client.chat.completions.create(
                    model=model, messages=single, max_tokens=max_tokens, temperature=temperature, **(kwargs or {})
                )
                content = resp.choices[0].message.content if resp.choices else ""
                usage = getattr(resp, "usage", {}) or {}
                return _norm_response("openai", model, content or "", resp, usage)
            except Exception as e:
                return _norm_response("openai", model, "", {"error": str(e)}, {})
        
        if provider in ("gemini", "google", "genai", "google-genai"):
            gemini_key = self.gemini_api_key or os.environ.get("GEMINI_API_KEY")
            if not gemini_key:
                return _norm_response("gemini", model, "", {"error": "Missing GEMINI_API_KEY"}, {})
            try:
                try:
                    from google import genai
                    from google.genai import types as genai_types
                except Exception as ie:
                    return _norm_response("gemini", model, "", {"error": f"google.genai SDK not installed: {ie}"}, {})
                
                client = genai.Client(api_key=gemini_key)
                prompt_text = prompt or ""
                if messages:
                    prompt_text = "\n".join([m.get("content", "") for m in messages])
                
                safe_max_tokens = max(max_tokens or 512, 256)
                
                config = genai_types.GenerateContentConfig(
                    max_output_tokens=safe_max_tokens,
                    temperature=temperature
                )
                
                resp = client.models.generate_content(model=model, contents=prompt_text, config=config)
                
                text = ""
                try:
                    text = resp.text
                except (AttributeError, ValueError):
                    try:
                        if hasattr(resp, 'candidates') and resp.candidates:
                            candidate = resp.candidates[0]
                            if hasattr(candidate, 'content') and candidate.content.parts:
                                text = candidate.content.parts[0].text
                    except Exception:
                        pass
                
                return _norm_response("gemini", model, text, resp, {})
            except Exception as e:
                return _norm_response("gemini", model, "", {"error": str(e)}, {})
