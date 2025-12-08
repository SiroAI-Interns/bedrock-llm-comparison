"""AWS Bedrock client for Titan, LLaMA, and Claude models."""

import json
import boto3
import botocore
import re
from typing import Dict, Any, Optional
from app.utils.text_processor import (
    extract_first_json_block, 
    extract_bullets_simple, 
    normalize_output_for_recommendation
)


class BedrockClient:
    """AWS Bedrock client for Titan, LLaMA, and Claude (Anthropic) models."""
    
    def __init__(self, model_id: str, model_type: str = "titan", region: str = "us-east-1"):
        self.model_id = model_id
        self.model_type = model_type.lower()
        self.region = region
        self.client = boto3.client("bedrock-runtime", region_name=self.region)
        
        self.DEFAULT_BUILDERS = {
            "titan": self._build_titan_payload,
            "llama": self._build_llama_native_payload,
            "anthropic": self._build_anthropic_payload,
            "claude": self._build_anthropic_payload,
        }
        self.DEFAULT_PARSERS = {
            "titan": self._parse_titan_response,
            "llama": self._parse_llama_native_response,
            "anthropic": self._parse_anthropic_response,
            "claude": self._parse_anthropic_response,
        }
        
        self.build_payload_fn = self.DEFAULT_BUILDERS.get(self.model_type, self._build_titan_payload)
        self.parse_response_fn = self.DEFAULT_PARSERS.get(self.model_type, self._parse_titan_response)
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, 
                 top_p: float = 0.9, **extra_kwargs) -> Dict[str, Any]:
        """Generate response from Bedrock model."""
        primary_payload = self.build_payload_fn(prompt=prompt, max_tokens=max_tokens,
                                                temperature=temperature, top_p=top_p, **extra_kwargs)
        candidates = [primary_payload] + self._construct_fallbacks(prompt, max_tokens, temperature, top_p, extra_kwargs, primary_payload)
        
        last_exc = None
        for idx, payload in enumerate(candidates):
            try:
                raw = self.client.invoke_model(modelId=self.model_id, body=json.dumps(payload))
                body_bytes = raw.get("body").read()
                try:
                    decoded = json.loads(body_bytes)
                except Exception:
                    decoded = {"raw": body_bytes.decode("utf-8", errors="replace")}
                
                parsed_text = self.parse_response_fn(decoded)
                
                # Retry if code-like response
                normalized = normalize_output_for_recommendation(parsed_text)
                looks_like_code = bool(re.search(r"\bdef\s+\w+\s*\(|\bclass\s+\w+|import\s+\w+|return\s+|`{3}", parsed_text or ""))
                
                if (not normalized or normalized == []) and looks_like_code:
                    try:
                        followup_prompt = (
                            "You are a helpful summarizer. From the text below, output EXACTLY 5 concise bullets, each on its own line. "
                            "Do NOT output code or explanations.\n\nTEXT:\n" + prompt
                        )
                        follow_payload = self.build_payload_fn(prompt=followup_prompt, max_tokens=max(128, max_tokens), temperature=0.0, top_p=1.0)
                        raw2 = self.client.invoke_model(modelId=self.model_id, body=json.dumps(follow_payload))
                        body_bytes2 = raw2.get("body").read()
                        try:
                            decoded2 = json.loads(body_bytes2)
                        except Exception:
                            decoded2 = {"raw": body_bytes2.decode("utf-8", errors="replace")}
                        parsed_text2 = self.parse_response_fn(decoded2)
                        parsed_text = parsed_text2 or parsed_text
                        decoded = {"first": decoded, "retry": decoded2}
                    except Exception:
                        pass
                
                return {"provider": "bedrock", "model": self.model_id, "text": parsed_text, "raw": decoded}
            except botocore.exceptions.ClientError as e:
                err_code = e.response.get("Error", {}).get("Code", "")
                err_msg = e.response.get("Error", {}).get("Message", str(e))
                last_exc = e
                if err_code == "ValidationException" or "required key" in err_msg or "not permitted" in err_msg:
                    continue
                else:
                    raise RuntimeError(f"Bedrock invoke_model error: {e}") from e
        
        raise RuntimeError(f"All payload variants rejected. Last error: {str(last_exc)}")
    
    def _construct_fallbacks(self, prompt, max_tokens, temperature, top_p, extra_kwargs, primary_payload):
        fallbacks = []
        fallbacks.append({"prompt": prompt})
        fallbacks.append({"prompt": prompt, "max_tokens_to_sample": max_tokens, "temperature": temperature})
        fallbacks.append({
            "inputText": prompt,
            "textGenerationConfig": {"maxTokenCount": max_tokens, "temperature": temperature, "topP": top_p}
        })
        
        seen = set()
        unique = []
        for p in fallbacks:
            key = json.dumps(p, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique
    
    @staticmethod
    def _build_titan_payload(prompt: str, max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9, **kwargs):
        return {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": temperature,
                "topP": top_p,
                **(kwargs.get("textGenerationConfig", {}))
            }
        }
    
    @staticmethod
    def _build_llama_native_payload(prompt: str, max_tokens: int = 512, max_gen_len: Optional[int] = None,
                                   temperature: float = 0.5, **kwargs):
        effective_max = max_gen_len if max_gen_len is not None else max_tokens
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
        return {
            "prompt": formatted_prompt,
            "max_gen_len": effective_max,
            "temperature": temperature
        }
    
    @staticmethod
    def _build_anthropic_payload(prompt: str, max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9, **kwargs):
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p
        }
    
    @staticmethod
    def _parse_titan_response(response: Dict[str, Any]) -> str:
        if "results" in response and isinstance(response["results"], list) and len(response["results"]) > 0:
            first_result = response["results"][0]
            if isinstance(first_result, dict):
                return first_result.get("outputText", "")
        if "outputText" in response:
            return response["outputText"]
        return response.get("raw", json.dumps(response))
    
    @staticmethod
    def _parse_llama_native_response(response: Dict[str, Any]) -> str:
        text_candidate = None
        if "generation" in response:
            gen = response["generation"]
            text_candidate = gen if isinstance(gen, str) else str(gen)
        elif "results" in response and response["results"]:
            first = response["results"][0]
            if isinstance(first, dict) and "outputText" in first:
                text_candidate = first["outputText"]
        elif "outputText" in response:
            text_candidate = response["outputText"]
        else:
            text_candidate = json.dumps(response)
        
        if not isinstance(text_candidate, str):
            text_candidate = str(text_candidate)
        
        try:
            candidate_json_block = extract_first_json_block(text_candidate)
            if candidate_json_block.strip().startswith(("{", "[")):
                parsed = json.loads(candidate_json_block)
                return json.dumps(parsed, indent=2, ensure_ascii=False)
        except Exception:
            pass
        
        bullets = extract_bullets_simple(text_candidate)
        if bullets:
            return "\n".join([f"- {b}" for b in bullets])
        
        cleaned = text_candidate.strip()
        return cleaned[:8000] + "\n...[truncated]" if len(cleaned) > 8000 else cleaned
    
    @staticmethod
    def _parse_anthropic_response(response: Dict[str, Any]) -> str:
        if "content" in response and isinstance(response["content"], list):
            text_parts = []
            for block in response["content"]:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            if text_parts:
                return "\n".join(text_parts)
        
        if "completion" in response:
            return response["completion"]
        
        return response.get("raw", json.dumps(response))
