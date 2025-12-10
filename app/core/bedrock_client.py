"""AWS Bedrock client with DeepSeek-R1 support."""

import boto3
import json
from typing import Dict, Any, Optional


class BedrockClient:
    """Client for AWS Bedrock models including DeepSeek-R1."""
    
    SUPPORTED_MODELS = {
        "claude": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "titan": "amazon.titan-text-express-v1",
        "llama": "meta.llama3-8b-instruct-v1:0",
        "deepseek": "us.deepseek.r1-v1:0"  # Cross-region inference profile
    }
    
    def __init__(self, model_id: str, model_type: str = "claude", region: str = "us-east-1"):
        """
        Initialize Bedrock client.
        
        Args:
            model_id: Full model ID or short name (claude, titan, llama, deepseek)
            model_type: Type of model for response parsing
            region: AWS region (use us-west-2 for DeepSeek)
        """
        self.model_type = model_type.lower()
        
        # Handle short names
        if model_id in self.SUPPORTED_MODELS:
            self.model_id = self.SUPPORTED_MODELS[model_id]
            self.model_type = model_id
        else:
            self.model_id = model_id
        
        # DeepSeek requires us-west-2 region
        if self.model_type == "deepseek":
            region = "us-west-2"
        
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region
        )
        
        print(f"[Bedrock] Initialized {self.model_type} in {region}")
        print(f"[Bedrock] Model ID: {self.model_id}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Generate text using the specified model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop_sequences: Stop sequences for generation
            
        Returns:
            Dict with 'text' key containing generated text
        """
        
        if self.model_type == "deepseek":
            return self._generate_deepseek(prompt, max_tokens, temperature, top_p, stop_sequences)
        elif self.model_type == "anthropic" or self.model_type == "claude":
            return self._generate_anthropic(prompt, max_tokens, temperature, top_p, stop_sequences)
        elif self.model_type == "titan":
            return self._generate_titan(prompt, max_tokens, temperature, top_p, stop_sequences)
        elif self.model_type == "llama":
            return self._generate_llama(prompt, max_tokens, temperature, top_p, stop_sequences)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _generate_deepseek(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: Optional[list]
    ) -> Dict[str, Any]:
        """Generate text using DeepSeek-R1."""
        
        # Format prompt with DeepSeek's special tokens
        formatted_prompt = f"""<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜><think>
"""
        
        body = {
            "prompt": formatted_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        if stop_sequences:
            body["stop"] = stop_sequences
        
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )
            
            response_body = json.loads(response["body"].read())
            
            # Extract text from choices
            if "choices" in response_body and len(response_body["choices"]) > 0:
                text = response_body["choices"][0]["text"]
                stop_reason = response_body["choices"][0].get("stop_reason", "")
                
                # DeepSeek shows reasoning in <think> tags - extract just the answer
                # Format: <think>reasoning...</think>actual answer
                if "</think>" in text:
                    # Extract answer after </think>
                    answer = text.split("</think>")[-1].strip()
                else:
                    answer = text
                
                return {
                    "text": answer,
                    "stop_reason": stop_reason,
                    "full_response": text,  # Keep full response with reasoning
                    "model": "deepseek-r1"
                }
            else:
                return {"text": "", "error": "No choices in response"}
                
        except Exception as e:
            print(f"[ERROR] DeepSeek generation failed: {e}")
            return {"text": "", "error": str(e)}
    
    def _generate_anthropic(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: Optional[list]
    ) -> Dict[str, Any]:
        """Generate text using Claude."""
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        if stop_sequences:
            body["stop_sequences"] = stop_sequences
        
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )
            
            response_body = json.loads(response["body"].read())
            
            if "content" in response_body:
                text = response_body["content"][0]["text"]
                return {
                    "text": text,
                    "stop_reason": response_body.get("stop_reason", ""),
                    "model": "claude"
                }
            else:
                return {"text": "", "error": "No content in response"}
                
        except Exception as e:
            print(f"[ERROR] Claude generation failed: {e}")
            return {"text": "", "error": str(e)}
    
    def _generate_titan(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: Optional[list]
    ) -> Dict[str, Any]:
        """Generate text using Titan."""
        
        body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": temperature,
                "topP": top_p
            }
        }
        
        if stop_sequences:
            body["textGenerationConfig"]["stopSequences"] = stop_sequences
        
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )
            
            response_body = json.loads(response["body"].read())
            
            if "results" in response_body:
                text = response_body["results"][0]["outputText"]
                return {
                    "text": text,
                    "model": "titan"
                }
            else:
                return {"text": "", "error": "No results in response"}
                
        except Exception as e:
            print(f"[ERROR] Titan generation failed: {e}")
            return {"text": "", "error": str(e)}
    
    def _generate_llama(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: Optional[list]
    ) -> Dict[str, Any]:
        """Generate text using Llama."""
        
        body = {
            "prompt": prompt,
            "max_gen_len": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )
            
            response_body = json.loads(response["body"].read())
            
            if "generation" in response_body:
                text = response_body["generation"]
                return {
                    "text": text,
                    "model": "llama"
                }
            else:
                return {"text": "", "error": "No generation in response"}
                
        except Exception as e:
            print(f"[ERROR] Llama generation failed: {e}")
            return {"text": "", "error": str(e)}
