"""Ultra-simplified unified Bedrock client - Single generate() function does everything."""

import json
import boto3
from typing import Dict, Any, Optional
from botocore.exceptions import ClientError


class UnifiedBedrockClient:
    """Single generate() method handles all Bedrock models - formatting, invoking, parsing."""
    
    def __init__(
        self,
        model_id: str,
        region_name: str = "us-east-1",
        max_tokens: int = None,  # ← Changed to None for auto-detection
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 50
    ):
        """
        Initialize unified Bedrock client.
        
        Args:
            model_id: Full Bedrock model ID (e.g., 'meta.llama3-70b-instruct-v1:0')
            region_name: AWS region
            max_tokens: Maximum tokens to generate (auto-detected if None)
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
        """
        self.model_id = model_id
        self.region_name = region_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        
        # ========== AUTO-DETECT MAX TOKENS BASED ON MODEL ==========
        if max_tokens is None:
            # Reasoning models need MORE tokens (they output thinking process)
            if any(x in model_id.lower() for x in ["gpt-oss", "llama3-1", "deepseek", "r1"]):
                self.max_tokens = 8000  # ← HIGH for reasoning chains
                print(f"   🧠 Reasoning model detected: using {self.max_tokens} max_tokens")
            else:
                self.max_tokens = 4000  # ← DEFAULT for normal models
        else:
            self.max_tokens = max_tokens
        
        # Initialize Bedrock client
        self.client = boto3.client("bedrock-runtime", region_name=region_name)
    
    def generate(self, prompt: str) -> str:
        """
        Single method that formats prompt, invokes model, and parses response.
        
        Args:
            prompt: Raw input prompt text
            
        Returns:
            Generated text string
            
        Raises:
            Exception: If generation fails
        """
        try:
            # ==================== STEP 1: FORMAT PROMPT ====================
            if self.model_id.startswith("meta.llama"):
                # Llama instruction format
                formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
            
            elif self.model_id.startswith("mistral.mistral"):
                # Mistral instruction format
                formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            
            else:
                # Claude, Titan, DeepSeek, OpenAI OSS, Gemma use plain prompt
                formatted_prompt = prompt
            
            # ==================== STEP 2: BUILD REQUEST BODY ====================
            if self.model_id.startswith("anthropic.claude"):
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "messages": [{"role": "user", "content": formatted_prompt}]
                })
            
            elif self.model_id.startswith("meta.llama"):
                body = json.dumps({
                    "prompt": formatted_prompt,
                    "max_gen_len": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p
                })
            
            elif self.model_id.startswith("mistral.mistral"):
                body = json.dumps({
                    "prompt": formatted_prompt,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k
                })
            
            elif self.model_id.startswith("amazon.titan"):
                body = json.dumps({
                    "inputText": formatted_prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": self.max_tokens,
                        "temperature": self.temperature,
                        "topP": self.top_p
                    }
                })
            
            elif self.model_id.startswith("us.deepseek"):
                body = json.dumps({
                    "prompt": formatted_prompt,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p
                })
            
            elif self.model_id.startswith("openai.gpt-oss"):
                body = json.dumps({
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_completion_tokens": self.max_tokens,  # ← Uses self.max_tokens now
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                })

            elif self.model_id.startswith("google.gemma"):
                body = json.dumps({
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k
                })
            
            else:
                raise ValueError(f"Unsupported model ID: {self.model_id}")
            
            # ==================== STEP 3: INVOKE MODEL ====================
            response = self.client.invoke_model(
                contentType='application/json',
                accept='application/json',
                body=body,
                modelId=self.model_id
            )
            
            response_body = json.loads(response["body"].read())
            
            # ==================== STEP 4: PARSE RESPONSE ====================
            if self.model_id.startswith("anthropic.claude"):
                return response_body.get("content", [{}])[0].get("text", "")
            
            elif self.model_id.startswith("meta.llama"):
                return response_body.get("generation", "")
            
            elif self.model_id.startswith("mistral.mistral"):
                outputs = response_body.get("outputs", [])
                return outputs[0].get("text", "") if outputs else ""
            
            elif self.model_id.startswith("amazon.titan"):
                results = response_body.get("results", [])
                return results[0].get("outputText", "") if results else ""
            
            elif self.model_id.startswith("us.deepseek"):
                choices = response_body.get("choices", [])
                return choices[0].get("text", "") if choices else ""
            
            elif self.model_id.startswith("openai.gpt-oss"):
                choices = response_body.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    return message.get("content", "")
                return ""
            
            elif self.model_id.startswith("google.gemma"):
                choices = response_body.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    if content:
                        return content
            return ""
        
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]
            raise Exception(f"Bedrock API error [{error_code}]: {error_message}")
        
        except Exception as e:
            raise Exception(f"Generation failed: {str(e)}")
    
    def __repr__(self) -> str:
        return f"UnifiedBedrockClient(model='{self.model_id}', tokens={self.max_tokens}, temp={self.temperature})"
