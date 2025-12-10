"""Rule extraction service - FIXED TAG PARSING."""

import json
import re
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

from app.core.bedrock_client import BedrockClient
from app.core.openai_client import PaidModelsClient
from app.models.validation_schema import ValidationDocument, ValidationRule, MetaData, Tag


class RuleExtractor:
    """Extract validation rules from pharmaceutical documents."""
    
    def __init__(self, use_model: str = "claude"):
        self.use_model = use_model.lower()
        self.bedrock_client = None
        self.openai_client = None
        
        if self.use_model in ["claude", "titan", "llama","deepseek"]:
            model_map = {
                "claude": ("anthropic.claude-3-5-sonnet-20240620-v1:0", "anthropic"),
                "titan": ("amazon.titan-text-express-v1", "titan"),
                "llama": ("meta.llama3-70b-instruct-v1:0", "llama"),
                "deepseek": ("us.deepseek.r1-v1:0", "deepseek"),
            }
            # In extract_rules_from_page, for Llama add:
            model_id, model_type = model_map[self.use_model]
            self.bedrock_client = BedrockClient(
                model_id=model_id,
                model_type=model_type,
                region="us-east-1"
            )
        elif self.use_model in ["openai", "gemini"]:
            self.openai_client = PaidModelsClient()
    
    def extract_rules_from_page(self, page_text: str, page_number: str) -> List[Dict[str, Any]]:
        """Extract validation rules from a single page."""

        # BASE PROMPT - shared by all models
        base_prompt = f"""You are extracting validation rules from a pharmaceutical regulatory guideline page.

    Page: {page_number}

    Task:
    - Identify every sentence or group of sentences that expresses a requirement, recommendation, or expectation.
    - Treat any sentence containing words like "should", "must", "shall", "required to", or "is expected to" as a validation rule.

    For each such rule, return:
    - "rule": the full sentence(s) describing the requirement.
    - "pretext": 1–2 sentences explaining when/where this applies, in plain language.
    - "tags": leave as an empty array [].

    Page content:
    {page_text[:2500]}
    """

        # MODEL-SPECIFIC PROMPT ADJUSTMENTS
        if self.use_model == "gemini":
            # Gemini prefers looser, conversational instructions
            prompt = base_prompt + """

    If there are no rules on this page, return: {"rules": []}

    Return the result as valid JSON in this format:
    {
    "rules": [
        {
        "rule": "Complete rule text here.",
        "pretext": "Context about when this rule applies.",
        "tags": []
        }
    ]
    }
    """
        
        elif self.use_model in ["llama", "titan"]:
            # Llama/Titan need very strict formatting instructions
            prompt = base_prompt + """

    If there are no rules on this page, return:
    { "rules": [] }

    You MUST return syntactically valid JSON. Do not include comments, trailing commas, or any text before or after the JSON.

    Format exactly:

    {
    "rules": [
        {
        "rule": "Complete rule text here.",
        "pretext": "1–2 sentences of context about when this rule applies.",
        "tags": []
        }
    ]
    }

    IMPORTANT: Return ONLY valid JSON. Add commas after every property except the last one.
    """
        
        else:
            # Claude, OpenAI, DeepSeek - moderate strictness
            prompt = base_prompt + """

    If there are no rules on this page, return:
    { "rules": [] }

    You MUST return syntactically valid JSON. Do not include comments, trailing commas, or any text before or after the JSON.

    Format exactly:

    {
    "rules": [
        {
        "rule": "Complete rule text here.",
        "pretext": "1–2 sentences of context about when this rule applies.",
        "tags": []
        }
    ]
    }
    """

        try:
            max_tokens = 3000 if self.use_model == "deepseek" else 2000

            # ===== LLAMA: Special handling =====
            if self.use_model == "llama" and self.bedrock_client:
                formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
    {prompt}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
                
                import boto3
                import json
                
                bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
                
                native_request = {
                    "prompt": formatted_prompt,
                    "max_gen_len": max_tokens,
                    "temperature": 0.2,
                }
                
                response = bedrock_runtime.invoke_model(
                    modelId="meta.llama3-70b-instruct-v1:0",
                    body=json.dumps(native_request)
                )
                
                model_response = json.loads(response["body"].read())
                response_text = model_response.get("generation", "")

            # ===== GEMINI =====
            elif self.use_model == "gemini":
                from google import genai

                gemini_client = genai.Client()
                
                response = gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                
                response_text = response.text

            # ===== OPENAI =====
            elif self.openai_client and self.use_model == "openai":
                response = self.openai_client.generate(
                    provider="openai",
                    model="gpt-4o-mini",
                    prompt=prompt,
                    max_tokens=2000,
                    temperature=0.2,
                )
                response_text = response.get("text", "")

            # ===== OTHER BEDROCK MODELS =====
            elif self.bedrock_client:
                response = self.bedrock_client.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.2,
                    top_p=0.9,
                )
                response_text = response.get("text", "")

            else:
                return []

        except Exception as e:
            print(f"[ERROR] LLM call failed for {self.use_model} page {page_number}: {e}")
            return []

        rules = self._parse_json_response(response_text, page_number)

        for rule in rules:
            rule["page_number"] = page_number

        return rules



    
    def _parse_json_response(self, response_text: str, page_number: str) -> List[Dict[str, Any]]:
        """Parse JSON response from any model using robust extraction and repair."""
        import re
        import json
        
        if not response_text or not response_text.strip():
            return []

        cleaned = response_text.strip()

        # Remove code fences if present
        cleaned = re.sub(r"^⁠  (?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"  ⁠$", "", cleaned).strip()


        # Try direct parse first
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and "rules" in parsed:
                return parsed.get("rules", [])
            return []
        except Exception:
            pass

        # Extract first JSON block using non-greedy matching
        match = re.search(r"(\{.*?\}|\[.*?\])", cleaned, re.DOTALL)
        if not match:
            return []
        
        json_str = match.group(1).strip()
        
        # ===== AGGRESSIVE JSON REPAIR FOR LLAMA =====
        
        # 1. Fix missing commas between properties (most common Llama issue)
        # Pattern: "text" \n "property" becomes "text", \n "property"
        json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)
        json_str = re.sub(r'"\s*\n\s*}', '"\n}', json_str)
        
        # 2. Fix missing commas before closing braces/brackets
        json_str = re.sub(r'(["\d])\s*\n\s*}', r'\1\n}', json_str)
        json_str = re.sub(r'(["\d])\s*\n\s*\]', r'\1\n]', json_str)
        
        # 3. Fix single quotes to double quotes
        json_str = json_str.replace("'", '"')
        
        # 4. Remove trailing commas (already had this)
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # 5. Fix common property name issues
        json_str = re.sub(r'(\w+):', r'"\1":', json_str)  # Unquoted keys
        json_str = re.sub(r'""(\w+)"":', r'"\1":', json_str)  # Double-quoted keys
        
        # Try parsing after repairs
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and "rules" in parsed:
                return parsed.get("rules", [])
            return []
        except Exception as e:
            # Last resort: try to extract rules array directly
            rules_match = re.search(r'"rules"\s*:\s*\[(.*?)\]', cleaned, re.DOTALL)
            if rules_match:
                rules_content = rules_match.group(1)
                # Try to parse individual rule objects
                rule_objects = re.findall(r'\{[^{}]*\}', rules_content)
                parsed_rules = []
                for rule_obj in rule_objects:
                    try:
                        # Apply same repairs
                        rule_obj = re.sub(r'"\s*\n\s*"', '",\n"', rule_obj)
                        rule_obj = rule_obj.replace("'", '"')
                        rule_obj = re.sub(r',(\s*})', r'\1', rule_obj)
                        parsed_rules.append(json.loads(rule_obj))
                    except:
                        continue
                if parsed_rules:
                    return parsed_rules
            
            print(f"[WARN] Parse failed for {self.use_model} page {page_number}: {str(e)[:100]}")
            return []

    
    def process_document(
        self,
        pdf_path: str,
        document_title: str,
        s3_url: str = ""
    ) -> ValidationDocument:
        """Process entire PDF and extract all validation rules."""
        
        from app.services.pdf_processor import PDFProcessor
        
        processor = PDFProcessor()
        
        print(f"\n{'='*70}")
        print(f"Processing: {document_title}")
        print(f"{'='*70}\n")
        
        full_text = processor.extract_text_from_pdf(pdf_path)
        pages = self._split_into_pages(full_text)
        
        print(f"Extracted {len(pages)} pages")
        print(f"Using model: {self.use_model}\n")
        
        all_rules = []
        
        for page_num, page_text in enumerate(pages, 1):
            page_number_str = f"{page_num:03d}"
            
            print(f"Processing page {page_number_str}... ", end="", flush=True)
            
            rules = self.extract_rules_from_page(page_text, page_number_str)
            
            if rules:
                all_rules.extend(rules)
                print(f"✓ Found {len(rules)} rule(s)")
            else:
                print("✓ No rules found")
        
        print(f"\n{'='*70}")
        print(f"Total rules extracted: {len(all_rules)}")
        print(f"{'='*70}\n")
        
        # Convert to Pydantic models
        validation_rules = []
        for rule_data in all_rules:
            try:
                # Create default tag if needed
                tags_list = rule_data.get("tags", [])
                if not tags_list:
                    tags_list = []
                
                # Parse tags (handle both dict and string)
                tags = []
                for tag_data in tags_list:
                    if isinstance(tag_data, dict):
                        if "main_tag" not in tag_data and "main-tag" in tag_data:
                            tag_data["main_tag"] = tag_data.pop("main-tag")
                        tags.append(Tag(**tag_data))
                    elif isinstance(tag_data, str):
                        # Skip string tags
                        continue
                
                validation_rule = ValidationRule(
                    page_number=rule_data["page_number"],
                    rule=rule_data["rule"],
                    pretext=rule_data.get("pretext", "No context provided"),
                    tags=tags
                )
                validation_rules.append(validation_rule)
            except Exception as e:
                print(f"[DEBUG] Skipping rule: {str(e)}")
                continue
        
        print(f"Valid rules after validation: {len(validation_rules)}\n")
        
        # Create metadata
        file_path = Path(pdf_path)
        s3_key = f"System Uploaded/general/{file_path.name}"
        
        if not s3_url:
            s3_url = f"s3://trial-guidelines/{s3_key}"
        
        meta_data = MetaData(
            title=document_title,
            description="",
            s3Key=s3_key,
            s3Uri=s3_url
        )
        
        # Create document - EXPLICIT PARAMETERS
        validation_doc = ValidationDocument(
            validation_ruleset=validation_rules,
            s3_url=s3_url,
            state="PROCESSING_RULES",
            meta_data=meta_data
        )
        
        return validation_doc
    
    def _split_into_pages(self, text: str) -> List[str]:
        """Split document text into pages."""
        
        if "--- Page" in text:
            pages = text.split("--- Page")
            return [p.strip() for p in pages if p.strip()]
        
        page_length = 3000
        pages = []
        current_page = ""
        
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            if len(current_page) + len(para) < page_length:
                current_page += para + "\n\n"
            else:
                if current_page:
                    pages.append(current_page.strip())
                current_page = para + "\n\n"
        
        if current_page:
            pages.append(current_page.strip())
        
        return pages
    
    def save_to_json(self, validation_doc: ValidationDocument, output_path: str):
        """Save validation document to JSON file."""
        
        doc_dict = json.loads(validation_doc.model_dump_json(by_alias=True))
        
        doc_dict["createdAt"] = {"$date": validation_doc.createdAt.isoformat() + "Z"}
        doc_dict["updatedAt"] = {"$date": validation_doc.updatedAt.isoformat() + "Z"}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(doc_dict, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved JSON to: {output_path}")
        print(f"   Rules: {len(validation_doc.validation_ruleset)}")
        print(f"   File size: {Path(output_path).stat().st_size / 1024:.1f} KB\n")
