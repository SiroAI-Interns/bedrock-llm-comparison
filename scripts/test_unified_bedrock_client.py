import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from app.core.unified_bedrock_client import UnifiedBedrockClient

client = UnifiedBedrockClient(
    model_id="google.gemma-3-27b-it",
    max_tokens=100,
    temperature=0.7
)

client2 = UnifiedBedrockClient(
    model_id="openai.gpt-oss-20b-1:0",  # ensure this ID is correct in console
    max_tokens=200,
    temperature=0.5
)
client3 = UnifiedBedrockClient(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    max_tokens=150,
    temperature=0.5
)
client4 = UnifiedBedrockClient(
    model_id="us.deepseek.r1-v1:0",
    max_tokens=6000,
    region_name="us-west-2",
    temperature=0.5
)   
client5 = UnifiedBedrockClient(
    model_id="meta.llama3-70b-instruct-v1:0",
    max_tokens=150,
    temperature=0.5
)   
client6 = UnifiedBedrockClient(
    model_id="amazon.titan-text-express-v1",
    max_tokens=150,
    temperature=0.5
)
client7 = UnifiedBedrockClient(
    model_id="mistral.mistral-7b-instruct-v0:2",
    max_tokens=150,
    temperature=0.5
)   

resp1 = client.generate(
    "Write exactly one short tagline (max 10 words) for a sustainable coffee brand. "
    "Return only the tagline, no explanations, no reasoning."
)
print("Gemma:", (resp1 or "").strip())

resp2 = client2.generate(
    "Write exactly one short tagline (max 10 words) for a sustainable coffee brand. "
    "Return only the tagline, no explanations, no reasoning."
)
print("GPT-OSS:", (resp2 or "").strip())

resp3 = client3.generate(
    "Write exactly one short tagline (max 10 words) for a sustainable coffee brand. "
    "Return only the tagline, no explanations, no reasoning."
)
print("Claude:", (resp3 or "").strip())

resp4 = client4.generate(
    "Write exactly one short tagline (max 10 words) for a sustainable coffee brand."
    "Return only the tagline, no explanations, strictly no reasoning just the tagline."
)
print("Deepseek:", (resp4 or "").strip())

resp5 = client5.generate(
    "Write exactly one short tagline (max 10 words) for a sustainable coffee brand. "
    "Return only the tagline, no explanations, no reasoning."
)
print("Llama 3:", (resp5 or "").strip())

resp6 = client6.generate(
    "Write exactly one short tagline (max 10 words) for a sustainable coffee brand. "
    "Return only the tagline, no explanations, no reasoning."
)
print("Titan:", (resp6 or "").strip())  

resp7 = client7.generate(
    "Write exactly one short tagline (max 10 words) for a sustainable coffee brand. "
    "Return only the tagline, no explanations, no reasoning."
)
print("Mistral:", (resp7 or "").strip())