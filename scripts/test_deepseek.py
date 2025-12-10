"""Quick test for DeepSeek-R1."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.bedrock_client import BedrockClient


def test_deepseek():
    """Test DeepSeek-R1 model."""
    
    print("="*70)
    print("TESTING DEEPSEEK-R1")
    print("="*70)
    
    try:
        client = BedrockClient(model_id="deepseek", model_type="deepseek")
        
        prompt = """Extract validation rules from this text:

"Installation Qualification (IQ) should be performed on equipment. 
IQ should include calibration of instrumentation and verification 
of materials of construction."

Return ONLY valid JSON with a rules array. Each rule should have "rule" and "pretext" fields."""
        
        response = client.generate(
            prompt=prompt,
            max_tokens=2000,  # INCREASED from 512 to 2000
            temperature=0.3
        )
        
        print(f"\n✅ SUCCESS!\n")
        print(f"📝 Final Answer:\n{'-'*70}")
        print(response['text'])
        print('-'*70)
        
        if 'full_response' in response and response['full_response'] != response['text']:
            print(f"\n🧠 DeepSeek's Reasoning Process:\n{'-'*70}")
            # Show just first 500 chars of reasoning
            reasoning = response['full_response'].replace(response['text'], '').strip()
            print(reasoning[:500] + "..." if len(reasoning) > 500 else reasoning)
            print('-'*70)
        
        print(f"\nStop Reason: {response.get('stop_reason', 'N/A')}")
        print(f"Model: {response.get('model', 'N/A')}")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_deepseek()
