
"""Text processing utilities - no external dependencies."""

import re
import json
from typing import List


def extract_first_json_block(raw: str) -> str:
    """Extract first JSON block from text."""
    if not isinstance(raw, str):
        return raw
    cleaned = raw.strip()
    cleaned = re.sub(r"^`{3}(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*`{3}$", "", cleaned)
    match = re.search(r"(\{[\s\S]*?\}|\[[\s\S]*?\])", cleaned, re.DOTALL)
    if match:
        return match.group(1).strip()
    return cleaned


def extract_bullets_simple(text: str) -> List[str]:
    """Extract bullet points from text."""
    if not isinstance(text, str):
        return []
    matches = re.findall(r'(?m)^\s*(?:•|\-|\*|\d+[\.\)])\s*(.+)\s*$', text)
    return [m.strip() for m in matches]


def top_sentences(text: str, n: int = 5) -> List[str]:
    """Extract top N sentences."""
    if not isinstance(text, str):
        return []
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()][:n]


def normalize_output_for_recommendation(raw_text: str, want_n: int = 5) -> List[str]:
    """Normalize output to list of strings."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return []
    
    try:
        jblk = extract_first_json_block(raw_text)
        if isinstance(jblk, str) and jblk.strip().startswith(("{", "[")):
            parsed = json.loads(jblk)
            if isinstance(parsed, dict):
                for k in ("summary", "summaries", "bullets", "results"):
                    if k in parsed and isinstance(parsed[k], (list, tuple)):
                        return [str(x).strip() for x in parsed[k]][:want_n]
                for v in parsed.values():
                    if isinstance(v, list):
                        return [str(x).strip() for x in v][:want_n]
            elif isinstance(parsed, list):
                return [str(x).strip() for x in parsed][:want_n]
    except Exception:
        pass
    
    bullets = extract_bullets_simple(raw_text)
    if bullets:
        return bullets[:want_n]
    
    return top_sentences(re.sub(r'\s+', ' ', raw_text), n=want_n)