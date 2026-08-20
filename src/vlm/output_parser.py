import json
import re

def parse_json_response(raw_text: str) -> dict:
    """
    Extracts and parses JSON from the VLM output.
    Attempts to safely recover if wrapped in markdown blocks.
    Never silently fabricates values; returns a specific failure object on error.
    """
    raw_text = raw_text.strip()
    
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass
        
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
            
    brace_match = re.search(r'(\{.*?\})', raw_text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(1))
        except json.JSONDecodeError:
            pass

    return {
        "error": "Failed to parse JSON",
        "raw_response": raw_text,
        "humorous": None,
        "confidence": None,
        "detected_text": None,
        "visual_description": None,
        "cultural_context": None,
        "reason": None
    }
