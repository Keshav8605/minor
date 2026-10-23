"""
Output parser for VLM-generated JSON responses.

Extracts structured data from the raw text output of Qwen2.5-VL.
Handles markdown wrapping, nested braces, and partial JSON.

CONSERVATIVE RECOVERY RULES:
- Attempt standard JSON extraction first.
- Attempt safe structural recovery only when possible.
- Validate recovered JSON using schema normalization.
- Never fabricate missing values.
- Never invent cultural context or reasoning.
- Never silently convert incomplete model output into a complete result.
- If recovery fails, return explicit deterministic fallback/error state.
- Token-level humor probability is preserved independently of JSON parsing.
"""

import json
import re
import logging

logger = logging.getLogger(__name__)

# Required fields that MUST be present for a valid result
_REQUIRED_FIELDS = {"humorous"}

# All expected fields (optional ones have default fallbacks)
_ALL_EXPECTED_FIELDS = {
    "humorous", "detected_text", "visual_description", "reason",
    "cultural_category", "cultural_dependency", "cultural_context",
    "humor_evidence", "non_humor_evidence"
}


def parse_json_response(raw_text: str) -> dict:
    """
    Extracts and parses JSON from the VLM output.
    Uses conservative multi-strategy recovery.

    Strategy order:
    1. Direct JSON parse
    2. Extract from markdown code block
    3. Extract outermost braces
    4. Clean common JSON formatting issues
    5. Attempt safe structural recovery on truncated JSON

    Returns:
        dict: Parsed JSON with at minimum 'humorous', 'detected_text', 'reason' keys.
              On failure, returns an error dict with 'error' and 'raw_response'.
    """
    if not raw_text:
        return _make_error_result("Empty response from model", "")

    raw_text = raw_text.strip()
    logger.debug("Parsing VLM output (%d chars): %s...", len(raw_text), raw_text[:200])

    # Strategy 1: Direct JSON parse
    result = _try_parse(raw_text)
    if result is not None:
        validated = _validate_and_normalize(result)
        if validated is not None:
            logger.info("Parsed VLM output successfully (direct JSON)")
            return validated

    # Strategy 2: Extract from markdown code block (```json ... ``` or ``` ... ```)
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
    if json_match:
        result = _try_parse(json_match.group(1))
        if result is not None:
            validated = _validate_and_normalize(result)
            if validated is not None:
                logger.info("Parsed VLM output successfully (markdown block)")
                return validated

    # Strategy 3: Find outermost braces (handles nested JSON)
    brace_content = _extract_outermost_braces(raw_text)
    if brace_content:
        result = _try_parse(brace_content)
        if result is not None:
            validated = _validate_and_normalize(result)
            if validated is not None:
                logger.info("Parsed VLM output successfully (brace extraction)")
                return validated

    # Strategy 4: Try to fix common JSON issues (trailing commas, single quotes, etc.)
    cleaned = _clean_json_text(raw_text)
    if cleaned:
        result = _try_parse(cleaned)
        if result is not None:
            validated = _validate_and_normalize(result)
            if validated is not None:
                logger.info("Parsed VLM output successfully (cleaned JSON)")
                return validated

    # Strategy 5: Conservative structural recovery for truncated JSON
    # Only attempt if we can detect a valid JSON start that was truncated
    recovered = _attempt_safe_structural_recovery(raw_text)
    if recovered is not None:
        validated = _validate_and_normalize(recovered)
        if validated is not None:
            logger.info("Parsed VLM output via safe structural recovery")
            return validated

    logger.error("All JSON parsing strategies failed for output: %s", raw_text[:300])
    return _make_error_result("Failed to parse JSON", raw_text)


def _try_parse(text: str):
    """Attempts to parse JSON, returns dict or None."""
    if not text:
        return None
    try:
        result = json.loads(text, strict=False)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        return None
    return None


def normalize_result_schema(data: dict) -> dict:
    """
    Normalizes any parsed VLM dict into ONE consistent canonical schema:
      - humorous: bool or None
      - prediction: 'Humorous', 'Not Humorous', or 'Unavailable'
      - detected_text: str
      - visual_description: str
      - cultural_category: str (properly formatted or sensible default)
      - cultural_dependency: 'Low', 'Medium', 'High', or 'Not analyzed'
      - cultural_context: str
      - reasoning: str (clean text, never raw JSON)
      - reason: str (alias pointing to reasoning)
    """
    if not isinstance(data, dict):
        return data

    norm = dict(data)

    # 1. Normalize humorous and prediction
    h_val = norm.get("humorous")
    if h_val is True or h_val == 1 or str(h_val).lower() == "true":
        norm["humorous"] = True
        norm["prediction"] = "Humorous"
    elif h_val is False or h_val == 0 or str(h_val).lower() == "false":
        norm["humorous"] = False
        norm["prediction"] = "Not Humorous"
    else:
        norm["humorous"] = None
        norm["prediction"] = "Unavailable"

    # 2. Normalize reasoning (priority: reasoning then reason)
    reasoning_text = norm.get("reasoning") or norm.get("reason") or ""
    reasoning_text = str(reasoning_text).strip()

    # Prevent raw JSON leakage into reasoning
    if reasoning_text.startswith("{") and ('"humorous"' in reasoning_text or '"reason"' in reasoning_text):
        inner = _try_parse(reasoning_text)
        if inner and isinstance(inner, dict):
            reasoning_text = inner.get("reasoning") or inner.get("reason") or ""
            if not norm.get("cultural_context") and inner.get("cultural_context"):
                norm["cultural_context"] = inner["cultural_context"]
            if not norm.get("cultural_category") and inner.get("cultural_category"):
                norm["cultural_category"] = inner["cultural_category"]
            if not norm.get("cultural_dependency") and inner.get("cultural_dependency"):
                norm["cultural_dependency"] = inner["cultural_dependency"]

    if not reasoning_text or reasoning_text.lower() in ("none", "null"):
        reasoning_text = "No reasoning provided."

    norm["reasoning"] = reasoning_text
    norm["reason"] = reasoning_text

    # 3. Normalize cultural context
    ctx = norm.get("cultural_context")
    if not ctx or str(ctx).lower() in ("none", "null", ""):
        ctx_used = norm.get("cultural_context_used")
        if isinstance(ctx_used, str) and ctx_used.lower() not in ("true", "false", "none", "null", ""):
            ctx = ctx_used
    if not ctx or str(ctx).lower() in ("none", "null", ""):
        ctx = "No significant cultural context detected"
    norm["cultural_context"] = str(ctx).strip()

    # 4. Normalize cultural category
    cat = norm.get("cultural_category")
    if not cat or str(cat).lower() in ("none", "null", ""):
        norm["cultural_category"] = "No specific cultural category detected"
    else:
        cat_str = str(cat).strip()
        parts = [p.strip().title() for p in re.split(r'[/,]', cat_str) if p.strip()]
        norm["cultural_category"] = " / ".join(parts) if parts else cat_str.title()

    # 5. Normalize cultural dependency (Low, Medium, High)
    dep = norm.get("cultural_dependency")
    if not dep or str(dep).lower() in ("null", ""):
        norm["cultural_dependency"] = "Not analyzed"
    else:
        dep_str = str(dep).strip().lower()
        if "high" in dep_str:
            norm["cultural_dependency"] = "High"
        elif "med" in dep_str:
            norm["cultural_dependency"] = "Medium"
        elif "low" in dep_str or "none" in dep_str:
            norm["cultural_dependency"] = "Low"
        else:
            norm["cultural_dependency"] = dep_str.title()

    # 6. Normalize detected_text
    dt = norm.get("detected_text")
    if not dt or str(dt).lower() in ("none", "null", ""):
        norm["detected_text"] = "No text detected in image"
    else:
        norm["detected_text"] = str(dt).strip()

    # 7. Normalize visual_description
    vd = norm.get("visual_description")
    if not vd or str(vd).lower() in ("none", "null", ""):
        norm["visual_description"] = "No visual description provided"
    else:
        norm["visual_description"] = str(vd).strip()

    return norm


def _validate_and_normalize(parsed: dict) -> dict:
    """
    Validates a parsed JSON dict against the expected schema.

    - Required fields must exist (humorous).
    - Optional fields get None defaults if missing (NOT fabricated values).
    - Rejects dicts that don't contain 'humorous' at all.

    Returns:
        dict or None: The normalized dict, or None if validation fails.
    """
    if not isinstance(parsed, dict):
        return None

    # 'humorous' is the only strictly required field
    if "humorous" not in parsed:
        logger.warning("Parsed JSON missing required 'humorous' field: %s",
                       list(parsed.keys()))
        return None

    # Normalize 'humorous' to bool
    h = parsed["humorous"]
    if isinstance(h, bool):
        pass  # already correct
    elif isinstance(h, str):
        if h.lower() in ("true", "yes", "1"):
            parsed["humorous"] = True
        elif h.lower() in ("false", "no", "0"):
            parsed["humorous"] = False
        else:
            logger.warning("Unrecognized humorous value: %s", h)
            return None
    elif isinstance(h, (int, float)):
        parsed["humorous"] = bool(h)
    else:
        logger.warning("Invalid humorous type: %s", type(h))
        return None

    # Set None for missing optional fields — do NOT fabricate values
    for field in _ALL_EXPECTED_FIELDS:
        if field not in parsed:
            parsed[field] = None

    # Derive prediction field if not explicitly present
    if parsed.get("prediction") is None:
        h_val = parsed.get("humorous")
        if h_val is True:
            parsed["prediction"] = "Humorous"
        elif h_val is False:
            parsed["prediction"] = "Not Humorous"
        else:
            parsed["prediction"] = "Unavailable"

    return parsed


def _extract_outermost_braces(text: str):
    """Extracts text between the first '{' and the last matching '}'."""
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    end = None
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                end = i
                break

    if end is not None:
        return text[start:end + 1]
    return None


def _clean_json_text(text: str):
    """Attempts to fix common JSON formatting issues."""
    # Find JSON-like content
    content = _extract_outermost_braces(text)
    if not content:
        return None

    # Fix trailing commas before closing brace
    content = re.sub(r',\s*}', '}', content)
    # Fix single quotes to double quotes for keys and values
    content = re.sub(r"'(\w+)':", r'"\1":', content)
    content = re.sub(r":\s*'([^']*)'", r': "\1"', content)
    # Fix True/False to true/false
    content = content.replace(': True', ': true').replace(': False', ': false')
    content = content.replace(':True', ':true').replace(':False', ':false')
    # Fix None to null
    content = content.replace(': None', ': null').replace(':None', ':null')

    return content


def _attempt_safe_structural_recovery(raw_text: str) -> dict:
    """
    Attempts conservative structural recovery on truncated JSON.

    RULES:
    - Only attempts recovery if JSON starts validly but appears truncated
      (i.e., has '{' but no matching '}').
    - Only recovers if we can extract at least the 'humorous' field from
      the partial content.
    - Does NOT blindly append '}' and treat as valid.
    - Does NOT fabricate missing field values.
    - Does NOT invent cultural context or reasoning.

    Returns:
        dict or None: Recovered dict if safe recovery succeeded, else None.
    """
    # Find the start of JSON
    start = raw_text.find('{')
    if start == -1:
        return None

    partial = raw_text[start:]

    # Check if it's actually truncated (has opening { but no matching })
    depth = 0
    for ch in partial:
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
    if depth <= 0:
        # Braces are balanced or over-closed — not a truncation issue
        return None

    logger.warning("Detected truncated JSON (unclosed depth=%d). Attempting safe key extraction.", depth)

    # Extract key-value pairs that were fully written before truncation.
    # We use regex to find complete "key": value pairs.
    extracted = {}

    # Match "key": "string_value" pairs (complete ones only)
    string_pairs = re.findall(
        r'"(\w+)"\s*:\s*"((?:[^"\\]|\\.)*)"',
        partial
    )
    for key, value in string_pairs:
        extracted[key] = value

    # Match "key": true/false (case-insensitive and quoted variants)
    bool_pairs = re.findall(r'"(\w+)"\s*:\s*(true|false|True|False)\b', partial, re.IGNORECASE)
    for key, value in bool_pairs:
        if key not in extracted:
            extracted[key] = value.lower() == "true"

    quoted_bool_pairs = re.findall(r'"(\w+)"\s*:\s*"(true|false|True|False)"', partial, re.IGNORECASE)
    for key, value in quoted_bool_pairs:
        if key not in extracted:
            extracted[key] = value.lower() == "true"

    # Match "key": number
    num_pairs = re.findall(r'"(\w+)"\s*:\s*(-?\d+(?:\.\d+)?)\b', partial)
    for key, value in num_pairs:
        if key not in extracted:
            try:
                extracted[key] = float(value) if '.' in value else int(value)
            except ValueError:
                pass

    # Match "key": null
    null_pairs = re.findall(r'"(\w+)"\s*:\s*null\b', partial)
    for key, _ in null_pairs:
        if key not in extracted:
            extracted[key] = None

    # Handle unclosed trailing string for text fields (e.g. reason or visual_description cut off)
    trailing_str = re.search(r'"(\w+)"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)$', partial)
    if trailing_str:
        t_key, t_val = trailing_str.group(1), trailing_str.group(2)
        if t_key not in extracted and t_key in _ALL_EXPECTED_FIELDS:
            extracted[t_key] = t_val.strip()

    if not extracted:
        logger.warning("Safe structural recovery failed: no complete key-value pairs found")
        return None

    # Only accept recovery if the critical 'humorous' field was extracted
    if "humorous" not in extracted or extracted["humorous"] is None:
        logger.warning("Safe structural recovery: 'humorous' field not found in partial content")
        return None

    logger.info("Safe structural recovery extracted %d fields: %s",
                len(extracted), list(extracted.keys()))

    # Mark recovered fields as partial — do NOT fabricate the rest
    extracted["_recovery"] = "partial_truncation"
    return extracted


def _make_error_result(error_msg: str, raw_response: str) -> dict:
    """Creates a standardized error result."""
    return {
        "error": error_msg,
        "raw_response": raw_response,
        "humorous": None,
        "prediction": "Unavailable",
        "confidence": None,
        "detected_text": None,
        "visual_description": None,
        "cultural_context": None,
        "cultural_category": None,
        "cultural_dependency": None,
        "reason": None,
        "reasoning": "Inference failed or output was unparseable.",
    }
