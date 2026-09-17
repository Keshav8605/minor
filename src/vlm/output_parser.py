"""
Output parser for VLM-generated JSON responses.

Extracts structured data from the raw text output of Qwen2.5-VL.
Handles markdown wrapping, nested braces, and partial JSON.
Never silently fabricates values; returns a specific failure object on error.
"""

import json
import re
import logging

logger = logging.getLogger(__name__)


def parse_json_response(raw_text: str) -> dict:
    """
    Extracts and parses JSON from the VLM output.
    Attempts multiple recovery strategies if the JSON is wrapped or malformed.
    Normalizes keys so reasoning, cultural context, and cultural dependency
    are cleanly extracted without raw JSON leakage.

    Returns:
        dict: Parsed and normalized JSON with humorous, detected_text, reasoning,
              cultural_category, cultural_dependency, cultural_context, etc.
              On failure, returns an error dict with 'error' and 'raw_response'.
    """
    if not raw_text:
        return _make_error_result("Empty response from model", "")

    raw_text = raw_text.strip()
    logger.debug("Parsing VLM output (%d chars): %s...", len(raw_text), raw_text[:200])

    result = None

    # Strategy 1: Direct JSON parse
    result = _try_parse(raw_text)
    if result is not None and isinstance(result, dict):
        logger.info("Parsed VLM output successfully (direct JSON)")
        return normalize_result_schema(result)

    # Strategy 2: Extract from markdown code block (```json ... ``` or ``` ... ```)
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
    if json_match:
        result = _try_parse(json_match.group(1))
        if result is not None and isinstance(result, dict):
            logger.info("Parsed VLM output successfully (markdown block)")
            return normalize_result_schema(result)

    # Strategy 3: Find outermost braces (handles nested JSON, string-aware)
    brace_content = _extract_outermost_braces(raw_text)
    if brace_content:
        result = _try_parse(brace_content)
        if result is not None and isinstance(result, dict):
            logger.info("Parsed VLM output successfully (brace extraction)")
            return normalize_result_schema(result)

    # Strategy 4: Try to fix common JSON issues
    cleaned = _clean_json_text(raw_text)
    if cleaned:
        result = _try_parse(cleaned)
        if result is not None and isinstance(result, dict):
            logger.info("Parsed VLM output successfully (cleaned JSON)")
            return normalize_result_schema(result)

    # Strategy 5: Field-level regex extraction fallback
    # If the JSON syntax is broken or truncated, extract individual fields
    field_result = _extract_fields_by_regex(raw_text)
    if field_result and (field_result.get("humorous") is not None or field_result.get("reason") or field_result.get("reasoning")):
        logger.info("Parsed VLM output successfully via field-level regex extraction")
        return normalize_result_schema(field_result)

    logger.error("All JSON parsing strategies failed for output: %s", raw_text[:300])
    return _make_error_result("Failed to parse JSON", raw_text)


def _try_parse(text: str):
    """Attempts to parse JSON with standard loads and strict=False for control characters."""
    if not text:
        return None
    try:
        return json.loads(text, strict=False)
    except (json.JSONDecodeError, ValueError):
        pass

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_outermost_braces(text: str):
    """Extracts text between the first '{' and the matching last '}' considering quotes."""
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    end = None

    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == '\\':
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if not in_string:
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break

    if end is not None:
        return text[start:end + 1]

    # Fallback to last index of '}'
    last_brace = text.rfind('}')
    if last_brace > start:
        return text[start:last_brace + 1]

    return None


def _clean_json_text(text: str):
    """Attempts to fix common JSON formatting issues."""
    content = _extract_outermost_braces(text)
    if not content:
        return None

    # Fix trailing commas before closing brace or bracket
    content = re.sub(r',\s*([\}\]])', r'\1', content)
    # Fix single quotes around keys and strings (when double quotes are absent)
    content = re.sub(r"\'(\w+)\'\s*:", r'"\1":', content)
    # Fix unquoted boolean literals
    content = re.sub(r':\s*True\b', ': true', content)
    content = re.sub(r':\s*False\b', ': false', content)
    content = re.sub(r':\s*None\b', ': null', content)

    return content


def _extract_fields_by_regex(text: str) -> dict:
    """
    Robust fallback to extract key-value pairs using regex
    when JSON structure is slightly broken or truncated.
    """
    data = {}

    # humorous
    humor_match = re.search(r'"humorous"\s*:\s*(true|false|True|False)', text)
    if humor_match:
        data["humorous"] = humor_match.group(1).lower() == "true"

    # string fields
    fields = [
        "detected_text", "visual_description", "cultural_category",
        "cultural_dependency", "cultural_context", "cultural_context_used",
        "reason", "reasoning"
    ]
    for field in fields:
        # Match string value with escaped quote support
        pattern = rf'"{field}"\s*:\s*"((?:\\.|[^"\\])*)"'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            # Unescape string
            val = match.group(1).replace(r'\"', '"').replace(r'\n', '\n').strip()
            data[field] = val

    return data


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

    # Prevent raw JSON leakage into reasoning:
    # If reasoning_text itself looks like a JSON object containing keys like "humorous"
    if reasoning_text.startswith("{") and ('"humorous"' in reasoning_text or '"reason"' in reasoning_text):
        inner = _try_parse(reasoning_text)
        if inner and isinstance(inner, dict):
            reasoning_text = inner.get("reasoning") or inner.get("reason") or ""
            # Also extract cultural context or category from inner if missing in parent
            if not norm.get("cultural_context") and inner.get("cultural_context"):
                norm["cultural_context"] = inner["cultural_context"]
            if not norm.get("cultural_category") and inner.get("cultural_category"):
                norm["cultural_category"] = inner["cultural_category"]
            if not norm.get("cultural_dependency") and inner.get("cultural_dependency"):
                norm["cultural_dependency"] = inner["cultural_dependency"]

    if not reasoning_text or reasoning_text.lower() in ("none", "null"):
        reasoning_text = "No reasoning provided."

    norm["reasoning"] = reasoning_text
    norm["reason"] = reasoning_text  # maintain backwards compatibility

    # 3. Normalize cultural context (from cultural_context or cultural_context_used)
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
        # If category string has multiple slash/comma separated items, title case each
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


def _make_error_result(error_msg: str, raw_response: str) -> dict:
    """Creates a standardized error result."""
    return {
        "error": error_msg,
        "raw_response": raw_response,
        "status": "error",
        "prediction": "Unavailable",
        "humorous": None,
        "confidence": None,
        "humor_probability": None,
        "non_humor_probability": None,
        "detected_text": "No text detected",
        "visual_description": "No visual description available",
        "cultural_context": "Not analyzed",
        "cultural_category": "No specific cultural category detected",
        "cultural_dependency": "Not analyzed",
        "reason": "Analysis could not be completed.",
        "reasoning": "Analysis could not be completed.",
    }
