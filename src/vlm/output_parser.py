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
        logger.info("Parsed VLM output successfully (direct JSON)")
        return result

    # Strategy 2: Extract from markdown code block (```json ... ``` or ``` ... ```)
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
    if json_match:
        result = _try_parse(json_match.group(1))
        if result is not None:
            logger.info("Parsed VLM output successfully (markdown block)")
            return result

    # Strategy 3: Find outermost braces (handles nested JSON)
    brace_content = _extract_outermost_braces(raw_text)
    if brace_content:
        result = _try_parse(brace_content)
        if result is not None:
            logger.info("Parsed VLM output successfully (brace extraction)")
            return result

    # Strategy 4: Try to fix common JSON issues
    cleaned = _clean_json_text(raw_text)
    if cleaned:
        result = _try_parse(cleaned)
        if result is not None:
            logger.info("Parsed VLM output successfully (cleaned JSON)")
            return result

    logger.error("All JSON parsing strategies failed for output: %s", raw_text[:300])
    return _make_error_result("Failed to parse JSON", raw_text)


def _try_parse(text: str):
    """Attempts to parse JSON, returns dict or None."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


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
    # Fix single quotes to double quotes (basic)
    content = re.sub(r"'(\w+)':", r'"\1":', content)
    # Fix True/False to true/false
    content = content.replace(': True', ': true').replace(': False', ': false')
    content = content.replace(':True', ':true').replace(':False', ':false')
    # Fix None to null
    content = content.replace(': None', ': null').replace(':None', ':null')

    return content


def _make_error_result(error_msg: str, raw_response: str) -> dict:
    """Creates a standardized error result."""
    return {
        "error": error_msg,
        "raw_response": raw_response,
        "humorous": None,
        "confidence": None,
        "detected_text": None,
        "visual_description": None,
        "cultural_context": None,
        "cultural_category": None,
        "cultural_dependency": None,
        "reason": None,
    }
