"""
Formatting utilities for the Gradio UI.

Converts structured backend result dicts into UI-ready display values.
Handles confidence visualization with SVG progress ring.
"""

import json
import math


def format_confidence(conf_val, humor_prob=None, non_humor_prob=None):
    """
    Format confidence as HTML with a circular SVG progress ring
    and dual probability bars.

    Args:
        conf_val: Confidence value (0-1 float, or string like "N/A").
        humor_prob: P(Humorous) float, optional.
        non_humor_prob: P(Non-Humorous) float, optional.
    """
    try:
        conf = float(conf_val)
        percent = round(conf * 100, 1)
        circumference = 2 * math.pi * 28  # radius = 28
        offset = circumference * (1 - conf)

        if conf >= 0.8:
            color = "#22C55E"
            label = "High Confidence"
        elif conf >= 0.5:
            color = "#F59E0B"
            label = "Medium Confidence"
        else:
            color = "#EF4444"
            label = "Low Confidence"

        html = (
            f"<div style='display:flex;align-items:center;gap:16px;'>"
            f"<div>"
            f"<div style='font-size:28px;font-weight:700;color:#F4F6FA;line-height:1.2;'>{percent}%</div>"
            f"<div style='font-size:13px;color:{color};font-weight:500;margin-top:3px;'>{label}</div>"
            f"</div>"
            f"<svg width='64' height='64' viewBox='0 0 64 64' style='flex-shrink:0;'>"
            f"<circle cx='32' cy='32' r='28' stroke='#1E2438' stroke-width='5' fill='none'/>"
            f"<circle cx='32' cy='32' r='28' stroke='{color}' stroke-width='5' fill='none' "
            f"stroke-dasharray='{circumference:.1f}' stroke-dashoffset='{offset:.1f}' "
            f"transform='rotate(-90 32 32)' stroke-linecap='round'/>"
            f"<text x='32' y='36' text-anchor='middle' font-size='13' font-weight='600' "
            f"fill='#F4F6FA' font-family='Inter,sans-serif'>{percent}%</text>"
            f"</svg>"
            f"</div>"
        )

        # Add probability breakdown if available
        if humor_prob is not None and non_humor_prob is not None:
            h_pct = round(humor_prob * 100, 1)
            n_pct = round(non_humor_prob * 100, 1)
            html += (
                f"<div style='margin-top:12px;font-size:12px;color:#AAB4C5;'>"
                f"<div style='display:flex;justify-content:space-between;margin-bottom:4px;'>"
                f"<span>P(Humorous)</span><span style='color:#F4F6FA;font-weight:500;'>{h_pct}%</span>"
                f"</div>"
                f"<div style='background:#1E2438;border-radius:4px;height:6px;margin-bottom:8px;'>"
                f"<div style='background:#22C55E;border-radius:4px;height:6px;width:{h_pct}%;'></div>"
                f"</div>"
                f"<div style='display:flex;justify-content:space-between;margin-bottom:4px;'>"
                f"<span>P(Non-Humorous)</span><span style='color:#F4F6FA;font-weight:500;'>{n_pct}%</span>"
                f"</div>"
                f"<div style='background:#1E2438;border-radius:4px;height:6px;'>"
                f"<div style='background:#EF4444;border-radius:4px;height:6px;width:{n_pct}%;'></div>"
                f"</div>"
                f"<div style='margin-top:6px;font-size:11px;color:#6B7A90;font-style:italic;'>"
                f"Derived from model logits (token-level probability)"
                f"</div>"
                f"</div>"
            )

        return html

    except (ValueError, TypeError):
        return f"<span style='color:#AAB4C5;font-size:14px;'>{conf_val if conf_val else 'Unavailable'}</span>"


def parse_vlm_output(result_dict):
    """
    Parses the structured VLM result dict into UI display values.

    Returns a 6-tuple:
        (prediction, confidence_html, detected_text, cultural_category,
         cultural_dependency, reasoning)
    """
    if not isinstance(result_dict, dict):
        return "Error", "N/A", "N/A", "N/A", "N/A", "Invalid response format from model."

    if result_dict.get("error"):
        return "Error", "N/A", "N/A", "N/A", "N/A", result_dict.get("raw_response", "Unknown Error")

    # ── Prediction ──
    humorous_val = result_dict.get("humorous")
    if humorous_val is True or humorous_val == 1 or str(humorous_val).lower() == "true":
        humorous = "Humorous"
    elif humorous_val is False or humorous_val == 0 or str(humorous_val).lower() == "false":
        humorous = "Not Humorous"
    else:
        humorous = str(humorous_val) if humorous_val is not None else "Unknown"

    # ── Confidence (with probability breakdown) ──
    conf_val = result_dict.get("confidence")
    humor_prob = result_dict.get("humor_probability")
    non_humor_prob = result_dict.get("non_humor_probability")

    if conf_val is not None:
        conf_html = format_confidence(conf_val, humor_prob, non_humor_prob)
    else:
        conf_html = (
            "<span style='color:#F59E0B;font-size:14px;'>"
            "Confidence unavailable (logit extraction failed)"
            "</span>"
        )

    # ── OCR / Detected Text ──
    detected_text = str(result_dict.get("detected_text", ""))
    if not detected_text.strip() or detected_text.lower() in ("none", "null"):
        detected_text = "No text detected"

    # ── Cultural Category ──
    category = str(result_dict.get("cultural_category", "Not analyzed"))
    if category.lower() in ("none", "null", ""):
        category = "No specific cultural category detected"
    else:
        # Title-case but preserve multi-word categories
        category = category.strip().title()

    # ── Cultural Dependency ──
    dependency = str(result_dict.get("cultural_dependency", "Not analyzed"))
    if dependency.lower() in ("null", ""):
        dependency = "Not analyzed"
    else:
        dependency = dependency.strip().title()

    # ── Reasoning ──
    reason = str(result_dict.get("reason", "No reasoning provided."))
    if reason.lower() in ("none", "null", ""):
        reason = "No reasoning provided."

    return humorous, conf_html, detected_text, category, dependency, reason

