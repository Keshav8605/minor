import json
import math

def format_confidence(conf_val):
    """Format confidence as HTML with a circular SVG progress ring."""
    try:
        conf = float(conf_val)
        percent = int(conf * 100)
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

        return (
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
    except (ValueError, TypeError):
        return f"<span style='color:#AAB4C5;font-size:14px;'>{conf_val}</span>"

def parse_vlm_output(result_dict):
    if not isinstance(result_dict, dict):
        return "Error", "N/A", "N/A", "N/A", "N/A", "Invalid response format from model."

    if result_dict.get("error"):
        return "Error", "N/A", "N/A", "N/A", "N/A", result_dict.get("raw_response", "Unknown Error")

    humorous_val = result_dict.get("humorous")
    if humorous_val is True or humorous_val == 1 or str(humorous_val).lower() == "true":
        humorous = "Humorous"
    elif humorous_val is False or humorous_val == 0 or str(humorous_val).lower() == "false":
        humorous = "Not Humorous"
    else:
        humorous = str(humorous_val) if humorous_val is not None else "Unknown"

    conf_html = format_confidence(result_dict.get("confidence", "N/A"))

    # Separate cultural fields for the two-card layout
    category = str(result_dict.get("cultural_category", "None")).title()
    dependency = str(result_dict.get("cultural_dependency", "None")).title()

    reason = str(result_dict.get("reason", "No reason provided."))

    detected_text = str(result_dict.get("detected_text", ""))
    if not detected_text.strip() or detected_text.lower() == "none":
        detected_text = "No text detected"

    return humorous, conf_html, detected_text, category, dependency, reason
