"""
Formatting utilities for the Gradio UI.

Converts structured backend result dicts into UI-ready display values.
Handles confidence visualization with SVG progress ring.
"""

import json
import math


def format_confidence(conf_val, humor_prob=None, non_humor_prob=None):
    """
    Format model-derived probability as HTML with a circular SVG progress ring
    and dual probability breakdown bars.
    Explicitly labeled as 'Model Probability' in accordance with scientific rigor.

    Args:
        conf_val: Confidence / probability value (0-1 float, or string like "N/A").
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
            label = "High Model Probability"
        elif conf >= 0.5:
            color = "#F59E0B"
            label = "Moderate Model Probability"
        else:
            color = "#EF4444"
            label = "Low Model Probability"

        html = (
            f"<div style='display:flex;align-items:center;gap:16px;'>"
            f"<div>"
            f"<div style='font-size:26px;font-weight:700;color:#F4F6FA;line-height:1.2;'>{percent}%</div>"
            f"<div style='font-size:12px;color:{color};font-weight:600;margin-top:2px;'>{label}</div>"
            f"</div>"
            f"<svg width='60' height='60' viewBox='0 0 64 64' style='flex-shrink:0;'>"
            f"<circle cx='32' cy='32' r='28' stroke='#1E2438' stroke-width='5' fill='none'/>"
            f"<circle cx='32' cy='32' r='28' stroke='{color}' stroke-width='5' fill='none' "
            f"stroke-dasharray='{circumference:.1f}' stroke-dashoffset='{offset:.1f}' "
            f"transform='rotate(-90 32 32)' stroke-linecap='round'/>"
            f"<text x='32' y='36' text-anchor='middle' font-size='12' font-weight='600' "
            f"fill='#F4F6FA' font-family='Inter,sans-serif'>{percent}%</text>"
            f"</svg>"
            f"</div>"
        )

        # Add probability breakdown if available
        if humor_prob is not None and non_humor_prob is not None:
            h_pct = round(humor_prob * 100, 1)
            n_pct = round(non_humor_prob * 100, 1)
            html += (
                f"<div style='margin-top:10px;font-size:12px;color:#AAB4C5;'>"
                f"<div style='display:flex;justify-content:space-between;margin-bottom:3px;'>"
                f"<span>P(Humorous)</span><span style='color:#22C55E;font-weight:600;'>{h_pct}%</span>"
                f"</div>"
                f"<div style='background:#1E2438;border-radius:4px;height:5px;margin-bottom:6px;'>"
                f"<div style='background:#22C55E;border-radius:4px;height:5px;width:{h_pct}%;'></div>"
                f"</div>"
                f"<div style='display:flex;justify-content:space-between;margin-bottom:3px;'>"
                f"<span>P(Non-Humorous)</span><span style='color:#EF4444;font-weight:600;'>{n_pct}%</span>"
                f"</div>"
                f"<div style='background:#1E2438;border-radius:4px;height:5px;'>"
                f"<div style='background:#EF4444;border-radius:4px;height:5px;width:{n_pct}%;'></div>"
                f"</div>"
                f"<div style='margin-top:6px;font-size:11px;color:#6B7A90;font-style:italic;'>"
                f"Model Probability (derived from token logits, uncalibrated)"
                f"</div>"
                f"</div>"
            )

        return html

    except (ValueError, TypeError):
        return f"<span style='color:#AAB4C5;font-size:14px;'>{conf_val if conf_val else 'Unavailable'}</span>"


def parse_vlm_output(result_dict, include_context=False):
    """
    Parses the structured VLM result dict into UI display values.

    Returns a 6-tuple by default (backwards-compatible with existing tests):
        (prediction, prob_html, detected_text, cultural_category,
         cultural_dependency, reasoning)

    Returns a 7-tuple when include_context=True:
        (prediction, prob_html, detected_text, cultural_category,
         cultural_dependency, cultural_context, reasoning)
    """
    if not isinstance(result_dict, dict):
        err_msg = "Invalid response format from model."
        if include_context:
            return "Error", "N/A", "N/A", "N/A", "N/A", "N/A", err_msg
        return "Error", "N/A", "N/A", "N/A", "N/A", err_msg

    # Handle explicit error status without leaking raw JSON or traceback
    if result_dict.get("error"):
        pred = result_dict.get("prediction", "Error")
        if pred not in ("Error", "Unavailable"):
            pred = "Error"
        prob_html = "N/A"
        dt = str(result_dict.get("detected_text", "N/A"))
        cat = str(result_dict.get("cultural_category", "N/A"))
        dep = str(result_dict.get("cultural_dependency", "N/A"))
        ctx = str(result_dict.get("cultural_context", "N/A"))
        raw_resp = result_dict.get("raw_response", "Analysis could not be completed.")
        reason = str(result_dict.get("reasoning") or result_dict.get("reason") or raw_resp)

        # Ensure raw JSON or error objects never leak into reason
        if reason.strip().startswith("{") and ('"humorous"' in reason or '"reason"' in reason or '"error"' in reason):
            reason = "Analysis could not be completed."

        if include_context:
            return pred, prob_html, dt, cat, dep, ctx, reason
        return pred, prob_html, dt, cat, dep, reason

    # ── Prediction ──
    humorous_val = result_dict.get("humorous")
    if humorous_val is True or humorous_val == 1 or str(humorous_val).lower() == "true":
        humorous = "Humorous"
    elif humorous_val is False or humorous_val == 0 or str(humorous_val).lower() == "false":
        humorous = "Not Humorous"
    else:
        pred_field = result_dict.get("prediction")
        if pred_field:
            humorous = str(pred_field).strip()
        else:
            humorous = "Unavailable"

    # ── Probability / Confidence ──
    conf_val = result_dict.get("confidence")
    humor_prob = result_dict.get("humor_probability")
    non_humor_prob = result_dict.get("non_humor_probability")

    if conf_val is not None:
        conf_html = format_confidence(conf_val, humor_prob, non_humor_prob)
    else:
        conf_html = (
            "<span style='color:#F59E0B;font-size:14px;'>"
            "Model probability unavailable"
            "</span>"
        )

    # ── OCR / Detected Text ──
    detected_text = str(result_dict.get("detected_text", ""))
    if not detected_text.strip() or detected_text.lower() in ("none", "null", "no text detected in image"):
        detected_text = "No text detected in image"

    # ── Cultural Category ──
    category = str(result_dict.get("cultural_category", ""))
    if not category.strip() or category.lower() in ("none", "null", "no specific cultural category detected"):
        category = "No specific cultural category detected"
    else:
        category = category.strip().title()

    # ── Cultural Dependency ──
    dependency = str(result_dict.get("cultural_dependency", ""))
    if not dependency.strip() or dependency.lower() in ("none", "null", "not analyzed"):
        dependency = "Not analyzed"
    else:
        dep_lower = dependency.lower()
        if "high" in dep_lower:
            dependency = "High"
        elif "med" in dep_lower:
            dependency = "Medium"
        elif "low" in dep_lower:
            dependency = "Low"
        else:
            dependency = dependency.strip().title()

    # ── Cultural Context ──
    cultural_context = str(
        result_dict.get("cultural_context") or
        result_dict.get("cultural_context_used") or
        ""
    ).strip()
    if not cultural_context or cultural_context.lower() in ("none", "null", "true", "false", ""):
        cultural_context = "No significant cultural context detected"

    # ── Reasoning ──
    reason = str(
        result_dict.get("reasoning") or
        result_dict.get("reason") or
        "No reasoning provided."
    ).strip()

    # Absolute safeguard against raw JSON leaking into reasoning
    if reason.startswith("{") and ('"humorous"' in reason or '"reason"' in reason or '"detected_text"' in reason):
        try:
            parsed_inner = json.loads(reason, strict=False)
            if isinstance(parsed_inner, dict):
                inner_reason = parsed_inner.get("reasoning") or parsed_inner.get("reason")
                if inner_reason:
                    reason = str(inner_reason).strip()
                if cultural_context == "No significant cultural context detected" and parsed_inner.get("cultural_context"):
                    cultural_context = str(parsed_inner["cultural_context"]).strip()
                if category == "No specific cultural category detected" and parsed_inner.get("cultural_category"):
                    category = str(parsed_inner["cultural_category"]).strip()
        except Exception:
            pass

    if not reason or reason.lower() in ("none", "null"):
        reason = "No reasoning provided."

    if include_context:
        return humorous, conf_html, detected_text, category, dependency, cultural_context, reason
    return humorous, conf_html, detected_text, category, dependency, reason


def format_multi_meme_cards(multi_result: dict, include_context: bool = True) -> str:
    """
    Renders structured, visually separated result cards for multiple uploaded memes.
    Every meme is rendered with its own independent section, preserving order
    and preventing cross-meme data mixing.

    Args:
        multi_result: Dict containing a 'memes' list of per-meme analysis dicts.
        include_context: Whether to include cultural context cards.

    Returns:
        HTML string containing cleanly separated, numbered card sections.
    """
    if not isinstance(multi_result, dict):
        return "<div style='color:#EF4444;padding:16px;'>Invalid multi-meme response format.</div>"

    memes = multi_result.get("memes", [])
    if not memes:
        return "<div style='color:#AAB4C5;padding:16px;'>No meme analyses available.</div>"

    sections_html = []
    total_memes = len(memes)

    for idx, meme in enumerate(memes):
        meme_num = meme.get("meme_number", idx + 1)
        img_p = meme.get("image_path", "")
        import os
        img_name = os.path.basename(img_p) if img_p else f"Image {meme_num}"

        # ── Prediction ──
        pred = meme.get("prediction", "Not Humorous")
        is_humorous = (pred == "Humorous") or meme.get("humorous", False)
        pred_color = "#34D399" if is_humorous else "#9CA3AF"
        pred_bg = "#064E3B" if is_humorous else "#1F2937"
        pred_border = "#059669" if is_humorous else "#374151"

        # ── Model Probability (Derived from Logits) ──
        h_prob = meme.get("humor_probability")
        nh_prob = meme.get("non_humor_probability")
        if h_prob is not None and nh_prob is not None:
            h_pct = round(float(h_prob) * 100, 1)
            nh_pct = round(float(nh_prob) * 100, 1)
            prob_html = f"""
            <div style="font-size:13px;color:#AAB4C5;">
              <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                <span>P(Humorous)</span><span style="color:#22C55E;font-weight:600;">{h_pct}%</span>
              </div>
              <div style="background:#1E2438;border-radius:4px;height:6px;margin-bottom:8px;overflow:hidden;">
                <div style="background:#22C55E;height:100%;width:{h_pct}%;"></div>
              </div>
              <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                <span>P(Non-Humorous)</span><span style="color:#EF4444;font-weight:600;">{nh_pct}%</span>
              </div>
              <div style="background:#1E2438;border-radius:4px;height:6px;overflow:hidden;">
                <div style="background:#EF4444;height:100%;width:{nh_pct}%;"></div>
              </div>
              <div style="margin-top:6px;font-size:11px;color:#6B7A90;font-style:italic;">
                Model Probability (derived from token logits, uncalibrated)
              </div>
            </div>
            """
        else:
            prob_html = "<span style='color:#F59E0B;font-size:13px;'>Probability unavailable</span>"

        # ── Text / Category / Dependency / Context / Reasoning ──
        dt = str(meme.get("detected_text", "No text detected in image")).strip()
        cat = str(meme.get("cultural_category", "No specific cultural category detected")).strip()
        dep = str(meme.get("cultural_dependency", "Not analyzed")).strip()
        ctx = str(meme.get("cultural_context", "No significant cultural context detected")).strip()
        reason = str(meme.get("reasoning", "No reasoning provided.")).strip()

        # Sanitize reasoning
        if reason.startswith("{") and ('"humorous"' in reason or '"reason"' in reason):
            reason = "Reasoning details could not be parsed."

        # Card container styles
        card_style = "background:#111625;border:1px solid #1E2640;border-radius:8px;padding:12px 14px;margin-bottom:10px;"
        label_style = "font-size:11px;font-weight:700;color:#94A3B8;letter-spacing:0.5px;margin-bottom:6px;display:flex;align-items:center;gap:6px;"
        val_style = "color:#F1F5F9;font-size:13px;line-height:1.45;"

        # Build meme block HTML
        meme_block = f"""
        <div class="multi-meme-section" style="margin-bottom: 24px;">
          <!-- Header for this specific meme -->
          <div style="display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #1E2640;padding-bottom:10px;margin-bottom:14px;">
            <div style="font-size:16px;font-weight:800;color:#38BDF8;letter-spacing:0.8px;">
              MEME {meme_num} ANALYSIS
            </div>
            <div style="font-size:12px;color:#64748B;background:#1E2640;padding:3px 10px;border-radius:12px;font-weight:500;">
              Image {meme_num}: {img_name}
            </div>
          </div>

          <!-- Row 1: Prediction & Probability -->
          <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(240px, 1fr));gap:12px;margin-bottom:10px;">
            <div style="{card_style}">
              <div style="{label_style}">🎯 HUMOR PREDICTION</div>
              <div style="display:inline-block;padding:6px 14px;border-radius:6px;background:{pred_bg};border:1px solid {pred_border};color:{pred_color};font-size:14px;font-weight:700;">
                {pred}
              </div>
            </div>
            <div style="{card_style}">
              <div style="{label_style}">◉ MODEL PROBABILITY</div>
              {prob_html}
            </div>
          </div>

          <!-- Row 2: Detected Text (OCR) -->
          <div style="{card_style}">
            <div style="{label_style}">📄 DETECTED TEXT (OCR)</div>
            <div style="{val_style}word-break:break-word;">{dt}</div>
          </div>

          <!-- Row 3: Category & Dependency -->
          <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(240px, 1fr));gap:12px;margin-bottom:10px;">
            <div style="{card_style}">
              <div style="{label_style}">🏷️ CULTURAL CATEGORY</div>
              <div style="{val_style}">{cat}</div>
            </div>
            <div style="{card_style}">
              <div style="{label_style}">🔗 CULTURAL DEPENDENCY</div>
              <div style="{val_style}">{dep}</div>
            </div>
          </div>

          <!-- Row 4: Cultural Context & AI Reasoning -->
          <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(240px, 1fr));gap:12px;">
            <div style="{card_style}">
              <div style="{label_style}">🌐 CULTURAL CONTEXT</div>
              <div style="{val_style}color:#CBD5E1;">{ctx}</div>
            </div>
            <div style="{card_style}">
              <div style="{label_style}">🧠 AI REASONING</div>
              <div style="{val_style}color:#E2E8F0;line-height:1.5;">{reason}</div>
            </div>
          </div>
        </div>
        """
        sections_html.append(meme_block)

        # Append separator if not last meme
        if idx < total_memes - 1:
            separator_html = """
            <div class="meme-separator" style="margin: 28px 0; height: 1px; background: linear-gradient(90deg, transparent, #38BDF8 30%, #818CF8 70%, transparent); opacity: 0.5;"></div>
            """
            sections_html.append(separator_html)

    return "\n".join(sections_html)


