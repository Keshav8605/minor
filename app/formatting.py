import json

def format_confidence(conf_val):
    try:
        conf = float(conf_val)
        percent = int(conf * 100)
        if conf >= 0.8:
            return f"<span style='background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); padding: 6px 14px; border-radius: 20px; font-weight: 600; display: inline-block;'>{percent}% High Confidence</span>"
        elif conf >= 0.5:
            return f"<span style='background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); padding: 6px 14px; border-radius: 20px; font-weight: 600; display: inline-block;'>{percent}% Medium Confidence</span>"
        else:
            return f"<span style='background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); padding: 6px 14px; border-radius: 20px; font-weight: 600; display: inline-block;'>{percent}% Low Confidence</span>"
    except (ValueError, TypeError):
        return f"<span style='background: rgba(156, 163, 175, 0.15); color: #9ca3af; padding: 6px 14px; border-radius: 20px; display: inline-block;'>{conf_val}</span>"

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
    
    # Extract cultural context dynamically
    cultural_context = result_dict.get("cultural_context")
    if not cultural_context or str(cultural_context).lower() == "none":
        category = result_dict.get("cultural_category")
        dependency = result_dict.get("cultural_dependency")
        parts = []
        if category and str(category).lower() != "none":
            parts.append(f"Category: {str(category).title()}")
        if dependency and str(dependency).lower() != "none":
            parts.append(f"Relevance: {str(dependency).title()}")
        cultural_context = " | ".join(parts) if parts else "No cultural context detected"
    else:
        cultural_context = str(cultural_context)
        
    reason = str(result_dict.get("reason", "No reason provided."))
    detected_text = str(result_dict.get("detected_text", ""))
    if not detected_text.strip() or detected_text.lower() == "none":
        detected_text = "No text detected"
        
    # The fifth value (previously dependency) is kept empty to allow unified card display
    return humorous, conf_html, detected_text, cultural_context, "", reason
