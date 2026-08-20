import json

def format_confidence(conf_val):
    try:
        conf = float(conf_val)
        if conf >= 0.8:
            return f"<span style='color: green; font-weight: bold;'>{conf:.2f} (High)</span>"
        elif conf >= 0.5:
            return f"<span style='color: orange; font-weight: bold;'>{conf:.2f} (Medium)</span>"
        else:
            return f"<span style='color: red; font-weight: bold;'>{conf:.2f} (Low)</span>"
    except (ValueError, TypeError):
        return f"<span>{conf_val}</span>"

def parse_vlm_output(result_dict):
    if not isinstance(result_dict, dict):
        return "Error", "N/A", "N/A", "N/A", "N/A", "Invalid response format from model."
        
    if result_dict.get("error"):
        return "Error", "N/A", "N/A", "N/A", "N/A", result_dict.get("raw_response", "Unknown Error")
        
    humorous = "Yes (Humorous)" if result_dict.get("humorous") else "No (Not Humorous)"
    conf_html = format_confidence(result_dict.get("confidence", "N/A"))
    
    category = str(result_dict.get("cultural_category", "None")).title()
    dependency = str(result_dict.get("cultural_dependency", "None")).title()
    reason = str(result_dict.get("reason", "No reason provided."))
    
    detected_text = str(result_dict.get("detected_text", "N/A"))
    
    return humorous, conf_html, detected_text, category, dependency, reason
