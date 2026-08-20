def build_prompt(image_path: str, ocr_text: str = "", retrieved_context: str = "", mode: str = "A"):
    """
    Builds the structured prompt for cultural awareness.
    Mode A: Plain VLM.
    Mode B: Culturally contextualized VLM.
    """
    system_prompt = (
        "You are an expert AI assistant specialized in analyzing multimodal humor in "
        "codemixed Hindi-English memes. Output your analysis ONLY as a valid JSON object."
    )
    
    user_content = "Analyze the following meme.\n"
    if ocr_text:
        user_content += f"Detected Text (OCR): {ocr_text}\n"
        
    if mode == "B" and retrieved_context:
        user_content += f"\n{retrieved_context}\n\nUse this external context to inform your analysis if relevant.\n"
        
    user_content += (
        "\nReturn a JSON object strictly matching this structure:\n"
        "{\n"
        '  "humorous": bool,\n'
        '  "confidence": float,\n'
        '  "cultural_category": "string (e.g., family, education, none)",\n'
        '  "cultural_dependency": "none, low, medium, or high",\n'
        '  "cultural_context_used": bool,\n'
        '  "reason": "explanation of humor"\n'
        "}\n\n"
        "Output ONLY the JSON object."
    )
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": user_content},
            ],
        }
    ]
    return messages
