def build_humor_analysis_prompt(image_path: str):
    """
    Builds the structured prompt for Qwen2.5-VL to analyze humor in a meme.
    The response format must be strictly requested as JSON.
    """
    system_prompt = (
        "You are an expert AI assistant specialized in analyzing culturally aware, "
        "multimodal humor in codemixed Hindi-English memes. You must output your "
        "analysis ONLY as a valid JSON object without any additional conversational text or markdown formatting outside the JSON block."
    )
    
    user_content = (
        "Analyze the following meme. Pay close attention to:\n"
        "1. Visual content (what is happening in the image, facial expressions, actions).\n"
        "2. Visible text (read any text in the image).\n"
        "3. Hindi/Devanagari text and Hinglish/code-mixed text.\n"
        "4. The relationship between the visual content and the text.\n"
        "5. The cultural context or implicit knowledge required to understand it.\n"
        "6. Whether the meme is humorous and why.\n\n"
        "Return a JSON object strictly matching this structure:\n"
        "{\n"
        '  "humorous": bool,\n'
        '  "confidence": float,\n'
        '  "detected_text": "text found in image",\n'
        '  "visual_description": "description of the image",\n'
        '  "cultural_context": "explanation of cultural references",\n'
        '  "reason": "why it is or is not funny"\n'
        "}\n\n"
        "Output ONLY the JSON object."
    )
    
    # Qwen2.5-VL format
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{image_path}",
                },
                {"type": "text", "text": user_content},
            ],
        }
    ]
    return messages
