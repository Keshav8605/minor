def build_prompt(image_path: str, ocr_text: str = "", retrieved_context: str = "", mode: str = "A"):
    """
    Builds the structured prompt for cultural awareness.
    Mode A: Plain VLM.
    Mode B: Culturally contextualized VLM.
    """
    system_prompt = (
        "You are a strict binary humor classifier for memes. "
        "Output your analysis ONLY as a valid JSON object. "
        "Your DEFAULT classification is NOT HUMOROUS. "
        "Only classify as humorous when you find clear, specific evidence of an actual joke or comedic mechanism. "
        "Cultural context helps UNDERSTAND memes, but cultural relevance does NOT equal humor."
    )

    user_content = "Analyze the following meme.\n"
    if ocr_text:
        user_content += f"Detected Text (OCR): {ocr_text}\n"

    if mode == "B" and retrieved_context:
        user_content += (
            f"\n{retrieved_context}\n\n"
            "WARNING: This cultural context is for UNDERSTANDING the meme, NOT evidence of humor. "
            "A cultural reference does NOT make a meme humorous. "
            "Use this context only if it helps explain an actual joke.\n"
        )

    user_content += (
        "\nCRITICAL: Before setting humorous=true, answer: 'What exactly is the joke?' "
        "If you cannot state the joke in one sentence, set humorous=false.\n\n"
        "Return a JSON object strictly matching this structure:\n"
        "{\n"
        '  "detected_text": "string (transcription of text found in meme)",\n'
        '  "cultural_category": "string (e.g., family, education, cricket, bollywood, none)",\n'
        '  "cultural_dependency": "none, low, medium, or high — high ONLY if the joke itself requires cultural knowledge",\n'
        '  "cultural_context_used": bool,\n'
        '  "humor_evidence": "Name the SPECIFIC comedic mechanism and cite evidence. If none found, write No clear comedic mechanism found.",\n'
        '  "non_humor_evidence": "Is this motivational, informational, serious, emotional? List indicators.",\n'
        '  "reason": "What exactly is the joke? If you cannot state it, classify as NOT humorous.",\n'
        '  "humorous": bool\n'
        "}\n\n"
        "Output ONLY the JSON object. Do not include markdown codeblocks or other text."
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

