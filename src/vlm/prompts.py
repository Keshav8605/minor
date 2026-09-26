"""
Prompt builders for Qwen2.5-VL humor analysis.

IMPORTANT: Confidence/probability is NOT requested from the VLM.
The model is only asked to classify (humorous: true/false) and reason.
Actual probability is derived from token logits in inference.py.
"""

# ─── JSON schema the VLM must return ───────────────────────────────────────────
# NOTE: "confidence" is intentionally ABSENT. We extract it from logits instead.

_GENERAL_JSON_SCHEMA = """{
  "detected_text": "all text visible in the image (Hindi, English, Hinglish)",
  "visual_description": "brief description of what the image shows",
  "humor_evidence": "What SPECIFIC comedic mechanism exists? Name it (sarcasm, punchline, absurdity, irony, wordplay, comedic contradiction, etc.) and cite the exact image/text element that creates it. If you cannot name a specific mechanism with specific evidence, write 'No clear comedic mechanism found'. Do NOT use the words 'true' or 'false'.",
  "non_humor_evidence": "Is this content motivational, informational, serious, emotional, educational, or a straightforward statement? Does the image simply reinforce the text's message without comedic contrast? List all non-humor indicators. Do NOT use the words 'true' or 'false'.",
  "reason": "Final decision (2-5 sentences). Answer: What exactly is the joke? If you cannot state what the joke is in one sentence, classify as NOT humorous. A visual-text relationship alone is NOT humor — explain WHY the relationship is FUNNY, not just that it exists. Do NOT use 'true' or 'false'.",
  "humorous": bool
}"""

_CULTURAL_JSON_SCHEMA = """{
  "detected_text": "all text visible in the image (Hindi, English, Hinglish)",
  "visual_description": "brief description of what the image shows",
  "cultural_category": "one of: family, education, college, JEE/exams, cricket, Bollywood, food, festivals, marriage/wedding, relationships, workplace, social_norms, religion, regional_culture, daily_life, hindi_slang, internet_culture, or none",
  "cultural_dependency": "low if no cultural knowledge needed, medium if some helps, high ONLY if the joke itself depends on cultural knowledge. Cultural setting alone does NOT make dependency high. Do NOT use 'true' or 'false'.",
  "cultural_context": "explain what cultural knowledge is needed to understand this meme, or 'No significant cultural context' if none. Do NOT use 'true' or 'false'.",
  "humor_evidence": "What SPECIFIC comedic mechanism exists? Name it and cite the exact evidence. A cultural reference is NOT automatically humor evidence. If you cannot name a specific joke or comedic mechanism, write 'No clear comedic mechanism found'. Do NOT use 'true' or 'false'.",
  "non_humor_evidence": "Is this content motivational, informational, serious, emotional, educational, or a straightforward cultural statement? Does the image reinforce the text without comedic contrast? List all non-humor indicators. Do NOT use 'true' or 'false'.",
  "reason": "Final decision (2-5 sentences). Answer: What exactly is the joke? If you cannot state the joke in one sentence, classify as NOT humorous. Cultural relevance is NOT humor. A visual-text relationship is NOT automatically funny. Do NOT use 'true' or 'false'.",
  "humorous": bool
}"""


def build_humor_analysis_prompt(image_path: str):
    """
    Builds the GENERAL mode prompt for Qwen2.5-VL humor analysis.
    Does NOT inject cultural context. Does NOT ask for self-reported confidence.
    """
    system_prompt = (
        "You are a strict binary humor classifier for memes. "
        "You read Hindi, Hinglish, and English. Output ONLY valid JSON. "
        "Your DEFAULT classification is NOT HUMOROUS. "
        "Only classify as humorous when you find clear, specific evidence of an actual joke or comedic mechanism. "
        "Most memes you analyze will NOT be humorous — many are motivational, informational, emotional, or simply relatable without being funny. "
        "A visual-text relationship, cultural reference, meme format, or relatable situation is NOT sufficient evidence for humor."
    )

    user_content = (
        "Analyze the following meme image.\n\n"
        "Instructions:\n"
        "1. Read ALL text visible in the image carefully (Hindi, English, Hinglish).\n"
        "2. Briefly describe the visual content.\n"
        "3. Classify whether the meme is humorous (true) or not humorous (false).\n\n"
        "CRITICAL CLASSIFICATION RULES:\n"
        "- Humorous REQUIRES: a joke, punchline, comedic twist, irony, absurdity, sarcasm, wordplay, or exaggeration that is INTENDED TO MAKE PEOPLE LAUGH.\n"
        "- NOT humorous: motivational quotes, inspirational messages, serious statements, emotional content, informational content, educational messages, cultural greetings, life advice, awareness messages, straightforward observations.\n"
        "- A visual-text relationship is NOT automatically humor. Ask: WHY is this relationship FUNNY? If you cannot answer, it is NOT humor.\n"
        "- 'Juxtaposition' is NOT a comedic mechanism by itself. A motivational quote paired with a matching scenic image is thematic consistency, NOT humor.\n"
        "- Relatability is NOT humor. A relatable situation without a comedic twist is NOT humorous.\n"
        "- A meme format does NOT guarantee humor.\n"
        "- If the text is a straightforward statement and the image reinforces that statement, it is NOT humor.\n\n"
        "DECISION TEST — before setting humorous=true, answer this question:\n"
        "'What exactly is the joke?'\n"
        "If you cannot state the joke in one clear sentence referencing specific image/text elements, set humorous=false.\n\n"
        "Return a JSON object strictly matching this structure:\n"
        f"{_GENERAL_JSON_SCHEMA}\n\n"
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


def build_cultural_analysis_prompt(image_path: str, ocr_text: str = "", retrieved_context: str = ""):
    """
    Builds the CULTURAL-AWARE mode prompt for Qwen2.5-VL.
    Injects OCR text and retrieved cultural knowledge to guide reasoning.
    Does NOT ask for self-reported confidence.
    """
    system_prompt = (
        "You are a strict binary humor classifier for Indian memes with cultural expertise. "
        "You read Hindi/Hinglish/English. Output ONLY valid JSON. "
        "Your DEFAULT classification is NOT HUMOROUS. "
        "Only classify as humorous when you find clear, specific evidence of an actual joke or comedic mechanism. "
        "Cultural context helps you UNDERSTAND memes, but cultural relevance does NOT equal humor. "
        "Most memes you analyze will NOT be humorous."
    )

    user_content = "Analyze the following meme image.\n\n"

    if ocr_text and ocr_text.strip():
        user_content += f"Previously detected text (OCR): {ocr_text}\n\n"

    if retrieved_context and retrieved_context.strip():
        user_content += (
            f"{retrieved_context}\n\n"
            "WARNING: This cultural context is for UNDERSTANDING the meme, NOT evidence of humor. "
            "A cultural reference does NOT make a meme humorous. "
            "Use this context only if it helps explain an actual joke. Do not invent unsupported facts.\n\n"
        )

    user_content += (
        "Instructions:\n"
        "1. Read all text & describe visuals.\n"
        "2. Identify Indian cultural references & assess cultural dependency.\n"
        "   - cultural_dependency should be 'low' unless the JOKE ITSELF requires cultural knowledge.\n"
        "   - An Indian setting alone does NOT make cultural dependency 'high'.\n"
        "3. Classify if humorous (true) or not (false).\n\n"
        "CRITICAL CLASSIFICATION RULES:\n"
        "   - Humor REQUIRES a joke, punchline, comedic twist, sarcasm, irony, absurdity, or exaggeration intended to cause laughter.\n"
        "   - NOT humorous: motivational quotes, inspirational messages, serious statements, emotional content, informational content, educational messages, cultural greetings, life advice.\n"
        "   - A cultural reference alone does NOT imply humor.\n"
        "   - A visual-text relationship is NOT automatically humor. Ask: WHY is this FUNNY?\n"
        "   - 'Juxtaposition' without a comedic effect is NOT humor. A motivational quote with a matching image is thematic consistency.\n"
        "   - Relatability is NOT humor unless there is a clear comedic mechanism.\n\n"
        "DECISION TEST — before setting humorous=true:\n"
        "'What exactly is the joke?'\n"
        "If you cannot state the joke in one sentence, set humorous=false.\n\n"
        "Return a JSON object strictly matching this structure:\n"
        f"{_CULTURAL_JSON_SCHEMA}\n\n"
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


def build_multi_image_prompt(image_paths: list, mode: str = "general", ocr_text: str = "", retrieved_context: str = ""):
    """
    Builds a multi-image prompt for Qwen2.5-VL.
    All images are included in a single conversation turn.
    """
    if mode == "cultural":
        system_prompt = (
            "You are an expert AI assistant specialized in analyzing culturally-grounded humor "
            "in Indian memes. You have deep knowledge of Indian culture. "
            "You can read Hindi (Devanagari), Hinglish (code-mixed Hindi-English), and English text. "
            "You must output your analysis ONLY as a valid JSON object. No markdown formatting."
        )
        json_schema = _CULTURAL_JSON_SCHEMA
    else:
        system_prompt = (
            "You are an expert AI assistant specialized in analyzing humor in memes. "
            "You can read Hindi (Devanagari), Hinglish (code-mixed Hindi-English), and English text. "
            "You must output your analysis ONLY as a valid JSON object. No markdown formatting."
        )
        json_schema = _GENERAL_JSON_SCHEMA

    # Build user content with multiple images
    user_content_parts = []
    for i, img_path in enumerate(image_paths):
        user_content_parts.append({"type": "image", "image": f"file://{img_path}"})

    text_content = (
        f"Analyze the above {len(image_paths)} meme images together as a combined set.\n"
        "They may represent different parts of one meme, multiple panels, or related memes.\n\n"
    )

    if ocr_text and ocr_text.strip():
        text_content += f"Previously detected text (OCR): {ocr_text}\n\n"

    if mode == "cultural" and retrieved_context and retrieved_context.strip():
        text_content += (
            f"{retrieved_context}\n\n"
            "Use the above cultural context only if relevant.\n\n"
        )

    text_content += (
        "Provide a COMBINED analysis considering all images together.\n"
        f"Return a JSON object strictly matching this structure:\n{json_schema}\n\n"
        "Output ONLY the JSON object."
    )

    user_content_parts.append({"type": "text", "text": text_content})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content_parts}
    ]
    return messages
