"""
Prompt builders for Qwen2.5-VL humor analysis.

IMPORTANT: Confidence/probability is NOT requested from the VLM.
The model is only asked to classify (humorous: true/false) and reason.
Actual probability is derived from token logits in inference.py.
"""

# ─── JSON schema the VLM must return ───────────────────────────────────────────
# NOTE: "confidence" is intentionally ABSENT. We extract it from logits instead.

_GENERAL_JSON_SCHEMA = """{
  "humorous": bool,
  "detected_text": "all text visible in the image (Hindi, English, Hinglish)",
  "visual_description": "brief description of what the image shows",
  "reason": "why the meme is or is not humorous"
}"""

_CULTURAL_JSON_SCHEMA = """{
  "humorous": bool,
  "detected_text": "all text visible in the image (Hindi, English, Hinglish)",
  "visual_description": "brief description of what the image shows",
  "cultural_category": "one of: family, education, college, JEE/exams, cricket, Bollywood, food, festivals, marriage/wedding, relationships, workplace, social_norms, religion, regional_culture, daily_life, hindi_slang, internet_culture, or none",
  "cultural_dependency": "low, medium, or high",
  "cultural_context": "explain what cultural knowledge is needed to understand this meme, or 'No significant cultural context' if none",
  "reason": "why the meme is or is not humorous, referencing cultural context if relevant"
}"""


def build_humor_analysis_prompt(image_path: str):
    """
    Builds the GENERAL mode prompt for Qwen2.5-VL humor analysis.
    Does NOT inject cultural context. Does NOT ask for self-reported confidence.
    """
    system_prompt = (
        "You are an expert AI assistant specialized in analyzing humor in memes. "
        "You can read Hindi (Devanagari), Hinglish (code-mixed Hindi-English), and English text. "
        "You must output your analysis ONLY as a valid JSON object. "
        "Do not include any text outside the JSON object. No markdown formatting."
    )

    user_content = (
        "Analyze the following meme image.\n\n"
        "Instructions:\n"
        "1. Read ALL text visible in the image carefully (Hindi, English, Hinglish).\n"
        "2. Describe the visual content (people, expressions, scene, objects).\n"
        "3. Determine whether the meme is humorous (true) or not humorous (false).\n"
        "4. Explain your reasoning.\n\n"
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
        "You are an expert AI assistant specialized in analyzing culturally-grounded humor "
        "in Indian memes. You have deep knowledge of Indian culture, society, Bollywood, cricket, "
        "education system, family dynamics, festivals, food, and social norms. "
        "You can read Hindi (Devanagari), Hinglish (code-mixed Hindi-English), and English text. "
        "You must output your analysis ONLY as a valid JSON object. "
        "Do not include any text outside the JSON object. No markdown formatting."
    )

    user_content = "Analyze the following meme image.\n\n"

    if ocr_text and ocr_text.strip():
        user_content += f"Previously detected text (OCR): {ocr_text}\n\n"

    if retrieved_context and retrieved_context.strip():
        user_content += (
            f"{retrieved_context}\n\n"
            "Use the above cultural context to inform your analysis ONLY if it is relevant "
            "to this specific meme. Do not force cultural references that are not present.\n\n"
        )

    user_content += (
        "Instructions:\n"
        "1. Read ALL text visible in the image carefully (Hindi, English, Hinglish).\n"
        "2. Describe the visual content (people, expressions, scene, objects).\n"
        "3. Identify any Indian cultural references (family, education, cricket, Bollywood, festivals, etc.).\n"
        "4. Assess how much the humor depends on understanding Indian culture (low/medium/high).\n"
        "5. Determine whether the meme is humorous (true) or not humorous (false).\n"
        "6. Explain your reasoning, referencing specific cultural context if applicable.\n\n"
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
