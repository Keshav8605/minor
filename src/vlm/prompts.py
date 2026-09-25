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
  "reason": "Explain WHY the meme works as humor (2-5 sentences). Identify the specific humor mechanism (e.g. sarcasm, exaggeration, relatable situation, visual-text mismatch) and explain the relationship between the image and text. If not humorous, explain why it lacks a comedic mechanism. Do not just describe the image."
}"""

_CULTURAL_JSON_SCHEMA = """{
  "humorous": bool,
  "detected_text": "all text visible in the image (Hindi, English, Hinglish)",
  "visual_description": "brief description of what the image shows",
  "cultural_category": "one of: family, education, college, JEE/exams, cricket, Bollywood, food, festivals, marriage/wedding, relationships, workplace, social_norms, religion, regional_culture, daily_life, hindi_slang, internet_culture, or none",
  "cultural_dependency": "low, medium, or high",
  "cultural_context": "explain what cultural knowledge is needed to understand this meme, or 'No significant cultural context' if none",
  "reason": "Explain WHY the meme works as humor (2-5 sentences). Identify the specific humor mechanism (e.g. sarcasm, exaggeration, relatable situation) and explain how the visual, text, and cultural context interact to create the joke. If not humorous, explain why. Do not just describe the image."
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
        "Do not include any text outside the JSON object. No markdown formatting. "
        "Keep your output concise."
    )

    user_content = (
        "Analyze the following meme image.\n\n"
        "Instructions:\n"
        "1. Read ALL text visible in the image carefully (Hindi, English, Hinglish).\n"
        "2. Briefly describe the visual content.\n"
        "3. Classify whether the meme is humorous (true) or not humorous (false).\n\n"
        "Classification criteria — carefully distinguish between:\n"
        "- Humorous: contains a joke, punchline, comedic twist, irony, absurdity, or exaggeration intended to make people laugh.\n"
        "- Sarcastic/satirical: uses irony or mockery — may or may not be humorous depending on comedic intent.\n"
        "- Cultural reference: references culture, traditions, or social norms — a cultural reference alone does NOT make something humorous.\n"
        "- Observational: makes a relatable observation — relatability alone is NOT humor unless there is a comedic element.\n"
        "- Informational/serious: conveys information, opinions, or emotional messages without comedic intent — classify as NOT humorous.\n"
        "- Emotional: expresses frustration, nostalgia, or sentiment — emotional content is NOT automatically humorous.\n\n"
        "An image being a 'meme format' does NOT automatically make it humorous. Evaluate actual comedic intent.\n\n"
        "4. Explain your reasoning (2-5 sentences). Focus on WHY the meme works as humor. Identify the specific humor mechanism (sarcasm, exaggeration, relatable situation, etc.) and explain the relationship between the image and text.\n\n"
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
        "You are an expert AI in Indian culture and meme humor analysis. "
        "You read Hindi/Hinglish/English. Output ONLY valid JSON, no markdown or text outside. Keep it concise."
    )

    user_content = "Analyze the following meme image.\n\n"

    if ocr_text and ocr_text.strip():
        user_content += f"Previously detected text (OCR): {ocr_text}\n\n"

    if retrieved_context and retrieved_context.strip():
        user_content += (
            f"{retrieved_context}\n\n"
            "Use this cultural context only if relevant. Do not invent unsupported facts.\n\n"
        )

    user_content += (
        "Instructions:\n"
        "1. Read all text & describe visuals.\n"
        "2. Identify Indian cultural references & assess cultural dependency (low/medium/high).\n"
        "3. Classify if humorous (true) or not (false).\n"
        "   - Humor requires a joke, comedic twist, absurdity, or exaggeration intended to cause laughter.\n"
        "   - Sarcasm, relatable observation, or emotional/informational content alone is NOT humor.\n"
        "   - A cultural reference alone does NOT imply humor.\n"
        "   - Being a 'meme format' does not guarantee humor. Evaluate actual comedic intent.\n"
        "4. Explain reasoning (2-5 sentences). Focus on WHY the meme works as humor. Identify the specific humor mechanism (sarcasm, exaggeration, relatable situation, etc.) and explain how the visual, text, and cultural context interact.\n\n"
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
