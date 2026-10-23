"""
Prompt builders for Qwen2.5-VL humor analysis.

IMPORTANT: Confidence/probability is NOT requested from the VLM.
The model is only asked to classify (humorous: true/false) and reason.
Actual probability is derived from token logits in inference.py.

TWO-STAGE ARCHITECTURE:
- Stage 1 (General Assessment): Determines whether humor exists based on
  image + text + their relationship. No cultural context injected.
- Stage 2 (Cultural-Aware): Uses RAG/DRISHTIKON context to help UNDERSTAND
  humor that was already identified. Cultural context must NOT create humor.
"""

# ─── JSON schema the VLM must return ───────────────────────────────────────────
# NOTE: "confidence" is intentionally ABSENT. We extract it from logits instead.

_GENERAL_JSON_SCHEMA = """{
  "humorous": bool,
  "detected_text": "all text visible in the image (Hindi, English, Hinglish). Write 'No text detected' if none.",
  "visual_description": "brief description (1 sentence)",
  "humor_evidence": "Specific comedic mechanism (sarcasm, punchline, absurdity, irony, wordplay, visual comedy, exaggeration, comedic contradiction) and cite the exact element. If none, write 'No clear comedic mechanism found'.",
  "non_humor_evidence": "List all non-humor indicators (motivational, informational, serious, emotional, educational, religious, straightforward statement). If none, write 'None'.",
  "reason": "Final decision (1-2 sentences). Explain why it is or is not humorous."
}"""

_CULTURAL_JSON_SCHEMA = """{
  "humorous": bool,
  "detected_text": "all text visible in the image (Hindi, English, Hinglish). Write 'No text detected' if none.",
  "visual_description": "brief description (1 sentence)",
  "cultural_category": "one of: family, education, college, JEE/exams, cricket, Bollywood, food, festivals, marriage/wedding, relationships, workplace, social_norms, religion, regional_culture, daily_life, hindi_slang, internet_culture, or none",
  "cultural_dependency": "low if no cultural knowledge needed, medium if some helps, high ONLY if the comedic mechanism requires cultural knowledge",
  "cultural_context": "brief cultural context (1 sentence) or 'None'. Cultural understanding is NOT humor evidence.",
  "humor_evidence": "Specific comedic mechanism and exact evidence. A cultural reference alone is NOT humor. If none, write 'No clear comedic mechanism found'.",
  "non_humor_evidence": "List all non-humor indicators (serious, motivational, emotional, educational, cultural greeting). If none, write 'None'.",
  "reason": "Final decision (1-2 sentences). Cultural relevance ≠ humor. Explain why it is or is not humorous."
}"""


# ─── Core classification rules shared across all prompts ─────────────────────

_CLASSIFICATION_RULES = (
    "CLASSIFICATION RULES:\n"
    "A. BALANCED CLASSIFICATION: Evaluate evidence symmetrically based on verifiable comedic intent versus genuine non-humor indicators. Do not presume an arbitrary prior bias toward either class. Classify as HUMOROUS when an intentional comedic mechanism is evident; classify as NOT HUMOROUS when non-humor indicators predominate or comedic intent is absent.\n"
    "B. HUMOROUS requires a specific comedic mechanism that is INTENDED TO AMUSE:\n"
    "   - Comedic twist, irony, absurdity, sarcasm with comedic intent, wordplay,\n"
    "     exaggeration for comedic effect, parody, satire, visual comedy, situational humor,\n"
    "     incongruity between image and text that creates amusement, relatable humor WITH a comedic element.\n"
    "   - Legitimate dark comedy and morbid absurdity: Fictional characters in cynical, morbid, or absurd situations\n"
    "     (e.g., comic strips featuring doctor/patient mishaps, grim reaper gags, cynical workplace/life cynicism,\n"
    "     or absurd stick-figure misadventures where no real-world platform or person is targeted with criminal accusations).\n"
    "   - Intentional satirical hypocrisy and self-contradictory personas (e.g. contrasting contradictory\n"
    "     traits or expectations for comedic effect, such as 'I have two sides' contrasts) when an amusing comedic effect is evident.\n"
    "   - Clearly satirical celebrity caricature cards, political parody, or parody merchandise created for amusement\n"
    "     (a conventional punchline is not strictly required if visual, situational, or parody comedy is evident;\n"
    "     distinguish caricature, novelty, political or celebrity parody greeting cards from genuine personal greetings).\n"
    "   - Situational irony and comedic incongruity created by contrasting serious public announcements or national headlines\n"
    "     with hyperbolic personal, romantic, or internet-subculture reactions (e.g. contrasting grave news or lockdown announcements\n"
    "     with melodramatic teenage romance panic).\n"
    "   - Visual, situational, linguistic, and culturally specific humor.\n"
    "C. NOT HUMOROUS includes:\n"
    "   - Motivational quotes, inspirational messages, serious statements\n"
    "   - Emotional content (sad, angry, nostalgic without comedic framing)\n"
    "   - Informational, educational, or protest content\n"
    "   - Political/social commentary without comedic mechanism\n"
    "   - Genuine personal, religious, or cultural greetings, life advice (distinguish from clearly satirical\n"
    "     celebrity caricature cards or parody merchandise created for amusement)\n"
    "   - News, awareness messages, advertisements, announcements\n"
    "   - Real-world criminal accusations on comic templates: Serious criminal allegations, malicious smears, defamatory accusations, or non-comedic attacks\n"
    "     that accuse real-world people, organizations, platforms, or professions of heinous crimes\n"
    "     (e.g. pedophilia, sexual abuse, violence, or corruption) simply as character assassination or platform warfare,\n"
    "     even when rendered in a cartoon, comic, or webcomic panel format.\n"
    "     Pasting allegations of heinous crimes against identifiable real-world entities onto a comic format does NOT constitute dark humor or comedic absurdity.\n"
    "     (Contrast with legitimate dark humor, which features purely fictional characters in hypothetical absurd scenarios without defamatory real-world criminal accusations).\n"
    "   - Dark/offensive/disturbing content without a clear comedic mechanism\n"
    "   - Complaints, opinions, observations without comedic twist\n"
    "   - Relatable situations WITHOUT a comedic element\n"
    "D. These alone are NOT evidence of humor:\n"
    "   - Meme, comic, or webcomic format alone (cartoons and comic strips are legitimate media for humor,\n"
    "     but a comic format alone does NOT prove humor, nor does it convert a defamatory criminal smear into comedy)\n"
    "   - Comic strip templates with heinous criminal accusations: pasting allegations of serious crimes onto a comic format does NOT constitute dark humor or comedic absurdity\n"
    "   - Cultural reference alone (understanding cultural context is essential for comprehension,\n"
    "     but cultural relevance alone does NOT establish humor)\n"
    "   - Hindi/Hinglish text (language ≠ humor)\n"
    "   - Expressive facial expression (surprise, shock, anger ≠ comedy)\n"
    "   - Sarcasm-like language (sarcastic tone in complaints is NOT humor)\n"
    "   - Image-text relationship (thematic consistency is NOT humor)\n"
    "   - Relatability (a relatable situation is NOT automatically funny)\n"
    "   - Plain juxtaposition (arbitrary contrast without an intentional joke is not humor,\n"
    "     though intentional satirical hypocrisy or comedic self-contradiction is)\n"
    "E. DECISION TEST before setting humorous=true:\n"
    "   1. Can you identify a specific comedic mechanism (visual, situational, linguistic, satirical hypocrisy, caricature parody, or fictional dark absurdity)?\n"
    "   2. Does the content level an accusation of a heinous crime against an identifiable real-world platform, organization, or person? If YES, set humorous=false.\n"
    "   3. Is the content INTENDED to amuse, or is it serious, defamatory, informational, or non-comedic protest?\n"
    "   4. Evaluate the full context and intended mechanism rather than applying blanket surface rules.\n"
    "   5. Are non-humor indicators WEAKER than humor evidence?\n"
    "   If ANY check fails, set humorous=false.\n"
)


def build_humor_analysis_prompt(image_path: str):
    """
    Builds the GENERAL mode prompt for Qwen2.5-VL humor analysis.
    Does NOT inject cultural context. Does NOT ask for self-reported confidence.
    """
    system_prompt = (
        "You are an objective binary classifier that determines whether content is "
        "humorous or not humorous. You read Hindi, Hinglish, and English. "
        "Output ONLY valid JSON. "
        "Evaluate whether comedic intent is present based on evidence. "
        "Weigh comedic mechanisms symmetrically against genuine non-humor indicators without prior class bias."
    )

    user_content = (
        "Analyze the following meme image and determine whether it is humorous or not humorous.\n\n"
        "STEP-BY-STEP ANALYSIS:\n"
        "1. Describe what the image shows.\n"
        "2. Read ALL text in the image (Hindi, English, Hinglish).\n"
        "3. Determine the literal meaning of the image+text combination.\n"
        "4. Search for a SPECIFIC comedic mechanism (punchline, irony, absurdity, "
        "wordplay, exaggeration, parody, satire, visual comedy, comedic contradiction).\n"
        "5. Search for NON-humor indicators (motivational, informational, emotional, "
        "political, serious, educational, dark/disturbing).\n"
        "6. Try to state the joke in ONE sentence. If you cannot, it is likely not humorous.\n"
        "7. Compare humor evidence against non-humor indicators.\n"
        "8. Make your final decision based on which evidence is stronger.\n\n"
        "Keep each field concise (1-2 sentences maximum). Do not write long explanations.\n\n"
        f"{_CLASSIFICATION_RULES}\n\n"
        f"Return a JSON object strictly matching this structure:\n{_GENERAL_JSON_SCHEMA}\n\n"
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

    CRITICAL: Cultural context helps UNDERSTAND content, but cultural relevance
    is NOT evidence of humor.
    """
    system_prompt = (
        "You are an objective binary classifier for content that may involve Indian cultural context. "
        "You read Hindi/Hinglish/English. Output ONLY valid JSON. "
        "IMPORTANT: Cultural context helps you UNDERSTAND content, but "
        "cultural relevance does NOT mean the content is humorous. "
        "Evaluate evidence symmetrically. You must find concrete evidence of a comedic mechanism to classify as humorous. "
        "Understanding a cultural reference is NOT the same as finding it funny."
    )

    user_content = "Analyze the following meme image and determine whether it is humorous or not humorous.\n\n"

    if ocr_text and ocr_text.strip():
        user_content += f"Previously detected text (OCR): {ocr_text}\n\n"

    if retrieved_context and retrieved_context.strip():
        user_content += (
            f"{retrieved_context}\n\n"
            "CRITICAL WARNING about cultural context:\n"
            "- This context is provided to help you UNDERSTAND the content, NOT as evidence of humor.\n"
            "- A culturally relevant reference does NOT make content humorous.\n"
            "- Understanding a reference ≠ finding it funny.\n"
            "- Use this context ONLY to understand what the content is about.\n"
            "- The humor decision must be based on whether a COMEDIC MECHANISM exists.\n\n"
        )

    user_content += (
        "STEP-BY-STEP ANALYSIS:\n"
        "1. Describe the image.\n"
        "2. Read all text and describe visuals.\n"
        "3. Determine the literal meaning of image+text.\n"
        "4. Identify Indian cultural references and assess cultural dependency.\n"
        "   - cultural_dependency = 'high' ONLY if the JOKE requires cultural knowledge.\n"
        "   - An Indian setting, Hindi text, or cultural topic alone is NOT 'high'.\n"
        "5. Search for a SPECIFIC comedic mechanism with concrete evidence.\n"
        "6. Search for NON-humor indicators.\n"
        "7. State the joke in ONE sentence. If you cannot, it is likely not humorous.\n"
        "8. Compare humor evidence against non-humor indicators.\n"
        "9. Make your final decision.\n\n"
        "Keep each field concise (1-2 sentences maximum). Do not write long explanations.\n\n"
        f"{_CLASSIFICATION_RULES}\n"
        f"Return a JSON object strictly matching this structure:\n{_CULTURAL_JSON_SCHEMA}\n\n"
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
    Uses the SAME classification rules as single-image prompts.
    """
    if mode == "cultural":
        system_prompt = (
            "You are a strict binary classifier for content that may involve Indian cultural context. "
            "You read Hindi/Hinglish/English. Output ONLY valid JSON. "
            "Cultural context helps UNDERSTAND content, but cultural relevance is NOT humor. "
            "Most content you analyze will NOT be humorous. "
            "You must find concrete evidence of a comedic mechanism to classify as humorous."
        )
        json_schema = _CULTURAL_JSON_SCHEMA
    else:
        system_prompt = (
            "You are a strict binary classifier that determines whether content is "
            "humorous or not humorous. You read Hindi, Hinglish, and English. "
            "Output ONLY valid JSON. "
            "Most content you analyze will NOT be humorous. "
            "You must find concrete evidence of a comedic mechanism to classify as humorous."
        )
        json_schema = _GENERAL_JSON_SCHEMA

    # Build user content with multiple images
    user_content_parts = []
    for i, img_path in enumerate(image_paths):
        user_content_parts.append({"type": "image", "image": f"file://{img_path}"})

    text_content = (
        f"Analyze the above {len(image_paths)} meme images together as a combined set.\n"
        "They may represent different parts of one piece of content, multiple panels, or related images.\n\n"
    )

    if ocr_text and ocr_text.strip():
        text_content += f"Previously detected text (OCR): {ocr_text}\n\n"

    if mode == "cultural" and retrieved_context and retrieved_context.strip():
        text_content += (
            f"{retrieved_context}\n\n"
            "CRITICAL: This cultural context helps UNDERSTAND the content. "
            "Cultural relevance is NOT evidence of humor. "
            "Use this context only to understand references, NOT to justify a humor classification.\n\n"
        )

    text_content += (
        "Provide a COMBINED analysis considering all images together.\n\n"
        f"{_CLASSIFICATION_RULES}\n"
        f"Return a JSON object strictly matching this structure:\n{json_schema}\n\n"
        "Output ONLY the JSON object."
    )

    user_content_parts.append({"type": "text", "text": text_content})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content_parts}
    ]
    return messages


def build_cultural_analysis_prompt_config_c(image_path: str, ocr_text: str = "", retrieved_context: str = ""):
    """
    Builds the Config C (Targeted Cultural-Irony Refinement) prompt for Qwen2.5-VL.
    Preserves all classification rules, output schema, and decision threshold (tau=0.50).
    Refines instructions to prevent over-suppression of satire and irony when cultural context
    describes solemn, serious, or sensitive real-world topics.
    """
    system_prompt = (
        "You are an objective binary classifier for content that may involve Indian cultural context. "
        "You read Hindi/Hinglish/English. Output ONLY valid JSON. "
        "Evaluate evidence symmetrically without prior class bias. "
        "Retrieved cultural context is supporting evidence to help understand the content, NOT proof that an image is humorous or non-humorous. "
        "Serious cultural subject matter does not automatically imply non-humour: political satire, irony, sarcasm, parody, "
        "and dark humour frequently discuss serious events while remaining humorous. "
        "Do not suppress a clear comedic mechanism solely because the retrieved context describes a solemn event, cultural practice, relationship, or social issue. "
        "Equally, cultural familiarity, provocative language, or an unusual image alone does not establish humour; you must verify an actual comedic mechanism."
    )

    user_content = "Analyze the following meme image and determine whether it is humorous or not humorous.\n\n"

    if ocr_text and ocr_text.strip():
        user_content += f"Previously detected text (OCR): {ocr_text}\n\n"

    if retrieved_context and retrieved_context.strip():
        user_content += (
            f"{retrieved_context}\n\n"
            "CULTURAL CONTEXT & SATIRE/IRONY GUIDELINES:\n"
            "- Retrieved cultural context is supporting evidence to help you UNDERSTAND references, NOT proof that an image is humorous or non-humorous.\n"
            "- Serious cultural subject matter does NOT automatically imply non-humour. Political satire, irony, sarcasm, parody, and dark humour often discuss serious events while still being humorous.\n"
            "- Do NOT suppress a clear comedic mechanism solely because the retrieved context describes a solemn event, cultural practice, relationship, or social issue.\n"
            "- Equally, cultural familiarity, provocative language, or an unusual image alone does not establish humour; identify an actual comedic mechanism.\n"
            "- Base your decision on whether a genuine COMEDIC MECHANISM exists when the cultural reference is understood.\n\n"
        )

    user_content += (
        "STEP-BY-STEP ANALYSIS:\n"
        "1. Describe the image.\n"
        "2. Read all text and describe visuals.\n"
        "3. Determine the literal meaning of image+text.\n"
        "4. Identify Indian cultural references and assess cultural dependency.\n"
        "   - cultural_dependency = 'high' ONLY if the JOKE requires cultural knowledge.\n"
        "   - An Indian setting, Hindi text, or cultural topic alone is NOT 'high'.\n"
        "5. Search for a SPECIFIC comedic mechanism with concrete evidence.\n"
        "6. Search for NON-humor indicators.\n"
        "7. State the joke in ONE sentence. If you cannot, it is likely not humorous.\n"
        "8. Compare humor evidence against non-humor indicators.\n"
        "9. Make your final decision.\n\n"
        "Keep each field concise (1-2 sentences maximum). Do not write long explanations.\n\n"
        f"{_CLASSIFICATION_RULES}\n\n"
        f"Return a JSON object strictly matching this structure:\n{_CULTURAL_JSON_SCHEMA}\n\n"
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

