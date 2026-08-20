# Problem Statement

The automated detection of humor in code-mixed Hindi-English memes presents a multifaceted problem that current state-of-the-art models fail to fully address.

1. **Inadequacy of Text-Only Models:** Purely textual models (like XLM-R or mBERT) process the extracted Optical Character Recognition (OCR) text but remain blind to the visual context. Since meme humor often hinges on the visual punchline or the facial expressions of subjects in the image, text-only models frequently misclassify memes.
2. **Inadequacy of Vision-Only Models:** Purely visual models (like ViT or ResNet) can identify objects and scenes but lack the linguistic capacity to parse code-mixed Hindi-English text or understand the semantic wordplay that drives the joke.
3. **The Cultural Gap:** Even multimodal Vision-Language Models (VLMs) often lack the localized cultural knowledge required to understand regional humor, idioms, or societal references specific to the Indian subcontinent.

Therefore, the core problem is designing a system that can simultaneously process both visual and textual inputs while contextualizing them within the appropriate cultural framework to accurately classify the presence and type of humor.
