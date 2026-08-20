# Academic Demonstration Interface

The Gradio web interface provides a visual, interactive way to evaluate the core research hypothesis: **Does injecting explicit cultural context improve VLM reasoning on codemixed Hindi/English memes?**

## How to Start
1. Ensure dependencies are synced: `uv pip install -e .` and `uv pip install gradio`
2. Run `python app/app.py`
3. Navigate to `http://127.0.0.1:7860` in your browser.

*(Note: The server loads the VLM weights lazily. If you do not have a GPU or the model is missing, the UI will still load but will display a friendly "Model Unavailable" message when you attempt to analyze an image.)*

## How Inference Works
- **Image Upload**: The user uploads an image.
- **Cultural Toggle**:
  - **Off**: The VLM receives the standard prompt (Mode A).
  - **On**: The system runs OCR, detects keywords, retrieves facts from `data/cultural/cultural_knowledge.json`, and injects them into the prompt (Mode B).
- **Backend**: The UI invokes the `VLMInferenceEngine` securely. Stack traces are caught and converted to clean HTML messages to prevent exposing system internals.

## Displayed Fields
- **Humor Prediction**: Binary outcome (Humorous vs Not Humorous).
- **Confidence**: The VLM's self-reported confidence, color-coded (Green for High, Red for Low).
- **Detected Text**: The raw OCR output.
- **Cultural Category**: The identified meme taxonomy tag (e.g., "Bollywood").
- **Cultural Dependency**: How heavily the joke relies on cultural knowledge (None, Low, Medium, High).
- **Reasoning**: The VLM's step-by-step logic explaining the humor.
