# Zero-Shot VLM Baseline Experiment

## Experimental Question
Can a vision-language model (Qwen2.5-VL-3B-Instruct) reliably detect humor in Hindi-English codemixed memes without any fine-tuning (zero-shot)?

## Model
**Qwen2.5-VL-3B-Instruct**: A 3 billion parameter model capable of OCR and visual reasoning.

## Prompt
A structured zero-shot prompt was used, asking the model to act as an expert AI assistant specialized in culturally aware multimodal humor. The model is forced to output JSON containing a binary `humorous` decision, a `confidence` score, and explanatory fields (`detected_text`, `visual_description`, `cultural_context`, `reason`).

## Dataset
Memotion 3 test set (`test.csv`). All records missing physical image files were omitted during processing.

## Evaluation Metrics
- **Accuracy**: Overall correctness.
- **Precision**: How many of the predicted humorous memes were actually humorous.
- **Recall**: How many of the actual humorous memes were successfully identified.
- **F1 Score**: Harmonic mean of precision and recall.
- **Confusion Matrix**: To identify if the model overpredicts humor.

## Limitations & Known Sources of Bias
- **Codemixed Language Bias**: VLMs are predominantly trained on English and standard Hindi. Hinglish slang is often misinterpreted.
- **OCR Reliance**: If the model fails to read messy meme text, the humor classification will likely fail.
- **Cultural Context Gap**: Zero-shot models may lack specific Indian cultural nuances required to understand the joke, leading to false negatives.
