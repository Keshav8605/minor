# Experiment Protocol

To scientifically validate the hypothesis that a VLM with explicit cultural awareness outperforms traditional pipelines on codemixed Hindi/English memes, we adhere to the following rigid experimental protocol.

## Runs

- **EXP-01**: Text-only classical baseline. Uses XLM-R to prove how much signal is purely in the Hinglish text.
- **EXP-02**: Image-only classical baseline. Uses ViT to prove how much signal is purely visual.
- **EXP-03**: Classical Multimodal (ViT + XLM-R). The traditional early-fusion approach.
- **EXP-04**: Qwen2.5-VL Zero-Shot (Mode A). The raw VLM reasoning over image and OCR, without cultural injection.
- **EXP-05**: Qwen2.5-VL Contextualized (Mode B). The VLM prompted with explicit factual context retrieved from the Cultural Module.

## Integrity Policies
1. **No Manual Overwrites**: Every experiment generates a unique UUID folder (e.g., `EXP-04_20260820_abcd1234`). Old data is never overwritten.
2. **Deterministic Evaluation**: All metrics are calculated by a single shared module (`report_generator.py`) to prevent discrepancies in how F1 or accuracy are calculated between baselines and VLMs.
3. **No Fabrication**: If an experiment fails due to VRAM limits or parsing errors, it is recorded as a failure. We measure `parsing_failure_rate` explicitly for VLMs.
