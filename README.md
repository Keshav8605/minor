# Culturally Aware Multimodal Humor Detection in Hindi Using Large Vision-Language Models

This repository contains the full academic research pipeline for detecting humor in codemixed Hindi-English ("Hinglish") memes. The core hypothesis investigates whether injecting explicit cultural factual context into a Vision-Language Model (VLM) improves its reasoning capabilities over purely visual or purely textual baselines.

---

## 1. System Architecture & Pipeline

This project implements a modular architecture comparing Zero-Shot General VLM Inference against Culturally-Augmented Inference:

```
                      ┌────────────────────────────────────────┐
                      │            User Meme Input             │
                      │       (Single or Multi-Panel)          │
                      └───────────────────┬────────────────────┘
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
            [ General Mode ]                         [ Cultural-Aware Mode ]
                     │                                         │
                     │                           ┌─────────────┴─────────────┐
                     │                           ▼                           ▼
                     │                   Dedicated EasyOCR           Knowledge Retrieval
                     │               (Devanagari + English)      (Cultural Knowledge KB)
                     │                           └─────────────┬─────────────┘
                     │                                         │
                     ▼                                         ▼
           Standard VLM Prompt                       Culturally-Injected Prompt
                     │                                         │
                     └────────────────────┬────────────────────┘
                                          │
                                          ▼
                            Qwen2.5-VL-3B-Instruct Engine
                                          │
                         ┌────────────────┴────────────────┐
                         ▼                                 ▼
                 Generated Output                 Per-Token Logits
                 (JSON Structure)             (P(Humorous) & P(Non-Humor))
                         │                                 │
                         └────────────────┬────────────────┘
                                          │
                                          ▼
                             Unified Prediction & Analysis
                               (7 Dedicated UI Cards)
```

### Key Highlights:
1. **Calibrated Logit-Based Confidence**:
   - Eliminates hallucinated ~90% self-reported confidence.
   - Extracts real softmax probabilities from model output logits (`output_scores=True`).
   - Yields dual probability estimates: $P(\text{Humorous})$ and $P(\text{Non-Humorous})$, maintaining $P(H) + P(NH) \approx 1.0$.
2. **Dedicated OCR Extraction**:
   - Integrated `src/cultural/ocr_engine.py` using EasyOCR (`['hi', 'en']`) for bilingual Devanagari and Latin script extraction.
   - Cultural retrieval runs on actual extracted meme text rather than relying solely on image filenames.
3. **Structured Cultural Knowledge Base**:
   - Taxonomy enriched with `hindi_slang`, `internet_culture`, Bollywood references, cricket, and desi lifestyle.
   - Fallback to Qwen2.5-VL visual text reading when image resolution is low.
4. **CPU Latency Optimization**:
   - Vision token resolution capped (`min_pixels: 200704`, `max_pixels: 401408`), preventing 2000+ visual token explosion on CPU.
   - Reduced CPU inference latency by 3x–4x while preserving full OCR and visual reasoning fidelity.

---

## 2. Gradio Web Application & 7 Output Components

The interactive interface provides 7 distinct, responsive output cards:
1. **🎯 HUMOR PREDICTION**: Primary classification (`Humorous` vs `Not Humorous`).
2. **◉ MODEL PROBABILITY**: Uncalibrated model probability derived from token logits with dual probability breakdown bars for $P(H)$ and $P(NH)$.
3. **📄 DETECTED TEXT (OCR)**: Bilingual OCR extracted text from the meme image.
4. **🏷️ CULTURAL CATEGORY**: Detected cultural domain (e.g., `family_relations`, `bollywood_pop`, `cricket_sports`, `hindi_slang`, `internet_culture`).
5. **🔗 CULTURAL DEPENDENCY**: Assessment of cultural necessity (`high`, `moderate`, `low`, `none`).
6. **🌐 CULTURAL CONTEXT**: Retrieved cultural explanation from the grounded knowledge base.
7. **🧠 AI REASONING**: Clean, step-by-step reasoning explaining the punchline, cultural nuance, and incongruity (guaranteed free of raw JSON `{...}` formatting).

### Responsive Design
The UI incorporates media queries supporting:
- Desktop screens (1366×768 and above)
- Tablet / Small Laptops (1024×768)
- Mobile devices (412×915) with automatic single-column stacking and zero horizontal overflow.

To launch the web interface:
```bash
python app/app.py
```
Then navigate to `http://127.0.0.1:7860`.

---

## 3. Dataset Audit & Scientific Honesty

- **Primary Dataset**: **Memotion 3** (10,000 codemixed Hindi/Hinglish memes: 7,000 train, 1,500 validation, 1,500 test).
- **Labels Available**: Humor classes (`not_funny`, `funny`, `very_funny`, `hilarious`), sarcasm, offense, motivation, sentiment.
- **Dataset Limitations**:
  - Memotion 3 contains **zero cultural labels** (no category annotations, no dependency labels).
  - Annotator agreement is moderate ($\kappa \approx 0.42$) due to the subjective nature of humor.
- **Scientific Claim**:
  - The model is **not fine-tuned on cultural humor annotations**.
  - Cultural awareness is introduced via **In-Context Grounded Knowledge Retrieval (RAG)** combined with zero-shot multimodal reasoning in `Qwen2.5-VL-3B-Instruct`.
  - Claims of "cultural expertise" are avoided; instead, the system is evaluated on whether factual cultural context improves classification accuracy and qualitative reasoning.

---

## 4. Test Suite & Verification

The automated test suite contains **36 passing tests** across 6 test modules:
```bash
python -m pytest tests/ -v
```

### Test Coverage:
- `tests/test_vlm_output_parser.py`: Strict & relaxed JSON parsing, markdown codeblock stripping, field normalization.
- `tests/test_regression_cultural_parsing.py`: Regression verification for multiline JSON, unescaped newlines, clean reasoning text, 7 UI output tuple unpacking.
- `tests/test_category_detector.py`: Keyword and regex cultural categorization across Hindi and English.
- `tests/test_context_retriever.py`: Grounded knowledge retrieval and similarity ranking.
- `tests/test_confidence.py`: Logit-based dual-probability derivation and formatting.
- `tests/test_smoke.py`: Pipeline smoke testing and schema validation.

---

## 5. Directory Structure

```
hindi-humor-vlm/
├── app/
│   ├── app.py                     # Gradio application server
│   ├── formatting.py              # Output formatting & probability parsing
│   └── ui_components.py           # Responsive UI layout & 7 output cards
├── configs/
│   ├── model.yaml                 # Model configuration & vision token bounds
│   ├── cultural.yaml              # Cultural taxonomy & KB paths
│   └── prompts.py                 # Structured VLM prompt templates
├── data/
│   ├── cultural/                  # Cultural categories & knowledge base
│   └── processed/                 # Processed Memotion 3 splits & images
├── src/
│   ├── cultural/
│   │   ├── ocr_engine.py          # Dedicated EasyOCR engine (Hindi+English)
│   │   ├── category_detector.py   # Heuristic cultural domain classifier
│   │   └── context_retriever.py   # Grounded knowledge retriever
│   ├── vlm/
│   │   ├── model_loader.py        # Optimized Qwen2.5-VL loader (CPU thread cap)
│   │   ├── inference.py           # Inference engine & logit confidence extraction
│   │   └── output_parser.py       # Robust JSON parser & schema normalizer
│   └── evaluation/
│       ├── evaluator.py           # Experiment runner
│       └── report_generator.py    # Comparative analysis reporter
├── tests/                         # 36 automated unit & regression tests
└── README.md                      # Project documentation
```