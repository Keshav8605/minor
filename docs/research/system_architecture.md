# System Architecture

The architecture of the Culturally-Aware VLM pipeline is designed for modularity and rigorous evaluation.

```mermaid
graph TD
    A[Meme Image] --> B{Pipeline Mode}
    B -- Mode A: Standard --> C[VLM Inference Engine]
    B -- Mode B: Cultural --> D[Entity Extraction]
    D --> E[Cultural DB / Retrieval]
    E --> F[Contextualized Prompt]
    F --> C
    C --> G[Humor Classification Output]
    G --> H[Evaluation Engine]
    H --> I[Results & Metrics]
```

## Key Components:
- **Experiment Manager:** Orchestrates the pipeline, ensuring exact configurations are tracked and repeatable.
- **VLM Inference Engine:** Handles the interface with models (e.g., Qwen-VL) to process images and prompts.
- **Cultural Retrieval Module:** Manages the injection of context (utilized in Mode B).
- **Evaluation Engine:** Computes metrics (Macro F1, Accuracy) and records parsing failures.
