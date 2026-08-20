# Experimental Setup

The project defines a series of structured experiments to evaluate the system iteratively.

- **EXP-01: Standard VLM (Zero-Shot)** - Evaluates the baseline capability of the VLM without any external context (Mode A).
- **EXP-02: Culturally-Augmented VLM** - Evaluates the VLM with injected cultural context to measure the uplift in humor detection (Mode B).
- **EXP-03: Few-Shot VLM Inference** - Provides the VLM with several labeled examples within the prompt to assess in-context learning capabilities.
- **EXP-04: Cross-Model Comparison** - Compares different VLM architectures (e.g., Qwen-VL vs. LLaVA) using the Mode B setup.
- **EXP-05: Error Taxonomy Analysis** - A qualitative review of failed inferences to categorize errors according to the 11-point taxonomy.

All experiments are executed via the `src.experiments.runner` and tracked meticulously in the `results/` directory to prevent metric fabrication.
