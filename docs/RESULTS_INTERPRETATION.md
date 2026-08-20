# Results Interpretation

The automated orchestrator generates reports, tables, and (if `matplotlib` is installed) figures for every run.

## Key Metrics
- **Macro F1**: This is the primary metric. Because humor datasets are often imbalanced (e.g., 70% humorous, 30% not), accuracy is misleading. Macro F1 averages the F1 scores of the positive and negative classes equally.
- **Parsing Failure Rate**: VLMs instructed to output JSON sometimes fail. This rate tracks how often the model hallucinates outside the schema.
- **Inference Failure Rate**: Measures outright API/CUDA crashes.

## Evaluating Cultural Awareness
Do not claim that the Cultural Awareness Module (EXP-05) is successful unless it achieves a statistically significant improvement in **Macro F1** over the Zero-Shot VLM (EXP-04). 

Furthermore, you should check the `cultural_dependency_analysis` table. The true test of the cultural module is whether its performance spikes specifically on memes tagged as `high` cultural dependency compared to EXP-04.
