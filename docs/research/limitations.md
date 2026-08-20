# Limitations

Scientific transparency requires acknowledging the limitations of the current architecture.

1. **Not Universal Intelligence:** The system does not possess human-level intelligence. It relies on statistical patterns and the quality of retrieved context.
2. **Retrieval Bottleneck:** The effectiveness of Mode B (Cultural Augmentation) is heavily dependent on the comprehensiveness of the external knowledge base. Missing entries will degrade performance back to Mode A levels.
3. **Inference Latency:** Processing large VLMs requires significant computational resources, resulting in high inference latency that may not be suitable for real-time applications.
4. **Hallucinations:** Despite structural constraints, VLMs are prone to occasional hallucinations, sometimes generating confident but incorrect justifications for a meme's humor.
