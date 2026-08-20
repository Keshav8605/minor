# Future Work

While the current architecture provides a solid foundation, several avenues remain open for future development:

1. **Vector Database Integration:** Migrating the cultural knowledge base from flat JSON files to a robust vector database (like Pinecone or ChromaDB) would enable semantic search and more precise context retrieval for Mode B inference.
2. **Larger Parameter Models:** Experimenting with larger, commercially available models via API (e.g., GPT-4V or Claude 3.5 Sonnet) could provide higher baseline performance against which open-source models can be benchmarked.
3. **End-to-End Fine-Tuning:** Rather than relying solely on Retrieval-Augmented Generation (RAG), future iterations could explore fine-tuning techniques (such as LoRA) to natively embed cultural understanding within the VLM weights.
4. **Expanded Taxonomies:** Refining the 11-point error taxonomy based on qualitative feedback to capture even more granular failure modes specific to varying regional dialects across India.
