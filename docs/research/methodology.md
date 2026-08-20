# Methodology

The research methodology revolves around comparing two primary inference modes for Vision-Language Models processing code-mixed memes.

## Mode A: Zero-Shot VLM Inference
In this mode, the VLM is provided only with the meme image and a prompt asking it to classify the humor. The model must rely entirely on its pre-trained internal knowledge to decipher the Hindi-English text and understand any visual references.

## Mode B: Culturally-Augmented Inference
This mode introduces a cultural retrieval step prior to inference.
1. **Entity Extraction:** Key visual and textual entities are identified from the meme.
2. **Context Retrieval:** These entities are queried against a knowledge base to retrieve cultural context (e.g., explaining a specific Bollywood reference, a political figure, or a regional idiom).
3. **Augmented Prompting:** The retrieved cultural context is appended to the prompt alongside the image.
4. **Inference:** The VLM classifies the humor, now armed with explicit localized knowledge.

By strictly comparing the outputs of Mode A and Mode B on identical datasets, we isolate the performance impact of explicit cultural context injection.
