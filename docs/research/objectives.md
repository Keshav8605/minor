# Objectives

The primary objective of this research is to evaluate and improve the classification of humor in Hindi-English code-mixed memes. The specific sub-objectives are:

1. **Baseline Evaluation:** To establish performance benchmarks using classical unimodal and multimodal approaches on the Memotion 3 dataset.
2. **VLM Integration:** To integrate state-of-the-art Vision-Language Models (specifically focusing on open-source variants like Qwen-VL) to perform zero-shot and few-shot humor inference (Mode A).
3. **Cultural Contextualization:** To design and implement a cultural retrieval mechanism that provides the VLM with contextual background information about the meme's contents prior to inference (Mode B).
4. **Comparative Analysis:** To rigorously compare the performance of Mode A (VLM alone) versus Mode B (VLM + Cultural Retrieval) to quantify the impact of cultural context on humor detection accuracy.
5. **Taxonomic Error Analysis:** To classify model failures into a predefined 11-point taxonomy to better understand the limitations of current VLM architectures in processing culturally specific humor.
