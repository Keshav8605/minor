# Literature Review

The study of computational humor detection has evolved from text-only jokes to multimodal meme analysis.

**Multimodal Meme Analysis:**
The Memotion dataset series (1, 2, and 3) has been instrumental in providing annotated data for meme sentiment and humor analysis. Memotion 3 specifically introduces code-mixed Hindi-English memes, highlighting the necessity for models that can process both multimodality and multilingualism. Previous approaches have largely relied on late-fusion architectures combining Vision Transformers (ViTs) and multilingual language models (like XLM-R).

**Vision-Language Models (VLMs):**
Recent advancements have shifted towards unified Vision-Language Models (e.g., LLaVA, Qwen-VL, GPT-4V) that can ingest both images and text to produce text outputs. Qwen-VL, for instance, has shown strong multilingual capabilities and fine-grained visual understanding. However, these models are often pre-trained on Western-centric datasets, leading to a gap in understanding regional humor.

**Retrieval-Augmented Generation (RAG):**
RAG has emerged as a powerful technique to provide LLMs with external knowledge, reducing hallucinations and improving contextual accuracy. Applying RAG principles to visual domains—by retrieving cultural context based on image and OCR text—represents a promising frontier for adapting generalized VLMs to specific, localized tasks without extensive fine-tuning.
