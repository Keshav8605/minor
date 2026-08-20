# Error Analysis

Quantitative metrics alone are insufficient for understanding the failure modes of VLMs on code-mixed humor. Therefore, this project implements a rigorous 11-point taxonomy for qualitative error analysis.

## Key Taxonomic Categories Include:
- **T1: Sarcasm Failure:** The model interprets sarcastic text literally.
- **T3: Cultural Knowledge Gap:** The model fails because it lacks specific Indian regional context (e.g., a reference to a local festival or political event).
- **T7: OCR/Text Extraction Failure:** The model incorrectly reads the code-mixed text from the image, propagating the error to the classification stage.

Human-in-the-loop review is required to categorize parsing failures and misclassifications into these taxonomic buckets. This qualitative analysis is critical for iteratively improving the cultural retrieval mechanisms.
