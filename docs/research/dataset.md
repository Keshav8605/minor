# Dataset

This study utilizes the **Memotion 3** dataset, a benchmark dataset designed for multimodal sentiment and humor analysis of code-mixed Hindi-English memes.

## Data Characteristics
- **Format:** Images (memes) containing superimposed text.
- **Language:** Hindi-English code-mixed text (Hinglish), often written in Latin script.
- **Annotations:** The dataset provides labels for various tasks, including sentiment (positive, negative, neutral) and humor presence/intensity.

## Preprocessing
To ensure data quality, the pipeline includes several preprocessing steps:
1. **Validation:** Checking for file existence and corrupt images.
2. **OCR Integration:** Extracting text from images (handled natively by VLMs or via explicit OCR tools depending on the pipeline mode).
3. **Filtering:** Ensuring that only memes with valid annotations and readable formats are included in the evaluation suite.

*Note: In adherence to our strict data policy, we do not fabricate dataset statistics. The system dynamically parses and utilizes the available subset of the Memotion 3 dataset during experimentation.*
