# Dataset Card: Memotion 3

## Source
- **Dataset Name**: Memotion 3
- **Venue**: De-Factify 2 workshop in AAAI-22

## Access Requirements
The dataset requires explicit authorization from the creators (via workshop registration) and must not be redistributed. Ensure compliance with their terms.

## Annotations
The original schema contains the following classes:
- `humour`: ["not_funny", "funny", "very_funny", "hilarious"]
- `sarcastic`: Degree of sarcasm
- `offensive`: Degree of offensiveness
- `motivational`: Degree of motivation
- `overall_sentiment`: Sentiment polarity
- `ocr`: Hindi-English Codemixed Text extracted from images

## Preprocessing Pipeline
- **Target Label (`is_humorous`)**: A binary label derived directly from the `humour` annotation. "not_funny" maps to `0`, while "funny", "very_funny", and "hilarious" map to `1`.
- **Validation**: Any records missing physical image files in `trainImages/` or `testImages/` are filtered out.
- **Duplicates**: Duplicate rows (by `image_url`) are identified and dropped.
- **Splits**: Since there is no `val.csv` in the downloaded archive, we stratify the `train.csv` (using the new binary label) to generate a 90% train / 10% validation split. The test set is processed separately. A fixed random seed (42) is used for reproducibility.

## Limitations
- OCR text inherently contains transcription errors.
- Codemixed text is extremely noisy.
- Annotation subjectivity: Humor perception varies significantly between individuals.
