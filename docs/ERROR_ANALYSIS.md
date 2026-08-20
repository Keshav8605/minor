# Error Analysis and Explainability

## Why Aggregate Metrics Are Insufficient
In multimodal cultural AI, aggregate metrics like F1 or Accuracy do not tell the whole story. A model might correctly predict a meme as "humorous" but for the completely wrong reason (e.g., misinterpreting Hindi text as gibberish, but recognizing a laughing face). Conversely, it might fail because of a nuanced cultural trope that we haven't mapped. 

Without systematic error analysis, we cannot improve the system or understand its biases.

## The Error Taxonomy
We define an 11-point taxonomy to categorize failures:
1. **OCR/text extraction failure**: The model misread the text.
2. **Hindi/Devanagari understanding failure**: The model read the text but failed to translate/understand the Hindi.
3. **Hinglish/code-mixing failure**: The model failed to grasp the blended slang.
4. **Visual interpretation failure**: Misidentified objects or facial expressions.
5. **Image-text relationship failure**: Understood both modalities separately but failed to connect them.
6. **Sarcasm**: Failed to detect sarcasm.
7. **Cultural context failure**: Missed a specific cultural trope (e.g., Bollywood reference).
8. **Ambiguous humor**: The joke is highly subjective.
9. **Missing context**: Requires external context not present in the meme or cultural module.
10. **Label ambiguity**: The ground truth label itself is questionable.
11. **Other**: Catch-all.

## Human Review Workflow
1. Run `python scripts/generate_error_report.py --run-dir results/EXP-XX/...`
2. This generates `human_review_worksheet.csv` in the `error_analysis` folder of the run.
3. Open this CSV. It filters out all correct predictions.
4. Review the `raw_response` and image, and manually select one of the 11 taxonomy categories to place in the `human_assigned_error_type` column.
5. Do NOT trust the `auto_inferred_reason`. The script explicitly tags it as a hypothesis requiring human review to prevent accidental automated claims.
