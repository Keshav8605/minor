# Repair Plan for Hindi-Humor-VLM Project

This document details the plan to fix all remaining genuine bugs, repair the evaluation pipelines, and fortify the test suite, adhering strictly to the deep audit findings.

## Status of Verification
- **Finding 12 (Data Leakage)**: VERIFIED. Image MD5 hashing run on the entire `trainImages` and `testImages` directories yielded 0 identical hashes across the 7000 and 1500 images respectively. The 1358 overlapping image names (e.g. `0.jpg`) were merely integer counters resetting in each directory. No data leakage exists, but name collisions during data processing must be prefixed (already fixed).
- **Finding 4, 5, 6, 7, 11**: VERIFIED. Previously identified during the first phase and already patched (e.g. `MultimodalClassifier` dimensions dynamically size based on config, `best_model.pth` properly tracked, and `(?:^|\W)` unicode boundaries applied).
- **Finding 1, 2, 3, 8, 9, 10**: VERIFIED. These issues require the following new logic and tests.

## Proposed Changes

### Tests & Validation

#### `tests/test_dataset_validation.py`
- **Fix Finding 1 & 2**: Add tests to ensure `train`, `validation`, and `test` data splits *must* contain the target label `"is_humorous"`. If the target label is missing, the validation must fail loudly.
- **Fix Finding 2**: Add explicit tests validating local paths vs HTTP URLs, and checking malformed paths, spaces, and unicode in paths for image processing.

#### `src/data_validation.py`
- **Fix Finding 2**: The `validate_images` API will be explicitly updated to handle remote URL vs local filepath scenarios properly. It will no longer blindly check `Path(image_dir) / image_url` for HTTP references.

#### `tests/test_classifier.py`
- **Fix Finding 6 & 8**: Replace existing "mock" unit tests with realistic synthetic tensor tests. The test will dynamically verify the `MultimodalClassifier` dynamically handles varying synthetic dimensions (e.g., `(2, 512)` and `(2, 768)`).

#### `tests/test_smoke.py`
- **Fix Finding 8**: Improve smoke testing so it genuinely exercises a lightweight pipeline component without massive downloads.

#### `tests/test_cultural_categories.py`
- **Fix Finding 7 & 8**: Add Unicode edge-case tests (Devanagari text, mixed Hinglish, Emojis) to confirm the new `(?:^|\W)` regex word boundary functions perfectly.

#### `tests/test_experiment_failure.py`
- **Fix Finding 3 & 8**: Add an explicit test that forces an experiment failure, verifying that `scripts/run_experiment.py` marks the status as `FAILED`, preserves error logs, and immediately halts downstream dependent stages (like reporting).

---

### Core Data & Pipeline Operations

#### `src/evaluation/prediction_store.py`
- **Fix Finding 10**: The current CSV writer uses basic `mode="a"`. We will implement threading locks for thread safety, write atomic rows using proper escaping, and explicitly document whether the system currently uses multi-processing for concurrent writes (it does not, but safety is guaranteed).

#### `scripts/run_experiment.py`
- **Fix Finding 3**: Ensure experiment registry entries are saved with status `FAILED` if the subprocess raises an exception. No metrics or reports will be generated upon failure.

## Verification Plan

### Automated Tests
- Run `pytest` on all updated tests (dataset, classifier, cultural, failure conditions).
- Run `python -m compileall .` to ensure syntax correctness globally.

### Manual Verification
- Execute `python scripts/run_experiment.py --exp EXP-03 --smoke-test` to guarantee the baseline multi-modal pipeline functions start to finish without generating mock labels.
- Execute `python app/app.py --smoke-test` to ensure Gradio loads.
