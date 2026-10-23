# Critical Evaluation-Safety Fixes & Implementation Verification

**Document Type**: Engineering Implementation & Verification Report  
**Target Plan**: `reports/heldout_evaluation_plan.md`  
**Reference Audit**: `reports/heldout_evaluation_plan_audit.md`  
**Auditor & Implementer**: Antigravity Assistant  
**Date**: October 20, 2026  
**Status**: All Critical Safety Fixes Implemented & Validated via 52/52 Passing Tests  

---

## Executive Summary

Following the methodological and execution-safety audit documented in `reports/heldout_evaluation_plan_audit.md`, this report details the implementation of all required safety patches across the evaluation harness, checkpoint manager, inference engine, deduplication pipeline, and sampling specification.

Every identified hazard—including the dangerous `prediction: 0` fallback bug, the unmonitored vision prefill vulnerability, configuration collision risks, incomplete record tracking, and the infeasible `NH3` sampling quota—has been systematically resolved and verified with offline tests. Zero VLM inference calls or application services were launched.

---

## 1. Summary of Changes by Phase

### Phase 1 — Fix Failed-Prediction Handling
1. **Eliminated `prediction: 0` Fallback Bug**:
   - In `run_robust_eval.py` (lines 62–74), changed `"prediction": 0` to `"prediction": None`, with `"timed_out": True`, `"json_error": "HUNG_PROCESS"`, and reason `"SKIPPED: Model hang/deadlock detected by wrapper timeout."`.
   - In `scripts/run_controlled_evaluation.py` (lines 369–387 & 454–478), ensured that any general or cultural inference suffering a timeout or unrecovered JSON error strictly records `prediction: None` rather than defaulting to 0 or boolean fallbacks.
   - In `src/evaluation/evaluator.py` (lines 92–98), patched `prediction_val = 1 if pred_bool else 0` to `prediction_val = 1 if pred_bool is True else (0 if pred_bool is False else None)`, setting `prediction_val = None` on errors or timeouts.
2. **Metric Denominator Integrity**:
   - In `scripts/run_controlled_evaluation.py` (`_generate_final_outputs`), updated `gen_valid_mask` and `cult_valid_mask` to strictly require `(~gen_timed_out) & res_df["general_json_error"].isna()`. Timed-out and error records are excluded from Accuracy, Precision, Recall, and F1 denominators.
   - Added explicit tracking and reporting in `evaluation_summary.json` under `inference_reliability`: `general_timeouts`, `cultural_timeouts`, `general_valid_rate`, and `cultural_valid_rate`.
3. **Resumption & Incomplete Record Safety**:
   - In `src/evaluation/checkpoint_manager.py`, added `is_record_complete(r)` and `is_record_valid_prediction(r)` helper methods.
   - Updated `get_completed_pairs()` to reject structurally incomplete or corrupt records, preventing dropped inferences from being skipped.
   - Added `get_valid_completed_pairs()` to cleanly separate valid predictions from failed/hung attempts during resume.
   - In `src/evaluation/prediction_store.py`, added `load_valid_completed_ids()` to load only successfully generated predictions (`parsing_status == 'SUCCESS'`).

### Phase 2 — Audit Timeout Behavior
1. **Runner Timeout Alignment**:
   - In `run_robust_eval.py`, increased `TIMEOUT_S` from 900 seconds (15 minutes) to **1,500 seconds (25 minutes)**, matching the conservative execution budget for slow memes (`train_4136.jpg` observed at 1,118.59s in v4).
2. **Analysis of Blocking Vision-Prefill Interruption**:
   - PyTorch vision-encoder prefill (`pixel_values` $\to$ ViT blocks $\to$ KV-cache forward pass) runs as a synchronous, blocking C++/OpenMP forward pass inside `model.generate()`.
   - HuggingFace `StoppingCriteria` is **only invoked during token generation** in the autoregressive decoding loop; it cannot execute during prefill.
   - In-process Python threading cannot interrupt a blocking native C-extension on Windows without crashing the interpreter.
   - **Architectural Guarantee**: The outer subprocess timeout wrapper in `run_robust_eval.py` (`subprocess.run(timeout=1500)`) serves as the true process-level watchdog. When `subprocess.TimeoutExpired` fires, the OS process handle is terminated via `process.kill()`, cleanly halting hung prefill and releasing system threads before recording a `timed_out: True` record.
3. **Decoupled Internal Decoding Timeout**:
   - In `src/vlm/inference.py`, made `generation_timeout_seconds` configurable from `generation_params` (defaulting to 900.0s).
   - In `_build_result()`, enforced that if `diagnostics["timed_out"]` is True, `humorous` is forced to `None` and `prediction` is forced to `"Unavailable"`, even if partial output text contained a boolean token prior to truncation.
4. **Atomic Checkpoint Flushing**:
   - Added immediate `f.flush()` and `os.fsync(f.fileno())` after JSONL writes.
   - Replaced direct in-place JSON writes in `run_robust_eval.py` with atomic temporary file replacement (`tempfile.mkstemp` + `os.replace`).

### Phase 3 — Test Checkpoint & Metric Integrity
1. **Configuration Collision Prevention**:
   - Added `validate_configuration(expected_hashes)` to `CheckpointManager`.
   - In `scripts/run_controlled_evaluation.py`, called `mgr.validate_configuration(config_hashes)` immediately after computing hashes. If an existing checkpoint was created with different prompt or model hashes, a `ValueError` is raised, preventing cross-configuration checkpoint corruption.
2. **Safety Test Suite Created**:
   - Created `tests/test_evaluation_safety.py` containing 11 tests covering all six required safety criteria.
   - Verified that 52/52 tests pass across the entire checkpoint, timeout, mitigation, and safety test suites.

### Phase 4 — Revalidate the Sampling Plan
1. **Quota Rebalancing in `heldout_evaluation_plan.md`**:
   - In §2.2: Corrected arithmetic typo (91 clean non-humorous candidates, 589 clean humorous; was 81).
   - In §4 & §4.2: Updated `NH3 (Factual & Educational Graphics)` quota from 10 to **4** (reflecting the true availability in Memotion 3).
   - Reallocated the 6 freed places among eligible negative strata:
     - `NH4 (Sarcastic Non-Humour)`: 8 $\to$ **10** (29 candidates available)
     - `NH5 (Hostile Statements)`: 5 $\to$ **6** (22 candidates available)
     - `NH6 (Commercial Spam / Ads)`: 3 $\to$ **6** (8 candidates available)
     - Total negative class quota remains exactly **50** ($12 + 12 + 4 + 10 + 6 + 6 = 50$).
   - In §7.2: Updated per-sample hard process timeout from 1,200s to **1,500s**.
2. **Pure PIL/NumPy Perceptual-Hash Deduplication Module**:
   - Created `src/evaluation/deduplication.py` using zero external dependencies (pure PIL and NumPy).
   - Implemented 8x8 average hash (`compute_average_hash`), Hamming distance (`hamming_distance`), and normalized OCR Levenshtein matching (`compute_ocr_similarity`).
   - Verified that known historical visual duplicates (`train_6993.jpg` vs `train_2837.jpg`) are detected with Hamming distance $d=0$.
   - **Guaranteed Constraint**: Confirmed that no 100-sample manifest has been generated or finalized.

---

## 2. Test Execution & Verification Matrix

The test suite was executed offline using `python -m pytest`:

```bash
python -m pytest tests/test_checkpoint.py tests/test_prefill_timeout.py tests/test_drishtikon_lexical_mitigation.py tests/test_evaluation_safety.py
```

### Test Results Breakdown

| Test File | Test Class / Scope | Tests Run | Passed | Failed | Key Verification Point |
|:---|:---|:---:|:---:|:---:|:---|
| `tests/test_evaluation_safety.py` | `TestTimeoutFailureHandling` | 2 | 2 | 0 | Timeout records `prediction=None`, `timed_out=True`, never 0. |
| `tests/test_evaluation_safety.py` | `TestJSONParsingFailures` | 2 | 2 | 0 | Unparseable JSON records `prediction=None`, never class 0. |
| `tests/test_evaluation_safety.py` | `TestMetricDenominatorsExcludeFailures` | 1 | 1 | 0 | Denominator strictly equals valid inferences (5/5), excluding 5 failures. |
| `tests/test_evaluation_safety.py` | `TestInterruptionAndResume` | 1 | 1 | 0 | Resumed evaluation executes exactly remaining samples without duplication. |
| `tests/test_evaluation_safety.py` | `TestIncompleteRecordRejection` | 1 | 1 | 0 | Incomplete/corrupt JSONL records are rejected from `completed_pairs`. |
| `tests/test_evaluation_safety.py` | `TestConfigurationIsolation` | 1 | 1 | 0 | `validate_configuration()` raises `ValueError` on config hash mismatch. |
| `tests/test_evaluation_safety.py` | `TestPerceptualHashDeduplication` | 3 | 3 | 0 | Pure PIL/NumPy aHash detects identical ($d=0$), noisy ($d\le 2$), and OCR dupes. |
| `tests/test_checkpoint.py` | `CheckpointManager` Core Suite | 30 | 30 | 0 | Atomic writing, JSONL append, stop signals, status reporting. |
| `tests/test_prefill_timeout.py` | Decoupled Prefill/Decoding | 4 | 4 | 0 | Slow prefill does not trigger token 1 timeout; decoding limit enforced. |
| `tests/test_drishtikon_lexical_mitigation.py` | Lexical Mitigation Suite | 7 | 7 | 0 | Section 144/302/377/420 negative controls pass; Metro positive control intact. |
| **Total** | **Full Safety & Regression Suite** | **52** | **52** | **0** | **100% Pass Rate** |

---

## 3. Detailed Code Modifications

### 3.1 `run_robust_eval.py`
```diff
-TIMEOUT_S = 900  # 15 minutes max per inference
+TIMEOUT_S = 1500  # 25 minutes max per inference (aligned with 1,500s timeout plan)
...
-        "prediction": 0,
+        "prediction": None,  # NEVER default to 0 on failure/timeout
...
     with open(jsonl_path, "a", encoding="utf-8") as f:
         f.write(json.dumps(dummy_record) + "\n")
+        f.flush()
+        os.fsync(f.fileno())
...
-    with open(ckpt_path, "w", encoding="utf-8") as f:
-        json.dump(ckpt, f, indent=2)
+    # Atomic write for checkpoint
+    dirpath = os.path.dirname(ckpt_path)
+    import tempfile
+    fd, tmp_path = tempfile.mkstemp(dir=dirpath, suffix=".tmp", prefix=".ckpt_")
+    with os.fdopen(fd, "w", encoding="utf-8") as f:
+        json.dump(ckpt, f, indent=2)
+        f.flush()
+        os.fsync(f.fileno())
+    os.replace(tmp_path, ckpt_path)
```

### 3.2 `src/evaluation/checkpoint_manager.py`
```python
    @staticmethod
    def is_record_complete(r: dict) -> bool:
      if not isinstance(r, dict) or r.get("incomplete", False):
        return False
      required_keys = {"sample_index", "condition", "image_filename"}
      return (
          required_keys.issubset(r.keys())
          and r.get("sample_index") is not None
          and bool(r.get("condition"))
      )

    @staticmethod
    def is_record_valid_prediction(r: dict) -> bool:
      if not CheckpointManager.is_record_complete(r):
        return False
      if r.get("timed_out", False) or r.get("json_error") is not None:
        return False
      return r.get("prediction") in (0, 1)

    def validate_configuration(
        self, expected_hashes: dict, config_name: Optional[str] = None
    ):
      ckpt = self.load_checkpoint()
      if not ckpt:
        return
      ckpt_hashes = ckpt.get("config_hashes", {})
      mismatches = [
          f"{k}: existing '{ckpt_hashes[k]}' != expected '{v}'"
          for k, v in expected_hashes.items()
          if k in ckpt_hashes and ckpt_hashes[k] != v
      ]
      if mismatches:
        raise ValueError(
            f"Configuration mismatch in checkpoint directory"
            f" '{self.results_dir}': {', '.join(mismatches)}"
        )
```

### 3.3 `scripts/run_controlled_evaluation.py`
```python
    # Ensure failed or timed out inferences NEVER record 0
    is_gen_timed_out = bool(res.get("timed_out", False))
    has_gen_error = res.get("error") is not None
    if is_gen_timed_out or has_gen_error:
      gen_pred_val = None
    elif res.get("humorous") is True:
      gen_pred_val = 1
    elif res.get("humorous") is False:
      gen_pred_val = 0
    else:
      gen_pred_val = None

    # Exclude failed inferences from metric calculation masks
    gen_valid_mask = (
        res_df["general_prediction"].notna()
        & res_df["general_prediction"].isin([0, 1])
        & (~gen_timed_out)
        & res_df["general_json_error"].isna()
    )
    cult_valid_mask = (
        res_df["cultural_prediction"].notna()
        & res_df["cultural_prediction"].isin([0, 1])
        & (~cult_timed_out)
        & res_df["cultural_json_error"].isna()
    )
```

### 3.4 `src/vlm/inference.py`
```python
    # Enforce prediction Unavailable if timed out
    is_timed_out = diagnostics.get("timed_out", False) if diagnostics else False

    if (
        not is_timed_out
        and humor_prob is not None
        and non_humor_prob is not None
    ):
      ...
    else:
      vlm_humorous = result.get("humorous")
      if not is_timed_out and vlm_humorous is not None:
        result["humorous"] = bool(vlm_humorous)
        result["prediction"] = (
            "Humorous" if result["humorous"] else "Not Humorous"
        )
      else:
        result["humorous"] = None
        result["prediction"] = "Unavailable"
```

---

## 4. Inventory of Modified and Created Files

### Files Created
1. `src/evaluation/deduplication.py`: Pure PIL/NumPy zero-dependency deduplication library implementing 8x8 average hash, Hamming distance, normalized OCR text sequence matcher, and quarantine collision checker.
2. `tests/test_evaluation_safety.py`: Comprehensive test suite containing 11 tests verifying all six critical evaluation safety requirements.
3. `reports/heldout_evaluation_safety_fixes.md`: This comprehensive implementation and verification report.

### Files Modified
1. `run_robust_eval.py`: Set `TIMEOUT_S = 1500`, changed hung dummy record `prediction` from 0 to `None`, added atomic temporary file replacement for checkpoint persistence.
2. `src/evaluation/checkpoint_manager.py`: Added record completion validation (`is_record_complete`), valid prediction check (`is_record_valid_prediction`), valid pair tracking (`get_valid_completed_pairs`), configuration validation (`validate_configuration`), and enriched status reporting.
3. `scripts/run_controlled_evaluation.py`: Added configuration validation at initialization, enforced `prediction: None` on timeouts/errors, excluded failures from accuracy/F1 masks, and added failure tracking metrics to `evaluation_summary.json`.
4. `src/evaluation/evaluator.py`: Ensured `prediction_val` is set strictly to `None` on errors or timeouts rather than converting to 0.
5. `src/evaluation/prediction_store.py`: Added `load_valid_completed_ids()` method.
6. `src/vlm/inference.py`: Made generation timeout configurable from parameters, and forced `humorous = None` / `prediction = "Unavailable"` on timeouts in `_build_result()`.
7. `reports/heldout_evaluation_plan.md`: Corrected clean candidate count to 91, rebalanced `NH3` quota from 10 to 4 (reallocating to NH4/5/6), and updated sample timeout to 1,500s.

---

## 5. Remaining Risks & Operational Safeguards

1. **Subprocess Termination Latency**:
   While `subprocess.run(timeout=1500)` will kill a hung child process via Windows OS process termination, cleanup of large memory buffers (approx. 14GB RAM for Qwen2.5-VL-7B) on Windows may take 5–10 seconds. The 5-second sleep in the outer loop (`time.sleep(5)`) ensures clean memory reclamation before launching the next inference.
2. **Pre-Inference Manifest Audit Required**:
   When the user authorizes manifest creation, `scripts/create_heldout_manifest.py` must execute `src.evaluation.deduplication` across all 680 candidate memes against the 20 historical evaluation memes. Any candidate with an exact MD5 match, an aHash Hamming distance $d \le 3$, or normalized OCR similarity $> 85\%$ must be rejected from the candidate pool prior to finalizing `sample_manifest.json`.

---

## 6. Execution Safety & Audit Statement

- **VLM Inference**: Zero VLM forward passes and zero Qwen model instances were loaded or executed during this work.
- **Application Startup**: No application services, web servers, or long-running evaluation loops were started.
- **Model Weights & Prompts**: No model weights, classification prompts, decision thresholds, or dataset labels were altered.
- **Historical Reports & Results**: All prior reports (`reports/v4_*.md`) and evaluation directories (`results/evaluation_twostage_v4/`) remain preserved intact.
- **100-Sample Manifest**: The 100-sample held-out manifest was **not** created or finalized; candidate pools and quotas have been validated and safety-hardened, awaiting user authorization for manifest generation.
