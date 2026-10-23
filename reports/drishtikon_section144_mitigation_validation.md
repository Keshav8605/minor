# DRISHTIKON Lexical Collision Mitigation: Implementation & Validation Report

**Audit Target**: Case 3a Single-Token Lexical Collision (*Section 144 Curfew matching Delhi Metro Phase-IV*)  
**Report File**: `reports/drishtikon_section144_mitigation_validation.md`  
**Implementation Status**: Successfully Implemented & Verified (34/34 Unit & Regression Tests Passing)  
**Execution Mode**: Offline Deterministic Retrieval Validation (Zero VLM Inference, Zero Web Startup)  

---

## 1. Executive Summary

In response to the forensic audit documented in `reports/drishtikon_section144_false_positive_audit.md`, a narrowly scoped, evidence-backed mitigation was implemented in `src/cultural/drishtikon_retriever.py` to eliminate single-token lexical collisions caused by corpus vocabulary sparsity (specifically the isolated token `"section"`, which appears in only 1 of 64,288 records with IDF 11.3780).

### Key Outcomes

1. **Complete Removal of Statutory False Positives**:
   - Case 3a (`"Police imposed Section 144 curfew in Delhi during protests"`): Score drops from **0.3133 to 0.0000** against Delhi Metro Phase-IV (Record `1806`). DRISHTIKON returns **0 spurious matches**.
   - All statutory controls (`Section 144`, `Section 302`, `Section 377`, `Section 420`) drop from **0.4038 to 0.0000** against Record `1806`.
   - The architectural query `"Inner section of Brihadisvara Temple"` drops from **0.2608 to 0.0000** against Record `1806`, eliminating the spurious temple-to-metro collision.
2. **100% Preservation of Legitimate Matches**:
   - Genuine Delhi Metro Phase-IV query (`"Delhi Metro Phase-IV Janakpuri Krishna Park section"`): Retains Record `1806` with score **0.6318**.
   - Delhi Political Protests query (`"Curfew in Delhi during protests"`): Retains Record `4152` (*Political protests*, score **0.3548**).
   - Cultural landmarks and festivals near the 0.25 threshold (`Qutub Minar` @ 0.3439, `Durga Puja` @ 0.3047, `Garba dance` @ 0.2599, `Kedarnath` @ 0.4002): **100% retained**.
3. **Seamless Fallback Execution**:
   - In `CulturalContextBuilder`, queries where DRISHTIKON matches are filtered (such as Section 144 curfew and Brihadisvara Temple) cleanly fall back to `PROJECT_JSON` without contaminating the prompt with Delhi Metro transit schedules.
4. **Comprehensive Test Suite Passing**:
   - **34 out of 34 tests passing** across 7 test suites (7 new regression tests + 27 existing tests). Zero regressions.

---

## 2. Phase 1 — Regression Test Suite Baseline

A dedicated regression test file [`tests/test_drishtikon_lexical_mitigation.py`](file:///c:/Users/prach/Downloads/Minor%20Project/minor/tests/test_drishtikon_lexical_mitigation.py) was established before code modification to codify the 7 requirements specified by the user:

| Test Case | Description | Pre-Mitigation Status | Post-Mitigation Status |
|:---|:---|:---:|:---:|
| `test_section144_does_not_retrieve_metro_record` | Case 3a query must not match Delhi Metro (1806) | **FAILED** (Matched 1806 @ 0.3133) | **PASSED** (0 matches) |
| `test_statutory_sections_do_not_retrieve_metro_record` | Sections 302, 377, 420, Mumbai 144 must not match 1806 | **FAILED** (Matched 1806 @ 0.4038) | **PASSED** (0 matches) |
| `test_legitimate_metro_query_retains_metro_record` | True Metro Phase-IV query retains Record 1806 | **PASSED** (Matched 1806 @ 0.6318) | **PASSED** (Matched 1806 @ 0.6318) |
| `test_curfew_in_delhi_during_protests_retains_protest_results` | Delhi protest query retains political protest records | **PASSED** (Matched 4152 @ 0.3548) | **PASSED** (Matched 4152 @ 0.3548) |
| `test_legitimate_cultural_queries_near_threshold_retained` | Qutub Minar, Durga Puja, Garba retained; Brihadisvara excludes 1806 | **FAILED** (Brihadisvara matched 1806 @ 0.2608) | **PASSED** (Clean retention & exclusion) |
| `test_existing_sanitization_and_numeric_collision_rules` | Legal whitelist intact, 10,000 numeric noise stripped & rejected | **PASSED** (Clean sanitization) | **PASSED** (Clean sanitization) |
| `test_project_json_fallback_functional_when_drishtikon_empty` | ContextBuilder cleanly routes to PROJECT_JSON without 1806 | **FAILED** (1806 injected into context) | **PASSED** (Clean PROJECT_JSON context) |

*Pre-mitigation baseline*: **4 failed, 3 passed** in 26.79s.  
*Post-mitigation verified*: **7 passed, 0 failed** in 16.11s.

---

## 3. Phase 2 — Architectural Inspection & Minimal Mitigation Design

### 3.1 Pre-Implementation Inspection of Token Availability

Before implementing the overlap rule, we evaluated the prompt's proposed rule:
> *"for weak matches below 0.45, require at least two distinct meaningful, non-state content-token overlaps. Do not count generic structural words such as 'section' as meaningful content tokens."*

Our inspection identified two critical edge cases:
1. **Document-Token Availability in Unit Test Fixtures**:
   - In `tests/fixtures/drishtikon_sample.json`, queries like `"temples kedarnath pilgrimage"` and `"mandir kedarnath puja temple pilgrimage"` were written to match across multiple sample documents. However, individual sample documents in the fixture only contain **one** matching keyword (e.g. Record 0 only contains `"temples"`, Record 3 only contains `"kedarnath"`).
   - A naive requirement of $\ge 2$ content-token overlaps for any query below 0.45 causes existing unit tests (`test_retriever_successful_query` and `test_builder_both_sources_match`) to fail.
2. **Corpus Vocabulary Gaps (`Curfew in Delhi during protests`)**:
   - In `"Curfew in Delhi during protests"`, the word `"curfew"` is Out-of-Vocabulary (OOV) in DRISHTIKON (appears 0 times in 64,288 records). The word `"delhi"` is a state name. Thus, the query contains only **one** in-vocabulary non-state content token: `"protests"`.
   - As a result, the relevant candidate record (`Record 4152`, *Political protests in Delhi*) can only overlap on **one** content token (`protests`) plus **one** state token (`delhi`).
   - Demanding $\ge 2$ non-state content tokens would indiscriminately drop this legitimate match, directly violating Requirement 4.

### 3.2 Implemented Mitigation Architecture

The mitigation in [`src/cultural/drishtikon_retriever.py`](file:///c:/Users/prach/Downloads/Minor%20Project/minor/src/cultural/drishtikon_retriever.py) resolves this by implementing a **structural-aware content-token constraint**:

1. **Definition of Token Partitions**:
   - `GENERIC_STRUCTURAL = {"section", "depicted", "image", "question", "answer", "option", "general"}`
   - `INDIAN_STATES = {"delhi", "haryana", "punjab", "uttarakhand", "bihar", "gujarat", "maharashtra", "karnataka", "tamil_nadu", "tamil", "nadu", "west_bengal", "bengal", "kerala", "rajasthan", "odisha", "jharkhand", "telangana", "uttar_pradesh", "madhya_pradesh", "assam", "sikkim", "puducherry", "chandigarh", "goa", "india", "indian"}`
   - `CONTENT_TOKENS = ALL_TOKENS - ENGLISH_STOP_WORDS - INDIAN_STATES - GENERIC_STRUCTURAL`
2. **Validation Logic for Matches Below 0.45**:
   - If the query contains a generic structural collision token (e.g. `"section"`):
     Require **multi-token evidence**: at least **two** distinct non-state content tokens (`len(content_overlap) >= 2`), OR at least **one** non-state content token combined with state agreement (`len(content_overlap) >= 1 and len(state_overlap) >= 1`).
   - For standard cultural queries (no structural collision word):
     Require at least **one** meaningful non-state, non-structural content token (`len(content_overlap) >= 1`).
   - High-confidence matches ($\ge 0.45$) remain unpenalized.
   - Global similarity threshold remains unchanged at **0.25**.

```python
# Content-token overlap constraint in DrishtikonRetriever.retrieve()
if score < 0.45:
    doc_search_text = record.get("search_text", "").lower()
    doc_content = _extract_content_tokens(doc_search_text)
    content_overlap = query_content.intersection(doc_content)

    doc_states = _extract_state_tokens(doc_search_text)
    state_overlap = query_states.intersection(doc_states)

    if query_has_structural:
        has_sufficient_overlap = (len(content_overlap) >= 2) or (
            len(content_overlap) >= 1 and len(state_overlap) >= 1
        )
    else:
        has_sufficient_overlap = len(content_overlap) >= 1

    if not has_sufficient_overlap:
        continue
```

---

## 4. Phase 3 — Empirical Validation & Before/After Comparison

### 4.1 Before vs. After Controls Comparison Table

All 14 controls from the forensic audit suite plus Case 3a and cultural probes were evaluated on the mitigated retriever:

| ID | Query Description | Pre-Mitigation Score | Pre-Mitigation Metro Match (1806)? | Post-Mitigation Score | Post-Mitigation Metro Match (1806)? | Top Match Post-Mitigation | Verdict |
|:---|:---|:---:|:---:|:---:|:---:|:---|:---:|
| **CASE_3A** | Section 144 curfew Delhi during protests | **0.3133** | **YES** | **0.0000** | **NO** | *None (0 matches)* | **Spurious Match Eliminated** |
| **CTRL_01** | Section 144 curfew in Delhi | **0.4038** | **YES** | **0.0000** | **NO** | *None (0 matches)* | **Spurious Match Eliminated** |
| **CTRL_02** | Section 144 curfew (no Delhi) | **0.4030** | **YES** | **0.0000** | **NO** | *None (0 matches)* | **Spurious Match Eliminated** |
| **CTRL_03** | Curfew in Delhi during protests (no Section) | **0.3548** | **NO** | **0.3548** | **NO** | `[Common Sense\|Delhi]` ID 4152 (Political protests) | **Legitimate Match Retained** |
| **CTRL_04** | Section 144 in Mumbai (different state) | **0.2938** | **YES** | **0.0000** | **NO** | *None (0 matches)* | **Spurious Match Eliminated** |
| **CTRL_05** | Police curfew in Delhi (no Section) | **0.2540** | **NO** | **0.0000** | **NO** | *None (Spurious cuisine match dropped)* | **Noise Suppressed** |
| **CTRL_06** | Dhara 144 curfew in Delhi (Hindi legal) | **0.0886** | **NO** | **0.0000** | **NO** | *None (0 matches)* | **Clean Rejection** |
| **CTRL_07** | Curfew under 144 in Delhi | **0.2540** | **NO** | **0.0000** | **NO** | *None (Spurious cuisine match dropped)* | **Noise Suppressed** |
| **CTRL_08** | **Delhi Metro Phase-IV Janakpuri Krishna Park** | **0.6318** | **YES** | **0.6318** | **YES** | `[Transport\|Delhi]` ID 1806 (Janakpuri section) | **Legitimate Metro Retained** |
| **CTRL_09** | Section 302 IPC murder in Delhi | **0.4038** | **YES** | **0.0000** | **NO** | *None (0 matches)* | **Spurious Match Eliminated** |
| **CTRL_10** | Section 377 IPC in Delhi | **0.4038** | **YES** | **0.0000** | **NO** | *None (0 matches)* | **Spurious Match Eliminated** |
| **CTRL_11** | Section 420 IPC fraud in Delhi | **0.4038** | **YES** | **0.0000** | **NO** | *None (0 matches)* | **Spurious Match Eliminated** |
| **CTRL_12** | Article 370 in Delhi | **0.1571** | **NO** | **0.0000** | **NO** | *None (0 matches)* | **Clean Rejection** |
| **CTRL_13** | 144 people gathered in Delhi | **0.1111** | **NO** | **0.0000** | **NO** | *None (0 matches)* | **Clean Rejection** |
| **CTRL_14** | 10,000 people gathered in Delhi | **0.1111** | **NO** | **0.0000** | **NO** | *None (0 matches)* | **Clean Rejection** |
| **CULT_01** | Inner section of Brihadisvara Temple | **0.2608** | **YES** | **0.0000** | **NO** | *None (0 matches in DRISHTIKON)* | **Spurious Metro Match Eliminated** |
| **CULT_02** | Qutub Minar Delhi monument | **0.3439** | **NO** | **0.3439** | **NO** | `[Tourism\|Delhi]` ID 1781 (Qutub area) | **Legitimate Cultural Match Retained** |
| **CULT_03** | Durga Puja in Kolkata West Bengal | **0.3047** | **NO** | **0.3047** | **NO** | `[Rituals\|West Bengal]` ID 433 (Durga ceremony) | **Legitimate Cultural Match Retained** |
| **CULT_04** | Garba dance in Gujarat | **0.2599** | **NO** | **0.2599** | **NO** | `[Rituals\|Gujarat]` ID 3249 (Navratri Garba) | **Legitimate Cultural Match Retained** |

---

## 5. Offline Test Suite Execution Results

All offline tests were executed via pytest across the active Python environment:

```text
tests/test_drishtikon_lexical_mitigation.py:
  test_section144_does_not_retrieve_metro_record ....................... PASSED [ 14%]
  test_statutory_sections_do_not_retrieve_metro_record ................. PASSED [ 28%]
  test_legitimate_metro_query_retains_metro_record ..................... PASSED [ 42%]
  test_curfew_in_delhi_during_protests_retains_protest_results ......... PASSED [ 57%]
  test_legitimate_cultural_queries_near_threshold_retained ............. PASSED [ 71%]
  test_existing_sanitization_and_numeric_collision_rules ............... PASSED [ 85%]
  test_project_json_fallback_functional_when_drishtikon_empty .......... PASSED [100%]
  ========================= 7 passed in 16.11s =========================

Existing Cultural & Retrieval Test Suites:
  tests/test_retrieval_precision.py (6 items) .......................... PASSED [ 22%]
  tests/test_drishtikon_retriever.py (4 items) ......................... PASSED [ 37%]
  tests/test_drishtikon_loader.py (3 items) ............................ PASSED [ 48%]
  tests/test_context_builder.py (4 items) .............................. PASSED [ 62%]
  tests/test_query_sanitization.py (7 items) ........................... PASSED [ 88%]
  tests/test_combined_cultural_pipeline.py (3 items) ................... PASSED [100%]
  ========================= 27 passed in 21.22s =========================
```

**Grand Total**: **34 passed, 0 failed, 0 skipped**.

---

## 6. Safety, Integrity, & Remaining Risks

### 6.1 Integrity Statement
- **Zero VLM Inference**: No multimodal models or external APIs were called.
- **Zero Application Startup**: No web servers or Streamlit apps were started.
- **Configuration Invariance**: `configs/cultural.yaml` and `configs/model.yaml` remain unmodified. Global threshold remains `0.25`.
- **Preservation of Evaluation Artifacts**: Historical directories (`results/evaluation_twostage_v4/`, etc.) and all historical markdown reports remain strictly untouched.
- **Files Changed**:
  - Modified: [`src/cultural/drishtikon_retriever.py`](file:///c:/Users/prach/Downloads/Minor%20Project/minor/src/cultural/drishtikon_retriever.py) (Added content-token overlap filter)
  - Created: [`tests/test_drishtikon_lexical_mitigation.py`](file:///c:/Users/prach/Downloads/Minor%20Project/minor/tests/test_drishtikon_lexical_mitigation.py) (Regression test suite)

### 6.2 Remaining Risks & Future Work
1. **Corpus Coverage Limitations**:
   - Certain statutory terms (e.g. `curfew`, `police`) and specific transliterations (e.g. `Brihadisvara` vs `Brihadeeswara`) are absent from DRISHTIKON. The existing `CulturalContextBuilder` fallback to `PROJECT_JSON` handles these gracefully, but expanding DRISHTIKON's synonym map or lemmatizer remains a future enhancement opportunity.
2. **Multi-Word Legal Phrases**:
   - The mitigation specifically neutralizes generic structural words. If future queries introduce other rare non-cultural vocabulary items that have $df=1$ in DRISHTIKON, the structural token set can be extended, or statutory prefixes (`Section \d+`, `Article \d+`) can be formally bypassed at the router level in `ContextBuilder`.
