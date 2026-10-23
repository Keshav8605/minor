# Forensic Investigation: DRISHTIKON Retrieval False Positive (Case 3a)

**Audit Date**: October 19, 2026  
**Investigation Mode**: Read-Only Offline Forensic Analysis (Zero VLM Inference, Zero Application Startup)  
**Target Case**: Case 3a (`Police imposed Section 144 curfew in Delhi during protests`)  
**Matched Record**: DRISHTIKON Record ID `1806` (*Delhi Metro Phase-IV Janakpuri-Krishna Park section*)  
**Reported Similarity**: `0.3133` (Configured Threshold: `0.2500`)  
**Audit Status**: Complete — Root Cause Verified Mathematically and Empirically  

---

## 1. Executive Summary & Forensic Verdict

In Case 3a of the end-to-end cultural context pipeline validation, an administrative public-order query regarding an IPC Section 144 curfew in Delhi unexpectedly matched a modern urban transit QA record in DRISHTIKON:
- **Input Text**: `"Police imposed Section 144 curfew in Delhi during protests"`
- **Retrieved Record**: `[Transport | Delhi] Which section of depicted Phase-IV was inaugurated in January 2025? -> Janakpuri-Krishna Park section` (ID: `1806`)
- **Cosine Similarity Score**: `0.3133` (exceeding the production threshold of `0.25`)

### Primary Forensic Findings

1. **Root Cause — Single-Token Lexical Collision via Corpus Sparsity**:
   The false positive is **not** caused by the legal number `144`, nor by the administrative terms `police` or `curfew`, nor by category detection augmentation. In the DRISHTIKON knowledge base of 64,288 records, **Record 1806 is the only record in the entire corpus that contains the standalone English token `"section"`** ($df=1$).
2. **Mathematical Breakdown of Score (0.3133)**:
   - The token `'section'` contributes **0.2930 (93.52%)** of the total cosine similarity score.
   - The token `'delhi'` contributes **0.0203 (6.48%)** of the total cosine similarity score.
   - Exactly two unigrams overlap between query and record. All other query tokens (`police`, `curfew`, `144`) are **Out-of-Vocabulary (OOV)** in DRISHTIKON ($df=0$).
3. **Appended Categories & Preprocessing Role**:
   - The category detector identified `['regional_culture']` (triggered by `"delhi"`). However, the string `'regional_culture'` is completely Out-of-Vocabulary (OOV) in DRISHTIKON and contributed **0.0000** to the score.
   - Query sanitization correctly preserved `"Section 144"` under its legal whitelist; raw and sanitized query scores are identically **0.3133**.
   - The score originates **100% from the base query's unigram overlap on `'section'` and `'delhi'`**.
4. **Generalization across Legal Sections**:
   Because `df('section') = 1` across all 64,288 documents, **any query containing the word `"Section"`** (e.g., *Section 302*, *Section 377*, *Section 420*, or *Section 144 without Delhi*) automatically scores between **0.2938 and 0.4038** against Delhi Metro Phase-IV, easily clearing the 0.25 threshold.
5. **Mitigation Recommendation**:
   Do **not** raise the global retrieval threshold (which would degrade legitimate cultural recalls scoring ~0.26). Implement a **Minimum Content-Token Overlap Constraint** ($\ge 2$ distinct non-state content unigrams if similarity $< 0.45$) and domain routing for statutory prefixes (`Section \d+`, `Article \d+`) to `PROJECT_JSON`.

---

## 2. Full Pipeline Query Reproduction & Step-by-Step State Tracking

The exact query was executed deterministically through each stage of the cultural context pipeline (`src/cultural/context_builder.py`, `src/cultural/category_detector.py`, `src/cultural/drishtikon_retriever.py`).

### 2.1 Query Transformation Stages

```
Raw Query:
"Police imposed Section 144 curfew in Delhi during protests"
  │
  ├── [1. Query Sanitization: sanitize_retrieval_query()]
  │     - Matches legal pattern: r'(?i)(\b(?:section|article|act|rule|no\.?|dhara)\s+)?\b(\d+)\b'
  │     - "Section 144" preserved intact; no generic meme boilerplate detected
  │     => Sanitized: "Police imposed Section 144 curfew in Delhi during protests"
  │
  ├── [2. Category Detection: detect_categories()]
  │     - Keyword "delhi" in CATEGORY_KEYWORDS['regional_culture'] matches
  │     => Detected Categories: ['regional_culture']
  │
  └── [3. ContextBuilder Query Construction: query = " ".join([sanitized_text] + clean_cats)]
        => Category-Augmented Query:
        "Police imposed Section 144 curfew in Delhi during protests regional_culture"
```

### 2.2 Comparative Top-5 Retrieval Ranking

Retrieval was evaluated with `min_similarity_threshold = 0.0` across all three query forms to trace the full candidate distribution:

| Rank | Metric / Query Form | Top Candidate (ID: 1806) | Candidate #2 (ID: 1783) | Candidate #3 (ID: 1742) | Candidate #4 (ID: 1754) | Candidate #5 (ID: 2435) |
|:---:|:---|:---|:---|:---|:---|:---|
| **#1** | **Raw Query** | **0.3133** *(Metro Phase-IV)* | 0.2436 *(Pol. Protests)* | 0.2189 *(Pol. Protests)* | 0.1932 *(Pol. Protests)* | 0.1097 *(Telangana)* |
| **#2** | **Sanitized Query** | **0.3133** *(Metro Phase-IV)* | 0.2436 *(Pol. Protests)* | 0.2189 *(Pol. Protests)* | 0.1932 *(Pol. Protests)* | 0.1097 *(Telangana)* |
| **#3** | **Category-Augmented** | **0.3133** *(Metro Phase-IV)* | 0.2436 *(Pol. Protests)* | 0.2189 *(Pol. Protests)* | 0.1932 *(Pol. Protests)* | 0.1097 *(Telangana)* |
| — | **Accepted @ 0.25?** | **YES (Threshold Exceeded)** | NO (0.2436 < 0.25) | NO (0.2189 < 0.25) | NO (0.1932 < 0.25) | NO (0.1097 < 0.25) |

> [!NOTE]
> All three query variations produce **identically ranked top-5 lists and identical scores to 4 decimal places**. The category augmentation string `"regional_culture"` has zero effect.

### 2.3 Matched DRISHTIKON Record Detail

- **Record ID**: `1806`
- **Attribute**: `Transport`
- **State**: `Delhi`
- **Language**: `English`
- **Question**: `"Which section of depicted Phase-IV was inaugurated in January 2025?"`
- **Options**:
  - Option 1: `"Janakpuri-Krishna Park section"` (Ground Truth Answer)
  - Option 2: `"Rithala-Kundli section"`
  - Option 3: `"Sahibabad-New Ashok Nagar section"`
  - Option 4: `"Dwarka-Najafgarh section"`
- **Answer**: `"Janakpuri-Krishna Park section"`
- **Image Link**: `Delhi_Metro_01.jpg`
- **Constructed `search_text`**:
  ```text
  "which section of depicted phase-iv was inaugurated in january 2025? janakpuri-krishna park section transport delhi english general question delhi_metro_01.jpg"
  ```

---

## 3. Mathematical Decomposition of the Similarity Score (0.3133)

DRISHTIKON retrieval uses `scikit-learn`'s `TfidfVectorizer(ngram_range=(1, 2), stop_words="english")` followed by cosine similarity. In this vectorizer, each document vector and query vector is normalized to unit Euclidean length ($\|v\|_2 = 1.0$), so the cosine similarity is the dot product $\sum (q_i \cdot d_i)$.

### 3.1 Vocabulary & Document Frequency Audit

Scrutiny of the 64,288 indexed DRISHTIKON records reveals an extreme vocabulary asymmetry:

| Query Token | In DRISHTIKON Vocab? | Document Frequency ($df$) | Inverse Document Frequency ($IDF$) | Notes |
|:---|:---:|:---:|:---:|:---|
| `'section'` | **True** | **1** | **11.3780** | **Unique in corpus (Record 1806 only)** |
| `'delhi'` | **True** | 1,489 | **4.2364** | Frequent geographic entity in corpus |
| `'protests'` | **True** | 5 | **9.8739** | Present in political protest records |
| `'police'` | **False (OOV)** | 0 | $0.0000$ | Not indexed in cultural corpus |
| `'curfew'` | **False (OOV)** | 0 | $0.0000$ | Not indexed in cultural corpus |
| `'144'` | **False (OOV)** | 0 | $0.0000$ | Not indexed in cultural corpus |
| `'regional_culture'` | **False (OOV)** | 0 | $0.0000$ | Category label not indexed |
| `'in'`, `'during'` | Filtered | — | — | Standard English stopwords |

> [!IMPORTANT]
> In `TfidfVectorizer(smooth_idf=True)`, the IDF formula is:
> $$\text{IDF}(t) = \ln\left(\frac{1 + N}{1 + df(t)}\right) + 1 = \ln\left(\frac{64289}{2}\right) + 1 = \ln(32144.5) + 1 \approx 11.3780$$
> An IDF of **11.3780** is the maximum possible IDF for any token in this corpus.

### 3.2 Term-by-Term Dot Product Calculation

For the query `"Police imposed Section 144 curfew in Delhi during protests"`:
1. The out-of-vocabulary terms (`police`, `curfew`, `144`) and stopwords (`in`, `during`) receive weight 0.
2. The remaining in-vocabulary unigrams are `{'section', 'delhi', 'protests'}`.
3. Their unnormalized TF-IDF weights in the query are:
   - $w_{\text{section}} = 1 \times 11.3780 = 11.3780$
   - $w_{\text{delhi}} = 1 \times 4.2364 = 4.2364$
   - $w_{\text{protests}} = 1 \times 9.8739 = 9.8739$
4. Query L2 norm: $\sqrt{11.3780^2 + 4.2364^2 + 9.8739^2} = \sqrt{129.4589 + 17.9471 + 97.4939} = \sqrt{244.8999} \approx 15.6493$.
5. Normalized query weights:
   - $Q_{\text{section}} = 11.3780 / 15.6493 = \mathbf{0.7271}$
   - $Q_{\text{delhi}} = 4.2364 / 15.6493 = \mathbf{0.2707}$
   - $Q_{\text{protests}} = 9.8739 / 15.6493 = \mathbf{0.6310}$

For Record 1806 (`search_text`):
- `'section'` appears twice (TF=2, sublinear TF=False $\to$ weight scales with IDF). Normalized $Doc_{\text{section}} = \mathbf{0.4030}$.
- `'delhi'` appears once. Normalized $Doc_{\text{delhi}} = \mathbf{0.0750}$.
- `'protests'` does not appear ($Doc_{\text{protests}} = 0.0000$).

Calculating the dot product:

$$\text{Cosine Similarity} = (Q_{\text{section}} \times Doc_{\text{section}}) + (Q_{\text{delhi}} \times Doc_{\text{delhi}}) + (Q_{\text{protests}} \times Doc_{\text{protests}})$$

$$\text{Cosine Similarity} = (0.7271 \times 0.4030) + (0.2707 \times 0.0750) + 0.0000$$

$$\text{Cosine Similarity} = 0.293021 + 0.020302 = \mathbf{0.313323} \approx \mathbf{0.3133}$$

| Overlapping Token | Query Weight ($Q$) | Document Weight ($Doc$) | Product Contribution | Score Fraction (%) |
|:---|:---:|:---:|:---:|:---:|
| **`section`** | 0.7271 | 0.4030 | **0.2930** | **93.52%** |
| **`delhi`** | 0.2707 | 0.0750 | **0.0203** | **6.48%** |
| **Total** | — | — | **0.3133** | **100.00%** |

---

## 4. Attribution of Retrieval Behavior

To address Task 3 (*"Determine whether the score came from the base query, appended category terms, preprocessing, or a different code path"*):

1. **Appended Category Terms (`regional_culture`)**:
   - Contribution: **0.0000 (0.0%)**.
   - The token `regional_culture` does not exist in DRISHTIKON's vocabulary. Its vector entry is empty.
2. **Sanitization Preprocessing**:
   - Effect: **Neutral (No change)**.
   - The legal number whitelist explicitly preserved `"Section 144"`. Preprocessing did not add or remove any content token.
3. **Base Query Lexical Overlap**:
   - Contribution: **0.3133 (100.0%)**.
   - The entire score is generated by lexical overlap between the English noun `"section"` (used in law as a statutory paragraph) and `"section"` (used in transit engineering as a railway segment).
4. **Code Path**:
   - The false positive occurs directly inside `DrishtikonRetriever.retrieve()`.
   - It is not an artifact of pipeline chaining, prompt formatting, or fallback logic.

---

## 5. Documented Negative & Positive Controls Suite

A structured suite of 14 controlled test queries was evaluated against `DrishtikonRetriever` to test the boundaries of this lexical collision:

| Control ID | Query Description | Raw Query Text | Top Retrieved Record ID & Title | Top Score | Accepted @ 0.25? | Matched Metro Phase-IV? |
|:---|:---|:---|:---|:---:|:---:|:---:|
| **CTRL_01** | Section 144 in Delhi (without protests) | `"Section 144 curfew in Delhi"` | `1806` [Transport\|Delhi] Metro Phase-IV Section | **0.4038** | **YES** | **YES** |
| **CTRL_02** | Section 144 curfew (without Delhi) | `"Section 144 curfew"` | `1806` [Transport\|Delhi] Metro Phase-IV Section | **0.4030** | **YES** | **YES** |
| **CTRL_03** | Curfew in Delhi during protests (no Section) | `"Curfew in Delhi during protests"` | `1783` [Common Sense\|Delhi] Political protests | **0.3548** | **YES** | **NO** |
| **CTRL_04** | Section 144 in Mumbai (different state) | `"Police imposed Section 144 in Mumbai"` | `1806` [Transport\|Delhi] Metro Phase-IV Section | **0.2938** | **YES** | **YES** |
| **CTRL_05** | Police curfew in Delhi (no Section) | `"Police imposed curfew in Delhi"` | `1807` [Cuisine\|Delhi] Delhi street cuisine | **0.2540** | **YES** | **NO** |
| **CTRL_06** | Dhara 144 curfew in Delhi (Hindi legal term) | `"Dhara 144 curfew in Delhi"` | `1807` [Cuisine\|Delhi] Delhi street cuisine | **0.0886** | **NO** | **NO** |
| **CTRL_07** | Curfew under 144 in Delhi (no "Section") | `"Curfew under 144 in Delhi"` | `1807` [Cuisine\|Delhi] Delhi street cuisine | **0.2540** | **YES** | **NO** |
| **CTRL_08** | **Target Delhi Metro query (Positive Control)** | `"Delhi Metro Phase-IV Janakpuri Krishna Park section"` | `1806` [Transport\|Delhi] Metro Phase-IV Section | **0.6318** | **YES** | **YES** |
| **CTRL_09** | Section 302 IPC murder in Delhi | `"Section 302 IPC murder in Delhi"` | `1806` [Transport\|Delhi] Metro Phase-IV Section | **0.4038** | **YES** | **YES** |
| **CTRL_10** | Section 377 IPC in Delhi | `"Section 377 IPC in Delhi"` | `1806` [Transport\|Delhi] Metro Phase-IV Section | **0.4038** | **YES** | **YES** |
| **CTRL_11** | Section 420 IPC fraud in Delhi | `"Section 420 IPC fraud in Delhi"` | `1806` [Transport\|Delhi] Metro Phase-IV Section | **0.4038** | **YES** | **YES** |
| **CTRL_12** | Article 370 in Delhi | `"Article 370 in Delhi"` | `2436` [History\|Jharkhand] PESA Article 243 | **0.1571** | **NO** | **NO** |
| **CTRL_13** | 144 people gathered in Delhi (144 as quantity) | `"144 people gathered in Delhi"` | `3249` [Common Sense\|Haryana] Spiritual rituals | **0.1111** | **NO** | **NO** |
| **CTRL_14** | 10,000 people gathered in Delhi (Numeric noise) | `"10,000 people gathered in Delhi"` | `3249` [Common Sense\|Haryana] Spiritual rituals | **0.1111** | **NO** | **NO** |

### Key Control Findings

1. **State Isolation (`CTRL_02`, `CTRL_04`)**:
   - In `CTRL_02` (`"Section 144 curfew"`), the query contains **no mention of Delhi**, yet scores **0.4030** against Delhi Metro Phase-IV. Because `curfew` and `144` are OOV, the query vector collapses 100% onto `'section'` ($Q_{\text{section}} = 1.0$), multiplying directly by $Doc_{\text{section}} = 0.4030$.
   - In `CTRL_04` (`"Police imposed Section 144 in Mumbai"`), the query specifies **Mumbai**, yet still retrieves the Delhi Metro record with score **0.2938**.
2. **Vocabulary Specificity (`CTRL_03`, `CTRL_06`, `CTRL_07`)**:
   - When `"Section"` is removed (`CTRL_03`: `"Curfew in Delhi during protests"`), the Metro Phase-IV record disappears from the top matches, and candidate `1783` (*Delhi political protests*) correctly becomes top-ranked (0.3548).
   - When the Hindi statutory equivalent `"Dhara"` is used (`CTRL_06`: `"Dhara 144 curfew in Delhi"`), the score drops to **0.0886**, cleanly rejecting DRISHTIKON.
3. **Cross-Statutory Contamination (`CTRL_09`, `CTRL_10`, `CTRL_11`)**:
   - Queries for IPC Sections 302 (murder), 377 (unnatural offenses), and 420 (cheating) all score **0.4038** against Delhi Metro Phase-IV.
   - Any Indian legal penal code query formatted as `"Section <N>"` triggers the Delhi Metro Phase-IV record.

---

## 6. Impact on Legitimate Cultural Queries & Vulnerability Analysis

To fulfill Task 5 (*"Check whether relevant cultural queries are affected by the same retrieval behavior"*), legitimate Indian cultural queries were tested across two categories:
1. Architectural queries that legitimately use the structural word *"section"*.
2. Genuine cultural queries involving Delhi or other Indian regions.

### 6.1 Cultural Queries Containing Structural Term *"Section"*

| Query | Intended Cultural Entity | Top Retrieved Record | Score | Accepted @ 0.25? | Impact Assessment |
|:---|:---|:---|:---:|:---:|:---|
| `"Inner section of Brihadisvara Temple"` | UNESCO Great Living Chola Temple (Tamil Nadu) | `1806` [Transport\|Delhi] Metro Phase-IV | **0.2608** | **YES** | **False Positive Collision**: A classical temple architecture query retrieves modern Delhi Metro rail transit. |
| `"Section of Konark Sun Temple architecture"` | UNESCO Sun Temple (Odisha) | `2492` [Language\|Odisha] Linguistic expression | **0.1877** | **NO** | Clean rejection, but Metro Phase-IV was runner-up (0.1702). |

### 6.2 Genuine Cultural Queries (Clean Entities)

| Query | Intended Cultural Domain | Top Retrieved Record | Score | Accepted @ 0.25? | Impact Assessment |
|:---|:---|:---|:---:|:---:|:---|
| `"Red Fort Delhi Mughal architecture"` | Delhi Historical Monument | `1711` [Rituals\|Delhi] Flag-hoisting ceremony | **0.3413** | **YES** | Correct Cultural Match (Unaffected by Metro 1806) |
| `"Qutub Minar Delhi monument"` | Delhi Historical Monument | `1781` [Tourism\|Delhi] Qutub area history | **0.3439** | **YES** | Correct Cultural Match (Unaffected by Metro 1806) |
| `"Chandni Chowk Delhi street food"` | Delhi Cuisine & Markets | `1807` [Cuisine\|Delhi] Chandni Chowk dessert | **0.3569** | **YES** | Correct Cultural Match (Unaffected by Metro 1806) |
| `"Garba dance in Gujarat"` | Gujarat Folk Dance | `3249` [Rituals\|Gujarat] Navratri festival artifact | **0.2599** | **YES** | Correct Cultural Match (Unaffected by Metro 1806) |
| `"Durga Puja in Kolkata West Bengal"` | Bengal Religious Festival | `433` [Rituals\|West Bengal] Durga Puja ritual | **0.3047** | **YES** | Correct Cultural Match (Unaffected by Metro 1806) |

### 6.3 Forensic Vulnerability Insight

- Clean cultural queries in Delhi (`Red Fort`, `Qutub Minar`, `Chandni Chowk`) are **not contaminated** by Record 1806. Because their domain unigrams (`qutub`, `chandni`, `chowk`, `fort`) have substantial TF-IDF representation, they match their intended records.
- The vulnerability is strictly confined to queries containing the token `"section"` where the remaining tokens are OOV or have weak TF-IDF weights, allowing the rare token's high IDF (11.3780) to dominate the normalized query vector.

---

## 7. Recommended Mitigations (Evaluation & Non-Implementation)

As instructed by the user constraints, **no mitigation code or configuration changes have been implemented**. Below is a forensic evaluation of candidate mitigations based on empirical evidence:

### 7.1 Evaluation of Candidate Mitigations

| Candidate Mitigation | Mechanism | Pros | Cons / Risks | Recommendation |
|:---|:---|:---|:---|:---:|
| **A. Raise Global Similarity Threshold (e.g. to 0.32 or 0.35)** | Increase `min_similarity_threshold` in `configs/cultural.yaml` from `0.25` to `0.35`. | Rejects Case 3a (0.3133). | **Ineffective & Harmful**: `CTRL_01`, `CTRL_02`, `CTRL_09`-`CTRL_11` score **0.4038** and would still pass. Meanwhile, valid cultural queries like *Garba dance* (0.2599) and *Brihadisvara Temple* (0.2608) would be discarded. | **REJECT** |
| **B. Add `"section"` to Default Stopwords** | Add `"section"` to custom stopword set in `DrishtikonRetriever`. | Immediately eliminates all `"section"` collisions. | **Ad-Hoc & Brittle**: Does not solve the general problem of $df=1$ tokens. May slightly reduce match fidelity on true transport queries like `CTRL_08`. | **NOT RECOMMENDED** |
| **C. Minimum Content-Token Overlap Constraint** | In `DrishtikonRetriever.retrieve()`, when score $< 0.45$, require $\ge 2$ distinct overlapping non-state content unigrams between query and record. | **Principled & Robust**: Case 3a has only 1 content unigram (`section`). `CTRL_02` has 1. `CTRL_09` has 1. Brihadisvara query has 1. All would be rejected. True Metro query (`CTRL_08`) has 5 overlapping unigrams and passes easily. | Requires a 4-line helper in `drishtikon_retriever.py` to count overlapping non-stopword tokens. | **RECOMMENDED (Primary)** |
| **D. Statutory Domain Routing in ContextBuilder** | In `CulturalContextBuilder`, if text matches legal section pattern `(?i)\b(?:section|dhara|article)\s+\d+`, route exclusively to `PROJECT_JSON` and bypass DRISHTIKON unless cultural keywords are also present. | Aligns with domain reality: DRISHTIKON is a cultural heritage QA dataset with 0 legal statutes. `PROJECT_JSON` handles Indian legal/slang terms (420, 370). | Bypasses DRISHTIKON for pure legal queries. | **RECOMMENDED (Secondary)** |

### 7.2 Specification of Recommended Primary Mitigation (For Future Reference)

When authorized for future implementation, the narrowest, most robust fix is the **Minimum Content-Token Overlap Constraint** in `src/cultural/drishtikon_retriever.py`:

```python
# Illustrative specification only — NOT IMPLEMENTED
def _count_content_overlap(query_tokens: set, doc_text: str) -> int:
    doc_tokens = set(re.findall(r'\b\w\w+\b', doc_text.lower())) - GENERIC_STATE_NAMES - STOPWORDS
    return len(query_tokens.intersection(doc_tokens))

# Inside retrieve():
# If record score < 0.45 and _count_content_overlap(query_content_tokens, doc_text) < 2:
#     continue  # reject single-word high-IDF spike
```

---

## 8. Audit Verification & Artifact Preservation Statement

### 8.1 Files Read During Audit
1. `reports/cultural_context_pipeline_validation.md` (Lines 120–155: Case 3a specifications and baseline results)
2. `reports/drishtikon_sanitization_annotation_audit.md` (Lines 64–75, 268–275: Annotation review of `C_NUM_01`)
3. `reports/drishtikon_sanitization_benchmark_expanded.md` (Lines 40–45, 200–210: Benchmark row `C_NUM_01`)
4. `src/cultural/drishtikon_retriever.py` (Lines 1–106: TF-IDF vectorizer configuration and retrieval logic)
5. `src/cultural/context_builder.py` (Lines 60–179: Query sanitization, category concatenation, and retrieval flow)
6. `src/cultural/category_detector.py` (Lines 1–99: Keyword dictionaries and regex detection logic)
7. `src/cultural/drishtikon_loader.py` (Lines 30–75: `search_text` construction and record schema)
8. `data/cultural/drishtikon/processed/drishtikon_knowledge.json` (Record ID `1806` search text and corpus vocabulary stats)
9. `data/cultural/cultural_knowledge.json` (Category definitions and baseline cultural slang)

### 8.2 Confirmation of Zero Modification
- **Source Code**: No files in `src/`, `scripts/`, or `tests/` were modified.
- **Configurations**: `configs/cultural.yaml` and `configs/model.yaml` remain unmodified.
- **Data & Indices**: Knowledge files, TF-IDF representations, and annotations remain unmodified.
- **Existing Reports**: All historical reports in `reports/` remain strictly preserved.
- **Inference**: Zero VLM calls and zero web services were launched.
