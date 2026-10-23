# In-Depth Error Analysis & Failure Pattern Decomposition: Drishtikon `v4` Evaluation

**Primary Data Sources**:
- `results/evaluation_twostage_v4/comparison_results.csv`
- `results/evaluation_twostage_v4/inference_results.jsonl`
- `results/evaluation_twostage_v4/evaluation_summary.json`
- `reports/v4_threshold_sweep_verified.md`

**Status**: Strictly Read-Only Audit (Zero VLM inference executed, zero source/config/label modifications)

---

## 1. Independent Verification of Ground Truth, Predictions, and Confusion Matrices

All 11 evaluation memes from `results/evaluation_twostage_v4/comparison_results.csv` were verified against ground truth labels in `data/processed/test.csv`. The evaluation subset comprises **5 Humorous memes ($\text{GT}=1$)** and **6 Non-Humorous memes ($\text{GT}=0$)**.

### Comprehensive Prediction & Probability Audit Table

| Index | Image Filename | GT Label | General Pred ($\tau=0.50$) | General $P(\text{Humour})$ | Cultural Pred ($\tau=0.50$) | Cultural $P(\text{Humour})$ | Logit Shift ($\Delta P$) | Stage-2 Gating Path | Classification Outcome |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **0** | `train_6346.jpg` | 1 | 1 | 0.8354 | 1 | 0.7309 | -0.1045 | Stage-2 Invoked (762.2s) | **True Positive (TP)** |
| **1** | `train_5723.jpg` | 0 | 1 | 0.7279 | 1 | 0.5272 | -0.2007 | Stage-2 Invoked (674.8s) | **False Positive (FP)** |
| **2** | `train_5266.jpg` | 0 | 0 | 0.2146 | 0 | 0.2146 | +0.0000 | Short-Circuit (0.62s) | **True Negative (TN)** |
| **3** | `train_4199.jpg` | 1 | 1 | 0.8856 | 1 | 0.8856 | +0.0000 | Short-Circuit (0.55s) | **True Positive (TP)** |
| **4** | `train_4136.jpg` | 1 | 1 | 0.7879 | 1 | 0.5812 | -0.2067 | Stage-2 Invoked (1118.6s)| **True Positive (TP)** |
| **5** | `train_3395.jpg` | 1 | 1 | 0.6653 | 1 | 0.6653 | +0.0000 | Short-Circuit (0.97s) | **True Positive (TP)** |
| **6** | `train_1611.jpg` | 0 | 1 | 0.8175 | 1 | 0.8175 | +0.0000 | Short-Circuit (0.29s) | **False Positive (FP)** |
| **7** | `train_5890.jpg` | 1 | 0 | 0.4609 | 0 | 0.4686 | +0.0077 | Stage-2 Invoked (620.8s) | **False Negative (FN)** |
| **8** | `train_3280.jpg` | 0 | 1 | 0.8991 | 1 | 0.8991 | +0.0000 | Short-Circuit (0.29s) | **False Positive (FP)** |
| **9** | `train_127.jpg`  | 0 | 1 | 0.5888 | 1 | 0.5888 | +0.0000 | Short-Circuit (0.23s) | **False Positive (FP)** |
| **10**| `train_5088.jpg` | 0 | 1 | 0.5659 | 1 | 0.5659 | +0.0000 | Short-Circuit (0.24s) | **False Positive (FP)** |

### Verified Confusion Matrix ($N=11$, Threshold $\tau=0.5000$)

#### General Mode
$$\begin{pmatrix} \text{TN}=1 & \text{FP}=5 \\ \text{FN}=1 & \text{TP}=4 \end{pmatrix}$$
- Accuracy: **0.4545** (5/11)
- Humour Precision: **0.4444** (4/9), Humour Recall: **0.8000** (4/5), Humour F1: **0.5714**
- Non-Humour Precision: **0.5000** (1/2), Non-Humour Recall: **0.1667** (1/6), Non-Humour F1: **0.2500**

#### Cultural-Aware Mode
$$\begin{pmatrix} \text{TN}=1 & \text{FP}=5 \\ \text{FN}=1 & \text{TP}=4 \end{pmatrix}$$
- Accuracy: **0.4545** (5/11)
- Humour Precision: **0.4444** (4/9), Humour Recall: **0.8000** (4/5), Humour F1: **0.5714**
- Non-Humour Precision: **0.5000** (1/2), Non-Humour Recall: **0.1667** (1/6), Non-Humour F1: **0.2500**

---

## 2. Granular Error Case Studies

The evaluation produced **5 False Positives** and **1 False Negative** across both conditions. Each error is examined below with full evidence from image content, extracted text, VLM reasoning, retrieved context, and dataset annotation characteristics.

---

### Case 1: `train_5723.jpg` (False Positive at $\tau=0.50$)
- **Ground Truth**: `0` (Non-Humorous)
- **Predictions**: General = `1` ($P = 0.7279$) | Cultural = `1` ($P = 0.5272$)
- **OCR / Caption**:
  ```text
  #TheFamilyMan #OnlyExcusesOctober
  When Wife Starts To Tell About Her Parents Instead of about Her Sister #SalliJi
  Le Husband: main baad me phone karta hoon
  ```
- **Visual Evidence**: Stills from the web series *The Family Man* showing protagonist Srikant Tiwari (Manoj Bajpayee) holding a mobile phone to his ear with an uncomfortable, distracted expression.
- **Model Reasoning**:
  - *General Mode*: Identified relatable marital situational comedy: a husband pretending to have poor signal or calling later to escape an uninteresting family discussion.
  - *Cultural Mode*: Identified the cultural trope of husband-in-law dynamics and marital avoidance. Stated: *"The man's response is a play on words that creates a humorous situation... The cultural context of Indian family dynamics adds depth to the humor, but the core comedic element lies in the wordplay and situational irony."*
- **Cultural Context Retrieved**:
  - Source: `PROJECT_JSON` | Category: `relationships` / `family`
  - Content: *Indian family dynamics, strict parenting, arranged marriage biodatas, nosy relatives asking 'shaadi kab karoge', and sibling/in-law banter.*
- **Failure Mechanism**: **Threshold Proximity**. Cultural reasoning correctly suppressed false humorous over-confidence by **-0.2007** (from 0.7279 down to 0.5272). However, because the operational threshold was pinned at $0.5000$, $0.5272$ remained a False Positive by a margin of only $0.0272$. At any threshold $\tau \ge 0.550$, this sample converts to a True Negative.
- **Ground-Truth Ambiguity Assessment**: **Moderate**. In Hindi pop-culture, *The Family Man* domestic memes are widely perceived as comedic. A human annotator marking this $0$ may have regarded it as mundane promotional dialogue rather than an active joke.

---

### Case 2: `train_5890.jpg` (False Negative)
- **Ground Truth**: `1` (Humorous)
- **Predictions**: General = `0` ($P = 0.4609$) | Cultural = `0` ($P = 0.4686$)
- **OCR / Caption**:
  ```text
  PM Modi Speech Live:
  Narendra Modi Says India Will Be Under Complete Lockdown From Midnight For 21 Days
  Nibbi *Nibba
  BABY HUM OR 21 DIN NAHI MIL SAKTE
  Indya Speak
  ```
- **Visual Evidence**: Split panel graphic. Top half: Official live news broadcast chyron announcing the nationwide 21-day COVID-19 lockdown. Bottom half: A melodramatic crying kitten image macro labeled *"Nibbi: BABY HUM OR 21 DIN NAHI MIL SAKTE"* (Baby, we cannot meet for 21 days).
- **Model Reasoning**:
  - *General Mode*: *"The text appears to be a humorous meme, but there is no clear comedic mechanism... the content is not intended to amuse. Therefore, it is classified as NOT HUMOROUS."*
  - *Cultural Mode*: *"The image contains a serious political announcement about a complete lockdown in India, which is a significant event... Therefore, the image is classified as not humorous due to the presence of a serious political announcement and the humorous cat meme."*
- **Cultural Context Retrieved**:
  - Source: `PROJECT_JSON` | Category: `internet_culture` / `relationships`
  - Content: *Conservative societal attitudes towards dating, sneaking calls, 'nibba-nibbi' cringe teenage romance tropes, dramatic couple separation laments ('milna / nahi mil sakte' during lockdowns).*
- **Failure Mechanism**: **Negative Heuristic Collision in Prompt Rules**. The prompt contains an explicit negative constraint: *"Do NOT mistake serious/informational political announcements for humor."* The VLM gave priority to the official news banner over the satirical contrast of the teenage couple reaction underneath. Even though Stage-2 gating recovered the cultural category (`relationships`), the model’s probability only shifted slightly ($0.4609 \to 0.4686$), remaining below $0.5000$.
- **Ground-Truth Ambiguity Assessment**: **Low**. This is an unambiguous satirical reaction meme contrasting national crisis announcements with trivial teenage relationship laments. The $\text{GT}=1$ label is well-founded.

---

### Case 3: `train_3280.jpg` (False Positive)
- **Ground Truth**: `0` (Non-Humorous)
- **Predictions**: General = `1` ($P = 0.8991$) | Cultural = `1` ($P = 0.8991$)
- **OCR / Caption**:
  ```text
  Do you support pedopihila
  Reddit Admin: Yes
  You're hired
  Cyanide and Happiness Explosm.net
  ```
- **Visual Evidence**: Minimalist 3-panel stick-figure comic strip from *Cyanide & Happiness* showing a job applicant entering an office and answering an absurd interview question.
- **Model Reasoning**: *"The comic strip uses absurdity and irony to create humor. The absurdity lies in the hiring of someone who supports pedophilia, while the irony is in the fact that the admin is entering the room... making the content humorous."*
- **Cultural Context Retrieved**: None (Short-circuited in 0.29s; recognized as non-Indian cultural context).
- **Failure Mechanism**: **Sentiment / Content-Moderation Confounding in Dataset Labels**. *Cyanide & Happiness* is a globally established dark comedy webcomic. The model analyzed the structural comedic mechanism (dark irony and taboo absurdity), yielding $P = 0.8991$. In crowdsourced datasets like Memotion 3, human annotators frequently mark offensive, taboo, or disturbing content as $\text{GT}=0$, conflating social acceptability with the presence of comedic intent.
- **Ground-Truth Ambiguity Assessment**: **High**. Structurally, the comic is explicitly constructed as a satirical joke. However, under safety and sensitive-content annotation guidelines, it was labeled non-humorous. The model's classification reflects comedic form rather than annotator sentiment.

---

### Case 4: `train_127.jpg` (False Positive at $\tau < 0.60$)
- **Ground Truth**: `0` (Non-Humorous)
- **Predictions**: General = `1` ($P = 0.5888$) | Cultural = `1` ($P = 0.5888$)
- **OCR / Caption**:
  ```text
  I DONT KNOW WHO YOU ARE
  AB KI BAAR MODI SARKAAR
  buffedd
  ```
- **Visual Evidence**: Close-up of actor Liam Neeson holding a flip phone in the film *Taken*, referencing the iconic monologue ("I don't know who you are... I will look for you").
- **Model Reasoning**: *"The image contains an absurdity in the form of a play on words that creates an unexpected and humorous situation."*
- **Cultural Context Retrieved**: None (Short-circuited in 0.23s).
- **Failure Mechanism**: **Template Over-Weighting on Low-Quality Political Slogans**. The model identified the *Taken* phone monologue template (universally associated with humor macros), assigning $P = 0.5888$. However, the punchline is a generic political campaign slogan (*"Ab Ki Baar Modi Sarkar"*), rendering it low-effort political spam rather than a well-formed joke. At $\tau \ge 0.600$, this sample correctly transitions to a True Negative.
- **Ground-Truth Ambiguity Assessment**: **Moderate**. A viewer may perceive this as a lighthearted template mashup (humorous) or as low-effort political slogan spam (non-humorous).

---

### Case 5: `train_1611.jpg` (False Positive)
- **Ground Truth**: `0` (Non-Humorous)
- **Predictions**: General = `1` ($P = 0.8175$) | Cultural = `1` ($P = 0.8175$)
- **OCR / Caption**:
  ```text
  Me and the boys after earning 10,000 for reporting fake Whatsapp videos
  amc
  ```
- **Visual Evidence**: A man in dark sunglasses and leather jacket posing boastfully before a backdrop completely covered with thick stacks of Indian rupee banknotes.
- **Model Reasoning**: *"The image uses visual comedy to create an exaggerated and humorous scenario, making it clear that the content is intended to be funny."*
- **Cultural Context Retrieved**: None (Short-circuited in 0.29s).
- **Failure Mechanism**: **Annotator Mislabeling / Inconsistency in Memotion 3**. The visual composition (swaggering in front of millions of rupees) paired with the minor reward (₹10,000 for reporting fake news) is a textbook hyperbole joke using the viral *"Me and the boys"* meme archetype. The model's detection of visual comedy ($P = 0.8175$) is semantically coherent.
- **Ground-Truth Ambiguity Assessment**: **High**. This image possesses all standard properties of internet meme humor. The $\text{GT}=0$ annotation is highly idiosyncratic and likely represents crowdsourcing noise.

---

### Case 6: `train_5088.jpg` (False Positive at $\tau < 0.575$)
- **Ground Truth**: `0` (Non-Humorous)
- **Predictions**: General = `1` ($P = 0.5659$) | Cultural = `1` ($P = 0.5659$)
- **OCR / Caption**:
  ```text
  THIS is not COURAGE. THIS is COURAGE.
  Backup: @gayretardmemes.v3
  eu farage nigel nigelfarage brexit trump donald trump farage ukip...
  ```
- **Visual Evidence**: Two-panel political graphic contrasting a politician pointing sternly with a picture of a man crying.
- **Model Reasoning**: *"The image uses visual comedy through the contrast between the man pointing and the man crying, which is intended to be humorous."*
- **Cultural Context Retrieved**: None (Short-circuited in 0.24s).
- **Failure Mechanism**: **Weak Visual Contrast Interpretation**. The model attributed humorous visual comedy to a partisan meme that was actually intended as serious political propaganda or partisan mockery. Because the model's confidence was weak ($P = 0.5659$), setting $\tau \ge 0.575$ immediately converts this to a True Negative.
- **Ground-Truth Ambiguity Assessment**: **Moderate**. Highly ideological content where perception of humor depends entirely on the viewer's political allegiance.

---

## 3. Recurring Failure Patterns & Ranked Solution Matrix

Based on the 6 error investigations, failures group into three primary categories:

```
[Failure Distribution]
├── 50% Annotation / Confounding Bias (train_1611, train_3280, train_127)
├── 33% Decision Boundary Rigidity (train_5723, train_5088)
└── 17% Prompt Rule Negative Heuristic Collision (train_5890)
```

### Ranked Proposed Interventions

| Rank | Proposed Fix | Target Error Pattern | Expected Impact | Implementation Risk | Operational Cost |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | **Calibrated Decision Thresholding ($\tau \in [0.550, 0.575]$)** | Converts borderline False Positives (`train_5723`, `train_5088`) into True Negatives while preserving True Positives (`train_6346`, `train_4199`, `train_4136`, `train_3395`). | **High**: Increases accuracy from $45.5\%$ to $63.6\%$ on this split; unlocks $+9.1\%$ cultural advantage. | **Zero**: Offline post-processing calculation only. | Zero compute (reuses saved probabilities). |
| **2** | **Satire Juxtaposition Prompt Refinement** | Resolves False Negative on `train_5890.jpg` by clarifying that serious news headlines juxtaposed with unrelated meme characters (cats, teenage slang) constitute satire. | **High**: Directly fixes prompt-induced False Negatives on political satire memes. | **Low**: Modifies prompt instructions without altering model architecture. | Requires VLM re-inference on affected samples. |
| **3** | **Pre-Tokenization Query Sanitization for DRISHTIKON** | Strips hashtags (`#TheFamilyMan`), punctuation (`*Nibba`), and OCR noise before TF-IDF vectorization to increase DRISHTIKON hit rate (currently 0/11 hits). | **Moderate**: Enhances external DRISHTIKON knowledge recall beyond Project JSON. | **Low**: Purely algorithmic string pre-processing in `src/cultural/drishtikon_retriever.py`. | Zero VLM compute; fast offline test. |
| **4** | **Label Noise Consensus Filtering / Soft Targets** | Mitigates distortion from annotator subjectivity and taboo censorship (`train_1611`, `train_3280`) by flagging ambiguous labels or evaluating with soft Brier scores. | **High**: Eliminates benchmark distortion caused by crowdsourced annotation errors. | **Moderate**: Requires multi-annotator review of benchmark test splits. | Zero VLM compute. |

---

## 4. Scope & Limitations of the 11-Sample Evaluation

### What the 11-Sample Evaluation Establishes
1. **Execution Pipeline Robustness**: Zero crashes, zero timeouts, zero JSON parsing errors, and zero fallback degradations across 22 consecutive VLM inference attempts.
2. **Two-Stage Gating Efficiency**: Correctly routed 4 culturally-dependent memes to Stage-2 while short-circuiting 7 universal/non-cultural memes in $<1\text{s}$, delivering a **48.7% latency reduction** ($289.1\text{s}$ vs $563.9\text{s}$).
3. **Latent Calibration Direction**: Injected cultural knowledge reliably moves logit probabilities in the theoretically expected direction (e.g., dropping false positive confidence by $0.2007$ on `train_5723.jpg`, and lowering mean non-humorous confidence from $0.6356$ to $0.6022$).

### What the 11-Sample Evaluation CANNOT Establish
1. **Statistical Significance**: With $N=11$, every individual prediction change alters accuracy by $\pm 9.09\%$. A 1-sample divergence cannot prove generalizable superiority.
2. **Generalizable Threshold Optimization**: Finding that $\tau = 0.575$ maximizes accuracy on these 11 memes is an in-sample artifact; applying it as a global constant risks overfitting.
3. **Sub-Dialect Diversity**: 11 memes cannot represent the wide linguistic variety of Hindi/Hinglish internet culture (e.g., Bhojpuri, Punjabi-influenced slang, South Indian English memes).

### Proposed Larger Held-Out Validation Protocol

To validate these findings with statistical rigor without excessive inference cost:

```
[Proposed Validation Protocol: N = 100 Memes]
├── Stratified Sampling:
│   ├── 50 Humorous (GT=1) vs 50 Non-Humorous (GT=0)
│   └── 30 High Cultural Dependency, 40 Medium, 30 Universal/Low
├── Multi-Metric Assessment:
│   ├── ROC-AUC & PR-AUC (threshold-independent discrimination)
│   ├── Brier Score / Expected Calibration Error (probability calibration)
│   └── Macro-F1 at out-of-fold calibrated thresholds
└── 5-Fold Cross-Validation:
    └── Fit optimal decision threshold on 4 folds, evaluate on held-out 5th fold
```

---

## 5. Recommended Zero-Inference Offline Experiments

Before launching any new CPU-intensive VLM inference runs, the following analytical experiments can be executed immediately using existing saved artifacts:

1. **ROC-AUC and Precision-Recall Curve Generation**:
   - Compute receiver operating characteristic (ROC) curves and area under the curve (AUC) directly from the saved probabilities in `results/evaluation_twostage_v4/comparison_results.csv`. This provides a threshold-free metric of whether Cultural mode ranks humorous memes higher than non-humorous memes compared to General mode.
2. **Expected Calibration Error (ECE) & Reliability Diagrams**:
   - Partition predicted probabilities into confidence bins $[0.0, 0.2), [0.2, 0.4), \dots, [0.8, 1.0]$ and plot empirical accuracy versus mean predicted probability for both modes to quantify calibration improvement mathematically.
3. **Historical Cross-Run Fleiss' Kappa / Agreement Analysis**:
   - Calculate inter-run prediction stability across `evaluation_twostage`, `evaluation_twostage_refined`, `evaluation_twostage_final`, and `evaluation_twostage_v4` to quantify which samples have stabilized and which remain volatile.
4. **Offline DRISHTIKON Tokenizer Benchmark**:
   - Run unit tests on the DRISHTIKON TF-IDF retriever using alternative regex tokenizers and query sanitization routines against the 64,288 indexed documents, benchmarking hit-rates on the 11 meme captions in $<5$ seconds without invoking Qwen.

---

## 6. Verification of Read-Only Constraint & Integrity

- **Files Read**:
  - `results/evaluation_twostage_v4/comparison_results.csv`
  - `results/evaluation_twostage_v4/inference_results.jsonl`
  - `results/evaluation_twostage_v4/evaluation_summary.json`
  - `reports/v4_threshold_sweep_verified.md`
  - `data/processed/test.csv`
- **Integrity Confirmation**:
  - Zero VLM inference was launched.
  - Zero application source code, prompts, configurations, thresholds, or labels were modified.
  - All historical evaluation directories (`evaluation_twostage/`, `evaluation_twostage_refined/`, `evaluation_twostage_final/`, `evaluation_twostage_v4/`) and previous reports remain completely untouched.
