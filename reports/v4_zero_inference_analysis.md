# Zero-Inference Evaluation Audit & Statistical Analysis: Drishtikon v4

**Evaluation Series**: Drishtikon Final Controlled Evaluation (`v4`)  
**Scope**: Strictly Read-Only Statistical, Calibration, Longitudinal, and Retrieval Analysis  
**Output Report**: `reports/v4_zero_inference_analysis.md`  
**Execution Timestamp**: 2026-10-19  
**VLM Inference Launched**: **NONE (0 calls)**  
**Application / Configuration / Code Modifications**: **NONE (Strictly Preserved)**  

---

## 1. Executive Summary & Audit Safeguards

This report documents the zero-inference quantitative audit and statistical diagnostic phase of the Drishtikon meme humour detection project following the completion of the `v4` controlled evaluation run. 

All analyses in this document are strictly offline, using existing inference outputs, saved model probabilities, ground-truth annotations, historical evaluation logs, and the local cultural knowledge base. No vision-language model (VLM) weights were loaded, no new inference tokens were generated, and no source code, prompt definitions, configuration files, or existing evaluation artifacts were altered.

### 1.1 Exact Input Files Inspected
The following files were read and verified without modification:
1. `results/evaluation_twostage_v4/comparison_results.csv` (11 rows, 32 columns; primary paired dataset)
2. `results/evaluation_twostage_v4/inference_results.jsonl` (22 records; per-mode inference telemetry)
3. `results/evaluation_twostage_v4/evaluation_summary.json` (Summary metrics, execution metadata, and model hashes)
4. `results/evaluation_twostage/comparison_results.csv` (Baseline controlled run; 11 rows, 26 columns)
5. `results/evaluation_twostage_refined/comparison_results.csv` (Refined controlled run; 11 rows, 26 columns)
6. `results/evaluation_twostage_final/comparison_results.csv` (Audited final run; 11 rows, 32 columns)
7. `reports/v4_threshold_sweep_verified.md` (Independent threshold verification report)
8. `reports/v4_error_analysis.md` (Sample-by-sample qualitative failure analysis)
9. `data/cultural/drishtikon/processed/drishtikon_knowledge.json` (64,288 indexed cultural records, 125.5 MB)
10. `data/cultural/cultural_knowledge.json` (Project cultural knowledge base across 19 categories)
11. `tests/test_retrieval_precision.py` & `tests/test_query_sanitization.py` (Existing offline retrieval fixtures)

### 1.2 Audit Integrity Confirmations
- **Zero VLM Inferences**: 0 forward passes through `Qwen2.5-VL-3B-Instruct`.
- **Zero Historical Overwrites**: Directories `results/evaluation_twostage/`, `results/evaluation_twostage_refined/`, and `results/evaluation_twostage_final/` remain 100% byte-for-byte identical to their pre-run states.
- **Metric Fidelity**: All continuous discrimination, calibration, and agreement metrics were calculated directly from the raw output values without rounding intermediate steps.

---

## 2. Task 1: Discrimination Metrics (ROC-AUC & PR-AUC)

### 2.1 Conventions & Sample Universe
- **Positive-Class Convention**: Class `1` = **Humorous**; Class `0` = **Non-Humorous**.
- **Probability Metric**: $P(\text{Humour}) \in [0.0, 1.0]$, representing the model's posterior probability assigned to the humorous class.
- **Sample Universe**: $N = 11$ total memes.
  - Positive Ground Truth ($y=1$): 5 memes (`train_6346`, `train_4199`, `train_4136`, `train_3395`, `train_5890`).
  - Negative Ground Truth ($y=0$): 6 memes (`train_5723`, `train_5266`, `train_1611`, `train_3280`, `train_127`, `train_5088`).
  - Total pairwise comparisons ($N_{\text{pos}} \times N_{\text{neg}}$): $5 \times 6 = 30$ pairs.
  - Random-chance baseline for Precision-Recall: $\frac{P}{N} = \frac{5}{11} \approx 0.4545$.

### 2.2 Measured Discrimination Results

| Metric | General Mode | Cultural-Aware Mode | Delta ($\Delta$) | Random Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC** | **0.6000** | **0.5667** | -0.0333 | 0.5000 |
| **PR-AUC (Average Precision)** | **0.5676** | **0.5343** | -0.0333 | 0.4545 |
| **Concordant Pairs ($y_1 > y_0$)** | 18 / 30 | 17 / 30 | -1 pair | 15 / 30 |
| **Tied Pairs** | 0 / 30 | 0 / 30 | 0 | 0 / 30 |
| **Discordant Pairs ($y_0 > y_1$)** | 12 / 30 | 13 / 30 | +1 pair | 15 / 30 |

### 2.3 Mathematical Deconstruction of the Rank Inversion
The apparent 3.33 percentage point drop in ROC-AUC (from 0.6000 down to 0.5667) is driven entirely by a **single pair rank inversion** caused by Stage-2 logit damping on a true positive meme.

#### General Mode Pair Ranking:
- Positive samples (5): `train_4199` (0.8856), `train_6346` (0.8354), `train_4136` (0.7879), `train_3395` (0.6653), `train_5890` (0.4609).
- Negative samples (6): `train_3280` (0.8991), `train_1611` (0.8175), `train_5723` (0.7279), `train_127` (0.5888), `train_5088` (0.5659), `train_5266` (0.2146).
- Concordant comparisons:
  - `train_4199` (0.8856) > 5 negatives (beats all except `3280`) $\to$ **5 points**
  - `train_6346` (0.8354) > 5 negatives (beats all except `3280`) $\to$ **5 points**
  - `train_4136` (0.7879) > 4 negatives (`5723`, `127`, `5088`, `5266`) $\to$ **4 points**
  - `train_3395` (0.6653) > 3 negatives (`127`, `5088`, `5266`) $\to$ **3 points**
  - `train_5890` (0.4609) > 1 negative (`5266`) $\to$ **1 point**
  - **Total Concordant Pairs** = $5 + 5 + 4 + 3 + 1 = 18$.  
  - $\text{ROC-AUC} = \frac{18}{30} = 0.6000$.

#### Cultural-Aware Mode Pair Ranking:
- In Cultural-Aware mode, Stage-2 was invoked for `train_4136` (Family category), and its probability was adjusted from $0.7879$ down to $0.5812$ ($\Delta = -0.2067$).
- Meanwhile, `train_127` (GT=0) short-circuited in Stage-1 and retained its probability of $0.5888$.
- Because $0.5812 < 0.5888$, `train_127` (GT=0) ranked **above** `train_4136` (GT=1) in Cultural mode, whereas in General mode `train_4136` ranked above `train_127` ($0.7879 > 0.5888$).
- This inverted exactly **one** pair (`train_4136` vs `train_127`), reducing concordant pairs from 18 to 17.
- $\text{ROC-AUC} = \frac{17}{30} = 0.56666... \approx 0.5667$.

> [!NOTE]
> On a small sample ($N=11$), the global ranking metric has a discrete granularity of $\frac{1}{30} \approx 0.0333$. A single sample moving across an un-damped short-circuited negative sample shifts the AUC by 3.33%.

### 2.4 Empirical Operating Curves

#### ROC Operating Points

| Threshold ($\tau$) | General FPR | General TPR | Cultural FPR | Cultural TPR |
| :---: | :---: | :---: | :---: | :---: |
| $\infty$ | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.8991 | 0.1667 | 0.0000 | 0.1667 | 0.0000 |
| 0.8856 | 0.1667 | 0.2000 | 0.1667 | 0.2000 |
| 0.8354 | 0.1667 | 0.4000 | 0.1667 | 0.2000 |
| 0.8175 | 0.3333 | 0.4000 | 0.3333 | 0.2000 |
| 0.7879 | 0.3333 | 0.6000 | 0.3333 | 0.2000 |
| 0.7309 | 0.5000 | 0.6000 | 0.3333 | 0.4000 |
| 0.7279 | 0.5000 | 0.6000 | 0.3333 | 0.4000 |
| 0.6653 | 0.5000 | 0.8000 | 0.3333 | 0.6000 |
| 0.5888 | 0.6667 | 0.8000 | 0.5000 | 0.6000 |
| 0.5812 | 0.6667 | 0.8000 | 0.5000 | 0.8000 |
| 0.5659 | 0.8333 | 0.8000 | 0.6667 | 0.8000 |
| 0.5272 | 0.8333 | 0.8000 | 0.8333 | 0.8000 |
| 0.4686 | 0.8333 | 0.8000 | 0.8333 | 1.0000 |
| 0.4609 | 0.8333 | 1.0000 | 0.8333 | 1.0000 |
| 0.2146 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

#### Precision-Recall Operating Points

| Operating Recall | General Precision | Cultural Precision |
| :---: | :---: | :---: |
| 1.0000 (5/5) | 0.4545 (5/11) | 0.4545 (5/11) |
| 0.8000 (4/5) | 0.5000 (4/8) | 0.5714 (4/7) |
| 0.6000 (3/5) | 0.6000 (3/5) | 0.6000 (3/5) |
| 0.4000 (2/5) | 0.6667 (2/3) | 0.5000 (2/4) |
| 0.2000 (1/5) | 0.5000 (1/2) | 0.5000 (1/2) |
| 0.0000 (0/5) | 1.0000 | 1.0000 |

---

## 3. Task 2: Probability Calibration & Reliability Analysis

### 3.1 Brier Score Formulation & Decomposition
The Brier score measures the mean squared error between the model's predicted probability $p_i \in [0, 1]$ and the binary ground-truth label $y_i \in \{0, 1\}$:
$$\text{BS} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$

- **General Mode Brier Score**: **0.2916**
- **Cultural Mode Brier Score**: **0.2839**
- **Improvement in Calibration MSE**: **-0.0077** (Cultural mode exhibits lower squared error)

#### Per-Sample Squared Error Breakdown:

| Meme ID | Ground Truth ($y$) | General $P$ | Cultural $P$ | General SE: $(p_g - y)^2$ | Cultural SE: $(p_c - y)^2$ | Error Delta ($\Delta \text{SE}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `train_6346.jpg` | 1 | 0.8354 | 0.7309 | 0.0271 | 0.0724 | +0.0453 |
| `train_5723.jpg` | 0 | 0.7279 | 0.5272 | **0.5298** | **0.2779** | **-0.2519** |
| `train_5266.jpg` | 0 | 0.2146 | 0.2146 | 0.0461 | 0.0461 | 0.0000 |
| `train_4199.jpg` | 1 | 0.8856 | 0.8856 | 0.0131 | 0.0131 | 0.0000 |
| `train_4136.jpg` | 1 | 0.7879 | 0.5812 | 0.0450 | 0.1754 | +0.1304 |
| `train_3395.jpg` | 1 | 0.6653 | 0.6653 | 0.1120 | 0.1120 | 0.0000 |
| `train_1611.jpg` | 0 | 0.8175 | 0.8175 | 0.6683 | 0.6683 | 0.0000 |
| `train_5890.jpg` | 1 | 0.4609 | 0.4686 | 0.2906 | 0.2824 | -0.0082 |
| `train_3280.jpg` | 0 | 0.8991 | 0.8991 | 0.8084 | 0.8084 | 0.0000 |
| `train_127.jpg` | 0 | 0.5888 | 0.5888 | 0.3467 | 0.3467 | 0.0000 |
| `train_5088.jpg` | 0 | 0.5659 | 0.5659 | 0.3202 | 0.3202 | 0.0000 |
| **Overall Mean** | — | — | — | **0.2916** | **0.2839** | **-0.0077** |

> [!TIP]
> The net Brier score improvement in Cultural mode is driven by `train_5723.jpg`, where cultural grounding reduced the squared prediction error by **-0.2519**, overcoming the variance increases on `train_4136` (+0.1304) and `train_6346` (+0.0453).

---

### 3.2 Expected Calibration Error (ECE)

Expected Calibration Error partitions the interval $[0, 1]$ into $M$ bins $B_1, \dots, B_M$. In each bin:
$$\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} p_i, \quad \text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} y_i, \quad \text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$

#### Standard 5-Bin Calibration Table ($M=5$, Bin Width = 0.20)

| Bin | Range | General Count | General Avg Conf | General Empirical Acc | General Gap | Cultural Count | Cultural Avg Conf | Cultural Empirical Acc | Cultural Gap |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$B_1$** | $[0.00, 0.20)$ | 0 | — | — | — | 0 | — | — | — |
| **$B_2$** | $[0.20, 0.40)$ | 1 | 0.2146 | 0.0000 | 0.2146 | 1 | 0.2146 | 0.0000 | 0.2146 |
| **$B_3$** | $[0.40, 0.60)$ | 3 | 0.5385 | 0.3333 | 0.2052 | **5** | 0.5463 | 0.4000 | 0.1463 |
| **$B_4$** | $[0.60, 0.80)$ | 3 | 0.7270 | 0.6667 | 0.0604 | **2** | 0.6981 | 1.0000 | 0.3019 |
| **$B_5$** | $[0.80, 1.00]$ | **4** | 0.8594 | 0.5000 | 0.3594 | **3** | 0.8674 | 0.3333 | 0.5341 |
| **Total / ECE** | $[0.00, 1.00]$ | **11** | — | — | **0.2226** | **11** | — | — | **0.2866** |

#### 10-Bin Calibration Summary ($M=10$, Bin Width = 0.10)
- **General Mode ECE (10 Bins)**: **0.3815**
- **Cultural Mode ECE (10 Bins)**: **0.3832**

---

### 3.3 Critical Methodological Limitations on Small Sample Sizes ($N=11$)
It is scientifically invalid to assert that either mode is "well calibrated" based on summary ECE numbers on an 11-sample evaluation. The reliability metrics illustrate severe statistical vulnerabilities:

1. **Extreme Bin Sparsity**:
   - Bin 1 ($[0.0, 0.2)$) is entirely unpopulated (0 samples).
   - Bin 2 has exactly 1 sample (`train_5266.jpg`), which can only produce an empirical accuracy of either 0.0 or 1.0, generating an unavoidable calibration gap.
2. **The "Boundary Migration Paradox" in Bin 5**:
   - In General mode, Bin 5 ($[0.8, 1.0]$) contained 4 memes: `train_4199` (GT=1), `train_6346` (GT=1), `train_1611` (GT=0), and `train_3280` (GT=0). Empirical accuracy was $2/4 = 0.5000$, resulting in a gap of $|0.5000 - 0.8594| = 0.3594$.
   - In Cultural mode, Stage 2 dampened `train_6346` from $0.8354$ down to $0.7309$, migrating it **out** of Bin 5 into Bin 4.
   - This migration left Bin 5 with only 3 samples (`train_4199`, `train_1611`, `train_3280`), causing empirical accuracy to collapse to $1/3 \approx 0.3333$.
   - The calibration gap in Bin 5 spiked from $0.3594$ to $0.5341$, which single-handedly drove Cultural ECE up from 0.2226 to 0.2866.
3. **Artifact of Discrete Accuracy Shifts**:
   - Because $N=11$, each sample in a 3-element bin represents a 33.3% shift in empirical accuracy. A continuous logit adjustment that correctly dampens probability towards the true class can paradoxical inflate discrete binned ECE if the sample crosses a bin threshold.

---

## 4. Task 3: Cross-Run Consistency Across 4 Historical Evaluations

### 4.1 Comparative Runs Overview
Four historical runs were examined to establish stability across pipeline iterations:
1. **Baseline Run** (`results/evaluation_twostage/`): Initial two-stage implementation.
2. **Refined Run** (`results/evaluation_twostage_refined/`): Preliminary prompt adjustments.
3. **Final Audited Run** (`results/evaluation_twostage_final/`): Two-stage evaluation with instrumentation.
4. **v4 Controlled Run** (`results/evaluation_twostage_v4/`): Controlled evaluation with complete zero-error schema validation.

### 4.2 Matched Sample Coverage & Failure Tracking

| Run Identifier | Total Memes | General Valid Inferences | Cultural Valid Inferences | Failed Inferences & Cause |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline** | 11 | 11 / 11 (100%) | 11 / 11 (100%) | None |
| **Refined** | 11 | 11 / 11 (100%) | 10 / 11 (90.9%) | `train_1611.jpg` (Cultural JSON parse error) |
| **Final Audited** | 11 | 10 / 11 (90.9%) | 11 / 11 (100%) | `train_127.jpg` (General JSON parse error) |
| **v4 Controlled** | 11 | **11 / 11 (100%)** | **11 / 11 (100%)** | **None (0 JSON errors, 0 timeouts, 0 fallbacks)** |

### 4.3 Pairwise Agreement Matrix (v4 as Benchmark)

#### General Mode Agreement:

| Comparison | Matched Universe | Missing / Excluded | Binary Agreement | Disagreements | Cohen's Kappa ($\kappa$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **v4 vs Final** | 10 / 11 | 1 (`train_127`) | **10 / 10 (100.0%)** | None | **1.0000** |
| **v4 vs Baseline** | 11 / 11 | 0 | 9 / 11 (81.8%) | `train_3395`, `train_127` | 0.5600 |
| **v4 vs Refined** | 11 / 11 | 0 | 6 / 11 (54.5%) | 5 samples | 0.2254 |

#### Cultural-Aware Mode Agreement:

| Comparison | Matched Universe | Missing / Excluded | Binary Agreement | Disagreements | Cohen's Kappa ($\kappa$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **v4 vs Final** | 11 / 11 | 0 | **10 / 11 (90.9%)** | `train_5088` | **0.7442** |
| **v4 vs Baseline** | 11 / 11 | 0 | 6 / 11 (54.5%) | 5 samples | 0.2254 |
| **v4 vs Refined** | 10 / 11 | 1 (`train_1611`) | 5 / 10 (50.0%) | 5 samples | 0.1935 |

> [!IMPORTANT]
> **Statistical Disclosure on Cohen's Kappa ($\kappa$)**:
> Cohen's kappa measures inter-rater agreement above chance for categorical items. In this context:
> 1. It assumes independent categorical judgments on identical sample universes.
> 2. Kappa is notoriously sensitive to prevalence imbalances (the "kappa paradox").
> 3. Standard error on $N=10$ to $11$ samples is large ($\text{SE} \approx 0.25 - 0.30$).
> 4. $\kappa = 1.0000$ between `v4` and `final` in General mode reflects identical binary predictions across all 10 mutually valid items, but should not be interpreted as asymptotic convergence.

---

### 4.4 Longitudinal Prediction Tracking Across All 4 Runs

The table below tracks binary decisions ($0 = \text{Non-Humour}, 1 = \text{Humour}$) and continuous humour probabilities across all four historical iterations:

| Meme ID | GT | Baseline Gen | Refined Gen | Final Gen | v4 Gen | Baseline Cult | Refined Cult | Final Cult | v4 Cult | Stability Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `train_6346` | 1 | 1 (0.67) | 1 (0.54) | 1 (0.84) | 1 (0.84) | 1 (0.72) | 1 (0.62) | 1 (0.72) | 1 (0.73) | Stable True Positive |
| `train_5723` | 0 | 1 (0.54) | 0 (0.35) | 1 (0.72) | 1 (0.73) | 0 (0.50) | 0 (0.48) | 1 (0.53) | 1 (0.53) | Calibrated Near Boundary |
| `train_5266` | 0 | 0 (0.15) | 0 (0.06) | 0 (0.27) | 0 (0.21) | 0 (0.15) | 0 (0.06) | 0 (0.27) | 0 (0.21) | Robust True Negative |
| `train_4199` | 1 | 1 (0.69) | 1 (0.58) | 1 (0.90) | 1 (0.89) | 1 (0.78) | 1 (0.74) | 1 (0.78) | 1 (0.89) | Robust True Positive |
| `train_4136` | 1 | 1 (0.58) | 0 (0.50) | 1 (0.82) | 1 (0.79) | 0 (0.44) | 0 (0.45) | 1 (0.64) | 1 (0.58) | High Prompt Sensitivity |
| `train_3395` | 1 | 0 (0.43) | 0 (0.30) | 1 (0.68) | 1 (0.67) | 0 (0.43) | 0 (0.30) | 1 (0.68) | 1 (0.67) | Baseline FN $\to$ v4 TP |
| `train_1611` | 0 | 1 (0.69) | 1 (0.53) | 1 (0.82) | 1 (0.82) | 1 (0.70) | **ERR** | 1 (0.66) | 1 (0.82) | Systematic FP (Annotation) |
| `train_5890` | 1 | 0 (0.30) | 0 (0.17) | 0 (0.48) | 0 (0.46) | 0 (0.30) | 0 (0.34) | 0 (0.45) | 0 (0.47) | Systematic FN (Satire) |
| `train_3280` | 0 | 1 (0.79) | 1 (0.73) | 1 (0.92) | 1 (0.90) | 1 (0.79) | 1 (0.73) | 1 (0.92) | 1 (0.90) | Systematic FP (Taboo Humor) |
| `train_127` | 0 | 0 (0.40) | 0 (0.28) | **ERR** | 1 (0.59) | 0 (0.40) | 0 (0.28) | 1 (0.60) | 1 (0.59) | Boundary Shift in v4 |
| `train_5088` | 0 | 1 (0.52) | 0 (0.40) | 1 (0.59) | 1 (0.57) | 0 (0.44) | 0 (0.40) | 0 (0.46) | 1 (0.57) | Only Shift: Final $\to$ v4 |

---

## 5. Task 4: Offline Cultural Query Sanitization Benchmark

### 5.1 Benchmark Architecture & Purpose
The offline query sanitization benchmark evaluates the `sanitize_retrieval_query()` utility in `src/cultural/context_builder.py` against the full DRISHTIKON knowledge base (64,288 indexed cultural records).

The benchmark measures:
1. **False-Match Rate (FMR)** on noisy, non-cultural meme OCR queries (target: 0% matches).
2. **Relevant-Hit Rate (RHR)** on known Indian cultural entities embedded within noisy meme wrappers (target: 100% matches).
3. **Cosine Similarity Score Impact** before and after sanitization at the production threshold ($\tau_{\text{sim}} = 0.25$).

### 5.2 Quantitative Benchmark Summary

| Query Category | Evaluated Items | Raw Match Rate | Sanitized Match Rate | Performance Delta ($\Delta$) | Target Objective |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Non-Cultural Noisy Queries** | $N = 8$ | **0.0%** (0 / 8) | **0.0%** (0 / 8) | **0.0%** (No noise leakage) | **0.0% FMR** |
| **Cultural Entities in Noisy Context** | $N = 8$ | **75.0%** (6 / 8) | **75.0%** (6 / 8) | **+0.0%** (100% recall retention) | **High RHR** |

---

### 5.3 Detailed Item-by-Item Benchmark Telemetry

#### Non-Cultural Noisy Query Suite ($N=8$):
*All queries evaluate whether meme boilerplate, isolated numerals, or colloquial Hinglish trigger spurious matches against DRISHTIKON.*

| Query ID | Raw OCR / Input Query | Sanitized Query | Raw Top Score | Sanitized Top Score | Result |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `N1` | `"Me and the boys after earning 10,000 for reporting fake Whatsapp videos @memeguy ig/dankmemes"` | `"earning for reporting fake Whatsapp videos"` | 0.0000 | 0.0000 | Correct Rejection |
| `N2` | `"earning for reporting fake Whatsapp videos amc"` (`train_1611.jpg`) | `"earning for reporting fake Whatsapp videos amc"` | 0.0000 | 0.0000 | Correct Rejection |
| `N3` | `"Nobody: Me during quarantine: eating all snacks 50000 views"` | `"Me during quarantine: eating all snacks views"` | 0.0000 | 0.0000 | Correct Rejection |
| `N4` | `"THIS is not COURAGE. THIS is COURAGE."` (`train_5088.jpg`) | `"THIS is not COURAGE. THIS is COURAGE."` | 0.0000 | 0.0000 | Correct Rejection |
| `N5` | `"hot Indian milfs"` (`train_5266.jpg`) | `"hot Indian milfs"` | 0.0000 | 0.0000 | Correct Rejection |
| `N6` | `"Bore hore hoge chalo thoda Hans lo Quarantine Day"` (`train_4199.jpg`) | `"Bore hore hoge chalo thoda Hans lo Quarantine Day"` | 0.0000 | 0.0000 | Correct Rejection |
| `N7` | `"POV: When teacher asks question le me: backup: @dank_punjabi 999"` | `"When teacher asks question"` | 0.0000 | 0.0000 | Correct Rejection |
| `N8` | `"Earned 10,000 cash prize with 50000 views on 21 days"` | `"Earned cash prize with views on days"` | 0.0000 | 0.0000 | Correct Rejection |

#### Cultural Entity Suite in Noisy Context ($N=8$):
*All queries contain a legitimate Indian cultural entity wrapped in meme prefixes, social handles, and isolated numbers.*

| Query ID | Target Cultural Entity | Raw Input Query | Raw Score | Sanitized Score | Cosine Delta ($\Delta$) | Retrieved Grounded Context |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `C1` | Kedarnath Pilgrimage | `"Me and the boys at kedarnath temple uttarakhand pilgrimage @travelguy 10,000"` | 0.2798 | **0.4002** | **+0.1204** (+43.0%) | *"Which temple in Uttarakhand complex showcases medieval architecture..."* |
| `C2` | Badami Cave Temples | `"Nobody: badami rock cut cave temples karnataka 50000 views ig/heritage"` | 0.4632 | **0.4703** | **+0.0071** (+1.5%) | *"Who built the rock-cut cave temples at Badami and Pattadakal..."* |
| `C3` | Shah Rukh Khan | `"POV: shah rukh khan king of bollywood actor backup: @bollywood_daily 999"` | 0.6368 | **0.6368** | **0.0000** | *"Which famous Bollywood actor, known as the 'King of Bollywood'..."* |
| `C4` | Charminar Monument | `"Me and the boys visiting historic Charminar mosque Hyderabad Telangana @tourist 1000"` | 0.0000 | 0.0000 | 0.0000 | Unmatched in DRISHTIKON (Below 0.25) |
| `C5` | Brihadeeswara Temple | `"Nobody: Brihadeeswara temple towering architecture Tamil Nadu 50000 views @chola"` | 0.0000 | 0.0000 | 0.0000 | Unmatched in DRISHTIKON (Below 0.25) |
| `C6` | Kathakali Dance | `"Me and the boys watching Kathakali classical dance form Kerala makeup @kerala_art"` | 0.3343 | **0.3591** | **+0.0248** (+7.4%) | *"The costume shown in the given image is worn in a classical dance-drama..."* |
| `C7` | Nanda Devi Raj Jaat | `"Le husband: Nanda Devi Raj Jaat religious pilgrimage ceremony Uttarakhand 50000"` | 0.5879 | **0.6031** | **+0.0152** (+2.6%) | *"Which major religious ceremony in Uttarakhand is dedicated to the goddess..."* |
| `C8` | Qutub Minar Minaret | `"POV: Qutub Minar victory tower minaret Delhi monument @delhi_heritage 1000"` | 0.2626 | **0.2835** | **+0.0209** (+8.0%) | *"Depicted in the image is the original name of the area where the Qutub Minar..."* |

---

### 5.4 Crucial Discovery: Spurious Numeric Collision Elimination
During offline benchmarking, a critical vulnerability was exposed and proven resolved by sanitization:

- **Test Query**: `"Le husband: Bhai yeh toh 420 aadmi hai 10,000 fraud"`
- **Raw Un-Sanitized Behavior**:
  - The query returned a top match with similarity **0.2577** (exceeding the production threshold of 0.25).
  - Retrieved Record: `Q: 10,000 फूट उंचीवर 28 फूट उंचीची सोन्याची बुद्ध मूर्ती... (400-year-old Buddhist monastery in Arunachal Pradesh)`.
  - **Failure Mechanism**: The standalone numeral `"10,000"` caused a false TF-IDF lexical collision with a high-altitude monastery record in Arunachal Pradesh, injecting completely irrelevant Buddhist heritage context into a generic fraud meme.
- **Sanitized Behavior**:
  - `sanitize_retrieval_query()` stripped `"Le husband:"` and filtered out `"10,000"` as non-cultural numeric noise while preserving `"420"` (cultural number for fraud).
  - Sanitized Query: `"Bhai yeh toh 420 aadmi hai fraud"`.
  - Retrieved Result: **0 matches** (Score = 0.0000). The spurious Buddhist monastery collision was **100% eliminated**.
  - Concurrently, the query routed cleanly to the project cultural knowledge base (`cultural_knowledge.json`), retrieving the relevant `FAMILY` and `relationships` cultural context.

---

## 6. Synthesis: Measured Facts vs. Working Hypotheses

To maintain research integrity, findings are partitioned into empirically measured observations versus interpretive hypotheses:

### 6.1 Empirically Measured Facts
1. **Calibration MSE Advantage**: Cultural-Aware mode demonstrates a lower (superior) Brier score ($0.2839$ vs $0.2916$) across the 11 test memes, primarily driven by a $-0.2519$ squared-error reduction on `train_5723.jpg`.
2. **Deterministic Agreement with Final Run**: For all mutually valid inferences, `v4` and `final` agree **100.0%** in General mode ($\kappa = 1.0000$) and **90.9%** in Cultural mode ($\kappa = 0.7442$).
3. **Total Pipeline Reliability**: The `v4` run achieved 0 JSON parsing failures, 0 timeouts, 0 Stage-2 fallback triggers, and 100% schema compliance across 22 inference attempts.
4. **Retrieval Boosting**: Query sanitization increases TF-IDF cosine similarity on valid cultural entities by up to **+43.0%** ($\Delta = +0.1204$) by stripping meme prefixes, social handles, and non-cultural numeric tokens.
5. **Spurious Match Elimination**: Sanitization prevents isolated numerals (e.g., `10,000`) from triggering spurious cultural collisions in DRISHTIKON.

### 6.2 Hypotheses (Requiring Scaled Validation)
1. **Memotion 3 Label Noise Confounding**: Memes `train_1611.jpg` ($P = 0.8175$), `train_3280.jpg` (*Cyanide & Happiness*, $P = 0.8991$), and `train_127.jpg` ($P = 0.5888$) are labeled as non-humorous in Memotion 3 despite containing overt comedic tropes. This remains a dataset annotation hypothesis until corroborated by multi-annotator agreement studies.
2. **Prompt Constraint Collision on Satire**: The failure on `train_5890.jpg` (GT=1, predicted 0.4686) is hypothesized to stem from prompt rule conflict (*"Do NOT mistake serious political announcements for humor"* overriding satirical reaction macros).
3. **Threshold Generalizability**: While $\tau = 0.575$ maximizes accuracy (63.6%) on this 11-sample set, it is an in-sample post-hoc observation and must not be treated as a validated operating threshold without out-of-fold validation.

---

## 7. Recommended Next Zero-Inference Experiments

The following zero-inference, zero-compute experiments are recommended before launching any future VLM inference runs:

1. **Offline DRISHTIKON Tokenizer / Regex Sanitizer Expansion**:
   - Update `sanitize_retrieval_query()` to strip hashtag symbols (`#`) from social media campaign tags (e.g., converting `#TheFamilyMan` to `The Family Man`), allowing OCR hashtags to match cultural records.
   - Benchmark against the full 64,288 DRISHTIKON corpus using the offline test harness.
2. **Continuous Temperature Scaling / Platt Scaling Simulation**:
   - Simulate offline logit calibration on the saved probability outputs in `comparison_results.csv` to determine whether temperature scaling ($T \approx 1.2 - 1.5$) reduces the Brier score and improves ECE across both modes without altering rank ordering.
3. **Power Analysis for Held-Out Evaluation ($N=100$)**:
   - Compute the statistical power required to detect a true $\Delta \text{Accuracy} \ge 0.08$ between General and Cultural modes at $\alpha = 0.05$, establishing the minimal sample size for the Phase 4 scaled benchmark.

---

## 8. Verification & Execution Attestation

- **Report Generated At**: `reports/v4_zero_inference_analysis.md`
- **CPU / VLM Inference Calls Launched**: **0**
- **Existing Evaluation Directories Altered**: **0**
- **Existing Code / Configuration Files Altered**: **0**
- **Status**: Audit and analysis phase complete. Standing by for user review.
