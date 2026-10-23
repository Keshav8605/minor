# Independently Verified Threshold Sweep Report (`v4`)

**Source Data**: `results/evaluation_twostage_v4/comparison_results.csv`  
**Reference Document**: `reports/v4_threshold_sweep.md` (original report preserved intact)  
**Verification Method**: Independent re-calculation from raw probabilities and ground-truth labels  
**Decision Rule Applied**: $\hat{y} = 1$ (Humorous) if $P(\text{Humour}) \ge \tau$, else $\hat{y} = 0$ (Non-Humorous)  
**Sample Space**: 11 Test Memes (5 Humorous ground-truth, 6 Non-Humorous ground-truth)

---

> [!CAUTION]
> ### Critical Sample-Size Disclosure & Overfitting Disclaimer
> - **Sample Size Limitation ($N=11$)**: This analysis is based strictly on the 11 audited memes from `results/evaluation_twostage_v4/` (22 total paired inferences).
> - **Do NOT Treat Optimal Thresholds as Validated Production Parameters**: Identifying optimal metric performance at $\tau \in [0.550, 0.575]$ represents post-hoc in-sample observation on a small test subset. It must not be deployed as a hardcoded operational parameter without validation across a broader distribution.
> - **Analytical Purpose**: This sweep serves to formally verify the **latent logit calibration effect** of cultural context injection. It evaluates whether cultural reasoning moves logit probabilities in the theoretically sound direction, independent of static default boundaries.

---

## 1. Verified Metrics & Confusion Matrices

All calculations follow standard definitions:
- $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$
- $\text{Precision}_{\text{humor}} = \frac{TP}{TP + FP}$, $\text{Recall}_{\text{humor}} = \frac{TP}{TP + FN}$, $\text{F1}_{\text{humor}} = \frac{2 \cdot P \cdot R}{P + R}$
- $\text{Precision}_{\text{non-humor}} = \frac{TN}{TN + FN}$, $\text{Recall}_{\text{non-humor}} = \frac{TN}{TN + FP}$, $\text{F1}_{\text{non-humor}} = \frac{2 \cdot P_{\text{nh}} \cdot R_{\text{nh}}}{P_{\text{nh}} + R_{\text{nh}}}$

### Master Verified Table

| Threshold ($\tau$) | Condition | Accuracy | Humour Precision | Humour Recall | Humour F1 | Non-Humour Precision | Non-Humour Recall | Non-Humour F1 | Confusion Matrix [TN, FP, FN, TP] | Mode $\Delta$ Accuracy | Mode $\Delta$ Humour F1 | Mode $\Delta$ Non-Humour F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.500** | General | 0.4545 | 0.4444 | 0.8000 | 0.5714 | 0.5000 | 0.1667 | 0.2500 | `[1, 5, 1, 4]` | - | - | - |
| | Cultural | 0.4545 | 0.4444 | 0.8000 | 0.5714 | 0.5000 | 0.1667 | 0.2500 | `[1, 5, 1, 4]` | +0.0000 | +0.0000 | +0.0000 |
| **0.525** | General | 0.4545 | 0.4444 | 0.8000 | 0.5714 | 0.5000 | 0.1667 | 0.2500 | `[1, 5, 1, 4]` | - | - | - |
| | Cultural | 0.4545 | 0.4444 | 0.8000 | 0.5714 | 0.5000 | 0.1667 | 0.2500 | `[1, 5, 1, 4]` | +0.0000 | +0.0000 | +0.0000 |
| **0.550** | General | 0.4545 | 0.4444 | 0.8000 | 0.5714 | 0.5000 | 0.1667 | 0.2500 | `[1, 5, 1, 4]` | - | - | - |
| | Cultural | **0.5455** | **0.5000** | **0.8000** | **0.6154** | **0.6667** | **0.3333** | **0.4444** | `[2, 4, 1, 4]` | **+0.0909** | **+0.0440** | **+0.1944** |
| **0.575** | General | 0.5455 | 0.5000 | 0.8000 | 0.6154 | 0.6667 | 0.3333 | 0.4444 | `[2, 4, 1, 4]` | - | - | - |
| | Cultural | **0.6364** | **0.5714** | **0.8000** | **0.6667** | **0.7500** | **0.5000** | **0.6000** | `[3, 3, 1, 4]` | **+0.0909** | **+0.0513** | **+0.1556** |
| **0.600** | General | 0.6364 | 0.5714 | 0.8000 | 0.6667 | 0.7500 | 0.5000 | 0.6000 | `[3, 3, 1, 4]` | - | - | - |
| | Cultural | 0.6364 | 0.6000 | 0.6000 | 0.6000 | 0.6667 | 0.6667 | 0.6667 | `[4, 2, 2, 3]` | +0.0000 | -0.0667 | **+0.0667** |
| **0.625** | General | 0.6364 | 0.5714 | 0.8000 | 0.6667 | 0.7500 | 0.5000 | 0.6000 | `[3, 3, 1, 4]` | - | - | - |
| | Cultural | 0.6364 | 0.6000 | 0.6000 | 0.6000 | 0.6667 | 0.6667 | 0.6667 | `[4, 2, 2, 3]` | +0.0000 | -0.0667 | **+0.0667** |
| **0.650** | General | 0.6364 | 0.5714 | 0.8000 | 0.6667 | 0.7500 | 0.5000 | 0.6000 | `[3, 3, 1, 4]` | - | - | - |
| | Cultural | 0.6364 | 0.6000 | 0.6000 | 0.6000 | 0.6667 | 0.6667 | 0.6667 | `[4, 2, 2, 3]` | +0.0000 | -0.0667 | **+0.0667** |

---

## 2. In-Depth Verification of Focus Memes

### A. `train_5723.jpg` (Ground Truth = 0, Non-Humorous)
- **OCR Text**: `#TheFamilyMan #OnlyExcusesOctober When Wife Starts To Tell About Her Parents Instead of about Her Sister #SalliJi Le Husband: main baad me phone karta hoon`
- **Probabilities**: General $P = 0.7279$ | Cultural $P = 0.5272$ ($\Delta = -0.2007$)
- **Classification Behavior Across Sweep**:
  - At $\tau = 0.500$ & $0.525$: Both General and Cultural predict $1$ (**False Positive**), since $0.7279 \ge \tau$ and $0.5272 \ge \tau$.
  - At $\tau \ge 0.550$ ($\tau \in \{0.550, 0.575, 0.600, 0.625, 0.650\}$):
    - General mode predicts $1$ (**False Positive**), because $0.7279 \ge 0.650$.
    - Cultural mode predicts $0$ (**True Negative**), because $0.5272 < 0.550$.
- **Verification Finding**: **Confirmed**. `train_5723.jpg` is the exact driver of the cultural accuracy and non-humour F1 advantage at $\tau = 0.550$ and $\tau = 0.575$. Cultural reasoning significantly reduced the model's false humorous confidence by $0.2007$, crossing the $\tau = 0.550$ decision boundary into a correct True Negative.

---

### B. `train_4136.jpg` (Ground Truth = 1, Humorous)
- **OCR Text**: `I Have Two Sides ig/memeshroud Dad Buy Me An All Men Are iPhone Dogs/ B apists`
- **Probabilities**: General $P = 0.7879$ | Cultural $P = 0.5812$ ($\Delta = -0.2067$)
- **Classification Behavior Across Sweep**:
  - At $\tau \le 0.575$ ($\tau \in \{0.500, 0.525, 0.550, 0.575\}$):
    - General mode predicts $1$ (**True Positive**), because $0.7879 \ge \tau$.
    - Cultural mode predicts $1$ (**True Positive**), because $0.5812 \ge \tau$.
  - At $\tau \ge 0.600$ ($\tau \in \{0.600, 0.625, 0.650\}$):
    - General mode predicts $1$ (**True Positive**), because $0.7879 \ge 0.650$.
    - Cultural mode predicts $0$ (**False Negative**), because $0.5812 < 0.600$.
- **Verification Finding**: **Confirmed**. `train_4136.jpg` illustrates the peril of aggressive threshold tuning ($\tau \ge 0.600$). Cultural context softened the model's certainty to $0.5812$, which remains correctly classified as humorous for $\tau \le 0.575$, but is erroneously truncated to non-humorous when $\tau$ is set to $0.600$ or above.

---

### C. `train_127.jpg` (Ground Truth = 0, Non-Humorous)
- **OCR Text**: `I DONT KNOW WHO YOU ARE AB KI BAAR MODI SARKAAR buffedd`
- **Probabilities**: General $P = 0.5888$ | Cultural $P = 0.5888$ ($\Delta = +0.0000$)
- **Two-Stage Gating State**: Reused Stage-1 assessment via short-circuit (0.23s runtime, zero redundant VLM call).
- **Classification Behavior Across Sweep**:
  - At $\tau \le 0.575$: Both General and Cultural predict $1$ (**False Positive**), because $0.5888 \ge \tau$.
  - At $\tau \ge 0.600$: Both General and Cultural predict $0$ (**True Negative**), because $0.5888 < 0.600$.
- **Verification Finding**: **Confirmed**. Because `train_127.jpg` short-circuited Stage-2, its General and Cultural probabilities are strictly identical ($0.5888$). Thus, it **never diverges between modes at any single threshold**. However, it transitions synchronously in both modes from a False Positive to a True Negative at the $\tau = 0.600$ boundary.

---

## 3. Discrepancy Reconciliation Summary

1. **Numerical Metrics**:
   - The original report [`reports/v4_threshold_sweep.md`](file:///c:/Users/prach/Downloads/Minor%20Project/minor/reports/v4_threshold_sweep.md) contained **100% numerically accurate calculations** for accuracy, precision, recall, F1-scores, and confusion matrices.
2. **Formatting / Character Encoding Discrepancy**:
   - In [`reports/v4_threshold_sweep.md`](file:///c:/Users/prach/Downloads/Minor%20Project/minor/reports/v4_threshold_sweep.md), markdown em-dash characters (`—`) in empty delta cells were corrupted during Windows text encoding into replacement symbols (``).
   - This verified report explicitly replaces all corrupted placeholder characters with standard ASCII hyphens (`-`).
3. **Clarity on `train_127.jpg`**:
   - The original report documented divergence cases where General and Cultural differed, which led to omitting `train_127.jpg` from the divergence notes. This verified report clarifies that `train_127.jpg` transitions synchronously across modes at $\tau = 0.600$ due to Stage-2 short-circuiting.

---

## 4. Preservation of Historical Artifacts

All evaluation artifacts and directories remain completely unmodified:
- `results/evaluation_twostage/` (Untouched)
- `results/evaluation_twostage_refined/` (Untouched)
- `results/evaluation_twostage_final/` (Untouched)
- `results/evaluation_twostage_v4/` (Untouched)
- `reports/v4_threshold_sweep.md` (Untouched, preserved as original)

No VLM inference was executed, and no application source code, prompts, configs, or labels were altered.
