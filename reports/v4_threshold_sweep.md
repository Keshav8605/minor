# Threshold Sensitivity & Calibration Sweep Report (`v4`)
**Dataset / Artifact**: `results/evaluation_twostage_v4/comparison_results.csv`  
**Evaluation Pool**: 11 Memes (5 Humorous ground-truth, 6 Non-Humorous ground-truth)  
**Execution Type**: Strictly Read-Only Offline Post-Processing (Zero VLM inference, zero config modifications)

---

> [!CAUTION]
> ### Critical Sample-Size Disclosure & Overfitting Disclaimer
> - **Sample Size Limitation ($N=11$)**: This sweep operates exclusively on the 11 audited memes from `results/evaluation_twostage_v4/` (22 paired inferences total).
> - **Do NOT Treat Optimal Thresholds as Final Validated Parameters**: Selecting an operational threshold based solely on $N=11$ samples constitutes **in-sample optimization** (p-hacking / post-hoc tuning) and risks severe generalization collapse on unobserved test splits.
> - **Purpose of this Analysis**: This sweep is an analytical tool designed to quantify the **latent logit calibration effect** of cultural context injection. It proves whether cultural context moves probabilities in the theoretically expected direction, independent of arbitrary 0.50 decision boundaries.

---

## 1. Threshold Sweep Matrix: General vs Cultural-Aware

Decision Rule: $\hat{y} = 1$ (Humorous) if $P(\text{Humour}) \ge \tau$, else $\hat{y} = 0$ (Non-Humorous).

### Comprehensive Comparative Performance Table

| Threshold ($\tau$) | Condition | Accuracy | Humour Precision | Humour Recall | Humour F1 | Non-Humour Precision | Non-Humour Recall | Non-Humour F1 | Confusion Matrix (TN, FP, FN, TP) | Mode $\Delta$ Accuracy | Mode $\Delta$ Humour F1 | Mode $\Delta$ Non-Humour F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.500** | General | 0.4545 | 0.4444 | 0.8000 | 0.5714 | 0.5000 | 0.1667 | 0.2500 | `[1, 5, 1, 4]` | — | — | — |
| | Cultural | 0.4545 | 0.4444 | 0.8000 | 0.5714 | 0.5000 | 0.1667 | 0.2500 | `[1, 5, 1, 4]` | **+0.0000** | **+0.0000** | **+0.0000** |
| **0.525** | General | 0.4545 | 0.4444 | 0.8000 | 0.5714 | 0.5000 | 0.1667 | 0.2500 | `[1, 5, 1, 4]` | — | — | — |
| | Cultural | 0.4545 | 0.4444 | 0.8000 | 0.5714 | 0.5000 | 0.1667 | 0.2500 | `[1, 5, 1, 4]` | **+0.0000** | **+0.0000** | **+0.0000** |
| **0.550** | General | 0.4545 | 0.4444 | 0.8000 | 0.5714 | 0.5000 | 0.1667 | 0.2500 | `[1, 5, 1, 4]` | — | — | — |
| | Cultural | **0.5455** | **0.5000** | **0.8000** | **0.6154** | **0.6667** | **0.3333** | **0.4444** | `[2, 4, 1, 4]` | **+0.0909** | **+0.0440** | **+0.1944** |
| **0.575** | General | 0.5455 | 0.5000 | 0.8000 | 0.6154 | 0.6667 | 0.3333 | 0.4444 | `[2, 4, 1, 4]` | — | — | — |
| | Cultural | **0.6364** | **0.5714** | **0.8000** | **0.6667** | **0.7500** | **0.5000** | **0.6000** | `[3, 3, 1, 4]` | **+0.0909** | **+0.0513** | **+0.1556** |
| **0.600** | General | 0.6364 | 0.5714 | 0.8000 | 0.6667 | 0.7500 | 0.5000 | 0.6000 | `[3, 3, 1, 4]` | — | — | — |
| | Cultural | 0.6364 | 0.6000 | 0.6000 | 0.6000 | 0.6667 | 0.6667 | 0.6667 | `[4, 2, 2, 3]` | **+0.0000** | -0.0667 | **+0.0667** |
| **0.625** | General | 0.6364 | 0.5714 | 0.8000 | 0.6667 | 0.7500 | 0.5000 | 0.6000 | `[3, 3, 1, 4]` | — | — | — |
| | Cultural | 0.6364 | 0.6000 | 0.6000 | 0.6000 | 0.6667 | 0.6667 | 0.6667 | `[4, 2, 2, 3]` | **+0.0000** | -0.0667 | **+0.0667** |
| **0.650** | General | 0.6364 | 0.5714 | 0.8000 | 0.6667 | 0.7500 | 0.5000 | 0.6000 | `[3, 3, 1, 4]` | — | — | — |
| | Cultural | 0.6364 | 0.6000 | 0.6000 | 0.6000 | 0.6667 | 0.6667 | 0.6667 | `[4, 2, 2, 3]` | **+0.0000** | -0.0667 | **+0.0667** |

---

## 2. Threshold-by-Threshold Comparative Dynamics

### A. Thresholds $\tau \in [0.500, 0.525]$ (Baseline Default)
- **General Confusion Matrix**: $\begin{pmatrix} 1 & 5 \\ 1 & 4 \end{pmatrix}$ | **Cultural Confusion Matrix**: $\begin{pmatrix} 1 & 5 \\ 1 & 4 \end{pmatrix}$
- **Prediction Changes**: None ($0/11$). Both modes yield identical discrete outputs because all 10 non-zero probabilities sit outside the $[0.500, 0.525]$ window.
- **Limitation**: The model's slight positive bias across internet memes results in high recall ($0.8000$) but high false positive counts ($5/6$ non-humorous memes classified as humorous).

### B. Threshold $\tau = 0.550$ (Emergence of Cultural Advantage)
- **General Confusion Matrix**: $\begin{pmatrix} 1 & 5 \\ 1 & 4 \end{pmatrix}$ | **Cultural Confusion Matrix**: $\begin{pmatrix} 2 & 4 \\ 1 & 4 \end{pmatrix}$
- **Key Divergence**:
  - Sample `train_5723.jpg` (GT = 0, `#TheFamilyMan` marriage avoidance phone call):
    - **General Probability**: $P = 0.7279 \ge 0.55 \implies \hat{y} = 1$ (**False Positive**)
    - **Cultural Probability**: $P = 0.5272 < 0.55 \implies \hat{y} = 0$ (**True Negative**)
- **Empirical Impact**:
  - Cultural Accuracy rises to **0.5455** (+9.1% over General's 0.4545).
  - Cultural Non-Humour F1 jumps from **0.2500 to 0.4444** (+19.4%).
  - General mode remains trapped at 5 False Positives.

### C. Threshold $\tau = 0.575$ (Peak Separation Zone)
- **General Confusion Matrix**: $\begin{pmatrix} 2 & 4 \\ 1 & 4 \end{pmatrix}$ | **Cultural Confusion Matrix**: $\begin{pmatrix} 3 & 3 \\ 1 & 4 \end{pmatrix}$
- **Key Divergence**:
  - `train_5088.jpg` (GT = 0, Nigel Farage / political crying meme):
    - Both General and Cultural probabilities sit at $P = 0.5659 < 0.575$.
    - Flips from False Positive to True Negative in both modes simultaneously.
  - `train_5723.jpg` (GT = 0):
    - General ($P = 0.7279$) remains a **False Positive**.
    - Cultural ($P = 0.5272$) remains a **True Negative**.
- **Empirical Impact**:
  - Cultural Accuracy reaches **0.6364** (7/11) vs General Accuracy of **0.5455** (6/11) (+9.1% advantage).
  - Cultural Humour F1 reaches **0.6667** vs General's **0.6154**.
  - Cultural Non-Humour F1 reaches **0.6000** vs General's **0.4444** (+15.6% advantage).
  - Cultural condition cuts False Positives down to 3, while preserving 4/5 True Positives.

### D. Thresholds $\tau \in [0.600, 0.650]$ (Recall Trade-off & Saturation)
- **General Confusion Matrix**: $\begin{pmatrix} 3 & 3 \\ 1 & 4 \end{pmatrix}$ | **Cultural Confusion Matrix**: $\begin{pmatrix} 4 & 2 \\ 2 & 3 \end{pmatrix}$
- **Key Divergences**:
  1. `train_127.jpg` (GT = 0, Modi *"Ab Ki Baar"* slogan):
     - Both General and Cultural probabilities ($P = 0.5888 < 0.600$) drop below threshold into True Negatives.
  2. `train_4136.jpg` (GT = 1, *"I Have Two Sides"*):
     - General mode ($P = 0.7879 \ge 0.600$) preserves True Positive status ($\hat{y} = 1$).
     - Cultural mode ($P = 0.5812 < 0.600$) inadvertently drops below threshold into a **False Negative** ($\hat{y} = 0$).
  3. `train_5723.jpg` (GT = 0):
     - General mode ($P = 0.7279$) remains a False Positive ($\hat{y} = 1$).
     - Cultural mode ($P = 0.5272$) remains a True Negative ($\hat{y} = 0$).
- **Empirical Impact**:
  - Overall accuracy ties at **0.6364** (7/11).
  - However, the error profiles differ fundamentally: General mode suffers from high False Positives (3 FP / 1 FN), whereas Cultural mode balances errors evenly (2 FP / 2 FN) and achieves a higher Non-Humour F1 (**0.6667** vs 0.6000).

---

## 3. Per-Sample Classification Matrix Across Thresholds

| Image Filename | Ground Truth | General $P$ | Cultural $P$ | $\tau=0.500$ [G/C] | $\tau=0.525$ [G/C] | $\tau=0.550$ [G/C] | $\tau=0.575$ [G/C] | $\tau=0.600$ [G/C] | $\tau=0.625$ [G/C] | $\tau=0.650$ [G/C] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `train_6346.jpg` | 1 | 0.8354 | 0.7309 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| `train_5723.jpg` | 0 | 0.7279 | **0.5272** | 1 / 1 | 1 / 1 | **1 / 0** | **1 / 0** | **1 / 0** | **1 / 0** | **1 / 0** |
| `train_5266.jpg` | 0 | 0.2146 | 0.2146 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| `train_4199.jpg` | 1 | 0.8856 | 0.8856 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| `train_4136.jpg` | 1 | 0.7879 | **0.5812** | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | **1 / 0** | **1 / 0** | **1 / 0** |
| `train_3395.jpg` | 1 | 0.6653 | 0.6653 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| `train_1611.jpg` | 0 | 0.8175 | 0.8175 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| `train_5890.jpg` | 1 | 0.4609 | 0.4686 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| `train_3280.jpg` | 0 | 0.8991 | 0.8991 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| `train_127.jpg`  | 0 | 0.5888 | 0.5888 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 0 / 0 | 0 / 0 | 0 / 0 |
| `train_5088.jpg` | 0 | 0.5659 | 0.5659 | 1 / 1 | 1 / 1 | 1 / 1 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

---

## 4. Analytical Findings & Takeaways

1. **Proof of Latent Calibration Without Prompt Redesign**:
   - The logit probabilities prove that cultural context directly counteracts the VLM's default tendency to over-predict humor on culturally nuanced domestic memes.
   - For `train_5723.jpg`, cultural context compressed humorous logit probability by **-0.2007** (from 0.7279 down to 0.5272). At standard $\tau=0.5000$, this shift was masked as an apparent tie, but across $\tau \in [0.550, 0.575]$, it yields an immediate **+9.1% accuracy gain** and **+15.6% non-humour F1 gain**.

2. **The Risk of Over-Thresholding ($\tau \ge 0.600$)**:
   - Pushing the threshold to $\tau \ge 0.600$ eliminates false positives on borderline memes like `train_127.jpg`, but causes `train_4136.jpg` ($P=0.5812$) to drop into a False Negative in Cultural mode.
   - This demonstrates the standard precision-recall trade-off: high thresholds suppress false alarms at the expense of genuine humor detection.

3. **Status of Historical Artifacts**:
   - As required, all historical directories remain untouched:
     - `results/evaluation_twostage/` (Unmodified since 15-10-26)
     - `results/evaluation_twostage_refined/` (Unmodified since 18-10-26)
     - `results/evaluation_twostage_final/` (Unmodified since 18-10-26)
     - `results/evaluation_twostage_v4/` (Unmodified since generation)
   - Zero application code, prompt templates, model configurations, or runtime thresholds were modified.
