# Algorithmic Fairness Analysis
### UCI Adult Income Dataset

**Name:** Om Dipak Patil  
**GH Number:** GH1044372  
**Module:** M515A — Ethical Issues for AI

---

## Overview

This project investigates **algorithmic fairness** using the UCI Adult Income dataset. A Logistic Regression binary classifier is trained to predict whether an individual earns more than **$50K/year**, and then audited for discriminatory behaviour across protected demographic groups.

No bias mitigation is applied — the analysis is **purely diagnostic**.

---

## Objectives

- Train a binary income classifier on the UCI Adult dataset
- Audit model predictions for fairness across **sex** and **race** subgroups
- Compute four error-rate fairness metrics per subgroup
- Visualise disparities using confusion matrices and bar charts

---

## Dataset

**Source:** [UCI Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)

| Property | Value |
|----------|-------|
| Total records | 32,561 |
| After cleaning | 30,162 |
| Features | 14 |
| Target | `salary` (≤50K / >50K) |
| Class balance | 75.9% ≤50K / 24.1% >50K |

---

## Fairness Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **FPR** | FP / (FP + TN) | Rate of incorrectly predicting high income |
| **FNR** | FN / (FN + TP) | Rate of missing true high earners |
| **FDR** | FP / (FP + TP) | Fraction of positive predictions that are wrong |
| **FOR** | FN / (FN + TN) | Fraction of negative predictions that are wrong |

---

## Results

### Overall Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | **81.75%** |
| Precision (>50K) | 0.71 |
| Recall (>50K) | 0.45 |
| F1-Score (>50K) | 0.55 |

### Fairness by Sex

| Group | FPR | FNR | FDR | FOR | n |
|-------|-----|-----|-----|-----|---|
| Male | 0.0879 | 0.5130 | 0.2851 | 0.2030 | 4065 |
| Female | 0.0133 | 0.7745 | 0.3026 | 0.0962 | 1968 |

> Women have a significantly higher FNR (0.77 vs 0.51) — the model systematically under-predicts high income for female individuals.

### Fairness by Race

| Group | FPR | FNR | FDR | FOR | n |
|-------|-----|-----|-----|-----|---|
| White | 0.0655 | 0.5420 | 0.2851 | 0.1722 | 5186 |
| Black | 0.0251 | 0.7500 | 0.3750 | 0.1139 | 559 |
| Asian-Pac-Islander | 0.0423 | 0.5227 | 0.2222 | 0.1447 | 186 |
| Amer-Indian-Eskimo | 0.0000 | 0.8333 | 0.0000 | 0.0794 | 64 |
| Other | 0.0286 | 0.6667 | 0.5000 | 0.0556 | 38 |

> Black and Amer-Indian-Eskimo individuals face the highest FNR — the model most frequently misses true high earners in these groups.

---

## Project Structure

```
ethical-issues/
├── algorithmic_fairness.ipynb        # Main analysis notebook
├── algorithmic_fairness.html         # Exported HTML version
├── algorithmic_fairness_report.docx  # Word report with charts
├── algorithmic_fairness_report.pdf   # PDF report
├── adult.data.csv                    # UCI Adult dataset
└── README.md                         # This file
```

---

## Setup & Usage

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Run the Notebook

```bash
jupyter notebook algorithmic_fairness.ipynb
```

Run all cells top to bottom (**Kernel → Restart & Run All**).

---

## Key Findings

- The model **does not treat both sexes equally** — females face a much higher false negative rate, meaning true high earners among women are frequently misclassified as low earners.
- The model **violates the equalised odds fairness criterion** across racial groups — Black individuals are disproportionately under-predicted for high income.
- These disparities likely stem from **historical income inequality** embedded in the training data rather than any explicit discriminatory feature.

---

## Limitations

1. **No causality** — observed disparities are correlational, not causal.
2. **Small subgroups** — race categories with few samples yield less reliable metric estimates.
3. **No mitigation** — this project is diagnostic only; no fairness-aware training or post-processing was applied.

---

## License

This project is for academic purposes only as part of module **M515A — Ethical Issues for AI**.