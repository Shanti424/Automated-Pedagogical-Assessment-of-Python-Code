# Automated Pedagogical Assessment of Python Code

## Overview
This project investigates automated pedagogical assessment of Python code using:
- handcrafted features
- CodeBERT embeddings
- hybrid feature + embedding representations

The goal is to predict educational quality scores for Python code snippets.

---

## Dataset

**Dataset:**  
`HuggingFaceTB/python-edu-annotations`

- ~491K Python code snippets
- Scores from 1–5 representing educational quality
- Labels generated using Llama-3-70B-Instruct

---

## Project Files

### File 1 — Feature-Based Pipeline
Includes:
- feature extraction
- baseline models
- feature ablation
- training size analysis
- documentation impact
- imbalance handling

### File 2 — CodeBERT Hyperparameter Tuning
Tunes:
- token length
- pooling strategy

Best configuration:
- `max length = 256`
- `pooling = CLS`

### File 3 — Embedding-Based Pipeline
Includes:
- embedding generation
- embedding-only experiments
- code truncation analysis
- imbalance handling

### File 4 — Hybrid Models
Includes:
- feature + embedding combinations
- hybrid tier comparison
- delta analysis
- confusion matrices

---

## Technical Stack

- Python 3.10+
- transformers (CodeBERT)
- xgboost
- scikit-learn
- pandas
- numpy
- datasets
- torch
- imbalanced-learn
- matplotlib
- seaborn

> CUDA-enabled GPU recommended for embedding extraction.

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Files in Order

Run all notebooks/scripts sequentially in the following order:

1. File 1 — Feature-Based Pipeline  
2. File 2 — Hyperparameter Tuning  
3. File 3 — Embedding Pipeline  
4. File 4 — Hybrid Pipeline  

For notebook execution:
- restart runtime/kernel before running a new file
- use “Run All” for each notebook to ensure all cells execute correctly

---
## Evaluation Metrics

- Accuracy
- Macro-F1

Macro-F1 is treated as the primary metric due to dataset imbalance.

---

## Best Results

### Best Macro-F1
- Hybrid (Lexical) + XGBoost = `0.4966`

### Best Accuracy
- Hybrid (ALL) + Random Forest = `0.5835`

---

## Reproducibility

- Train / Validation / Test split:
  - 70% / 15% / 15%

- Random seed:

```python
random_state = 42
```

---

## Author

**Shanti Tamang**  
Computer Science Department  
The University of Memphis
