# Assignment 4: Optimizing Transformer Translation with Ray Tune & Optuna

**Course:** MLOps  
**Name:** Shikhar Dave  
**Roll No:** B22CH032  
**Branch:** Assignment_4  

---

## Overview

This assignment optimizes a custom **English → Hindi Transformer** translation model using **Ray Tune** with **Optuna** (Bayesian search) and **ASHA** (early stopping). The goal: match or exceed baseline BLEU quality in significantly fewer epochs.

---

## Repository Structure

```
├── B22CH032_ass_4_tuned_en_to_hi.ipynb   # Main notebook (baseline + Ray Tune + evaluation)
├── English-Hindi.tsv                      # Dataset (3,270 sentence pairs)
├── B22CH032_ass_4_report.pdf              # 1-2 page report with findings
├── loss_comparison.png                    # Baseline vs Tuned loss curves
├── tune_results.png                       # Ray Tune sweep analysis plots
└── README.md                              # This file
```

## Model Weights (HuggingFace)

The `.pth` model files are hosted on HuggingFace due to GitHub file size limits:

🔗 **[HuggingFace Model Repository ](https://huggingface.co/Shikhar16/assignment_4_Models)**


---

## Part 1: Baseline Metrics (100 Epochs)

The original Transformer was trained as-is for 100 epochs with hardcoded hyperparameters.

| Metric | Value |
|---|---|
| **Epochs** | 100 |
| **Training Time** | 1967.9s (32.8 min) |
| **Final Loss** | 0.0963 |
| **BLEU Score** | **0.7698 (76.98%)** |

### Baseline Hyperparameters

| Parameter | Value |
|---|---|
| d_model | 512 |
| num_layers | 6 |
| num_heads | 8 |
| d_ff | 2048 |
| dropout | 0.1 |
| learning rate | 1e-4 |
| batch_size | 60 |
| optimizer | Adam |

---

## Part 2: Hyperparameter Tuning Setup

### Search Space (6 Hyperparameters)

| Hyperparameter | Search Type | Range / Values |
|---|---|---|
| Learning Rate | Log-Uniform | 1e-5 → 1e-3 |
| Batch Size | Categorical | {16, 32, 64} |
| Num Attention Heads | Categorical | {4, 8} |
| Feed-Forward Dim (d_ff) | Categorical | {1024, 2048} |
| Dropout Rate | Uniform | 0.1 → 0.4 |
| Num Layers (enc + dec) | Categorical | {2, 4, 6} |

### Tuning Configuration

- **Search Algorithm:** OptunaSearch (TPE sampler, Bayesian optimization)
- **Scheduler:** ASHAScheduler (grace_period=5, max_t=30, reduction_factor=3)
- **Trials:** 20 configurations sampled
- **Epochs per Trial:** Capped at 30
- **Resources per Trial:** 2 GPU + 2 CPUs

### Best Configuration Found

| Hyperparameter | Best Value |
|---|---|
| **Learning Rate** | 0.000212 |
| **Batch Size** | 64 |
| **Num Heads** | 4 |
| **d_ff** | 2048 |
| **Dropout** | 0.1638 |
| **Num Layers** | 2 |

---

## Part 3: Final Comparison

| Metric | Baseline (100 ep) | Tuned Best (30 ep) |
|---|---|---|
| **Epochs** | 100 | 30 |
| **Training Time** | 1967.9s (32.8 min) | 209.7s (3.5 min) |
| **Final Loss** | 0.0963 | 0.3075 |
| **BLEU Score** | 0.7698 (76.98%) | 0.5816 (58.16%) |
| **Speedup** | — | **9.4x faster** |

### Key Findings

1. **9.4x Speedup:** The tuned model trains in ~3.5 minutes vs ~33 minutes for the baseline, a dramatic reduction in compute time.

2. **BLEU Score of 0.58:** The tuned model achieves a BLEU score above the 0.50 threshold specified in the assignment, using only 30 epochs (30% of the baseline epochs).

3. **Fewer Layers, Faster Convergence:** Optuna discovered that 2 layers (vs baseline's 6) with a slightly higher learning rate and larger batch size converges efficiently — the model is 3x smaller yet still produces reasonable translations.

4. **ASHA Early Stopping:** The ASHAScheduler terminated underperforming trials after just 5 epochs, saving significant GPU time across the 20-trial sweep.

5. **Gradient Clipping:** Adding `clip_grad_norm_(max_norm=1.0)` in the tuned training loop stabilized training with the higher learning rates discovered during search.

---

## How to Run

### Prerequisites

```bash
pip install torch pandas matplotlib seaborn nltk ray[tune] optuna
```

### Execution

1. Clone the repo and switch to the branch:
   ```bash
   git clone https://github.com/shikhar5647/MLOps-Shikhar-Dave-B22CH032.git
   cd MLOps-Shikhar-Dave-B22CH032
   git checkout Assignment_4
   ```

2. Download model weights from HuggingFace (link above) and place `.pth` files in the repo root.

3. Open and run the notebook:
   ```bash
   jupyter notebook B22CH032_ass_4_tuned_en_to_hi.ipynb
   ```

4. The notebook runs in order:
   - **Part 1:** Baseline training (100 epochs) → records time, loss, BLEU
   - **Part 2:** Ray Tune + Optuna sweep (20 trials × 30 epochs max)
   - **Part 3:** Retrain best config → final comparison

> **Note:** For GPU execution, ensure CUDA is available. The Ray Tune sweep uses `1 GPU per trial`.

---

## Tools & Frameworks

- **PyTorch** — Custom Transformer implementation (from scratch)
- **Ray Tune** — Distributed hyperparameter tuning
- **Optuna** — Bayesian optimization (TPE sampler)
- **ASHA Scheduler** — Early stopping of underperforming trials
- **NLTK** — BLEU score evaluation (SmoothingFunction method4)
