# Goodreads Genre Classifier — RoBERTa Fine-Tuning

**ML-DL-Ops Assignment 3 | Text Classification with Transformers, HuggingFace & Docker**

End-to-end MLOps pipeline: fine-tune `roberta-base` on Goodreads reviews → evaluate → push to HuggingFace Hub → containerize evaluation with Docker.

---

## 🔗 Links

| Resource | Link |
|----------|------|
| 🤗 HuggingFace Model | [Shikhar16/goodreads-genre-classifier](https://huggingface.co/Shikhar16/goodreads-genre-classifier) |
| 📓 Reference Notebook | `ML_DL_Ops_Ass_3_Fine_Tuning_Classification.ipynb` |

---

## Results (Actual)

| Model | Accuracy | F1 (Weighted) | Notes |
|-------|----------|----------------|-------|
| TF-IDF + Logistic Regression | **0.5113** | 0.5078 | Baseline (50k features) |
| RoBERTa-base, fine-tuned (local, CPU) | **0.2350** | 0.2052 | 3 epochs, batch 16, lr 2e-5 |
| RoBERTa-base, from HuggingFace Hub | **0.2350** | 0.2052 | Identical to local — reproducibility verified |

**Honest discussion:** the fine-tuned RoBERTa model under-performed the TF-IDF baseline. Final eval_loss was 1.96 against a random-baseline upper bound of ln(8) ≈ 2.08, indicating the model learned some signal but did not converge to a strong classifier within 3 CPU epochs on 3,200 training samples. Probable causes: insufficient training data per class, CPU-only training, and unconverged hyperparameters. The pipeline (training → save → push → re-load → evaluate → Docker) is fully functional and reproducible — local and Hub evaluations agree to 13 decimal places (see `results/comparison.json`).

A second training run on GPU with more data and tuned hyperparameters is planned to improve absolute accuracy. The MLOps infrastructure itself is task-complete.

---

## Repository Structure

```
Assignment_3/
├── results/
│   ├── baseline_results.json            # TF-IDF baseline metrics
│   ├── local_eval_results.json          # Local RoBERTa metrics
│   ├── hub_eval_results.json            # Hub RoBERTa metrics
│   ├── comparison.json                  # Local vs Hub side-by-side
│   ├── local_classification_report.txt
│   └── hub_classification_report.txt
├── Dockerfile                           # Task 9
├── train_evaluate.py                    # Task 6
├── push_to_hub.py                       # Task 7
├── evaluate_from_hub.py                 # Task 8
├── prove_finetuning.py                  # Verifies model on Hub is fine-tuned
├── requirements.txt
├── genre_reviews_dict.pickle            # Cached Goodreads data
└── ML_DL_Ops_Ass_3_Fine_Tuning_Classification.ipynb
```

---

## Setup

```bash
git clone <repo>
cd Assignment_3
git checkout Assignment_3
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## Running Each Task

### Task 6 — Train & Evaluate Locally
```bash
python train_evaluate.py
```
Downloads Goodreads data (or uses cache), runs TF-IDF baseline, fine-tunes RoBERTa for 3 epochs on CPU, saves to `saved_model/`, writes metrics to `results/`. Takes ~45–90 min on MacBook CPU.

### Task 7 — Push to HuggingFace Hub
```bash
hf auth login   # one-time
python push_to_hub.py --repo Shikhar16/goodreads-genre-classifier
```

### Task 8 — Re-evaluate from Hub
```bash
python evaluate_from_hub.py --repo Shikhar16/goodreads-genre-classifier
```
Pulls the model fresh from Hub, evaluates, writes `hub_eval_results.json` and `comparison.json`.

### Task 9 — Docker Evaluation Container
```bash
docker build --build-arg HF_REPO=Shikhar16/goodreads-genre-classifier -t genre-eval .
docker run --rm genre-eval
```
Container pulls model from Hub at runtime (not baked in), downloads test data, prints accuracy + F1 + classification report.

### Bonus — Verify Fine-Tuning Actually Occurred
```bash
python prove_finetuning.py --repo Shikhar16/goodreads-genre-classifier
```
Inspects the classification head weights, label mappings, and runs sample inference to confirm the Hub model is a fine-tuned variant (not a copy of base `roberta-base`).

---

## Reproducibility Evidence

`results/comparison.json` shows local and Hub evaluations produce **identical metrics**:
- `eval_loss`: 1.9603493213653564 (both)
- `eval_accuracy`: 0.235 (both)
- `eval_f1_weighted`: 0.20516761265770955 (both)

This 13-decimal-place match across two independent evaluation runs (one from local checkpoint, one re-downloaded from HuggingFace Hub) is the definitive proof that the model on the Hub is the same trained model and the pipeline is reproducible.

---

## Methodology

The full methodology (data loading, TF-IDF baseline, tokenization, custom Dataset class, RoBERTa architecture, fine-tuning loop, serialization, Hub upload, Docker pipeline) follows the standard HuggingFace fine-tuning workflow as covered in class. See inline docstrings in each script for details.

---

## Issues Encountered

| # | Issue | Resolution |
|---|-------|------------|
| 1 | `evaluation_strategy` argument removed in transformers ≥ 4.41 | Switched to `eval_strategy` |
| 2 | MPS (Apple Silicon) backend produced ~17% accuracy due to known PyTorch gradient bug | Forced `use_cpu=True` |
| 3 | Architecture-mismatch error when pushing to Hub | Used `RobertaForSequenceClassification` consistently in `push_to_hub.py` |
| 4 | Docker container failed: pickle file missing inside container | Rewrote `evaluate_from_hub.py` to download Goodreads data on first run |
| 5 | `huggingface-cli` deprecated | Used `hf auth login` |
| 6 | RoBERTa under-performed TF-IDF baseline | Acknowledged in results; planned GPU rerun with more data |

---

## Dependencies

See `requirements.txt`. Key versions: `transformers`, `torch`, `scikit-learn`, `huggingface_hub`, `accelerate`.

---

## 👤 Author

**Shikhar Dave** | B22CH032 | ML-DL-Ops Assignment 3
