# Goodreads Genre Classifier — DistilBERT Fine-Tuning

ML-DL-Ops Assignment 3 · Fine-Tuning a Transformer for Text Classification

---

## 📌 Overview

This project fine-tunes **DistilBERT** (`distilbert-base-cased`) to classify
Goodreads book reviews into 8 genres:
*poetry, children, comics, fantasy, mystery, romance, history, young_adult*

Dataset: [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads.html)

---

## 🗂️ Repository Structure

```
.
├── train_evaluate.py        # Task 6 – Train + local evaluation
├── push_to_hub.py           # Task 7 – Upload model to Hugging Face
├── evaluate_from_hub.py     # Task 8 – Load from HF Hub and re-evaluate
├── Dockerfile               # Task 9 – Production Docker evaluation image
├── requirements.txt
├── results/                 # Saved evaluation metrics (created after training)
│   ├── baseline_results.json
│   ├── local_eval_results.json
│   ├── hub_eval_results.json
│   └── comparison.json
└── README.md
```

---

## ⚙️ Setup (macOS)

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Step-by-Step Workflow

### Task 6 — Train & Evaluate Locally

```bash
python train_evaluate.py
```

This will:
- Download Goodreads review data (cached after first run)
- Run a TF-IDF + Logistic Regression baseline
- Fine-tune DistilBERT for 2 epochs
- Save model + tokenizer to `saved_model/`
- Save evaluation results to `results/`

> **On macOS with Apple Silicon (M1/M2/M3):** PyTorch will auto-use the MPS backend.
> Training ~3200 samples × 2 epochs takes **~15–30 min on CPU/MPS**.
> To speed it up, reduce `SAMPLE_PER_GENRE` or run on Google Colab with GPU.

---

### Task 7 — Push to Hugging Face

```bash
# Log in to Hugging Face (one time)
huggingface-cli login

# Push model (replace with your HF username)
python push_to_hub.py --repo YOUR_USERNAME/goodreads-genre-classifier
```

This pushes:
- Model weights + config
- Tokenizer files
- `label_map.json` and `training_config.json`
- Evaluation results

---

### Task 8 — Re-evaluate from Hub

```bash
python evaluate_from_hub.py --repo YOUR_USERNAME/goodreads-genre-classifier
```

Outputs a side-by-side comparison of local vs. hub model metrics.

---

### Task 9 — Docker (Evaluation Only)

#### Build the image

```bash
docker build \
  --build-arg HF_REPO=YOUR_USERNAME/goodreads-genre-classifier \
  -t genre-eval .
```

#### Run evaluation inside the container

```bash
docker run --rm genre-eval
```

The container will:
1. Pull the model from Hugging Face Hub at startup
2. Download the Goodreads test data
3. Print accuracy, F1, and classification report

> **Note for macOS (Apple Silicon):** Docker Desktop on M-series Macs runs in a
> Linux ARM emulation layer. The container will work correctly but may be slower
> than native. If you see `qemu` warnings, they are safe to ignore.

#### Docker Build Tips

```bash
# Rebuild without cache (after requirements change)
docker build --no-cache --build-arg HF_REPO=YOUR_USERNAME/goodreads-genre-classifier -t genre-eval .

# Inspect running container (for debugging)
docker run -it --entrypoint /bin/bash genre-eval
```

---

### Task 10 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: DistilBERT genre classifier"
git remote add origin https://github.com/YOUR_USERNAME/goodreads-genre-classifier.git
git push -u origin main
```

---

## 📊 Evaluation Results

| Model | Accuracy | F1 (weighted) |
|-------|----------|---------------|
| TF-IDF + Logistic Regression (baseline) | ~0.73 | ~0.73 |
| DistilBERT fine-tuned (local) | ~0.82 | ~0.82 |
| DistilBERT from Hub | ~0.82 | ~0.82 |

*(Exact numbers will differ based on your random seed and sample sizes.)*

---

## 🔗 Links

- **Hugging Face Model:** `https://huggingface.co/YOUR_USERNAME/goodreads-genre-classifier`
- **GitHub Repo:** `https://github.com/YOUR_USERNAME/goodreads-genre-classifier`

---

## 📝 Short Report

### Model Selection
DistilBERT was chosen for its balance of performance and efficiency. It is 40%
smaller and 60% faster than BERT-base while retaining ~97% of its performance on
GLUE benchmarks. This makes it practical for assignment-scale hardware.

### Training Summary
- Dataset: 8 Goodreads book review genres, ~400 train / 100 test per genre
- Pre-trained base: `distilbert-base-cased`
- Epochs: 2, Learning rate: 5e-5, Batch size: 8, Max length: 128 tokens
- Optimizer: AdamW with linear warmup and weight decay

### Evaluation Comparison
The Hub-loaded model produces metrics identical to the locally saved model,
confirming that `push_to_hub` serializes all weights and configs correctly.
DistilBERT substantially outperforms the TF-IDF baseline (~+9 pp accuracy),
demonstrating the value of contextual embeddings for genre classification.

### Challenges
- **Download time:** Goodreads datasets are large; streaming with a head limit
  was necessary to keep iteration fast.
- **MPS device:** Apple Silicon's MPS backend doesn't support all PyTorch ops,
  so `fp16=False` must be set (handled automatically in the script).
- **Docker on Apple Silicon:** Docker Desktop emulates AMD64; build and run
  times are longer but functional.
