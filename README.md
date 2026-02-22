# 📚 Goodreads Genre Classifier — RoBERTa Fine-Tuning

**ML-DL-Ops Assignment 3 | Text Classification with Transformers, HuggingFace & Docker**

> Fine-tuning `roberta-base` to classify Goodreads book reviews into 8 genres, with full MLOps pipeline: local training → HuggingFace Hub → Docker evaluation container.

---

## 🔗 Important Links

| Resource | Link |
|----------|------|
| 🤗 HuggingFace Model | [Shikhar16/goodreads-genre-classifier](https://huggingface.co/Shikhar16/goodreads-genre-classifier) |
| 📓 Original Notebook | `ML_DL_Ops_Ass_3_Fine_Tuning_Classification.ipynb` |
| 🐙 GitHub Repo | This repository |

---

## 🗂️ Repository Structure

```
Assignment_3/
├── results/                                         # Auto-generated evaluation outputs
│   ├── baseline_results.json                        # TF-IDF baseline metrics
│   ├── local_eval_results.json                      # Local RoBERTa metrics
│   ├── hub_eval_results.json                        # Hub RoBERTa metrics
│   └── comparison.json                              # Side-by-side comparison
│
├── Dockerfile                                       # Task 9  — Production eval container
├── ML_DL_Ops_Ass_3_Fine_Tuning_Classification.ipynb # Original reference notebook
├── README.md                                        # This file
├── evaluate_from_hub.py                             # Task 8  — Load from Hub + re-evaluate
├── genre_reviews_dict.pickle                        # Cached Goodreads data (auto-generated)
├── push_to_hub.py                                   # Task 7  — Upload model to HF Hub
├── requirements.txt                                 # Python dependencies
└── train_evaluate.py                                # Task 6  — Fine-tune + local evaluation
```

---

## 📌 Project Overview

This project implements an end-to-end MLOps pipeline for text classification:

1. **Fine-tune** `roberta-base` on Goodreads book reviews
2. **Evaluate** locally and record metrics
3. **Push** the trained model to HuggingFace Hub
4. **Re-evaluate** from the Hub and compare
5. **Containerize** the evaluation pipeline using Docker

**Dataset:** [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads.html)  
**Task:** Classify reviews into one of 8 genres: `poetry` · `children` · `comics` · `fantasy` · `mystery` · `romance` · `history` · `young_adult`

---

## ⚙️ Setup

```bash
# 1. Clone the repository
git clone https://github.com/shikhar5647/MLOps-Shikhar-Dave-B22CH032.git
cd MLOps-Shikhar-Dave-B22CH032

# 2. Checkout the Assignment_3 branch
git checkout Assignment_3

# 3. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run — Step by Step

### Task 6 — Train & Evaluate Locally

```bash
python train_evaluate.py
```

**What this does:**
- Loads Goodreads review data (downloads on first run, then uses cached `genre_reviews_dict.pickle`)
- Runs a TF-IDF + Logistic Regression baseline
- Fine-tunes `roberta-base` for 3 epochs on CPU
- Saves the trained model to `saved_model/`
- Saves evaluation metrics to `results/`

> ⏱️ Training takes approximately **45–90 minutes** on a MacBook CPU.  
> You will see training loss drop from ~2.0 → ~0.4 across epochs.

---

### Task 7 — Push Model to HuggingFace Hub

```bash
# Login to HuggingFace (one time only)
hf auth login

# Push model, tokenizer, and config
python push_to_hub.py --repo Shikhar16/goodreads-genre-classifier
```

**What gets uploaded:**
- `model.safetensors` — trained weights
- `config.json` — architecture + label mappings
- Tokenizer files (`tokenizer.json`, `vocab.json`, `merges.txt`)
- `label_map.json` — genre-to-integer mapping
- `training_config.json` — full hyperparameter record
- `local_eval_results.json` and `local_classification_report.txt`

---

### Task 8 — Re-evaluate from HuggingFace Hub

```bash
python evaluate_from_hub.py --repo Shikhar16/goodreads-genre-classifier
```

**What this does:**
- Pulls the model and tokenizer directly from HuggingFace Hub
- Downloads test data (uses local pickle cache if available)
- Runs evaluation and prints accuracy, F1, and per-genre classification report
- Saves `hub_eval_results.json` and `comparison.json` to `results/`

---

### Task 9 — Docker Evaluation Container

```bash
# Build the image (replaces YOUR repo name if different)
docker build \
  --build-arg HF_REPO=Shikhar16/goodreads-genre-classifier \
  -t genre-eval .

# Run evaluation
docker run --rm genre-eval
```

**What the container does on startup:**
1. Pulls `roberta-base` fine-tuned weights from HuggingFace Hub
2. Downloads Goodreads test data from the UCSD server
3. Tokenizes and evaluates the model
4. Prints Accuracy, F1 Score, and full classification report

> Model weights are **not baked into the image** — they are pulled at runtime.  
> This keeps the image lightweight and model-version-agnostic.

---

## 🧠 Methodology

### Step 1 — Data Collection and Caching

The Goodreads review data is hosted on the UCSD McAuley Lab server as gzipped JSON files, one per genre. The `load_reviews()` function in `train_evaluate.py` streams each file directly over HTTP using Python's `requests` library and decompresses it on-the-fly with `gzip.open()`. It reads up to 5,000 reviews per genre and randomly samples 500 from those, giving a manageable but representative dataset.

The resulting dictionary of `{genre: [review_texts]}` is serialized to `genre_reviews_dict.pickle` after the first download so that subsequent runs skip the ~5-minute download entirely.

```
UCSD Server (gz files)
        ↓  stream + decompress
  load_reviews()  →  500 reviews × 8 genres
        ↓
  genre_reviews_dict.pickle  (cached for reuse)
```

---

### Step 2 — Train/Test Split

The `split_data()` function iterates over each genre and applies an 80/20 split:
- **3,200 training samples** (400 per genre)
- **800 test samples** (100 per genre)

Splitting is done per genre (stratified by design) to ensure balanced class representation in both sets. This is important because an unbalanced test set would give misleading accuracy scores.

---

### Step 3 — Baseline Model (TF-IDF + Logistic Regression)

Before training a neural model, a classical NLP baseline is established using scikit-learn:

1. **TF-IDF Vectorizer** (50,000 features) converts review text into sparse numerical vectors. TF-IDF weights terms by how frequently they appear in a document relative to the whole corpus, capturing genre-specific vocabulary.
2. **Logistic Regression** (max 1,000 iterations) is trained on the TF-IDF vectors.

This baseline serves two purposes: it gives a performance floor to beat, and it reveals which genres have the most distinctive vocabulary (useful for interpreting BERT's improvements).

**Baseline Result: ~51% accuracy** — meaningful above the 12.5% random chance for 8 classes, but limited because it ignores word order and context.

---

### Step 4 — Tokenization

`RobertaTokenizerFast` is loaded from the `roberta-base` checkpoint. It performs:

- **Byte-Pair Encoding (BPE):** Splits words into subword tokens based on frequency statistics. Rare words like genre-specific terms are split into meaningful pieces rather than discarded as unknown.
- **Truncation:** Reviews longer than 128 tokens are truncated. This covers the majority of short reviews while keeping memory usage manageable.
- **Padding:** Shorter sequences are padded to the same length within each batch using the `[PAD]` token.
- **Special tokens:** `<s>` (start) and `</s>` (end) tokens are automatically added by RoBERTa's tokenizer.

Labels (genre strings) are simultaneously converted to integers via the `label2id` dictionary.

---

### Step 5 — Custom PyTorch Dataset

The `GenreDataset` class wraps the tokenized encodings and integer labels into a `torch.utils.data.Dataset`. The `__getitem__` method converts each encoding to a `torch.tensor` on access, which allows PyTorch's `DataLoader` to efficiently batch samples during training.

```python
class GenreDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
```

---

### Step 6 — Model Architecture

`RobertaForSequenceClassification` is loaded from the `roberta-base` pretrained checkpoint. The architecture is:

```
Input Text
    ↓
RobertaTokenizerFast  →  token IDs + attention mask
    ↓
RoBERTa Encoder (12 transformer layers, 768 hidden dim, 12 attention heads)
    ↓  [CLS] token representation
Dropout (0.1)
    ↓
Linear layer (768 → 8)   ← newly initialized classification head
    ↓
Softmax → genre probabilities
```

The **12 transformer layers** are initialized from the pretrained `roberta-base` weights (trained on 160GB of text). Only the final **classification head** (768 → 8 linear layer) is randomly initialized, as it is specific to our 8-class task. Fine-tuning updates all weights — both the pretrained encoder and the new head — using backpropagation.

---

### Step 7 — Fine-Tuning with HuggingFace Trainer

The HuggingFace `Trainer` API handles the training loop, gradient accumulation, evaluation scheduling, and checkpointing. Key training decisions:

| Decision | Choice | Why |
|----------|--------|-----|
| Learning rate | 2e-5 | Standard for RoBERTa fine-tuning; large enough to adapt quickly, small enough not to destroy pretrained weights |
| Epochs | 3 | Sufficient convergence on 3,200 samples; more epochs risk overfitting |
| Batch size | 16 | Fits comfortably in MacBook RAM |
| Warmup steps | 100 | Prevents large gradient updates in early training while the head is random |
| Weight decay | 0.01 | L2 regularization to reduce overfitting |
| Best model | `load_best_model_at_end=True` | Saves the checkpoint with best F1, not just the last epoch |
| Device | `use_cpu=True` | MPS (Apple Silicon) has a known PyTorch bug causing incorrect gradient updates; CPU produces correct results |

The `compute_metrics` function is called after each epoch and returns accuracy and weighted F1, both of which are logged to the console and used to select the best checkpoint.

---

### Step 8 — Evaluation and Results

After training, `trainer.evaluate()` runs the model on the test set. `trainer.predict()` is then called to get raw logits, which are converted to predicted labels via `argmax(-1)`. The `classification_report` from scikit-learn gives per-genre precision, recall, and F1.

**Expected Results:**

| Model | Accuracy | F1 (Weighted) |
|-------|----------|----------------|
| TF-IDF + Logistic Regression | ~0.51 | ~0.51 |
| RoBERTa-base Fine-tuned | ~0.84 | ~0.84 |

The ~33 percentage point improvement comes from RoBERTa's ability to model semantic meaning and context — it understands that "haunted castle" and "mysterious stranger" both signal mystery/fantasy even without those exact words appearing in training reviews.

---

### Step 9 — Serialization and HuggingFace Hub

`trainer.save_model()` saves the full model checkpoint in `safetensors` format (safer and faster than pickle-based `.bin` files). The tokenizer, label map, and training config are saved alongside so the model is fully self-contained for inference.

`push_to_hub.py` uses `model.push_to_hub()` and `tokenizer.push_to_hub()` from the `transformers` library to upload directly to the HuggingFace model repository with one command.

---

### Step 10 — Docker Pipeline

The `Dockerfile` uses `python:3.11-slim` as the base to minimize image size. Dependencies are installed from `requirements.txt` in a separate layer (so Docker cache is preserved between rebuilds if only code changes). The `evaluate_from_hub.py` script is copied in, and the `HF_REPO` environment variable is injected at build time via `ARG/ENV`.

On container startup, the evaluation script:
1. Downloads the model from HuggingFace using `from_pretrained(repo_id)`
2. Fetches Goodreads test data from the UCSD server (or uses a local cache)
3. Tokenizes the test set, runs inference, and prints the full evaluation report

---

## 📊 Evaluation Results

*(Update with your actual numbers after training completes)*

| Model | Accuracy | F1 (Weighted) | Notes |
|-------|----------|----------------|-------|
| TF-IDF + Logistic Regression | 0.5112 | 0.5078 | Baseline |
| RoBERTa-base (Local, CPU) | ~0.84 | ~0.84 | 3 epochs |
| RoBERTa-base (from Hub) | ~0.84 | ~0.84 | Identical to local |

---

## ⚠️ Key Issues Encountered and Fixed

| # | Issue | Fix |
|---|-------|-----|
| 1 | `TypeError: unexpected keyword argument 'evaluation_strategy'` | Renamed to `eval_strategy` (changed in Transformers v4.41) |
| 2 | 17% accuracy on Apple Silicon MPS backend | Set `use_cpu=True` — MPS has a known gradient propagation bug |
| 3 | Architecture mismatch when pushing to Hub | Fixed `push_to_hub.py` to use `RobertaForSequenceClassification` |
| 4 | Docker container crash: "Cache not found" | `evaluate_from_hub.py` now downloads its own data; no pickle dependency |
| 5 | `huggingface-cli` not found | Used `hf auth login` (correct CLI command for newer versions) |

---

## 📦 Dependencies

```
transformers==4.40.2
torch==2.3.0
datasets==2.19.1
huggingface_hub==0.23.2
scikit-learn==1.5.0
numpy==1.26.4
pandas==2.2.2
requests==2.32.2
gdown==5.2.0
accelerate==0.30.1
```

---

## 👤 Author

**Shikhar Dave** | B22CH032 

ML-DL-Ops Assignment 3
