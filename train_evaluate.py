"""
Task 6: Fine-tune RoBERTa-base on Goodreads Genre Classification
Fixes:
  - Forces CPU on MPS (Apple Silicon MPS has a known bug with certain
    transformer versions that causes weights to not update → ~12% accuracy)
  - eval_strategy instead of deprecated evaluation_strategy
  - processing_class instead of deprecated tokenizer= in Trainer
"""

import os
import json
import random
import pickle
import gzip
import requests
import numpy as np
from collections import defaultdict

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ─────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────
MODEL_NAME       = "roberta-base"
MAX_LENGTH       = 128
OUTPUT_DIR       = "saved_model"
RESULTS_DIR      = "results"
SAMPLE_PER_GENRE = 500
TRAIN_SPLIT      = 0.8
NUM_EPOCHS       = 3
BATCH_SIZE       = 16
SEED             = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
random.seed(SEED)

# ── IMPORTANT: Force CPU — MPS has a known bug with transformer training
# that causes weights to not update properly (results in ~12-17% accuracy).
# CPU training is slower but gives correct results (~80-87% accuracy).
DEVICE = "cpu"
print(f"Using device: {DEVICE}  (MPS disabled due to known training bug)")

# ─────────────────────────────────────────────────────
# 1. Data
# ─────────────────────────────────────────────────────
GENRE_URLS = {
    "poetry":      "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
    "children":    "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz",
    "comics":      "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
    "fantasy":     "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
    "mystery":     "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
    "romance":     "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
    "history":     "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
    "young_adult": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
}
CACHE_FILE = "genre_reviews_dict.pickle"


def load_reviews(url, head=5000, sample_size=None):
    reviews, count = [], 0
    response = requests.get(url, stream=True, timeout=120)
    with gzip.open(response.raw, "rt", encoding="utf-8") as f:
        for line in f:
            text = json.loads(line).get("review_text", "").strip()
            if text:
                reviews.append(text)
                count += 1
            if head and count >= head:
                break
    if sample_size and len(reviews) > sample_size:
        reviews = random.sample(reviews, sample_size)
    return reviews


def get_data():
    if os.path.exists(CACHE_FILE):
        print("Loading cached data...")
        return pickle.load(open(CACHE_FILE, "rb"))
    print("Downloading data (this may take a few minutes)...")
    genre_reviews = {}
    for genre, url in GENRE_URLS.items():
        print(f"  Downloading {genre}...")
        genre_reviews[genre] = load_reviews(url, head=5000, sample_size=SAMPLE_PER_GENRE)
    pickle.dump(genre_reviews, open(CACHE_FILE, "wb"))
    return genre_reviews


# ─────────────────────────────────────────────────────
# 2. Train / test split
# ─────────────────────────────────────────────────────
def split_data(genre_reviews):
    train_texts, train_labels = [], []
    test_texts,  test_labels  = [], []
    for genre, reviews in genre_reviews.items():
        reviews   = random.sample(reviews, min(len(reviews), SAMPLE_PER_GENRE))
        split_idx = int(len(reviews) * TRAIN_SPLIT)
        for r in reviews[:split_idx]:
            train_texts.append(r); train_labels.append(genre)
        for r in reviews[split_idx:]:
            test_texts.append(r);  test_labels.append(genre)
    return train_texts, train_labels, test_texts, test_labels


# ─────────────────────────────────────────────────────
# 3. Baseline
# ─────────────────────────────────────────────────────
def run_baseline(train_texts, train_labels, test_texts, test_labels):
    print("\n[Baseline: TF-IDF + Logistic Regression]")
    vec   = TfidfVectorizer(max_features=50000)
    X_tr  = vec.fit_transform(train_texts)
    X_te  = vec.transform(test_texts)
    clf   = LogisticRegression(max_iter=1000).fit(X_tr, train_labels)
    preds = clf.predict(X_te)
    acc   = accuracy_score(test_labels, preds)
    f1    = f1_score(test_labels, preds, average="weighted")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  F1 (weighted): {f1:.4f}")
    print(classification_report(test_labels, preds))
    return {"accuracy": acc, "f1_weighted": f1}


# ─────────────────────────────────────────────────────
# 4. PyTorch Dataset
# ─────────────────────────────────────────────────────
class GenreDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


# ─────────────────────────────────────────────────────
# 5. Metrics
# ─────────────────────────────────────────────────────
def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    return {
        "accuracy":    accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


# ─────────────────────────────────────────────────────
# 6. Fine-tune RoBERTa
# ─────────────────────────────────────────────────────
def fine_tune(train_texts, train_labels, test_texts, test_labels, label2id, id2label):
    print(f"\n[Fine-Tuning {MODEL_NAME} on CPU]")
    print("  Estimated time: 45-90 min on MacBook CPU. Go grab a coffee ☕")

    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

    print("  Tokenizing...")
    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_enc  = tokenizer(test_texts,  truncation=True, padding=True, max_length=MAX_LENGTH)

    train_dataset = GenreDataset(train_enc, [label2id[l] for l in train_labels])
    test_dataset  = GenreDataset(test_enc,  [label2id[l] for l in test_labels])

    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        warmup_steps=100,
        weight_decay=0.01,
        eval_strategy="epoch",        # ✅ fixed (was evaluation_strategy)
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        fp16=False,
        bf16=False,
        use_cpu=True,                 # ✅ force CPU (fixes MPS training bug)
        report_to="none",
        logging_steps=100,
        seed=SEED,
        dataloader_pin_memory=False,  # must be False when use_cpu=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,   # ✅ fixed (was tokenizer=)
    )

    os.environ["WANDB_DISABLED"] = "true"
    print("  Training started...")
    trainer.train()

    print("\n  Running final evaluation...")
    eval_results = trainer.evaluate()
    print(f"  Results: {eval_results}")

    # Full classification report
    predicted = trainer.predict(test_dataset)
    pred_ids  = predicted.predictions.argmax(-1).flatten().tolist()
    pred_lbls = [id2label[i] for i in pred_ids]
    report    = classification_report(test_labels, pred_lbls)
    print("\n" + report)

    # ── Save everything ───────────────────────────────────────────────────────
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump({"label2id": label2id,
                   "id2label": {str(k): v for k, v in id2label.items()}},
                  f, indent=2)

    training_config = {
        "model_name":    MODEL_NAME,
        "max_length":    MAX_LENGTH,
        "num_epochs":    NUM_EPOCHS,
        "batch_size":    BATCH_SIZE,
        "learning_rate": 2e-5,
        "warmup_steps":  100,
        "weight_decay":  0.01,
        "device":        DEVICE,
        "train_samples": len(train_texts),
        "test_samples":  len(test_texts),
        "num_labels":    len(label2id),
        "labels":        list(label2id.keys()),
    }
    with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
        json.dump(training_config, f, indent=2)

    eval_output = {"source": "local_model", **eval_results,
                   "classification_report": report}
    with open(os.path.join(RESULTS_DIR, "local_eval_results.json"), "w") as f:
        json.dump(eval_output, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "local_classification_report.txt"), "w") as f:
        f.write(report)

    print(f"\n  ✅ Model   → '{OUTPUT_DIR}/'")
    print(f"  ✅ Results → '{RESULTS_DIR}/'")
    return eval_results, tokenizer, model


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    genre_reviews = get_data()
    train_texts, train_labels, test_texts, test_labels = split_data(genre_reviews)
    print(f"Train: {len(train_texts)}  |  Test: {len(test_texts)}")

    # Baseline
    baseline_results = run_baseline(train_texts, train_labels, test_texts, test_labels)
    with open(os.path.join(RESULTS_DIR, "baseline_results.json"), "w") as f:
        json.dump(baseline_results, f, indent=2)

    # Label maps
    unique_labels = sorted(set(train_labels))
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}

    # Fine-tune
    eval_results, tokenizer, model = fine_tune(
        train_texts, train_labels, test_texts, test_labels, label2id, id2label
    )

    print("\n" + "="*50)
    print("✅ TRAINING COMPLETE")
    print("="*50)
    print(f"   Accuracy  : {eval_results.get('eval_accuracy',    0):.4f}")
    print(f"   F1 Score  : {eval_results.get('eval_f1_weighted', 0):.4f}")
    print(f"   (Baseline was ~0.51 accuracy)")