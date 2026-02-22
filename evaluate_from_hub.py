"""
Task 8: Load RoBERTa model from Hugging Face Hub and re-evaluate.
- Downloads test data fresh (no pickle dependency — works inside Docker too)

Usage:
    python evaluate_from_hub.py --repo YOUR_HF_USERNAME/goodreads-genre-classifier
"""

import argparse
import json
import os
import pickle
import random
import gzip
import requests

import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
)

RESULTS_DIR      = "results"
CACHE_FILE       = "genre_reviews_dict.pickle"
SAMPLE_PER_GENRE = 500
TRAIN_SPLIT      = 0.8
MAX_LENGTH       = 128
SEED             = 42
random.seed(SEED)

os.makedirs(RESULTS_DIR, exist_ok=True)

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
    """Load from cache if available, otherwise download."""
    if os.path.exists(CACHE_FILE):
        print("  Using cached data...")
        return pickle.load(open(CACHE_FILE, "rb"))
    print("  Downloading data (first run only, ~5 min)...")
    genre_reviews = {}
    for genre, url in GENRE_URLS.items():
        print(f"    {genre}...")
        genre_reviews[genre] = load_reviews(url, head=5000, sample_size=SAMPLE_PER_GENRE)
    pickle.dump(genre_reviews, open(CACHE_FILE, "wb"))
    return genre_reviews


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


def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    return {
        "accuracy":    accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def load_test_data():
    genre_reviews = get_data()
    test_texts, test_labels = [], []
    for genre, reviews in genre_reviews.items():
        reviews   = random.sample(reviews, min(len(reviews), SAMPLE_PER_GENRE))
        split_idx = int(len(reviews) * TRAIN_SPLIT)
        for r in reviews[split_idx:]:
            test_texts.append(r); test_labels.append(genre)
    return test_texts, test_labels


def evaluate_from_hub(repo_id: str):
    print(f"Loading RoBERTa model from Hub: {repo_id}")
    tokenizer = RobertaTokenizerFast.from_pretrained(repo_id)
    model     = RobertaForSequenceClassification.from_pretrained(repo_id)

    label2id = model.config.label2id
    id2label = model.config.id2label

    print("Loading test data...")
    test_texts, test_labels = load_test_data()
    print(f"  Test samples: {len(test_texts)}")

    print("Tokenizing...")
    test_enc = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_dataset = GenreDataset(test_enc, [label2id[l] for l in test_labels])

    training_args = TrainingArguments(
        output_dir="hub_eval_tmp",
        per_device_eval_batch_size=32,
        use_cpu=True,
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    print("Evaluating...")
    hub_results = trainer.evaluate()

    predicted = trainer.predict(test_dataset)
    pred_ids  = predicted.predictions.argmax(-1).flatten().tolist()
    pred_lbls = [id2label[i] for i in pred_ids]
    report    = classification_report(test_labels, pred_lbls)
    print("\n" + report)

    hub_output = {"source": "hub_model", "repo_id": repo_id,
                  **hub_results, "classification_report": report}
    with open(os.path.join(RESULTS_DIR, "hub_eval_results.json"), "w") as f:
        json.dump(hub_output, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "hub_classification_report.txt"), "w") as f:
        f.write(report)

    # ── Comparison ────────────────────────────────────────────────────────────
    local_path = os.path.join(RESULTS_DIR, "local_eval_results.json")
    if os.path.exists(local_path):
        with open(local_path) as f:
            local = json.load(f)
        print("\n── Comparison ───────────────────────────────────────")
        print(f"{'Metric':<25} {'Local':>10} {'Hub':>10}")
        print("-" * 50)
        for key in ["eval_accuracy", "eval_f1_weighted"]:
            lv   = local.get(key, "N/A")
            hv   = hub_results.get(key, "N/A")
            lv_s = f"{lv:.4f}" if isinstance(lv, float) else str(lv)
            hv_s = f"{hv:.4f}" if isinstance(hv, float) else str(hv)
            print(f"  {key:<23} {lv_s:>10} {hv_s:>10}")
        comparison = {"local": local, "hub": hub_output}
        with open(os.path.join(RESULTS_DIR, "comparison.json"), "w") as f:
            json.dump(comparison, f, indent=2)
        print("\nComparison saved → results/comparison.json")

    print(f"\n✅ Hub evaluation complete!")
    print(f"   Accuracy : {hub_results.get('eval_accuracy',    0):.4f}")
    print(f"   F1 Score : {hub_results.get('eval_f1_weighted', 0):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    evaluate_from_hub(args.repo)