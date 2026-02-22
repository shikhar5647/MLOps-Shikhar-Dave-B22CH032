"""
Task 7: Push fine-tuned RoBERTa model to Hugging Face Hub.

Usage:
    python push_to_hub.py --repo YOUR_HF_USERNAME/goodreads-genre-classifier
"""

import argparse
import json
from huggingface_hub import HfApi
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
)

MODEL_DIR = "saved_model"


def push(repo_id: str, private: bool = False):
    print(f"Loading RoBERTa model from '{MODEL_DIR}'...")

    with open(f"{MODEL_DIR}/label_map.json") as f:
        maps = json.load(f)
    label2id = maps["label2id"]
    id2label = {int(k): v for k, v in maps["id2label"].items()}

    model     = RobertaForSequenceClassification.from_pretrained(
        MODEL_DIR, label2id=label2id, id2label=id2label
    )
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR)

    print(f"Pushing to: {repo_id} ...")
    model.push_to_hub(repo_id, private=private)
    tokenizer.push_to_hub(repo_id, private=private)

    api = HfApi()
    for filename in [
        f"{MODEL_DIR}/label_map.json",
        f"{MODEL_DIR}/training_config.json",
        "results/local_eval_results.json",
        "results/local_classification_report.txt",
    ]:
        try:
            api.upload_file(
                path_or_fileobj=filename,
                path_in_repo=filename.split("/")[-1],
                repo_id=repo_id,
            )
            print(f"  Uploaded: {filename}")
        except FileNotFoundError:
            print(f"  Skipped (not found): {filename}")

    print(f"\n✅ Model pushed!")
    print(f"   View at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    push(args.repo, args.private)