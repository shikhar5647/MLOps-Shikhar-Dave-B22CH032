from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu
import re

INPUT_FILE = "/Users/shikhar/Desktop/MLDLOPS_Exams/MLOps-Shikhar-Dave-B22CH032/MLOps-Shikhar-Dave-B22CH032/input.rtf"                 # Bengali (given)
REFERENCE_FILE = "/Users/shikhar/Desktop/MLDLOPS_Exams/MLOps-Shikhar-Dave-B22CH032/MLOps-Shikhar-Dave-B22CH032/reference_english.txt" # English ground truth


# 🔹 Minimal cleaning (ONLY to fix spacing issue)
def read_bengali(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # collapse weird spacing (do NOT over-clean)
    text = re.sub(r'\s+', ' ', text)

    # split sentences (based on danda or period)
    sentences = re.split(r'[।.!?]+', text)

    return [s.strip() for s in sentences if s.strip()]


# 🔹 Read English reference (already clean)
def read_english(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# 🔹 Load model
def load_model():
    model_name = "Helsinki-NLP/opus-mt-bn-en"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


# 🔹 Translate
def translate(texts, tokenizer, model):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    outputs = model.generate(**inputs, max_new_tokens=100)

    return [
        tokenizer.decode(o, skip_special_tokens=True)
        for o in outputs
    ]


def main():
    inputs_bn = read_bengali(INPUT_FILE)
    references_en = read_english(REFERENCE_FILE)

    print(f"Loaded {len(inputs_bn)} Bengali sentences")
    print(f"Loaded {len(references_en)} reference sentences")

    # ⚠️ Important: lengths may differ due to bad formatting
    min_len = min(len(inputs_bn), len(references_en))
    inputs_bn = inputs_bn[:min_len]
    references_en = references_en[:min_len]

    tokenizer, model = load_model()

    predictions_en = translate(inputs_bn, tokenizer, model)

    # BLEU
    bleu = sacrebleu.corpus_bleu(predictions_en, [references_en])

    print("\n" + "="*50)
    print("BLEU Evaluation Result")
    print("="*50)
    print(f"Samples used       : {min_len}")
    print(f"BLEU score         : {bleu.score:.2f}")
    print(f"Detailed signature : {bleu}")
    print("="*50)


if __name__ == "__main__":
    main()