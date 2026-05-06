import sys
import torch
sys.path.append(".")

from transformers import AutoTokenizer
from src.a3.dataset import WiCDataset, collate_fn
from src.a3.model import TargetWordModel
from src.train import train, evaluate

MMBERT_PATH = "/fp/projects01/ec403/hf_models/mmBERT-base"
DATA_BASE   = "/fp/projects01/ec403/IN5550/obligatories/2"

TRAIN_LANGS = {
    "de": f"{DATA_BASE}/de_train.jsonl.gz",
}

VAL_LANGS = {
    "en": f"{DATA_BASE}/en_dev.jsonl.gz",
    "de": f"{DATA_BASE}/de_dev.jsonl.gz",
    "zh": f"{DATA_BASE}/zh_dev.jsonl.gz",
    "ru": f"{DATA_BASE}/ru_dev.jsonl.gz",
    "es": f"{DATA_BASE}/es_dev.jsonl.gz",
    "no": f"{DATA_BASE}/no_dev.jsonl.gz",
}

tokenizer = AutoTokenizer.from_pretrained(MMBERT_PATH)
collate = lambda batch: collate_fn(batch, tokenizer)

# load all val sets once
val_sets = {lang: WiCDataset(path) for lang, path in VAL_LANGS.items()}

for train_lang, train_path in TRAIN_LANGS.items():
    print(f"\n{'='*50}")
    print(f"Training on: {train_lang}")
    print(f"{'='*50}")

    train_dataset = WiCDataset(train_path)
    model = TargetWordModel(MMBERT_PATH)

    # train on one language
    train(
        model,
        train_dataset,
        val_sets["en"],
        val_sets["en"].data,
        collate_fn=collate,
        epochs=5,     
        batch_size=4,
        lr=1e-5
    )

    # evaluate on all languages after training
    print(f"\n--- Results for model trained on {train_lang} ---")
    for val_lang, val_dataset in val_sets.items():
        from src.train import evaluate
        metrics = evaluate(model, val_dataset, val_dataset.data, collate_fn=collate)
        print(f"{val_lang}: Alpha={metrics['krippendorff_alpha']:.4f} | Acc={metrics['accuracy']:.4f}")
    
    del model
    torch.cuda.empty_cache()