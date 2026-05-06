import sys
import torch
sys.path.append(".")

from transformers import AutoTokenizer
from torch.utils.data import ConcatDataset
from src.a3.dataset import WiCDataset, collate_fn
from src.a3.model import TargetWordModel
from src.train import train, evaluate

MMBERT_PATH = "/fp/projects01/ec403/hf_models/mmBERT-base"
DATA_BASE   = "/fp/projects01/ec403/IN5550/obligatories/2"

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
val_sets = {lang: WiCDataset(path) for lang, path in VAL_LANGS.items()}

experiments = {
    "all": ["en", "de", "zh", "ru", "es"],
    "all_except_zh": ["en", "de", "ru", "es"],
    "european": ["en", "de", "ru", "es"],  # same as above but named clearly
}

for name, langs in experiments.items():
    print(f"\n{'='*50}")
    print(f"Training on: {name} ({', '.join(langs)})")
    print(f"{'='*50}")

    train_dataset = ConcatDataset([
        WiCDataset(f"{DATA_BASE}/{lang}_train.jsonl.gz") for lang in langs
    ])

    model = TargetWordModel(MMBERT_PATH)
    train(model, train_dataset, val_sets["en"], val_sets["en"].data,
          collate_fn=collate, epochs=10, batch_size=16, lr=1e-5)

    print(f"\n--- Results: {name} ---")
    for val_lang, val_dataset in val_sets.items():
        metrics = evaluate(model, val_dataset, val_dataset.data, collate_fn=collate)
        print(f"{val_lang}: Alpha={metrics['krippendorff_alpha']:.4f} | Acc={metrics['accuracy']:.4f}")

    del model
    torch.cuda.empty_cache()