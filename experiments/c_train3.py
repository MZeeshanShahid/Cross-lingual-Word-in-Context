import sys
import torch
import os
sys.path.append(".")

from transformers import AutoTokenizer
from torch.utils.data import ConcatDataset, Subset
from src.a3.dataset import WiCDataset, collate_fn
from src.a3.model import TargetWordModel
from src.train import train, evaluate

MMBERT_PATH = "/fp/projects01/ec403/hf_models/mmBERT-base"
DATA_BASE   = "/fp/projects01/ec403/IN5550/obligatories/2"

tokenizer = AutoTokenizer.from_pretrained(MMBERT_PATH)
collate = lambda batch: collate_fn(batch, tokenizer)

# split norsk 80/20
no_full = WiCDataset(f"{DATA_BASE}/no_dev.jsonl.gz")
n = len(no_full)
split = int(0.8 * n)
no_train = Subset(no_full, list(range(split)))
no_val   = Subset(no_full, list(range(split, n)))

# lage no_val.data for evaluate_batch
no_val_data = [no_full.data[i] for i in range(split, n)]

print(f"Norsk train: {len(no_train)} | Norsk val: {len(no_val)}")

experiments = {
    "no_split":        (["no_train_subset"], None),
    "de_en_no_split":  (["de_train", "en_train"], None),
    "de_en_es_no_split": (["de_train", "en_train", "es_train"], None),
}

os.makedirs("models", exist_ok=True)

for name, (extra_langs, _) in experiments.items():
    print(f"\n{'='*50}")
    print(f"Training on: {name}")
    print(f"{'='*50}")

    if extra_langs == ["no_train_subset"]:
        train_dataset = no_train
    else:
        train_dataset = ConcatDataset([
            WiCDataset(f"{DATA_BASE}/{l}.jsonl.gz") for l in extra_langs
        ] + [no_train])

    model = TargetWordModel(MMBERT_PATH)
    train(model, train_dataset, no_val, no_val_data,
          collate_fn=collate, epochs=10, batch_size=16, lr=1e-5)

    model.load_state_dict(torch.load("best_model.pt"))
    torch.save(model.state_dict(), f"models/c_{name}.pt")
    print(f"Saved models/c_{name}.pt")

    del model
    torch.cuda.empty_cache()