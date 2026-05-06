import sys
import torch
import os
sys.path.append(".")

from transformers import AutoTokenizer
from torch.utils.data import ConcatDataset
from src.a3.dataset import WiCDataset, collate_fn
from src.a3.model import TargetWordModel
from src.train import train, evaluate

MMBERT_PATH = "/fp/projects01/ec403/hf_models/mmBERT-base"
DATA_BASE   = "/fp/projects01/ec403/IN5550/obligatories/2"

tokenizer = AutoTokenizer.from_pretrained(MMBERT_PATH)
collate = lambda batch: collate_fn(batch, tokenizer)

no_val = WiCDataset(f"{DATA_BASE}/no_dev.jsonl.gz")

experiments = {
    "no_only":     ["no_dev"],
    "de_en_no":    ["de_train", "en_train", "no_dev"],
    "de_en_es_no": ["de_train", "en_train", "es_train", "no_dev"],
}

os.makedirs("models", exist_ok=True)

for name, files in experiments.items():
    print(f"\n{'='*50}")
    print(f"Training on: {name}")
    print(f"{'='*50}")

    train_dataset = ConcatDataset([
        WiCDataset(f"{DATA_BASE}/{f}.jsonl.gz") for f in files
    ])

    model = TargetWordModel(MMBERT_PATH)

    # monitor on norsk val under trening
    train(model, train_dataset, no_val, no_val.data,
          collate_fn=collate, epochs=10, batch_size=16, lr=1e-5)

    # last beste modell og lagre med navn
    model.load_state_dict(torch.load("best_model.pt"))
    torch.save(model.state_dict(), f"models/c_{name}.pt")
    print(f"Saved models/c_{name}.pt")

    del model
    torch.cuda.empty_cache()