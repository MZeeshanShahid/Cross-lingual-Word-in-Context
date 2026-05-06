import sys
sys.path.append(".")

from transformers import AutoTokenizer
from src.a3.dataset import WiCDataset, collate_fn
from src.a3.model import TargetWordModel
from src.train import train

TRAIN_PATH = "/fp/projects01/ec403/IN5550/obligatories/2/en_train.jsonl.gz"
VAL_PATH   = "/fp/projects01/ec403/IN5550/obligatories/2/en_dev.jsonl.gz"
MMBERT_PATH = "/fp/projects01/ec403/hf_models/mmBERT-base"

tokenizer = AutoTokenizer.from_pretrained(MMBERT_PATH)
collate = lambda batch: collate_fn(batch, tokenizer)

train_dataset = WiCDataset(TRAIN_PATH)
val_dataset   = WiCDataset(VAL_PATH)

print("--- mmBERT | Target-word | lr=1e-5 | batch size 16 | feature engineering --- ")
train(TargetWordModel(MMBERT_PATH), train_dataset, val_dataset, val_dataset.data,
      collate_fn=collate, epochs=15, batch_size=16, lr=1e-5)