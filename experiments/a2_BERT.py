import sys
sys.path.append(".")

import torch
from transformers import AutoTokenizer
from src.a2.dataset import BERTDataset, collate_fn
from src.a2.model import BertModel
from src.train import train

TRAIN_PATH = "/fp/projects01/ec403/IN5550/obligatories/2/en_train.jsonl.gz"
VAL_PATH   = "/fp/projects01/ec403/IN5550/obligatories/2/en_dev.jsonl.gz"

MODEL_PATH = "/fp/projects01/ec403/hf_models/mmBERT-base"
#MODEL_PATH = "/fp/projects01/ec403/hf_models/bert-base-cased"
print("Laster tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Laster data...")
train_dataset = BERTDataset(TRAIN_PATH)
val_dataset   = BERTDataset(VAL_PATH)

print("Laster modell...")
model = BertModel(MODEL_PATH)

print("Trener...")
train(
    model,
    train_dataset,
    val_dataset,
    val_dataset.data,
    collate_fn=lambda batch: collate_fn(batch, tokenizer),
    epochs=10,
    batch_size=16,
    lr=2e-5
)