import sys
sys.path.append(".")

from src.embeddings import load_embedding
from src.dataset import WiCDataset
from src.model import StaticMLP
from src.train import train

TRAIN_PATH = "/fp/projects01/ec403/IN5550/obligatories/2/en_train.jsonl.gz"
VAL_PATH   = "/fp/projects01/ec403/IN5550/obligatories/2/en_dev.jsonl.gz"
EMB_PATH   = "/fp/projects01/ec403/models/static/40/model.bin"

print("Laster embeddings...")
emb_model = load_embedding(EMB_PATH)

print("Laster data...")
train_dataset = WiCDataset(TRAIN_PATH, emb_model, N=3)
val_dataset   = WiCDataset(VAL_PATH,   emb_model, N=3)

input_dim = emb_model.vector_size * 3
model = StaticMLP(input_dim=input_dim)

print("Trener...")
train(model, train_dataset, val_dataset, val_dataset.data, epochs=10, batch_size=64, lr=1e-3)