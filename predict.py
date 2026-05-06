import json
from src.a3.dataset import WiCDataset, collate_fn
from src.a3.model import TargetWordModel
import torch
import sys
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader


MMBERT_PATH = "/fp/projects01/ec403/hf_models/mmBERT-base"
def prediction(model, loader, device):
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            encodings, target_tokens_a, target_tokens_b, _, _ = batch
            encodings = {k: v.to(device) for k, v in encodings.items()}
            logits = model(**encodings, target_tokens_a=target_tokens_a, target_tokens_b=target_tokens_b)
            preds = logits.argmax(dim=1).tolist()
            all_preds.extend([p + 1 for p in preds])
    return all_preds

def save_predictions(preds, output_path):
    with open(output_path, "w") as f:
        for pred in preds:
            json.dump({"median_label": pred}, f)
            f.write("\n")
    print(f"Saved {len(preds)} predictions to {output_path}")

def main(model_path, data_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Last modell
    model = TargetWordModel(MMBERT_PATH)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Last dataset
    tokenizer = AutoTokenizer.from_pretrained(MMBERT_PATH)
    collate = lambda batch: collate_fn(batch, tokenizer)
    dataset = WiCDataset(data_path)
    loader = DataLoader(dataset, batch_size=16, collate_fn=collate)

    preds = prediction(model, loader, device)
    save_predictions(preds, output_path)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python predict.py <model_path> <data_path> <output_path>")
        print("Example: python predict.py best_model.pt test.jsonl.gz predictions.jsonl")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])