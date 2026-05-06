import torch
from torch.utils.data import Dataset
import json
import gzip

class BERTDataset(Dataset):
    def __init__(self, path):
        self.data = self.load_data(path)
    
    def load_data(self, path):
        data = []
        open_fn = gzip.open if path.endswith(".gz") else open
        with open_fn(path, "rt") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return (            
            item["sentence_a"],
            item["sentence_b"],
            item["median_label"] - 1)
        
def collate_fn(batch, tokenizer):
    sentences_a, sentences_b, labels = zip(*batch)

    encodings = tokenizer(
        list(sentences_a),
        list(sentences_b),
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    labels = torch.tensor(labels)

    return encodings, labels
        