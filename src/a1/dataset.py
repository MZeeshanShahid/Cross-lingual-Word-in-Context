import gzip
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class WiCDataset(Dataset):
    def __init__(self, path, emb_model, N=3):
        self.data = self.load_data(path)
        self.emb_model = emb_model
        self.N = N

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
        vec_a = self.get_context_embedding(
            item["sentence_a"],
            item["word_indices_a"][0],
            item["word_indices_a"][1]
        )
        vec_b = self.get_context_embedding(
            item["sentence_b"],
            item["word_indices_b"][0],
            item["word_indices_b"][1]
        )
        combined = np.concatenate([vec_a, vec_b, np.abs(vec_a - vec_b)])
        label = item["median_label"] - 1
        return torch.tensor(combined, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def get_context_embedding(self, sentence, word_start, word_end):
        words = sentence.split()
        target_idx = sentence[:word_start].count(" ")
        start = max(0, target_idx - self.N)
        end = min(len(words), target_idx + self.N + 1)
        context_words = words[start:end]

        vecs = []
        for w in context_words:
            w_clean = w.lower().strip(".,!?\"'")
            if w_clean in self.emb_model:
                vecs.append(self.emb_model[w_clean])

        if len(vecs) == 0:
            return np.zeros(self.emb_model.vector_size)
        return np.mean(vecs, axis=0)