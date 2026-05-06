import torch
from torch.utils.data import Dataset
import json
import gzip

class WiCDataset(Dataset):
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
        
        # convert raw labels to distribution over [1,2,3,4]
        dist = [0.0, 0.0, 0.0, 0.0]
        for l in item["labels"]:
            dist[l - 1] += 1
        total = sum(dist)
        dist = [x / total for x in dist]
        
        return (            
            item["sentence_a"],
            item["sentence_b"],
            item["word_indices_a"],
            item["word_indices_b"],
            item["median_label"] - 1,
            dist)

        
def collate_fn(batch, tokenizer):
    sentences_a, sentences_b, indices_a, indices_b, labels, dists = zip(*batch)

    encodings = tokenizer(
        list(sentences_a),
        list(sentences_b),
        padding=True,
        truncation=True,
        max_length=256,
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    target_tokens_a = get_target_token_indices(
        encodings["offset_mapping"], indices_a, encodings, sentence_part=0
    )
    target_tokens_b = get_target_token_indices(
        encodings["offset_mapping"], indices_b, encodings, sentence_part=1
    )

    encodings.pop("offset_mapping")

    labels = torch.tensor(labels)
    dists = torch.tensor(dists, dtype=torch.float32)

    return encodings, target_tokens_a, target_tokens_b, labels, dists

def get_target_token_indices(offset_mapping, word_indices, encodings, sentence_part):
    batch_indices = []
    for i, (word_idx, offsets) in enumerate(zip(word_indices, offset_mapping)):
        word_start, word_end = word_idx
        seq_ids = encodings.sequence_ids(i)

        token_indices = [
            j for j, (s, e) in enumerate(offsets.tolist())
            if e > s and s < word_end and e > word_start
            and seq_ids[j] == sentence_part
        ]

        if len(token_indices) == 0:
            token_indices = [1]
        batch_indices.append(token_indices)
    return batch_indices