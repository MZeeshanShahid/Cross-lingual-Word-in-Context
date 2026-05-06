import torch
from transformers import AutoModel
import torch.nn as nn


class BOSModel(nn.Module):
    # same as a2, just renamed for clarity when comparing
    def __init__(self, model_path):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        self.bert.gradient_checkpointing_enable()
        self.dropout = nn.Dropout(0.1)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 4)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            #token_type_ids=token_type_ids
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_embedding)
        return self.classifier(x)


class TargetWordModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.1)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size * 2, 4)

    def forward(self, input_ids, attention_mask, token_type_ids=None, target_tokens_a=None, target_tokens_b=None, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state

        vec_a = self.pool_target_tokens(hidden_states, target_tokens_a)
        vec_b = self.pool_target_tokens(hidden_states, target_tokens_b)

        # gives the model a direct hint about how similar the two word vectors are
        #failed experiment
        #combined = torch.cat([vec_a, vec_b, (vec_a - vec_b).abs()], dim=-1)
        combined = torch.cat([vec_a, vec_b], dim=-1)
        x = self.dropout(combined)
        return self.classifier(x)

    def pool_target_tokens(self, hidden_states, target_tokens):
        pooled = []
        for i, token_indices in enumerate(target_tokens):
            vecs = hidden_states[i, token_indices, :]
            pooled.append(vecs.mean(dim=0))
        return torch.stack(pooled)