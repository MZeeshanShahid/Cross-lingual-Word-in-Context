import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
# Import model

class BertModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
        model_path,
        )
        self.dropout = nn.Dropout(0.1)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 4)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None
        )

        cls_embedding = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_embedding)
        logits = self.classifier(x)
        return logits

