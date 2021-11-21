import torch
from torch._C import DisableTorchFunction
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import utils


class ClassificationHead(nn.Module):
    def __init__(self, l_model, i_model, r_model, dropout):
        super(ClassificationHead, self).__init__()
        self.dance = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(l_model, i_model),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(i_model, r_model)
        )
    
    def forward(self, x):
        return self.dance(x)


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.is_pooling = False
        self.bert = AutoModel.from_pretrained(args.pretrain_path)
        self.classification = ClassificationHead(args.l_model, args.l_model, 1, args.dropout)
    
    def forward(self, input_ids, attention_mask):
        utils.debug("input_ids", input_ids)
        x = self.bert(input_ids, attention_mask=attention_mask)
        if self.is_pooling:
            x = x.pooler_output
        else:
            x = x.last_hidden_state[:, 0, :]
        terminal = self.classification(x)
        return terminal
