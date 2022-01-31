import torch
from torch._C import DisableTorchFunction
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, XLNetModel
import utils
import logging


class ClassificationHead(nn.Module):
    def __init__(self, l_model, i_model, r_model, dropout):
        super(ClassificationHead, self).__init__()
        self.dance = nn.Sequential(
            nn.LayerNorm(l_model),
            # nn.Dropout(dropout),
            # nn.Linear(l_model, i_model),
            # nn.Dropout(dropout),
            nn.Linear(i_model, 256),
            # nn.Tanh(),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(256, r_model)
        )

    def forward(self, x):
        return self.dance(x)


class ClassificationHeadV2(nn.Module):
    def __init__(self, l_model, i_model, r_model, dropout):
        super(ClassificationHeadV2, self).__init__()
        self.dance = nn.Sequential(
            nn.LayerNorm(l_model),
            # nn.Dropout(dropout),
            # nn.Linear(l_model, i_model),
            # nn.Dropout(dropout),
            nn.Linear(l_model, l_model),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(l_model, i_model),
            # nn.Tanh(),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(i_model, r_model)
        )

    def forward(self, x):
        return self.dance(x)


class ClassificationHeadV3(nn.Module):
    def __init__(self, l_model, i_model, r_model, dropout):
        super(ClassificationHeadV3, self).__init__()
        self.dance = nn.Sequential(
            nn.LayerNorm(l_model),
            # nn.Dropout(dropout),
            # nn.Linear(l_model, i_model),
            # nn.Dropout(dropout),
            nn.Linear(l_model, l_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.LayerNorm(l_model),
            nn.Linear(l_model, i_model),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.LayerNorm(i_model),
            nn.Linear(i_model, r_model)
        )

    def forward(self, x):
        return self.dance(x)


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.bert = AutoModel.from_pretrained(args.pretrain_path)
        self.classification = ClassificationHead(args.l_model, args.l_model, 1, args.dropout)
    
    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids, attention_mask=attention_mask)
        pooler_output = x.pooler_output
        terminal = self.classification(pooler_output)
        return terminal


class BaseModelV2(nn.Module):
    def __init__(self, args):
        super(BaseModelV2, self).__init__()
        self.bert = AutoModel.from_pretrained(args.pretrain_path)
        self.classification = ClassificationHeadV2(args.l_model * 3, args.l_model, 1, args.dropout)
    
    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids, attention_mask=attention_mask)
        # pooler_output = x.pooler_output
        last_hidden = x.last_hidden_state * attention_mask.unsqueeze(-1)
        cls = last_hidden[:, 0, :]
        pooled = torch.max(last_hidden, axis=1)[0]
        mean_pooled = last_hidden.sum(axis=1) / attention_mask.sum(axis=-1).unsqueeze(-1)
        # logging.info(f"pooled: {pooled.shape}")
        # logging.info(f"cls: {cls.shape}")
        # logging.info(f"mean_pooled: {mean_pooled.shape}")
        terminal = self.classification(torch.cat([cls, pooled, mean_pooled], dim=-1))
        return terminal


class BaseModelV3(nn.Module):
    def __init__(self, args):
        super(BaseModelV3, self).__init__()
        self.bert = AutoModel.from_pretrained(args.pretrain_path)
        self.dance = nn.Linear(args.l_model, args.l_model)
        self.classification = ClassificationHead(args.l_model, args.l_model, 1, args.dropout)

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids, attention_mask=attention_mask)
        cls = x.last_hidden_state[:, 0, :]
        terminal = self.dance(cls)
        terminal = self.classification(terminal)
        return terminal


class XLNETModel(nn.Module):
    def __init__(self, args):
        super(XLNETModel, self).__init__()
        self.is_pooling = True
        self.bert = XLNetModel.from_pretrained(args.pretrain_path)
        self.classification = ClassificationHeadV3(args.l_model * 3, args.l_model, 1, args.dropout)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        x = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # pooler_output = x.pooler_output
        last_hidden = x.last_hidden_state * attention_mask.unsqueeze(-1)
        cls = last_hidden[:, -1, :]
        pooled = torch.max(last_hidden, axis=1)[0]
        mean_pooled = last_hidden.sum(axis=1) / attention_mask.sum(axis=-1).unsqueeze(-1)
        # logging.info(f"pooled: {pooled.shape}")
        # logging.info(f"cls: {cls.shape}")
        # logging.info(f"mean_pooled: {mean_pooled.shape}")
        terminal = self.classification(torch.cat([cls, pooled, mean_pooled], dim=-1))
        return terminal
