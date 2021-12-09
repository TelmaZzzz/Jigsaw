import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from model import *
import logging
import utils
from torch.optim import lr_scheduler


class ModelConfig(object):
    def __init__(self, args):
        self.l_model = args.l_model
        self.dropout = args.dropout
        self.pretrain_path = args.pretrain_path
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epoch = args.epoch
        self.opt_step = args.opt_step
        self.eval_step = args.eval_step
        self.margin = args.margin
        self.Tmax = args.Tmax
        self.min_lr = args.min_lr


class Trainer(object):
    def __init__(self, args):
        self.config = ModelConfig(args)
        self.pretrain_path = args.pretrain_path
        self.tokenizer_path = args.tokenizer_path
        self.model_load = args.model_load
        self.device = args.device
        self.model_save = args.model_save
        self.local_rank = args.local_rank
    
    def reset(self, tokenizer_path=None, model_save=None, model_load=None, pretrain_path=None):
        self.tokenizer_path = tokenizer_path if tokenizer_path is not None else self.tokenizer_path
        self.pretrain_path = pretrain_path if pretrain_path is not None else self.pretrain_path
        self.model_load = model_load if model_load is not None else self.model_load
        self.model_save = model_save if model_save is not None else self.model_save
    
    def get_tokenizer(self):
        return self.tokenizer

    def model_init(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = BaseModel(self.config)
        if self.model_load:
            self.model.load_state_dict(torch.load(self.model_load, map_location=torch.device('cpu')))
        else:
            self.model.bert.config.device = self.device
        self.model.to(self.device)
        if self.local_rank != -1:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=False, find_unused_parameters=True)
    
    def optimizer_init(self, train_size):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
                # "lr": args.learning_rate * 0.1,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                # "lr": args.learning_rate * 0.1,
            },
        ]
        self.step = 0
        self.score = []
        self.Loss_fn = torch.nn.MarginRankingLoss(self.config.margin)
        self.optimizer = AdamW(optimizer_grouped_parameters, self.config.lr, correct_bias=False)
        # self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=train_size // self.config.opt_step, \
        #     num_training_steps=train_size * self.config.epoch // self.config.opt_step)
        # self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(self.optimizer, num_warmup_steps=train_size // self.config.opt_step, \
            # num_training_steps=train_size * self.config.epoch // self.config.opt_step)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=self.config.Tmax, eta_min=self.config.min_lr)
    def _get_logits(self, batch):
        sen1_input_ids = batch["sen1_input_ids"].to(self.device)
        sen2_input_ids = batch["sen2_input_ids"].to(self.device)
        sen1_attention_mask = batch["sen1_attention_mask"].to(self.device)
        sen2_attention_mask = batch["sen2_attention_mask"].to(self.device)
        sen1_logits = self.model(sen1_input_ids, sen1_attention_mask)
        sen2_logits = self.model(sen2_input_ids, sen2_attention_mask)
        return sen1_logits.view(-1), sen2_logits.view(-1)
    
    def train(self, batch):
        labels = batch["labels"].to(self.device)
        sen1_logits, sen2_logits = self._get_logits(batch)
        utils.debug("sen1_logits shape", sen1_logits.shape)
        utils.debug("sen2_logits shape", sen2_logits.shape)
        loss = self.Loss_fn(sen1_logits, sen2_logits, labels).mean()
        utils.debug("loss", loss)
        loss.backward()
        self.step += 1
        if self.step % self.config.opt_step == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

    def eval(self, valid_iter, valid_len):
        self.model.eval()
        acc = 0
        total = 0
        mean_loss = 0
        sen1_predicts, sen2_predicts = [], []
        with torch.no_grad():
            for batch in valid_iter:
                sen1_logits, sen2_logits = self._get_logits(batch)
                sen1_predicts.append(sen1_logits)
                sen2_predicts.append(sen2_logits)
            sen1_logits = utils.distributed_concat(torch.cat(sen1_predicts, dim=0), valid_len)
            dist.barrier()
            sen2_logits = utils.distributed_concat(torch.cat(sen2_predicts, dim=0), valid_len)
            if dist.get_rank() == 0:
                acc = self.metrics(sen1_logits, sen2_logits)
                if len(self.score) < 5:
                    self.score.append(acc)
                    self.save(acc)
                else:
                    if self.score[0] < acc:
                        self.rm(self.score[0])
                        self.score = self.score[1:]
                        self.save(acc)
                        self.score.append(acc)
                self.score = sorted(self.score)
                dist.barrier()
            else:
                dist.barrier()
        self.model.train()

    def predict(self, batch):
        sen1_input_ids = batch["sen1_input_ids"].to(self.device)
        sen1_attention_mask = batch["sen1_attention_mask"].to(self.device)
        sen1_logits = self.model(sen1_input_ids, sen1_attention_mask)
        return sen1_logits.view(-1)
    
    def metrics(self, sen1_logits, sen2_logits):
        acc = (sen1_logits < sen2_logits).sum()
        total = sen1_logits.size(0)
        loss = self.Loss_fn(sen1_logits, sen2_logits, torch.tensor([-1] * sen1_logits.size(0), dtype=torch.long).to(self.device))
        logging.info("acc:{:.4f}%({}/{}) loss:{:.4f}".format(acc / total * 100, acc, total, loss))
        return acc / total * 100
    
    def save(self, score):
        path = self.model_save + "_score_{:.4f}.pkl".format(score)
        if self.local_rank == -1:
            torch.save(self.model.state_dict(), path)
        else:
            torch.save(self.model.module.state_dict(), path)
        logging.info("Save Model")

    def rm(self, score):
        path = self.model_save + "_score_{:.4f}.pkl".format(score)
        if os.path.exists(path):
            os.remove(path)
            logging.info("Remove Model")

    def freeze(self):
        for k, v in self.model.named_parameters():
            if "layer" in k:
                v.requires_grad = False
    
    def unfreeze(self, layer):
        flag = False
        if layer == -1:
            flag = True
        for k, v in self.model.named_parameters():
            if f"layer.{layer}" in k:
                flag = True
                utils.debug("k", k)
            v.requires_grad = flag
    
    def log_parameters(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                logging.info(f"train name: {name}")
            else:
                logging.info(f"force name: {name}")
