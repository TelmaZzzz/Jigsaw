import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, XLNetTokenizer, AdamW, get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from model import *
import logging
import utils
from torch.optim import lr_scheduler, optimizer


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
        self.scheduler = args.scheduler
        self.name = args.name
        self.fgm = args.fgm
        self.margin_split = args.margin_split
        self.max_norm = args.max_norm


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
        if self.config.name == "roberta":
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            self.model = BaseModel(self.config)
        elif self.config.name == "xlnet":
            self.tokenizer = XLNetTokenizer.from_pretrained(self.tokenizer_path)
            self.model = XLNETModel(self.config)
        elif self.config.name == "roberta_v2":
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            self.model = BaseModelV2(self.config)
        elif self.config.name == "roberta_v3":
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            self.model = BaseModelV3(self.config)
        if self.model_load:
            self.model.load_state_dict(torch.load(self.model_load, map_location=torch.device('cpu')))
            # logging.info(torch.load(self.model_load, map_location=torch.device('cpu')))
        else:
            self.model.bert.config.device = self.device
        self.model.to(self.device)
        if self.local_rank != -1:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=False, find_unused_parameters=True)

    def get_Loss_fn(self):
        self.Loss_fn = torch.nn.MarginRankingLoss(self.config.margin)
        if self.config.margin_split:
            self.Loss_fn_less = torch.nn.MarginRankingLoss(self.config.margin / 2)

    def get_optimizer_grouped_parameters(self):
        no_decay = ["bias", "LayerNorm.weight"]
        lr_layer = ["classification", "pooler"]
        lr_layer_2 = ["layer.11", "layer.10", "layer.9", "layer.8"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and not any(ll in n for ll in lr_layer) and not any(ll in n for ll in lr_layer_2)
                ],
                "weight_decay": self.config.weight_decay,
                "lr": self.config.lr,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and not any(ll in n for ll in lr_layer) and not any(ll in n for ll in lr_layer_2)
                ],
                "weight_decay": 0.0,
                "lr": self.config.lr,
            },

            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and any(ll in n for ll in lr_layer_2)
                ],
                "weight_decay": self.config.weight_decay,
                "lr": self.config.lr,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and any(ll in n for ll in lr_layer_2)
                ],
                "weight_decay": 0.0,
                "lr": self.config.lr,
            },

            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and any(ll in n for ll in lr_layer)
                ],
                "weight_decay": self.config.weight_decay,
                "lr": self.config.lr,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and any(ll in n for ll in lr_layer)
                ],
                "weight_decay": 0.0,
                "lr": self.config.lr,
            },
        ]
        return optimizer_grouped_parameters

    def get_optimizer_grouped_parameters_v2(self):
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                ],
                "weight_decay": self.config.weight_decay,
                "lr": self.config.lr,
            }
        ]
        return optimizer_grouped_parameters

    def optimizer_init(self, train_size):
        optimizer_grouped_parameters = self.get_optimizer_grouped_parameters()
        # optimizer_grouped_parameters = self.get_optimizer_grouped_parameters_v2()
        self.step = 0
        self.score = []
        self.get_Loss_fn()
        self.optimizer = AdamW(optimizer_grouped_parameters)
        scheduler_map = {
            "get_linear_schedule_with_warmup": get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=train_size // self.config.opt_step, \
                                num_training_steps=train_size * self.config.epoch // self.config.opt_step),
            "get_cosine_with_hard_restarts_schedule_with_warmup": get_cosine_with_hard_restarts_schedule_with_warmup(self.optimizer, num_warmup_steps=train_size // self.config.opt_step // 5, \
                                num_training_steps=train_size * self.config.epoch // self.config.opt_step, num_cycles=int(self.config.epoch * 1.5)),
            "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.Tmax, eta_min=self.config.min_lr),
            "CosineAnnealingWarmRestarts": lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,T_0=self.config.Tmax, T_mult=2, eta_min=self.config.min_lr),
        }
        self.scheduler = scheduler_map[self.config.scheduler]
        if self.config.fgm:
            self.fgm = utils.FGM(self.model)
        # self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=train_size // self.config.opt_step, \
        #     num_training_steps=train_size * self.config.epoch // self.config.opt_step)
        # self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(self.optimizer, num_warmup_steps=train_size // self.config.opt_step, \
            # num_training_steps=train_size * self.config.epoch // self.config.opt_step)
        # self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=self.config.Tmax, eta_min=self.config.min_lr)
        # self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,T_0=self.config.Tmax, T_mult=2, eta_min=self.config.min_lr)
    def _get_logits(self, batch):
        sen1_input_ids = batch["sen1_input_ids"].to(self.device)
        sen2_input_ids = batch["sen2_input_ids"].to(self.device)
        sen1_attention_mask = batch["sen1_attention_mask"].to(self.device)
        sen2_attention_mask = batch["sen2_attention_mask"].to(self.device)
        if "roberta" in self.config.name: 
            sen1_logits = self.model(sen1_input_ids, sen1_attention_mask)
            sen2_logits = self.model(sen2_input_ids, sen2_attention_mask)
        else:
            sen1_token_type_ids = batch["sen1_token_type_ids"].to(self.device)
            sen2_token_type_ids = batch["sen2_token_type_ids"].to(self.device)
            sen1_logits = self.model(sen1_input_ids, sen1_attention_mask, sen1_token_type_ids)
            sen2_logits = self.model(sen2_input_ids, sen2_attention_mask, sen2_token_type_ids)
        return sen1_logits, sen2_logits

    def train(self, batch):
        labels = batch["labels"].to(self.device)
        sen1_logits, sen2_logits = self._get_logits(batch)
        utils.debug("sen1_logits shape", sen1_logits.shape)
        utils.debug("sen2_logits shape", sen2_logits.shape)
        utils.debug("labels shape", labels.shape)
        if self.config.margin_split:
            loss = self.margin_split_Loss_fn(sen2_logits, sen1_logits, labels, batch["margin"].to(self.device))
        else:
            loss = self.Loss_fn(sen2_logits, sen1_logits, labels)
        utils.debug("loss", loss)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_norm)
        if self.config.fgm:
            self.fgm.attack()
            sen1_logits_adv, sen2_logits_adv = self._get_logits(batch)
            if self.config.margin_split:
                loss_adv = self.margin_split_Loss_fn(sen2_logits_adv, sen1_logits_adv, labels, batch["margin"].to(self.device))
            else:
                loss_adv = self.Loss_fn(sen2_logits_adv, sen1_logits_adv, labels)
            loss_adv.backward()
            self.fgm.restore()
        self.step += 1
        if self.step % self.config.opt_step == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            # logging.info(self.optimizer)
        return loss.cpu()

    def eval(self, valid_iter, valid_len):
        # logging.info(self.optimizer)
        if self.local_rank == -1:
            self.eval_(valid_iter)
            return
        self.model.eval()
        acc = 0
        sen1_predicts, sen2_predicts = [], []
        with torch.no_grad():
            for batch in valid_iter:
                sen1_logits, sen2_logits = self._get_logits(batch)
                sen1_predicts.append(sen1_logits.view(-1))
                sen2_predicts.append(sen2_logits.view(-1))
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

    def eval_(self, valid_iter):
        self.model.eval()
        acc = 0
        sen1_predicts, sen2_predicts = [], []
        with torch.no_grad():
            for batch in valid_iter:
                sen1_logits, sen2_logits = self._get_logits(batch)
                sen1_predicts.append(sen1_logits.view(-1))
                sen2_predicts.append(sen2_logits.view(-1))
                # utils.debug("sen1_logits", sen1_logits)
                # utils.debug("sen2_logits", sen2_logits)
            sen1_logits = torch.cat(sen1_predicts, dim=0)
            sen2_logits = torch.cat(sen2_predicts, dim=0)
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
        self.model.train()

    def predict(self, batch):
        sen1_input_ids = batch["sen1_input_ids"].to(self.device)
        sen1_attention_mask = batch["sen1_attention_mask"].to(self.device)
        sen1_logits = self.model(sen1_input_ids, sen1_attention_mask)
        return sen1_logits.view(-1)

    def metrics(self, sen1_logits, sen2_logits):
        acc = (sen1_logits < sen2_logits).sum()
        total = sen1_logits.size(0)
        label = torch.tensor([1] * sen1_logits.size(0), dtype=torch.long).to(self.device)
        utils.debug("sen1.shape", sen1_logits.shape)
        utils.debug("sen2.shape", sen2_logits.shape)
        utils.debug("label.shape", label.shape)
        utils.debug("sen1", sen1_logits)
        utils.debug("sen2", sen2_logits)
        loss = self.Loss_fn(sen2_logits, sen1_logits, label)
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
    
    def clear(self):
        del self.model, self.optimizer, self.scheduler
    
    def margin_split_Loss_fn(self, sen2_logits, sen1_logits, labels, margin):
        less_mask = (margin==0).view(-1)
        less_margin_sen2 = torch.masked_select(sen2_logits.view(-1), less_mask)
        less_margin_sen1 = torch.masked_select(sen1_logits.view(-1), less_mask)
        less_margin_labels = torch.masked_select(labels.view(-1), less_mask)
        less_loss = self.Loss_fn_less(less_margin_sen2, less_margin_sen1, less_margin_labels)
        more_mask = (margin==1).view(-1)
        more_margin_sen2 = torch.masked_select(sen2_logits.view(-1), more_mask)
        more_margin_sen1 = torch.masked_select(sen1_logits.view(-1), more_mask)
        more_margin_labels = torch.masked_select(labels.view(-1), more_mask)
        more_loss = self.Loss_fn(more_margin_sen2, more_margin_sen1, more_margin_labels)
        return (less_loss * less_margin_sen2.size(0) + more_loss * more_margin_sen2.size(0)) / (less_margin_sen2.size(0) + more_margin_sen2.size(0))


class ScoreTrainer(Trainer):
    def __init__(self, args):
        super(ScoreTrainer, self).__init__(args)
    
    def get_Loss_fn(self):
        self.Loss_fn = nn.MSELoss()

    def _get_logits_single(self, batch):
        sen1_input_ids = batch["sen1_input_ids"].to(self.device)
        sen1_attention_mask = batch["sen1_attention_mask"].to(self.device)
        sen1_logits = self.model(sen1_input_ids, sen1_attention_mask)
        return sen1_logits.view(-1)

    def train(self, batch):
        labels = batch["labels"].to(self.device)
        sen1_logits = self._get_logits_single(batch)
        utils.debug("sen1_logits shape", sen1_logits.shape)
        utils.debug("labels shape", labels.shape)
        loss = self.Loss_fn(sen1_logits, labels.view(-1)).mean()
        utils.debug("loss", loss)
        loss.backward()
        self.step += 1
        if self.step % self.config.opt_step == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        return loss

    def metrics(self, sen1_logits, sen2_logits):
        acc = (sen1_logits < sen2_logits).sum()
        total = sen1_logits.size(0)
        logging.info("acc:{:.4f}%({}/{})".format(acc / total * 100, acc, total))
        return acc / total * 100