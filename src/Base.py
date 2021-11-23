import os
from config import *
import logging
import torch
import utils
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
from train import Trainer
import datetime
import csv


class Example(object):
    def __init__(self, id, sen1, sen2="", label=-1):
        self.id = id
        self.sen1 = utils.text_cleaning(sen1)
        self.sen2 = utils.text_cleaning(sen2)
        self.label = label


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, Examples, tokenizer, args):
        super(BaseDataset, self).__init__()
        self.sen1_input_ids = []
        self.sen2_input_ids = []
        self.sen1_attention_mask = []
        self.sen2_attention_mask = []
        self.labels = []
        self.tokenizer = tokenizer
        self.is_train = args.train
        self.eos_token_id = args.eos_id
        self.bos_token_id = args.bos_id
        self.build(Examples)
 
    def __getitem__(self, idx):
        return {
            "sen1_input_ids": self.sen1_input_ids[idx],
            "sen2_input_ids": self.sen2_input_ids[idx],
            "sen1_attention_mask": self.sen1_attention_mask[idx],
            "sen2_attention_mask": self.sen2_attention_mask[idx],
            "labels": self.labels[idx],
        }
    
    def __len__(self):
        return len(self.labels)
    
    def _convert(self, s):
        s_list = self.tokenizer.tokenize(s)
        while len(s_list) > 510:
            s_list.pop()
        return self.tokenizer.convert_tokens_to_ids(s_list)

    def build(self, Examples):
        for item in Examples:
            sen1_input_ids = [self.bos_token_id] + self._convert(item.sen1) + [self.eos_token_id]
            sen1_attention_mask = [1] * len(sen1_input_ids)
            sen2_input_ids = [self.bos_token_id] + self._convert(item.sen2) + [self.eos_token_id]
            sen2_attention_mask = [1] * len(sen2_input_ids)
            labels = [int(item.label)]
            self.sen1_input_ids.append(sen1_input_ids)
            self.sen1_attention_mask.append(sen1_attention_mask)
            self.sen2_input_ids.append(sen2_input_ids)
            self.sen2_attention_mask.append(sen2_attention_mask)
            self.labels.append(labels)
            

class Collection(object):
    def __init__(self, args):
        self.config = {}
        self.config["BUCKET"] = True
        self.config["FIX_LENGTH"] = args.fix_length
        self.config["PAD_ID"] = args.pad_id

    def __call__(self, batch):
        out = {
            "sen1_input_ids": [],
            "sen2_input_ids": [],
            "sen1_attention_mask": [],
            "sen2_attention_mask": [],
            "labels": [],
        }
        for mini_batch in batch:
            for k, v in mini_batch.items():
                out[k].append(v)
        sen1_input_max_pad = 0
        sen2_input_max_pad = 0
        if self.config["BUCKET"]:
            for p in out["sen1_input_ids"]:
                sen1_input_max_pad = max(sen1_input_max_pad, len(p))
            for p in out["sen2_input_ids"]:
                sen2_input_max_pad = max(sen2_input_max_pad, len(p))
        else:
            sen1_input_max_pad = self.config["FIX_LENGTH"]
            sen2_input_max_pad = self.config["FIX_LENGTH"]
        utils.debug("sen1_max_pad", sen1_input_max_pad)
        utils.debug("sen2_max_pad", sen2_input_max_pad)
        for i in range(len(batch)):
            out["sen1_input_ids"][i] = out["sen1_input_ids"][i] + [self.config["PAD_ID"]] * (sen1_input_max_pad - len(out["sen1_input_ids"][i]))
            out["sen1_attention_mask"][i] = out["sen1_attention_mask"][i] + [0] * (sen1_input_max_pad - len(out["sen1_attention_mask"][i]))
            out["sen2_input_ids"][i] = out["sen2_input_ids"][i] + [self.config["PAD_ID"]] * (sen2_input_max_pad - len(out["sen2_input_ids"][i]))
            out["sen2_attention_mask"][i] = out["sen2_attention_mask"][i] + [0] * (sen2_input_max_pad - len(out["sen2_attention_mask"][i]))
        out["sen1_input_ids"] = torch.tensor(out["sen1_input_ids"], dtype=torch.long)
        out["sen1_attention_mask"] = torch.tensor(out["sen1_attention_mask"], dtype=torch.long)
        out["sen2_input_ids"] = torch.tensor(out["sen2_input_ids"], dtype=torch.long)
        out["sen2_attention_mask"] = torch.tensor(out["sen2_attention_mask"], dtype=torch.long)
        out["labels"] = torch.tensor(out["labels"], dtype=torch.long)
        return out 


def prepare_examples(path, is_predict=False):
    data = utils.read_from_csv(path)
    Examples = []
    for item in data:
        if is_predict:
            Examples.append(Example(id=item[0], sen1=item[1]))
        else:
            Examples.append(Example(id=item[0], sen1=item[1], sen2=item[2]))
    return Examples


def predict(args):
    logging.info("Config Init")
    torch.cuda.set_device(args.local_rank if args.local_rank != -1 else 0)
    dist.init_process_group(backend='nccl')
    args.device = torch.device("cuda", args.local_rank if args.local_rank != -1 else 0)
    logging.info("Prepare Data")
    data = prepare_examples(args.test_path, True)
    BaseTester = Trainer(args)
    BaseTester.model_init()
    tokenizer = BaseTester.get_tokenizer()
    args.pad_id = tokenizer.pad_token_id
    test_dataset = BaseDataset(data, tokenizer, args)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=Collection(args))
    BaseTester.model.eval()
    predicts = []
    with torch.no_grad():
        for batch in test_iter:
            predicts.extend(BaseTester.predict(batch).tolist())
    with open(args.output_path, "w", encoding="utf-8") as f:
        csv_write = csv.writer(f)
        csv_write.writerow(["comment_id", "score"])
        for g, p in zip(data, predicts):
            csv_write.writerow([g.id, p])
    logging.info("END")


def train(args):
    logging.info("Config Init")
    torch.cuda.set_device(args.local_rank if args.local_rank != -1 else 0)
    dist.init_process_group(backend='nccl')
    args.device = torch.device("cuda", args.local_rank if args.local_rank != -1 else 0)
    logging.info("Prepare Data")
    data = prepare_examples(args.train_path)
    train_data = data[:int(len(data) * 0.8)]
    valid_data = data[int(len(data) * 0.8):]
    BaseTrainer = Trainer(args)
    BaseTrainer.model_init()
    tokenizer = BaseTrainer.get_tokenizer()
    args.pad_id = tokenizer.pad_token_id
    args.eos_id = tokenizer.eos_token_id
    args.bos_id = tokenizer.bos_token_id
    logging.info(f"pad_id:{args.pad_id}. eos_id:{args.eos_id}. bos_id:{args.bos_id}")
    train_dataset = BaseDataset(train_data, tokenizer, args)
    valid_dataset = BaseDataset(valid_data, tokenizer, args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    valid_sampler = utils.SequentialDistributedSampler(valid_dataset, args.batch_size)
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=Collection(args))
    valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler, collate_fn=Collection(args))
    # test_iter = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=valid_sampler, collate_fn=Collection(args))
    logging.info(f"Train Size: {len(train_iter)}")
    BaseTrainer.optimizer_init(len(train_iter))
    BaseTrainer.model.train()
    for step in range(args.epoch):
        logging.info(f"Start Training Epoch {step + 1}")
        if args.local_rank != -1:
            train_iter.sampler.set_epoch(step)
        for batch in train_iter:
            BaseTrainer.train(batch)
            if BaseTrainer.step % BaseTrainer.config.eval_step == 0:
                BaseTrainer.eval(valid_iter, len(valid_dataset))
        BaseTrainer.eval(valid_iter, len(valid_dataset))


if __name__ == "__main__":
    args = Baseconfig()
    logging.getLogger().setLevel(logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    utils.set_seed(959794)
    if args.train:
        args.model_save = '/'.join([args.model_save, utils.d2s(datetime.datetime.now(), time=True)])
        train(args)
    if args.predict:
        predict(args)