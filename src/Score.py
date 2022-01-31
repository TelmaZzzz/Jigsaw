import os
from config import *
import logging
import torch
import utils
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
from train import Trainer, ScoreTrainer
import datetime
import csv
import DatasetPrepare as DP
from Base import BaseDataset, Collection


class ScoreDataset(torch.utils.data.Dataset):
    def __init__(self, Examples):
        super(ScoreDataset, self).__init__()
        self.sen1_text = []
        self.labels = []
        self.build(Examples)
 
    def __getitem__(self, idx):
        return {
            "sen1_text": self.sen1_text[idx],
            "labels": self.labels[idx],
        }
    
    def __len__(self):
        return len(self.sen1_text)

    def build(self, Examples):
        for item in Examples:
            self.sen1_text.append(item.sen1)
            self.labels.append([float(item.label)])


class ScoreCollection(object):
    def __init__(self, args, tokenizer):
        self.config = {}
        self.config["BUCKET"] = True
        self.config["FIX_LENGTH"] = args.fix_length
        self.tokenizer = tokenizer

    def __call__(self, batch):
        batch_in = {
            "sen1_text": [],
            "labels": [],
        }
        out = {
            "sen1_input_ids": [],
            "sen1_mask_attention": [],
            "labels": [],
        }
        for mini_batch in batch:
            for k, v in mini_batch.items():
                batch_in[k].append(v)
        sen1_encoding = self.tokenizer(batch_in["sen1_text"], padding="max_length", max_length=self.config["FIX_LENGTH"], truncation=True)
        out["sen1_input_ids"] = sen1_encoding.input_ids
        out["sen1_attention_mask"] = sen1_encoding.attention_mask
        out["labels"] = batch_in["labels"]
        out["sen1_input_ids"] = torch.tensor(out["sen1_input_ids"], dtype=torch.long)
        out["sen1_attention_mask"] = torch.tensor(out["sen1_attention_mask"], dtype=torch.long)
        out["labels"] = torch.tensor(out["labels"], dtype=torch.float)
        return out 


def predict(args):
    logging.info("Config Init")
    torch.cuda.set_device(args.local_rank if args.local_rank != -1 else 0)
    # dist.init_process_group(backend='nccl')
    args.device = torch.device("cuda", args.local_rank if args.local_rank != -1 else 0)
    logging.info("Prepare Data")
    data = DP.prepare_examples(args.test_path)
    BaseTester = Trainer(args)
    BaseTester.model_init()
    tokenizer = BaseTester.get_tokenizer()
    test_dataset = BaseDataset(data)
    # test_sampler = utils.SequentialDistributedSampler(test_dataset, args.valid_batch_size)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=Collection(args, tokenizer))
    BaseTester.model.eval()
    BaseTester.eval_(test_iter)
    # predicts = []
    # with torch.no_grad():
    #     for batch in test_iter:
    #         predicts.extend(BaseTester.predict(batch).tolist())
    # with open(args.output_path, "w", encoding="utf-8") as f:
    #     csv_write = csv.writer(f)
    #     csv_write.writerow(["comment_id", "score"])
    #     for g, p in zip(data, predicts):
    #         csv_write.writerow([g.id, p])
    logging.info("END")


def train(args):
    logging.info("Config Init")
    torch.cuda.set_device(args.local_rank if args.local_rank != -1 else 0)
    dist.init_process_group(backend='nccl')
    args.device = torch.device("cuda", args.local_rank if args.local_rank != -1 else 0)
    logging.info("Prepare Data")
    PREFIX = args.model_save
    for step in range(args.fold):
        args.model_save = PREFIX + f"_fold_{step}"
        if args.train_path.endswith(".csv"):
            train_data = DP.prepare_examples_ruddit(args.train_path)
        else:
            train_data = DP.prepare_examples_ruddit(args.train_path + f"_fold_{step}.csv")
        if args.valid_path.endswith(".csv"):
            valid_data = DP.prepare_examples(args.valid_path)
        else:
            valid_data = DP.prepare_examples(args.valid_path + f"_fold_{step}.csv")
        # random.shuffle(train_data)
        BaseTrainer = ScoreTrainer(args)
        BaseTrainer.model_init()
        tokenizer = BaseTrainer.get_tokenizer()
        train_dataset = ScoreDataset(train_data)
        valid_dataset = BaseDataset(valid_data)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        valid_sampler = utils.SequentialDistributedSampler(valid_dataset, args.valid_batch_size)
        train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=ScoreCollection(args, tokenizer))
        valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=args.valid_batch_size, sampler=valid_sampler, collate_fn=Collection(args, tokenizer))
        # test_iter = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=valid_sampler, collate_fn=Collection(args))
        logging.info(f"Train Size: {len(train_iter)}")
        BaseTrainer.optimizer_init(len(train_iter))
        # BaseTrainer.log_parameters()
        if args.freeze:
            BaseTrainer.freeze()
        else:
            # BaseTrainer.freeze()
            BaseTrainer.unfreeze(args.unfreeze)
        # BaseTrainer.log_parameters()
        BaseTrainer.model.train()
        for step in range(args.epoch):
            logging.info(f"Start Training Epoch {step + 1}")
            # if step + 1 >= args.continue_epoch:
            #     continue
            if args.local_rank != -1:
                train_iter.sampler.set_epoch(step)
            loss_mean = 0
            for batch in train_iter:
                loss = BaseTrainer.train(batch)
                loss_mean += loss
                if BaseTrainer.step % BaseTrainer.config.eval_step == 0:
                    BaseTrainer.eval(valid_iter, len(valid_dataset))
            logging.info(f"train loss:{loss_mean / len(train_iter)}")
            loss_mean = 0
            BaseTrainer.eval(valid_iter, len(valid_dataset))


if __name__ == "__main__":
    args = Baseconfig()
    logging.getLogger().setLevel(logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    utils.set_seed(args.seed + args.local_rank)
    if args.train:
        args.model_save = '/'.join([args.model_save, utils.d2s(datetime.datetime.now(), time=True)])
        logging.info(f"mode_save:{args.model_save}")
        train(args)
    if args.predict:
        predict(args)
