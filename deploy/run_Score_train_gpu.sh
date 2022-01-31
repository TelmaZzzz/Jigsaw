#!/bin/sh
source ~/.bashrc
source activate telma
export PYTHONPATH="$HOME/opt/tiger/jigsaw"
ROOT="$HOME/opt/tiger/jigsaw"
MODEL="Score"
# TOKENIZER="bert-base-uncased"
# PRETRAIN="bert-base-uncased"
# TOKENIZER="bert-base-cased"
# PRETRAIN="bert-base-cased"
# TOKENIZER="roberta-base"
# PRETRAIN="roberta-base"
TOKENIZER="unitary/unbiased-toxic-roberta"
PRETRAIN="unitary/unbiased-toxic-roberta"
# TOKENIZER="roberta-large"
# PRETRAIN="roberta-large"
# TRAIN_PATH="$ROOT/data/jigsaw_cls/ranking_fold_0.csv"
# VALID_PATH="$ROOT/data/validation_data.csv"
TRAIN_PATH="$ROOT/data/tmp/submission_all.csv"
VALID_PATH="$ROOT/data/validation_data.csv"
# python ../src/Base.py \
python -m torch.distributed.launch --nproc_per_node 1 --master_port 11794 ../src/Score.py \
--train \
--train_path="$TRAIN_PATH" \
--valid_path="$VALID_PATH" \
--tokenizer_path="$TOKENIZER" \
--pretrain_path="$PRETRAIN" \
--model_save="$ROOT/model/$MODEL" \
--lr=1e-4 \
--min_lr=1e-7 \
--Tmax=2000 \
--batch_size=48 \
--valid_batch_size=64 \
--epoch=3 \
--opt_step=1 \
--dropout=0.2 \
--l_model=768 \
--eval_step=1000 \
--fold=1 \
--fix_length=128 \
--name="roberta_v2" \
# --scheduler="get_cosine_with_hard_restarts_schedule_with_warmup" \
# > ../log/Score.log 2>&1 &
