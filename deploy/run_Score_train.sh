#!/bin/sh
# source ~/.bashrc
# source activate telma
export PYTHONPATH="$HOME/opt/tiger/jigsaw"
ROOT="$HOME/opt/tiger/jigsaw"
MODEL="Score"
# TOKENIZER="bert-base-uncased"
# PRETRAIN="bert-base-uncased"
# TOKENIZER="bert-base-cased"
# PRETRAIN="bert-base-cased"
TOKENIZER="roberta-base"
PRETRAIN="roberta-base"
# TOKENIZER="roberta-large"
# PRETRAIN="roberta-large"
# TRAIN_PATH="$ROOT/data/jigsaw_cls/ranking_fold_0.csv"
# VALID_PATH="$ROOT/data/validation_data.csv"
TRAIN_PATH="$ROOT/data/jigsaw_cls/train_score.csv"
VALID_PATH="$ROOT/data/validation_data.csv"

# python ../src/Base.py \
python -m torch.distributed.launch --nproc_per_node 1 --master_port 11959 ../src/Score.py \
--train \
--train_path="$TRAIN_PATH" \
--valid_path="$VALID_PATH" \
--tokenizer_path="$TOKENIZER" \
--pretrain_path="$PRETRAIN" \
--model_save="$ROOT/model/$MODEL" \
--lr=3e-5 \
--batch_size=32 \
--valid_batch_size=64 \
--epoch=5 \
--opt_step=1 \
--dropout=0.2 \
--l_model=768 \
--eval_step=1000 \
--fold=1 \
--fix_length=128 \
--scheduler="get_cosine_with_hard_restarts_schedule_with_warmup" \
> ../log/Score.log 2>&1 &
