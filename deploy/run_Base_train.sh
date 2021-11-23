#!/bin/sh
# source ~/.bashrc
# source activate telma
export PYTHONPATH="$HOME/opt/tiger/jigsaw"
ROOT="$HOME/opt/tiger/jigsaw"
MODEL="Base"
# TOKENIZER="bert-base-uncased"
# PRETRAIN="bert-base-uncased"
# TOKENIZER="bert-base-cased"
# PRETRAIN="bert-base-cased"
# TOKENIZER="roberta-base"
# PRETRAIN="roberta-base"
TOKENIZER="roberta-large"
PRETRAIN="roberta-large"
TRAIN_PATH="$ROOT/data/validation_data.csv"

# python ../src/Base.py \
python -m torch.distributed.launch --nproc_per_node 1 ../src/Base.py \
--train \
--train_path="$TRAIN_PATH" \
--tokenizer_path="$TOKENIZER" \
--pretrain_path="$PRETRAIN" \
--model_save="$ROOT/model/$MODEL" \
--lr=0.000003 \
--batch_size=3 \
--epoch=8 \
--opt_step=1 \
--dropout=0.5 \
--l_model=1024 \
--eval_step=200 \
> ../log/Base.log 2>&1 &