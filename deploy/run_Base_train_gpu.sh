#!/bin/sh
source ~/.bashrc
source activate telma
export PYTHONPATH="$HOME/opt/tiger/jigsaw"
ROOT="$HOME/opt/tiger/jigsaw"
MODEL="Base"
# TOKENIZER="bert-base-uncased"
# PRETRAIN="bert-base-uncased"
# TOKENIZER="bert-base-cased"
# PRETRAIN="bert-base-cased"
TOKENIZER="roberta-base"
PRETRAIN="roberta-base"
# TOKENIZER="roberta-large"
# PRETRAIN="roberta-large"
# TRAIN_PATH="$ROOT/data/official_train"
# VALID_PATH="$ROOT/data/official_valid"
TRAIN_PATH="$ROOT/data/jigsaw_cls/ranking"
VALID_PATH="$ROOT/data/validation_data.csv"

# python ../src/Base.py \
python -m torch.distributed.launch --nproc_per_node 5 --master_port 11959 ../src/Base.py \
--train \
--train_path="$TRAIN_PATH" \
--valid_path="$VALID_PATH" \
--tokenizer_path="$TOKENIZER" \
--pretrain_path="$PRETRAIN" \
--model_save="$ROOT/model/$MODEL" \
--lr=1e-5 \
--min_lr=1e-7 \
--batch_size=12 \
--valid_batch_size=32 \
--epoch=3 \
--opt_step=1 \
--dropout=0.2 \
--l_model=768 \
--margin=0.5 \
--eval_step=1000 \
--Tmax=5000 \
--fold=3 \
# > ../log/Base.log 2>&1 &