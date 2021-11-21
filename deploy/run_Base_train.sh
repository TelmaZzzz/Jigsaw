#!/bin/sh
# source ~/.bashrc
# source activate telma
export PYTHONPATH="$HOME/opt/tiger/jigsaw"
ROOT="$HOME/opt/tiger/jigsaw"
MODEL="Base"
TOKENIZER="bert-base-cased"
PRETRAIN="bert-base-cased"
TRAIN_PATH="$ROOT/data/validation_data.csv"

# python ../src/Base.py \
python -m torch.distributed.launch --nproc_per_node 5 ../src/Base.py \
--train \
--train_path="$TRAIN_PATH" \
--tokenizer_path="$TOKENIZER" \
--pretrain_path="$PRETRAIN" \
--model_save="$ROOT/model/$MODEL" \
--lr=0.00003 \
--batch_size=5 \
--epoch=15 \
--opt_step=3 \
--l_model=768 \
--eval_step=400 \
> ../log/Base.log 2>&1 &