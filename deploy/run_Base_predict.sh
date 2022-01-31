#!/bin/sh
# source ~/.bashrc
# source activate telma
export PYTHONPATH="$HOME/opt/tiger/jigsaw"
ROOT="$HOME/opt/tiger/jigsaw"
MODEL="Base"
TOKENIZER="roberta-base"
PRETRAIN="roberta-base"
TEST_PATH="$ROOT/data/validation_data.csv"

# python -m torch.distributed.launch --nproc_per_node 1 ../src/Base.py \
python ../src/Base.py \
--predict \
--test_path="$TEST_PATH" \
--tokenizer_path="$TOKENIZER" \
--pretrain_path="$PRETRAIN" \
--model_load="$ROOT/model/Base/2021_12_09_22_51_fold_0_score_70.0213.pkl" \
--output_path="$ROOT/output/rank_1.csv" \
--batch_size=32 \
> ../log/Base_predict.log 2>&1 &