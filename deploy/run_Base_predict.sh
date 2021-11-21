#!/bin/sh
# source ~/.bashrc
# source activate telma
export PYTHONPATH="$HOME/opt/tiger/jigsaw"
ROOT="$HOME/opt/tiger/jigsaw"
MODEL="Base"
TOKENIZER="bert-base-cased"
TEST_PATH="$ROOT/data/comments_to_score.csv"

# python -m torch.distributed.launch --nproc_per_node 1 ../src/OrderBase.py \
python ../src/Base.py \
--predict \
--test_path="$TEST_PATH" \
--tokenizer_path="$TOKENIZER" \
--model_load="$ROOT/model/Base/2021_11_20_23_27_score_68.6981.pkl" \
--output_path="$ROOT/output/rank_1.csv" \
--batch_size=6 \
> ../log/Base_predict.log 2>&1 &