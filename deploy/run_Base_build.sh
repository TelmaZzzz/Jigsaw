#!/bin/sh
# source ~/.bashrc
# source activate telma

# --------------------------
# Type Define:
# kfold
# unique
# jigsaw_cls
# groupkfold
# ruddit
# data_argumentation
# jigsaw_score
# margin_split
# submit
# --------------------------
export PYTHONPATH="$HOME/opt/tiger/jigsaw"
ROOT="$HOME/opt/tiger/jigsaw"
TRAIN_PATH="$ROOT/data/group/group_v14.csv"
TYPE="groupkfold"

python ../src/Base.py \
--build \
--build_type="$TYPE" \
--test_path="$ROOT/data/validation_data.csv" \
--train_path="$TRAIN_PATH" \
--output_path="$ROOT/data/group/group_v14" \
--seed=19980917 \
--fold=5 \
> ../log/Base_build.log 2>&1 &