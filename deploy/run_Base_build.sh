#!/bin/sh
# source ~/.bashrc
# source activate telma
export PYTHONPATH="$HOME/opt/tiger/jigsaw"
ROOT="$HOME/opt/tiger/jigsaw"
MODEL="Base"
TRAIN_PATH="$ROOT/data/validation_data.csv"
TYPE="groupkfold"

python ../src/Base.py \
--build \
--build_type="$TYPE" \
--train_path="$TRAIN_PATH" \
--output_path="$ROOT/data/official/group" \
> ../log/Base_build.log 2>&1 &