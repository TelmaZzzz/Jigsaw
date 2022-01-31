#!/bin/sh
source ~/.bashrc
source activate telma
export PYTHONPATH="$HOME/opt/tiger/jigsaw"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
ROOT="$HOME/opt/tiger/jigsaw"
MODEL="Base"
# TOKENIZER="bert-base-uncased"
# PRETRAIN="bert-base-uncased"
# TOKENIZER="bert-base-cased"
# PRETRAIN="bert-base-cased"
# TOKENIZER="roberta-base"
# PRETRAIN="roberta-base"
TOKENIZER="unitary/toxic-bert"
PRETRAIN="unitary/toxic-bert"
# TOKENIZER="unitary/unbiased-toxic-roberta"
# PRETRAIN="unitary/unbiased-toxic-roberta"
# TOKENIZER="SkolkovoInstitute/roberta_toxicity_classifier"
# PRETRAIN="SkolkovoInstitute/roberta_toxicity_classifier"
# TOKENIZER="xlnet-base-cased"
# PRETRAIN="xlnet-base-cased"
# TOKENIZER="microsoft/deberta-v3-base"
# PRETRAIN="microsoft/deberta-v3-base"
# TOKENIZER="unitary/toxic-bert"
# PRETRAIN="unitary/toxic-bert"
# TOKENIZER="roberta-large"
# PRETRAIN="roberta-large"
# TOKENIZER="xlnet-large-cased"
# PRETRAIN="xlnet-large-cased"
# TRAIN_PATH="$ROOT/data/official_train"
# VALID_PATH="$ROOT/data/official_valid"
# TRAIN_PATH="$ROOT/data/official/random_noclean_da_train"
# TRAIN_PATH="$ROOT/data/official/random_noclean_train_fold_2.csv"
# VALID_PATH="$ROOT/data/official/random_noclean_valid_fold_2.csv"
# TRAIN_PATH="$ROOT/data/official/random_noclean_v2_train"
# VALID_PATH="$ROOT/data/official/random_noclean_v2_valid"

# TRAIN_PATH="$ROOT/data/group/group_v1_margin_train"
# TRAIN_PATH="$ROOT/data/group/group_v1_train"
# VALID_PATH="$ROOT/data/group/group_v1_valid"


# TRAIN_PATH="$ROOT/data/group/group_v3_train_fold_4.csv"
# VALID_PATH="$ROOT/data/group/group_v3_valid_fold_4.csv"

# TRAIN_PATH="$ROOT/data/group/group_v4"
# VALID_PATH="$ROOT/data/group/group_v4"

TRAIN_PATH="$ROOT/data/group/group_v13_train"
VALID_PATH="$ROOT/data/group/group_v13_valid"

# TRAIN_PATH="$ROOT/data/group_all_v1_train"
# VALID_PATH="$ROOT/data/group_all_v1_valid"


# TRAIN_PATH="$ROOT/data/group/group_v2_train_fold_4.csv"
# VALID_PATH="$ROOT/data/group/group_v2_valid_fold_4.csv"
# TRAIN_PATH="$ROOT/data/official/group_train"
# VALID_PATH="$ROOT/data/official/group_valid"
# TRAIN_PATH="$ROOT/data/official/group_da_train"
# TRAIN_PATH="$ROOT/data/official/group_train"
# VALID_PATH="$ROOT/data/official/group_valid"
# TRAIN_PATH="$ROOT/data/jigsaw_cls/ranking_v5_fold_1.csv"
# VALID_PATH="$ROOT/data/validation_data.csv"
# MODEL_LOAD="$ROOT/model/Score/2021_12_14_21_25_fold_0_score_68.6296.pkl"
# python -m torch.distributed.launch --nproc_per_node 8 --master_port 11959 ../src/Base.py \
python ../src/Base.py \
--train \
--train_path="$TRAIN_PATH" \
--valid_path="$VALID_PATH" \
--tokenizer_path="$TOKENIZER" \
--pretrain_path="$PRETRAIN" \
--model_save="$ROOT/model/$MODEL" \
--lr=1e-4 \
--min_lr=1e-7 \
--batch_size=32 \
--valid_batch_size=64 \
--epoch=3 \
--opt_step=1 \
--dropout=0.2 \
--l_model=768 \
--margin=0.5 \
--Tmax=2000 \
--eval_step=1000 \
--fix_length=128 \
--weight_decay=1e-6 \
--fold=5 \
--random \
--name="roberta" \
# --model_load="$MODEL_LOAD" \
# > ../log/Base.log 2>&1 &