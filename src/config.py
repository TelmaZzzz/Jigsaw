import argparse


def Baseconfig():
    parser = argparse.ArgumentParser()
    
    # OS config
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--valid_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--model_save", type=str)
    # Training config
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--continue_epoch", type=int, default=7)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--l_model", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--fix_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--valid_batch_size", type=int, default=16)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--pretrain_path", type=str)
    parser.add_argument("--opt_step", type=int, default=1)
    parser.add_argument("--eval_step", type=int, default=100)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--model_load", type=str, default=None)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--unfreeze", type=int, default=-1)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--Tmax", type=int, default=500)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    # Predict config
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--output_path", type=str)
    # Build config
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--build_type", type=str, default="kfold")
    parser.add_argument("--fold", type=int, default=5)
    args = parser.parse_args()
    return args