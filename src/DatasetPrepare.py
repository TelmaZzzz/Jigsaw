import logging
import random
from sklearn.model_selection import KFold, GroupKFold
import csv
import utils
import pandas as pd
import tqdm


class Example(object):
    def __init__(self, id, sen1, sen2="", label=-1):
        self.id = id
        self.sen1 = utils.text_cleaning(sen1)
        self.sen2 = utils.text_cleaning(sen2)
        self.label = label


class UnionFind():
    def __init__(self, n):
        self.n = n
        self.parents = [-1] * n

    def find(self, x):
        if self.parents[x] < 0:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return
        if self.parents[x] > self.parents[y]:
            x, y = y, x
        self.parents[x] += self.parents[y]
        self.parents[y] = x


def prepare_examples(path, is_predict=False):
    logging.info(f"Load Data From {path}.")
    data = utils.read_from_csv(path)
    Examples = []
    for item in data:
        if is_predict:
            Examples.append(Example(id=item[0], sen1=item[1]))
        else:
            Examples.append(Example(id=item[0], sen1=item[1], sen2=item[2]))
    return Examples


def get_group_unionfind(train):
    unique_text = set(train['less_toxic']) | set(train['more_toxic'])
    text2num = {text: i for i, text in enumerate(unique_text)}
    num2text = {num: text for text, num in text2num.items()}
    train['num_less_toxic'] = train['less_toxic'].map(text2num)
    train['num_more_toxic'] = train['more_toxic'].map(text2num)

    uf = UnionFind(len(unique_text))
    for seq1, seq2 in train[['num_less_toxic', 'num_more_toxic']].to_numpy():
        uf.union(seq1, seq2)

    text2group = {num2text[i]: uf.find(i) for i in range(len(unique_text))}
    train['group'] = train['less_toxic'].map(text2group)
    train = train.drop(columns=['num_less_toxic', 'num_more_toxic'])
    return train


def build_KFold(args):
    data = prepare_examples(args.train_path)
    logging.info(f"data len: {len(data)}")
    kfl = KFold(n_splits=args.fold, shuffle=False)
    xtrain = list(range(len(data)))
    for step, (train_idx, valid_idx) in enumerate(kfl.split(xtrain)):
        train_data = [data[idx] for idx in train_idx]
        valid_data = [data[idx] for idx in valid_idx]
        utils.draw(args.output_path + f"_train_fold_{step}.csv", train_data)
        utils.draw(args.output_path + f"_valid_fold_{step}.csv", valid_data)


def build_unique(args):
    data = prepare_examples(args.train_path)
    sen_set = set([item.sen1 for item in data] + [item.sen2 for item in data])
    sen2id = {sen:id for idx, sen in enumerate(sen_set)}
    logging.info(f"sen size:{len(sen_set)}")


def build_jigsaw_cls(args):
    df = pd.read_csv(args.train_path)
    df['sever_toxic'] = df.severe_toxic * 2
    df['y'] = (df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) ).astype(int)
    df = df[['comment_text', 'y']].rename(columns={'comment_text': 'text'})
    df = utils.clean(df, "text")
    mp = {i: [] for i in range(8)}
    for text, y in zip(df["text"], df["y"]):
        mp[y].append(text)
    for i in range(8):
        logging.info(f"y={i} have {len(mp[i])} sentences")
    data = []
    worker_id = 0
    for i in range(4):
        j = i + 3
        logging.info(f"start at {i}.")
        for sen1 in mp[i]:
            T = 4
            while T > 0:
                T -=1
                idx = random.randint(0, len(mp[j])-1)
                data.append(Example(id=worker_id, sen1=sen1, sen2=mp[j][idx]))
                worker_id += 1
    for i in range(4):
        j = i + 3
        logging.info(f"start at {i}.")
        for sen2 in mp[j]:
            T = 4
            while T > 0:
                T -=1
                idx = random.randint(0, len(mp[i])-1)
                data.append(Example(id=worker_id, sen1=mp[i][idx], sen2=sen2))
                worker_id += 1
    for item in data[:10]:
        logging.info(f"sen1:{item.sen1}.  sen2:{item.sen2}")
    utils.draw(args.output_path, data)


def build_groupKFold(args):
    data = pd.read_csv(args.train_path)
    data = get_group_unionfind(data)
    group_kfold = GroupKFold(n_splits=5)
    for fold, (_, valid_idx) in enumerate(group_kfold.split(data, data, data["group"])):
        data.loc[valid_idx, "fold"] = fold
    data["fold"] = data["fold"].astype(int)
    for fold in range(5):
        train_data = []
        valid_data = []
        for worker, less_toxic, more_toxic, fold_idx in zip(data["worker"], data["less_toxic"], data["more_toxic"], data["fold"]):
            if fold != fold_idx:
                train_data.append(Example(id=worker, sen1=less_toxic, sen2=more_toxic))
            else:
                valid_data.append(Example(id=worker, sen1=less_toxic, sen2=more_toxic))
        utils.draw(args.output_path + f"_train_fold_{fold}.csv", train_data)
        utils.draw(args.output_path + f"_valid_fold_{fold}.csv", valid_data)


def build(args):
    if args.build_type == "kfold":
        build_KFold(args)
    elif args.build_type == "unique":
        build_unique(args)
    elif args.build_type == "jigsaw_cls":
        build_jigsaw_cls(args)
    elif args.build_type == "groupkfold":
        build_groupKFold(args)


if __name__ == "__main__":
    pass
