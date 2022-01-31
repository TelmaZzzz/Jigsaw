import logging
import random
from sklearn.model_selection import KFold, GroupKFold
import csv
import utils
import pandas as pd
import numpy as np
from tqdm import tqdm
import eda
import copy


class Example(object):
    def __init__(self, id, sen1, sen2="", label=1, margin=1):
        self.id = id
        # self.sen1 = utils.sample_clean(sen1)
        # self.sen2 = utils.sample_clean(sen2)
        self.sen1 = sen1
        self.sen2 = sen2
        self.label = label
        self.margin = margin


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


def prepare_examples(path, is_predict=False, margin_tag=False):
    logging.info(f"Load Data From {path}.")
    data = utils.read_from_csv(path)
    Examples = []
    for item in data:
        if is_predict:
            Examples.append(Example(id=item[0], sen1=item[1]))
        else:
            if margin_tag:
                Examples.append(Example(id=item[0], sen1=item[1], sen2=item[2], margin=item[3]))
            else:    
                Examples.append(Example(id=item[0], sen1=item[1], sen2=item[2]))
    return Examples


def prepare_examples_ruddit(path, is_predict=False):
    logging.info(f"Load Data From {path}.")
    data = utils.read_from_csv(path)
    Examples = []
    for item in data:
        if is_predict:
            Examples.append(Example(id=item[0], sen1=item[1]))
        else:
            Examples.append(Example(id=item[0], sen1=item[1], label=item[2]))
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


def get_group_unionfind_v2(train):
    unique_text = set(train['less_toxic']) | set(train['more_toxic'])
    text2num = {text: i for i, text in enumerate(unique_text)}
    num2text = {num: text for text, num in text2num.items()}
    train['num_less_toxic'] = train['less_toxic'].map(text2num)
    train['num_more_toxic'] = train['more_toxic'].map(text2num)
    cnt = 0
    vis = dict()
    LEN = len(text2num) + 10
    group = []
    for seq1, seq2 in train[['num_less_toxic', 'num_more_toxic']].to_numpy():
        if vis.get(seq1 * LEN + seq2, 0) == 0:
            cnt += 1
            vis[seq1 * LEN + seq2] = cnt
            vis[seq2 * LEN + seq1] = cnt
        group.append(vis[seq1 * LEN + seq2])
    train["group"] = np.array(group)
    train = train.drop(columns=['num_less_toxic', 'num_more_toxic'])
    return train


def build_KFold(args):
    data = prepare_examples(args.train_path)
    random.shuffle(data)
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


def find_pos(y, lr_list):
    for idx, (l, r) in enumerate(lr_list):
        if l<=y and y<r:
            return idx


def build_jigsaw_score(args):
    df = pd.read_csv(args.train_path)
    cat_mtpl = {'obscene': 0.16, 'toxic': 0.32, 'threat': 1.5, 
            'insult': 0.64, 'severe_toxic': 0.64, 'identity_hate': 1.5}
    for cat in cat_mtpl:
        df[cat] = df[cat] * cat_mtpl[cat]
    df['y'] = (df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) ).astype(float)
    df = df[['comment_text', 'y']].rename(columns={'comment_text': 'text'})
    df_pos = df[df["y"] == 0]
    df_neg = df[df["y"] > 0]
    for i in range(7):
        df = pd.concat([df_pos.sample(int(len(df_neg) * 0.75), random_state=959+i), df_neg.sample(int(len(df_neg) * 0.75), random_state=959+i)])
    # num = len(df_pos) // len(df_neg)
    # for _ in range(num):
    #     df_pos = pd.concat([df_pos, copy.deepcopy(df_neg)])
        logging.info(f"df len: {len(df)}")
        df["y"] = (df["y"] - df["y"].min()) / (df["y"].max() - df["y"].min())
        df.to_csv(args.output_path+f"_fold_{i}.csv", index=True)


def log_len(name, df):
    logging.info(f"{name} len: {len(df)}")


def make_pair(df1, df2, num1, num2):
    res = []
    sen1_list = df1["comment_text"].tolist()
    sen2_list = df2["comment_text"].tolist()
    worker_id = 0
    for sen1 in sen1_list:
        T = num1
        while T > 0:
            T -= 1
            idx = random.randint(0, len(sen2_list)-1)
            res.append(Example(id=worker_id, sen1=sen1, sen2=sen2_list[idx]))
            worker_id += 1
    for sen2 in sen2_list:
        T = num2
        while T > 0:
            T -= 1
            idx = random.randint(0, len(sen1_list)-1)
            res.append(Example(id=worker_id, sen1=sen1_list[idx], sen2=sen2))
            worker_id += 1
    return res


def build_jigsaw_cls(args):
    df = pd.read_csv(args.train_path)
    df["score"] = (df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) ).astype(int)
    df_pos = df[df["score"] == 0]
    df_neg = df[df["score"] > 0]
    df_toxic = df[df["toxic"] == 1]
    df_toxic = df_toxic[df_toxic["severe_toxic"] == 0]
    df_severe_toxic = df[df["severe_toxic"] == 1]
    df_toxic["score"] = (df_toxic[['obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) ).astype(int)
    df_severe_toxic["score"] = (df_severe_toxic[['obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) ).astype(int)
    df_toxic_pos = df_toxic[df_toxic["score"] == 0]
    df_toxic_neg = df_toxic[df_toxic["score"] >1 ]
    df_severe_toxic_pos = df_severe_toxic[df_severe_toxic["score"] == 0]
    df_severe_toxic_neg = df_severe_toxic[df_severe_toxic["score"] >1 ]
    data = []
    worker_id = 0
    log_len("df_pos", df_pos)
    log_len("df_neg", df_neg)
    data.extend(make_pair(df_pos, df_neg, 1, 3))
    data.extend(make_pair(df_toxic_pos, df_severe_toxic, 9, 9))
    data.extend(make_pair(df_toxic_pos, df_toxic_neg, 6, 6))
    # log_len("df_toxic", df_toxic)
    # log_len("df_severe_toxic", df_severe_toxic)
    # data.extend(make_pair(df_toxic, df_severe_toxic, 6, 18))
    # log_len("df_toxic_pos", df_toxic_pos)
    # log_len("df_toxic_neg", df_toxic_neg)
    # data.extend(make_pair(df_toxic_pos, df_toxic_neg, 5, 5))
    # log_len("df_severe_toxic_pos", df_severe_toxic_pos)
    # log_len("df_severe_toxic_neg", df_severe_toxic_neg)
    # data.extend(make_pair(df_severe_toxic_pos, df_severe_toxic_neg, 5, 5))

    random.shuffle(data)
    for item in data[:10]:
        logging.info(f"sen1:{item.sen1}.  sen2:{item.sen2}")
    utils.draw(args.output_path, data)


def build_groupKFold(args):
    data = pd.read_csv(args.train_path)
    # data = get_group_unionfind(data)
    data = get_group_unionfind_v2(data)
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
        random.shuffle(train_data)
        utils.draw(args.output_path + f"_train_fold_{fold}.csv", train_data)
        utils.draw(args.output_path + f"_valid_fold_{fold}.csv", valid_data)


def build_ruddit(args):
    df = pd.read_csv(args.train_path)
    df_ = df[['comment_id', 'txt', 'offensiveness_score']]
    df_ = df_.rename(columns={'comment_id': 'worker_id', 'txt': 'text', 'offensiveness_score':'y'})
    df_ = df_.loc[(df_.text != "[deleted]") & (df_.text != "[removed]")]
    df_['y'] = (df_['y'] - df_.y.min()) / (df_.y.max() - df_.y.min())
    df_['text'] = df_['text'].str.replace("> ", "")
    df_.to_csv(args.output_path, index=False)


def data_argumentation(args):
    df = pd.read_csv(args.train_path)
    data = []
    worker_id = 0
    for sen1, sen2 in zip(df["sen1"], df["sen2"]):
        da_sen1 = eda.eda(sen1, alpha_sr=0.1, alpha_ri=0, alpha_rs=0, p_rd=0, num_aug=3)
        da_sen2 = eda.eda(sen2, alpha_sr=0.05, alpha_ri=0, alpha_rs=0, p_rd=0, num_aug=3)
        # da_sen1.append(sen1)
        # da_sen2.append(sen2)
        for s1 in da_sen1:
            for s2 in da_sen2:
                data.append(Example(id=worker_id, sen1=s1, sen2=s2))
                worker_id += 1
    random.shuffle(data)
    for item in data[:10]:
        logging.info(f"sen1:{item.sen1}.  sen2:{item.sen2}")
    utils.draw(args.output_path, data)


def margin_split(args):
    for fold in range(args.fold):
        train = pd.read_csv(args.train_path + f"_fold_{fold}.csv")
        logging.info(f"train_size: {len(train)}")
        unique_text = set(train['sen1']) | set(train['sen2'])
        text2num = {text: i for i, text in enumerate(unique_text)}
        num2text = {num: text for text, num in text2num.items()}
        train['num_less_toxic'] = train['sen1'].map(text2num)
        train['num_more_toxic'] = train['sen2'].map(text2num)
        LEN = len(text2num) + 10
        vis = dict()
        count = dict()
        data = []
        for num_less, num_more in zip(train["num_less_toxic"], train["num_more_toxic"]):
            if count.get(LEN * num_less + num_more, None) is None:
                count[LEN * num_less + num_more] = 0
            count[LEN * num_less + num_more] += 1
        NUM = {0: 0, 1: 0, 2: 0, 3: 0}
        for num_less, num_more, less_toxic, more_toxic in zip(train["num_less_toxic"], train["num_more_toxic"], train["sen1"], train["sen2"]):
            if vis.get(LEN * num_less + num_more, None) is None:
                less = count.get(LEN * num_less + num_more, 0)
                more = count.get(LEN * num_more + num_less, 0)
                if less < more:
                    data.append(Example(id=0, sen1=more_toxic, sen2=less_toxic, margin=1 if less == 0 else 0))
                else:
                    data.append(Example(id=0, sen1=less_toxic, sen2=more_toxic, margin=1 if more == 0 else 0))
                vis[LEN * num_less + num_more] = 1
                vis[LEN * num_more + num_less] = 1
        random.shuffle(data)
        utils.draw(args.output_path + f"_fold_{fold}.csv", data, margin_tag=True)


def find_pos(score, score_list):
    l = 0
    r = len(score_list) - 1
    while l <= r:
        mi = (l+r) // 2
        if score_list[mi] >= score:
            r = mi-1
        else:
            l = mi+1
    return l


def get_id(dn, up, mode):
    if mode == "sqrt_up":
        rg = (up - dn + 1) * (up - dn + 1) - 1
        id = int(random.randint(1, rg)**(0.5))
        return up - min(id, up - dn)
    elif mode == "sqrt_dn":
        rg = (up - dn + 1) * (up - dn + 1) - 1
        id = int(random.randint(1, rg)**(0.5))
        return dn + min(id, up - dn)
    else:
        return random.randint(dn, up)


def build_submit(args):
    df = pd.read_csv(args.train_path)
    df_test = pd.read_csv(args.test_path)
    df = df.sort_values(by="score")
    score_list = df["score"].tolist()
    text_list = df["text"].tolist()
    data = []
    Time_1 = 3
    Time_2 = 1
    mp = {i:0 for i in range(12)}
    for score in score_list:
        mp[int(score*10)] += 1
    logging.info(mp)
    for less_toxic, more_toxic in zip(df_test["less_toxic"], df_test["more_toxic"]):
        data.append(Example(id=0, sen1=less_toxic, sen2=more_toxic))
    a, b = 0, 0
    for text, score in zip(df["text"], df["score"]):
        dn = find_pos(score + 0.25, score_list)
        mi = find_pos(score + 0.15, score_list)
        up = len(score_list) - 1
        # logging.info(f"dn: {dn}, mi: {mi}, up: {up}")
        if up - dn > 100:
            T = Time_1
            a += 1
            while T:
                T -= 1
                # idx = random.randint(dn, up)
                idx = get_id(dn, up, "sqrt_up")
                data.extend([Example(id=0, sen1=text, sen2=text_list[idx])] * 3)
        if dn - mi > 100:
            T = Time_2
            b += 1
            while T:
                T -= 1
                # idx = random.randint(mi, dn)
                idx = get_id(mi, dn, "sqrt_up")
                data.extend([Example(id=0, sen1=text, sen2=text_list[idx])] * 2)
                data.append(Example(id=0, sen1=text_list[idx], sen2=text))
        dn = 0
        up = find_pos(score - 0.25, score_list)
        mi = find_pos(score - 0.15, score_list)
        # logging.info(f"-----dn: {dn}, mi: {mi}, up: {up}")
        if up - dn > 100:
            T = Time_1
            a += 1
            while T:
                T -= 1
                # idx = random.randint(dn, up)
                idx = get_id(dn, up, "sqrt_up")
                data.extend([Example(id=0, sen1=text_list[idx], sen2=text)] * 3)
        if mi - up > 100:
            T = Time_2
            b += 1
            while T:
                T -= 1
                # idx = random.randint(up, mi)
                idx = get_id(up, mi, "sqrt_up")
                data.extend([Example(id=0, sen1=text_list[idx], sen2=text)] * 2)
                data.append(Example(id=0, sen1=text, sen2=text_list[idx]))
        
        # dn = find_pos(score - 0.15, score_list)
        # up = find_pos(score + 0.15, score_list)
        # if up - dn > 100:
        #     T = Time_2
        #     while T:
        #         T -= 1
        #         idx = random.randint(dn, up)
        #         data.append(Example(id=0, sen1=text, sen2=text_list[idx]))
        #         data.append(Example(id=0, sen1=text_list[idx], sen2=text))
    logging.info(f"a: {a}, b: {b}")
    random.shuffle(data)
    logging.info(f"Write path: {args.output_path}. data length: {len(data)}")
    with open(args.output_path, "w", encoding="utf-8") as f:
        csv_write = csv.writer(f)
        csv_write.writerow(["worker", "less_toxic", "more_toxic"])
        for item in data:
            csv_write.writerow([item.id, item.sen1, item.sen2])


def build(args):
    if args.build_type == "kfold":
        build_KFold(args)
    elif args.build_type == "unique":
        build_unique(args)
    elif args.build_type == "jigsaw_cls":
        build_jigsaw_cls(args)
    elif args.build_type == "groupkfold":
        build_groupKFold(args)
    elif args.build_type == "ruddit":
        build_ruddit(args)
    elif args.build_type == "data_argumentation":
        data_argumentation(args)
    elif args.build_type == "jigsaw_score":
        build_jigsaw_score(args)
    elif args.build_type == "margin_split":
        margin_split(args)
    elif args.build_type == "submit":
        build_submit(args)


if __name__ == "__main__":
    pass
